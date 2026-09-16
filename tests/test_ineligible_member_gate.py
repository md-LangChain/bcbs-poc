"""Regression tests: an ineligible member must gate at intake and be reported."""

import json
import os
import sys
from pathlib import Path
from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agents"))

# fa_2 builds its agent at import time, which needs a key even for offline tests.
HAS_MODEL_ACCESS = bool(os.getenv("OPENAI_API_KEY"))
os.environ.setdefault("OPENAI_API_KEY", "test-placeholder")

import fa_1  # noqa: E402
import fa_2  # noqa: E402

INELIGIBLE_CASE = {
    "case_id": "PA-1024",
    "CaseID": "PA-1024",
    "ServiceRequested": "Radiation Therapy (IMRT)",
    "CPTCode": "77385",
    "ICD10Code": "C61",
    "ClinicalIndication": "Prostate cancer localized",
    "InterQualCriteriaSet": "Onc-RadTherapy",
    "RequestedUrgency": "Routine",
    "ClinicalNoteFreeText": "Localized prostate CA, intermediate risk.",
    "EstimatedCost": 21000.0,
    "error": None,
    "missing_fields": [],
    "high_cost": True,
    "eligible": False,
    "eligibility_reason": "Coverage terminated before service date",
}


class _StubIntake:
    def __init__(self, case: dict):
        self._case = case

    def invoke(self, _inputs, config=None) -> dict:
        return dict(self._case)


class _ToolState(TypedDict):
    result: str


def _run_tool_in_graph(tool, case_id: str) -> dict:
    """Run a HITL tool inside a checkpointed graph so interrupt() can surface."""

    def call_tool(_state: _ToolState) -> dict:
        return {"result": tool.invoke({"case_id": case_id})}

    builder = StateGraph(_ToolState)
    builder.add_node("call_tool", call_tool)
    builder.add_edge(START, "call_tool")
    builder.add_edge("call_tool", END)
    graph = builder.compile(checkpointer=MemorySaver())
    return graph.invoke(
        {"result": ""},
        {"configurable": {"thread_id": f"test-{case_id}"}},
    )


@pytest.fixture
def ineligible_intake(monkeypatch):
    monkeypatch.setattr(fa_1, "intake_agent", _StubIntake(INELIGIBLE_CASE))
    return INELIGIBLE_CASE


def test_run_intake_interrupts_for_ineligible_member(ineligible_intake):
    out = _run_tool_in_graph(fa_2.run_intake, "PA-1024")

    assert "__interrupt__" in out
    value = out["__interrupt__"][0].value
    assert value["type"] == "ineligible_member_hitl"
    assert value["payload"]["eligible"] is False
    assert value["payload"]["eligibility_reason"] == (
        "Coverage terminated before service date"
    )


def test_evaluate_criteria_skips_ineligible_member(ineligible_intake):
    result = json.loads(fa_2.evaluate_criteria.invoke({"case_id": "PA-1024"}))

    assert result["skipped"] is True
    assert result["reason"] == (
        "Member not eligible; resolve eligibility before clinical review"
    )
    assert result["eligible"] is False
    assert "meets_criteria" not in result


@pytest.mark.skipif(
    not HAS_MODEL_ACCESS, reason="requires model access for the agent reply"
)
def test_agent_reply_states_member_is_not_eligible(ineligible_intake):
    from langchain.agents import create_agent
    from langgraph.types import Command

    agent = create_agent(
        model="openai:gpt-4.1-mini",
        tools=[fa_2.run_intake, fa_2.evaluate_criteria],
        system_prompt=fa_2.SYSTEM_PROMPT,
        name="fa2-ineligible-regression",
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "test-ineligible-reply"}}

    paused = agent.invoke(
        {"messages": [{"role": "user", "content": "Validate case PA-1024."}]},
        config,
    )
    assert "__interrupt__" in paused

    resumed = agent.invoke(Command(resume="deny"), config)
    reply = resumed["messages"][-1].content.lower()
    assert "not eligible" in reply
