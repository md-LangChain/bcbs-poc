"""FA-2 regression tests: criteria eval must not depend on tool-call ordering."""

import json
import os
import sys
from pathlib import Path
from typing import ClassVar

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

# fa_2 builds its agent at import time, so a model key must exist before the import
os.environ.setdefault("OPENAI_API_KEY", "test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agents"))

import fa_2

HIGH_COST_CASE = "PA-1002"


class _StubCriteriaLLM:
    """Stand-in for the structured-output criteria model."""

    def invoke(self, prompt: str) -> fa_2.CriteriaResult:
        return fa_2.CriteriaResult(
            meets_criteria=True,
            borderline=False,
            rationale="stubbed criteria decision",
        )


class _RecordingModel(GenericFakeChatModel):
    """Fake chat model that records the kwargs create_agent binds tools with."""

    bind_kwargs: ClassVar[dict] = {}

    def bind_tools(self, tools, **kwargs):
        self.bind_kwargs.update(kwargs)
        return self.bind(tools=tools, **kwargs)


def _tool_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("tools", ToolNode([fa_2.run_intake, fa_2.evaluate_criteria]))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    return builder.compile(checkpointer=MemorySaver())


def _tool_result(messages, tool_name):
    for message in messages:
        if getattr(message, "name", None) == tool_name:
            return json.loads(message.content)
    raise AssertionError(f"no {tool_name} tool message in {messages}")


def test_evaluate_criteria_in_same_batch_as_run_intake(monkeypatch):
    """evaluate_criteria still evaluates when emitted beside run_intake."""
    monkeypatch.setattr(fa_2, "_criteria_llm", _StubCriteriaLLM)

    graph = _tool_graph()
    batch = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "run_intake",
                "args": {"case_id": HIGH_COST_CASE},
                "id": "call_intake",
            },
            {
                "name": "evaluate_criteria",
                "args": {"case_id": HIGH_COST_CASE},
                "id": "call_criteria",
            },
        ],
    )
    config = {"configurable": {"thread_id": f"parallel-{HIGH_COST_CASE}"}}

    result = graph.invoke({"messages": [batch]}, config)
    if "__interrupt__" in result:
        result = graph.invoke(Command(resume="approve"), config)

    criteria = _tool_result(result["messages"], "evaluate_criteria")
    assert "skipped" not in criteria
    assert criteria["meets_criteria"] is True
    assert criteria["case_id"] == HIGH_COST_CASE


def test_agent_binds_tools_without_parallel_tool_calls():
    """The validation agent binds its tools with parallel tool calls disabled."""
    model = _RecordingModel(messages=iter([AIMessage(content="done")]))
    agent = create_agent(
        model=model,
        tools=[fa_2.run_intake, fa_2.evaluate_criteria],
        middleware=[fa_2.sequential_tool_calls],
        system_prompt=fa_2.SYSTEM_PROMPT,
    )

    agent.invoke({"messages": [{"role": "user", "content": HIGH_COST_CASE}]})

    assert model.bind_kwargs["parallel_tool_calls"] is False
