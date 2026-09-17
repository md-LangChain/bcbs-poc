"""Validation agent — intake tool + POC clinical criteria eval."""

import json
import sys
from pathlib import Path
from typing import Any, NotRequired, TypedDict
import os

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

_AGENT_DIR = Path(__file__).resolve().parent
_AGENTS_DIR = _AGENT_DIR.parent
_REPO_ROOT = _AGENTS_DIR.parent

# langgraph loads this file by path; add intake sibling for run_intake
sys.path.insert(0, str(_AGENTS_DIR / "intake"))

load_dotenv(_REPO_ROOT / ".env")
os.environ["LANGSMITH_PROJECT"] = "bcbs-validation-agent"


class IntakeInput(TypedDict):
    case_id: str


class Case(TypedDict):
    case_id: str
    CaseID: NotRequired[str | None]
    SyntheticMemberID: NotRequired[str | None]
    SyntheticDOB: NotRequired[str | None]
    PlanType: NotRequired[str | None]
    RequestingProviderNPI: NotRequired[str | None]
    ServiceRequested: NotRequired[str | None]
    CPTCode: NotRequired[str | None]
    ICD10Code: NotRequired[str | None]
    ClinicalIndication: NotRequired[str | None]
    InterQualCriteriaSet: NotRequired[str | None]
    SubmittedDate: NotRequired[str | None]
    RequestedUrgency: NotRequired[str | None]
    EstimatedCost: NotRequired[float | None]
    PriorAuthStatus: NotRequired[str | None]
    ReviewPath: NotRequired[str | None]
    ClinicalNoteFreeText: NotRequired[str | None]
    PlantedTestCondition: NotRequired[str | None]

    error: NotRequired[str | None]
    missing_fields: NotRequired[list[str]]
    high_cost: NotRequired[bool]
    eligible: NotRequired[bool]


class Fa2State(AgentState):
    """Agent messages + full FA-1 case for evaluate_criteria (no second FA-1 call)."""

    intake_case: NotRequired[dict[str, Any] | None]


class CriteriaResult(BaseModel):
    meets_criteria: bool = Field(
        description="True only if evidence clearly supports medical necessity"
    )
    borderline: bool = Field(
        description=(
            "True if clinically inconclusive (neither clear meet nor clear fail), "
            "e.g. inconclusive prior test or explicit uncertainty"
        )
    )
    rationale: str = Field(description="Short rationale; use only provided fields")


def _criteria_llm():
    return init_chat_model("openai:gpt-4.1-mini").with_structured_output(CriteriaResult)


def _intake_matches(case: dict[str, Any] | None, case_id: str) -> bool:
    if not case:
        return False
    return str(case.get("case_id") or case.get("CaseID") or "") == case_id


@tool
def run_intake(case_id: str, runtime: ToolRuntime) -> Command:
    """Run FA-1 intake/validation for a prior-auth case id (e.g. PA-1001).

    Returns JSON with case fields plus error, missing_fields, high_cost, eligible.
    Stores the full case in graph state for evaluate_criteria.
    Does not pause for HITL — that happens in evaluate_criteria.
    """
    from intake import intake_agent

    result = intake_agent.invoke(
        {"case_id": case_id},
        config={"configurable": {"thread_id": case_id}},
    )
    # Slim payload for the model; full case stays in state.intake_case.
    payload = {
        "case_id": case_id,
        "CaseID": result.get("CaseID"),
        "ServiceRequested": result.get("ServiceRequested"),
        "CPTCode": result.get("CPTCode"),
        "ICD10Code": result.get("ICD10Code"),
        "EstimatedCost": result.get("EstimatedCost"),
        "error": result.get("error"),
        "missing_fields": result.get("missing_fields"),
        "high_cost": result.get("high_cost"),
        "eligible": result.get("eligible"),
    }

    return Command(
        update={
            "intake_case": result,
            "messages": [
                ToolMessage(
                    content=json.dumps(payload, default=str),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def evaluate_criteria(case_id: str, runtime: ToolRuntime) -> str:
    """POC InterQual-style clinical criteria check for a case id.

    Call after run_intake. Uses intake_case from graph state (does not re-run FA-1).
    HITL pauses: intake validation failure, high_cost, or borderline criteria.
    Returns meets_criteria, borderline, rationale (or skipped + human_decision).
    """
    case = runtime.state.get("intake_case") if runtime.state else None
    if not _intake_matches(case, case_id):
        return json.dumps(
            {
                "case_id": case_id,
                "skipped": True,
                "reason": "No intake_case in state for this case_id; call run_intake first.",
            },
            default=str,
        )
    assert case is not None

    intake_payload = {
        "case_id": case_id,
        "CaseID": case.get("CaseID"),
        "ServiceRequested": case.get("ServiceRequested"),
        "CPTCode": case.get("CPTCode"),
        "ICD10Code": case.get("ICD10Code"),
        "EstimatedCost": case.get("EstimatedCost"),
        "error": case.get("error"),
        "missing_fields": case.get("missing_fields"),
        "high_cost": case.get("high_cost"),
        "eligible": case.get("eligible"),
    }

    if case.get("error") or case.get("missing_fields"):
        decision = interrupt(
            {
                "type": "intake_validation_failed",
                "message": (
                    "Intake found problems. Review missing fields / error, "
                    "then resume with a note (e.g. how you will fix) or cancel."
                ),
                "payload": intake_payload,
            }
        )
        return json.dumps(
            {
                **intake_payload,
                "skipped": True,
                "reason": "Case failed intake validation; fix intake first.",
                "human_decision": decision,
            },
            default=str,
        )

    if case.get("high_cost"):
        decision = interrupt(
            {
                "type": "high_cost_hitl",
                "message": (
                    "Estimated cost exceeds the high-cost threshold. "
                    "Human review required before continuing. "
                    "Resume with approve, deny, or request-more-info."
                ),
                "payload": intake_payload,
            }
        )
        intake_payload["human_decision"] = decision
        decision_text = str(decision).strip().lower()
        if decision_text in {"deny", "denied", "cancel", "cancelled", "reject"}:
            return json.dumps(
                {
                    **intake_payload,
                    "skipped": True,
                    "reason": "High-cost HITL did not approve continuing to criteria.",
                },
                default=str,
            )

    system_prompt = f"""You are a prior-auth clinical criteria reviewer (POC).
Decide meets_criteria and borderline using only these fields. Do not invent facts.

- meets_criteria=true only if evidence clearly supports medical necessity.
- borderline=true if inconclusive (neither clear meet nor clear fail).
- If clearly insufficient / denial-leaning, meets_criteria=false and borderline=false
  unless the note is explicitly uncertain/inconclusive.

ServiceRequested: {case.get('ServiceRequested')}
CPTCode: {case.get('CPTCode')}
ICD10Code: {case.get('ICD10Code')}
ClinicalIndication: {case.get('ClinicalIndication')}
InterQualCriteriaSet: {case.get('InterQualCriteriaSet')}
RequestedUrgency: {case.get('RequestedUrgency')}
ClinicalNoteFreeText: {case.get('ClinicalNoteFreeText')}
"""
    result = _criteria_llm().invoke(system_prompt)
    payload = {
        "case_id": case_id,
        "CaseID": case.get("CaseID"),
        "meets_criteria": result.meets_criteria,
        "borderline": result.borderline,
        "rationale": result.rationale,
        "high_cost": case.get("high_cost"),
    }
    if "human_decision" in intake_payload:
        payload["high_cost_human_decision"] = intake_payload["human_decision"]

    if result.borderline:
        decision = interrupt(
            {
                "type": "borderline_hitl",
                "message": (
                    "Clinical criteria are borderline/inconclusive. "
                    "Human override required. Resume with approve, deny, or more-info."
                ),
                "payload": payload,
            }
        )
        payload["human_decision"] = decision

    return json.dumps(payload, default=str)


SYSTEM_PROMPT_NAME = os.getenv("VALIDATION_SYSTEM_PROMPT", "system-prompt2")
if SYSTEM_PROMPT_NAME not in {"system-prompt1", "system-prompt2"}:
    raise ValueError(
        "VALIDATION_SYSTEM_PROMPT must be 'system-prompt1' or 'system-prompt2'"
    )
SYSTEM_PROMPT = (
    (_AGENT_DIR / f"{SYSTEM_PROMPT_NAME}.txt").read_text(encoding="utf-8").strip()
)


def create_validation_agent(
    *,
    intake_tool=run_intake,
    checkpointer=None,
    name: str = "validation",
):
    """Build validation consistently for deployment and evaluation."""
    return create_agent(
        model="openai:gpt-4.1-mini",
        tools=[intake_tool, evaluate_criteria],
        system_prompt=SYSTEM_PROMPT,
        state_schema=Fa2State,
        checkpointer=checkpointer,
        name=name,
    )


validation_agent = create_validation_agent()


if __name__ == "__main__":
    import uuid

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command as ResumeCommand

    local = create_agent(
        model="openai:gpt-4.1-mini",
        tools=[run_intake, evaluate_criteria],
        system_prompt=SYSTEM_PROMPT,
        state_schema=Fa2State,
        name="validation-local",
        checkpointer=MemorySaver(),
    )
    config = {
        "configurable": {"thread_id": f"validation-demo-{uuid.uuid4().hex[:8]}"},
        "tags": ["bcbs", "validation"],
    }
    first = local.invoke(
        {"messages": [{"role": "user", "content": "I need to validate a prior auth case."}]},
        config,
    )
    print(first["messages"][-1].content)
    second = local.invoke(
        {"messages": [{"role": "user", "content": "PA-1001"}]},
        config,
    )
    if "__interrupt__" in second:
        print("INTERRUPTED:", second["__interrupt__"])
        resumed = local.invoke(ResumeCommand(resume="approve"), config)
        print(resumed["messages"][-1].content)
    else:
        print(second["messages"][-1].content)
        print("intake_case in state:", bool(second.get("intake_case")))
