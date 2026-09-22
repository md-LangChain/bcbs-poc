"""Validation agent — intake tool + POC clinical criteria eval."""

import ast
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, NotRequired, TypedDict
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ToolCallRequest, ToolErrorMiddleware
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command, interrupt
from langsmith import traceable
from pydantic import BaseModel, Field

_AGENT_DIR = Path(__file__).resolve().parent
_AGENTS_DIR = _AGENT_DIR.parent
_REPO_ROOT = _AGENTS_DIR.parent

load_dotenv(_REPO_ROOT / ".env")
os.environ["LANGSMITH_PROJECT"] = "bcbs-validation-agent"

# Model string for init_chat_model / create_agent. Override to route through the
# LangSmith LLM Gateway, which requires provider/model form:
#   VALIDATION_MODEL=openai:openai/gpt-4.1-mini
VALIDATION_MODEL = os.getenv("VALIDATION_MODEL", "openai:gpt-4.1-mini")

INTAKE_URL = os.getenv(
    "INTAKE_URL",
    "https://intake-agent-86af57ffbd905e1db5ab112132a2e494.us.langgraph.app",
).rstrip("/")
INTAKE_GRAPH = os.getenv("INTAKE_GRAPH", "intake")
INTAKE_ASSISTANT_ID = os.getenv("INTAKE_ASSISTANT_ID")


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
    return init_chat_model(VALIDATION_MODEL).with_structured_output(CriteriaResult)


def _intake_matches(case: dict[str, Any] | None, case_id: str) -> bool:
    if not case:
        return False
    return str(case.get("case_id") or case.get("CaseID") or "") == case_id


def _post_intake_json(path: str, body: dict[str, Any]) -> Any:
    """POST JSON to the intake deployment without logging credentials."""
    request = Request(
        f"{INTAKE_URL}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "X-Api-Key": os.environ["LANGSMITH_API_KEY"],
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Intake request failed ({exc.code}): {detail}") from exc


@lru_cache(maxsize=1)
def _resolve_intake_assistant_id() -> str:
    """Resolve the deployed assistant UUID for the intake graph."""
    if INTAKE_ASSISTANT_ID:
        return INTAKE_ASSISTANT_ID

    assistants = _post_intake_json(
        "/assistants/search",
        {"graph_id": INTAKE_GRAPH, "limit": 1},
    )
    if not isinstance(assistants, list) or not assistants:
        raise RuntimeError(f"No assistant found for graph {INTAKE_GRAPH!r}")
    return str(assistants[0]["assistant_id"])


@traceable(
    name="a2a:intake",
    run_type="tool",
    tags=["a2a", "intake"],
    metadata={
        "protocol": "a2a",
        "a2a_method": "message/send",
        "target_graph": "intake",
    },
)
def _run_intake_a2a(case_id: str) -> dict[str, Any]:
    """Invoke intake through A2A and extract its structured output artifact."""
    assistant_id = _resolve_intake_assistant_id()
    response = _post_intake_json(
        f"/a2a/{assistant_id}",
        {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": str(uuid4()),
                    "parts": [{"kind": "data", "data": {"case_id": case_id}}],
                }
            },
        },
    )
    if not isinstance(response, dict):
        raise RuntimeError("A2A intake response was not a JSON object")
    if "error" in response:
        raise RuntimeError(f"A2A intake request failed: {response['error']}")

    result: dict[str, Any] = {}
    task = response.get("result", {})
    if isinstance(task, dict):
        for artifact in task.get("artifacts", []):
            for part in artifact.get("parts", []):
                if part.get("kind") == "data" and isinstance(part.get("data"), dict):
                    result.update(part["data"])
                elif part.get("kind") == "text" and isinstance(part.get("text"), str):
                    try:
                        parsed = ast.literal_eval(part["text"])
                    except (SyntaxError, ValueError):
                        continue
                    if isinstance(parsed, dict):
                        result.update(parsed)

    if not result:
        raise RuntimeError("A2A intake response contained no structured data artifact")
    return result


@tool
def run_intake(case_id: str, runtime: ToolRuntime) -> Command:
    """Call deployed FA-1 intake over A2A for a case id (e.g. PA-1001).

    Returns JSON with case fields plus error, missing_fields, high_cost, eligible.
    Stores the full case in graph state for evaluate_criteria.
    Does not pause for HITL — that happens in evaluate_criteria.
    """
    try:
        result = _run_intake_a2a(case_id)
    except GraphBubbleUp:
        raise
    except Exception as exc:  # noqa: BLE001 - intake faults are reported to the model
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=json.dumps(
                            {
                                "case_id": case_id,
                                "error": "intake_unavailable",
                                "detail": f"{type(exc).__name__}: {exc}",
                            },
                            default=str,
                        ),
                        tool_call_id=runtime.tool_call_id,
                        status="error",
                    )
                ]
            }
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


SYSTEM_PROMPT = (
    "You help providers validate BCBS prior-authorization cases.\n"
    "1) If the user has not given a case id (like PA-1001), ask for one.\n"
    "2) Call run_intake with that case id.\n"
    "3) Always call evaluate_criteria for the same case id after intake "
    "(even when high_cost is true; HITL lives there).\n"
    "4) Explain results clearly:\n"
    "   - evaluate_criteria HITL: intake validation failure, high_cost, or borderline — "
    "use human_decision + payload.\n"
    "   - Criteria: report meets_criteria, borderline, and rationale.\n"
    "Do not invent clinical values. Keep replies concise."
)


def _tool_error_content(exc: Exception, request: ToolCallRequest) -> str:
    """Report an unhandled tool failure to the model as an error ToolMessage."""
    return json.dumps(
        {
            "tool": request.tool_call.get("name"),
            "error": "tool_failed",
            "detail": type(exc).__name__,
        },
        default=str,
    )


# Without this every unhandled tool exception crashes the tools superstep before any
# ToolMessage is committed, leaving the checkpoint with unanswered tool_call_ids that
# make later turns on the thread unreplayable.
TOOL_ERROR_MIDDLEWARE = ToolErrorMiddleware(_tool_error_content)


validation_agent = create_agent(
    model=VALIDATION_MODEL,
    tools=[run_intake, evaluate_criteria],
    system_prompt=SYSTEM_PROMPT,
    state_schema=Fa2State,
    middleware=[TOOL_ERROR_MIDDLEWARE],
    name="validation",
)


if __name__ == "__main__":
    import uuid

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command as ResumeCommand

    local = create_agent(
        model=VALIDATION_MODEL,
        tools=[run_intake, evaluate_criteria],
        system_prompt=SYSTEM_PROMPT,
        state_schema=Fa2State,
        middleware=[TOOL_ERROR_MIDDLEWARE],
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
