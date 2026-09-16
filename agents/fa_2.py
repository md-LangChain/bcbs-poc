"""FA-2 validation agent — intake tool + POC clinical criteria eval."""

import json
import sys
from pathlib import Path
from typing import NotRequired, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

# langgraph dev loads this file by path, so siblings are not importable by default
sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

#delete comment 
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
    eligible: NotRequired[bool | None]
    eligibility_error: NotRequired[str | None]


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


@tool
def run_intake(case_id: str) -> str:
    """Run FA-1 intake/validation for a prior-auth case id (e.g. PA-1001).

    Returns JSON with case fields plus error, missing_fields, high_cost, eligible,
    and eligibility_error (set when eligibility could not be verified).
    Pauses for human input when intake reports errors, missing fields, or high cost.
    """

    #import and use intake agent 
    from fa_1 import intake_agent

    result = intake_agent.invoke(
        {"case_id": case_id},
        config={"configurable": {"thread_id": case_id}},
    )
    # Drop bulky free-text for the tool payload; agent can still see key flags.
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
        "eligibility_error": result.get("eligibility_error"),
    }

    needs_hitl = bool(
        result.get("error") or result.get("missing_fields") or result.get("high_cost")
    )
    if needs_hitl:
        if result.get("error") or result.get("missing_fields"):
            interrupt_type = "intake_validation_failed"
            message = (
                "Intake found problems. Review missing fields / error, "
                "then resume with a note (e.g. how you will fix) or cancel."
            )
        else:
            interrupt_type = "high_cost_hitl"
            message = (
                "Estimated cost exceeds the high-cost threshold. "
                "Human review required before continuing. "
                "Resume with approve, deny, or request-more-info."
            )
        decision = interrupt(
            {
                "type": interrupt_type,
                "message": message,
                "payload": payload,
            }
        )
        payload["human_decision"] = decision

    return json.dumps(payload, default=str)


@tool
def evaluate_criteria(case_id: str) -> str:
    """POC InterQual-style clinical criteria check for a case id.

    Call after a successful run_intake. Returns meets_criteria, borderline, rationale.
    Pauses for human review when borderline.
    """
    from fa_1 import intake_agent

    case = intake_agent.invoke(
        {"case_id": case_id},
        config={"configurable": {"thread_id": f"criteria-{case_id}"}},
    )
    if case.get("error") or case.get("missing_fields"):
        return json.dumps(
            {
                "case_id": case_id,
                "skipped": True,
                "reason": "Case failed intake validation; fix intake first.",
                "error": case.get("error"),
                "missing_fields": case.get("missing_fields"),
            },
            default=str,
        )

    prompt = f"""You are a prior-auth clinical criteria reviewer (POC).
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
    result = _criteria_llm().invoke(prompt)
    payload = {
        "case_id": case_id,
        "CaseID": case.get("CaseID"),
        "meets_criteria": result.meets_criteria,
        "borderline": result.borderline,
        "rationale": result.rationale,
    }

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
    "3) If intake is valid (no blocking error/missing fields), call evaluate_criteria "
    "for the same case id.\n"
    "4) Explain results clearly:\n"
    "   - Intake HITL (validation failure or high_cost): use human_decision + payload.\n"
    "   - Eligibility: when eligibility_error is set, state plainly that member "
    "eligibility could not be verified and that field validation still completed; "
    "never call the member eligible or ineligible in that case.\n"
    "   - Criteria: report meets_criteria, borderline, and rationale. "
    "If borderline HITL paused, summarize the human decision after resume.\n"
    "Do not invent clinical values. Keep replies concise."
)


fa2 = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[run_intake, evaluate_criteria],
    system_prompt=SYSTEM_PROMPT,
    name="fa2-validation-agent",
    # No custom checkpointer — langgraph dev / API provide persistence.
)


if __name__ == "__main__":
    import uuid

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    local = create_agent(
        model="openai:gpt-4.1-mini",
        tools=[run_intake, evaluate_criteria],
        system_prompt=SYSTEM_PROMPT,
        name="fa2-validation-agent-local",
        checkpointer=MemorySaver(),
    )
    config = {
        "configurable": {"thread_id": f"fa2-demo-{uuid.uuid4().hex[:8]}"},
        "tags": ["bcbs", "fa-2"],
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
        resumed = local.invoke(Command(resume="approve"), config)
        print(resumed["messages"][-1].content)
    else:
        print(second["messages"][-1].content)
