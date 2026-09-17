from pathlib import Path
import math
import os
import sys
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph_sdk import get_sync_client

_AGENT_DIR = Path(__file__).resolve().parent
_AGENTS_DIR = _AGENT_DIR.parent
_REPO_ROOT = _AGENTS_DIR.parent

# langgraph loads this file by path; add sibling agent dirs + shared agents/
sys.path.insert(0, str(_AGENTS_DIR / "validation"))
sys.path.insert(0, str(_AGENTS_DIR))

from validation import Case, IntakeInput
from redact_phi import redact_clinical_note

load_dotenv(_REPO_ROOT / ".env")
os.environ["LANGSMITH_PROJECT"] = "bcbs-intake-agent"

CSV_PATH = _REPO_ROOT / "data" / "BCBSRI_Synthetic_PriorAuth_Datasetv1.csv"

"""
Intake: case_id → read CSV → redact_phi → validate → pass Case state to validation.
State schema inherited from validation.Case.
If incomplete: missing_fields = [...], error = message string.
Also sets high_cost when EstimatedCost exceeds HIGH_COST_THRESHOLD (HITL gate).
Eligibility: calls deployed eligibility agent via LangGraph SDK.
"""

HIGH_COST_THRESHOLD = 10000.0

# After redeploy, graph id is "eligibility" (was "fa_4" on the first revision).
ELIGIBILITY_URL = os.getenv(
    "ELIGIBILITY_URL",
    "https://eligibility-agent-f8686db694005ad78a24ab815f4b90a4.us.langgraph.app",
)
ELIGIBILITY_GRAPH = os.getenv("ELIGIBILITY_GRAPH", "eligibility")

REQUIRED_FIELDS = (
    "CaseID",
    "SyntheticMemberID",
    "SyntheticDOB",
    "PlanType",
    "RequestingProviderNPI",
    "ServiceRequested",
    "CPTCode",
    "ICD10Code",
    "ClinicalIndication",
    "InterQualCriteriaSet",
    "SubmittedDate",
    "RequestedUrgency",
    "EstimatedCost",
    "PriorAuthStatus",
    "ReviewPath",
    "ClinicalNoteFreeText",
    "PlantedTestCondition",
)


def read_case(state: Case) -> dict:
    """Load one case from CSV into state."""
    df = pd.read_csv(CSV_PATH)
    match = df[df["CaseID"] == state["case_id"]]

    if match.empty:
        return {"error": f"CaseID not found: {state['case_id']}"}

    row = {
        field: None if pd.isna(value) else value
        for field, value in match.iloc[0].to_dict().items()
    }
    if row.get("EstimatedCost") is not None:
        row["EstimatedCost"] = float(row["EstimatedCost"])
    row["error"] = None
    return row


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _is_high_cost(state: Case) -> bool:
    """True when EstimatedCost should trigger HITL (e.g. PA-1002)."""
    cost = state.get("EstimatedCost")
    if cost is None:
        return False
    return float(cost) > HIGH_COST_THRESHOLD


def _is_eligible(state: Case) -> bool:
    """Call deployed eligibility agent (SyntheticMemberID only)."""
    member_id = (state.get("SyntheticMemberID") or "").strip()
    if not member_id:
        return False
    client = get_sync_client(
        url=ELIGIBILITY_URL,
        api_key=os.environ["LANGSMITH_API_KEY"],
    )
    result = client.runs.wait(
        None,
        ELIGIBILITY_GRAPH,
        input={"member_id": member_id},
    )
    values = result.get("values", result) if isinstance(result, dict) else {}
    return bool(values.get("eligible"))


def validate_case(state: Case) -> dict:
    """Validate required fields; set missing_fields, error, high_cost, eligible."""
    high_cost = _is_high_cost(state)
    eligible = _is_eligible(state)
    missing = [f for f in REQUIRED_FIELDS if _is_missing(state.get(f))]
    if missing:
        return {
            "missing_fields": missing,
            "error": f"Missing required fields: {', '.join(missing)}",
            "high_cost": high_cost,
            "eligible": eligible,
        }
    return {
        "missing_fields": [],
        "error": None,
        "high_cost": high_cost,
        "eligible": eligible,
    }


def redact_phi(state: Case) -> dict:
    """After read_case: redact PHI in clinical free text."""
    original = state.get("ClinicalNoteFreeText")
    return {
        "ClinicalNoteFreeText": redact_clinical_note(original),
    }


builder = StateGraph(Case, input_schema=IntakeInput)
builder.add_node("read_case", read_case)
builder.add_node("redact_phi", redact_phi)
builder.add_node("validate_case", validate_case)

builder.add_edge(START, "read_case")
builder.add_edge("read_case", "redact_phi")
builder.add_edge("redact_phi", "validate_case")
builder.add_edge("validate_case", END)

intake_agent = builder.compile(name="intake")


if __name__ == "__main__":
    case_id = "PA-1020"  # planted SSN for redaction test
    result = intake_agent.invoke({"case_id": case_id})
    note = result.get("ClinicalNoteFreeText") or ""
    print("CaseID:", result.get("CaseID"))
    print("error:", result.get("error"))
    print("missing_fields:", result.get("missing_fields"))
    print("high_cost:", result.get("high_cost"))
    print("eligible:", result.get("eligible"))
    print("ssn_redacted:", "041-86-7735" not in note and "[REDACTED_SSN]" in note)
    print("ClinicalNoteFreeText:", note)
