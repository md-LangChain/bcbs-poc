from pathlib import Path
import math
import sys
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

# langgraph dev loads this file by path, so siblings are not importable by default
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eligibility import lookup_eligibility
from fa_2 import Case, IntakeInput
from redact_phi import redact_clinical_note

load_dotenv()

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "BCBSRI_Synthetic_PriorAuth_Datasetv1.csv"

"""
FA-1 Intake: case_id → read CSV → redact_phi → validate → pass Case state to FA-2.
State schema inherited from fa_2.Case.
If incomplete: missing_fields = [...], error = message string.
Also sets high_cost when EstimatedCost exceeds HIGH_COST_THRESHOLD (HITL gate).
Eligibility: looked up in validate; None when the member id cannot be resolved.
"""

HIGH_COST_THRESHOLD = 10000.0

UNRESOLVED_PLAN_STATUSES = ("not_found", "invalid_member_id")

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


def _eligibility(state: Case) -> bool | None:
    """Member eligibility from the lookup; None when the member id cannot be resolved."""
    result = lookup_eligibility({"member_id": state.get("SyntheticMemberID")})
    if result["plan_status"] in UNRESOLVED_PLAN_STATUSES:
        return None
    return result["eligible"]


def validate_case(state: Case) -> dict:
    """Validate required fields; set missing_fields, error, high_cost, eligible."""
    high_cost = _is_high_cost(state)
    eligible = _eligibility(state)
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

# node to redact PHI 
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

intake_agent = builder.compile(name="fa-1-intake")


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
