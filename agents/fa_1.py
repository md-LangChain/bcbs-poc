from pathlib import Path
import logging
import math
import os
import sys
from typing import Any

import httpx
import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph_sdk import get_sync_client

# langgraph dev loads this file by path, so siblings are not importable by default
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fa_2 import Case, IntakeInput
from redact_phi import redact_clinical_note

load_dotenv()

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "BCBSRI_Synthetic_PriorAuth_Datasetv1.csv"

"""
FA-1 Intake: case_id → read CSV → redact_phi → validate → pass Case state to FA-2.
State schema inherited from fa_2.Case.
If incomplete: missing_fields = [...], error = message string.
Also sets high_cost when EstimatedCost exceeds HIGH_COST_THRESHOLD (HITL gate).
Eligibility: looked up on the separately deployed eligibility graph. The lookup is
advisory — when it fails, eligible is None and eligibility_error explains why, so
field validation still completes.
"""

logger = logging.getLogger(__name__)

HIGH_COST_THRESHOLD = 10000.0

# Must match the graph name the eligibility deployment registers in its langgraph.json.
ELIGIBILITY_GRAPH = os.getenv("ELIGIBILITY_GRAPH", "eligibility")
ELIGIBILITY_URL = os.getenv("ELIGIBILITY_URL") or None
ELIGIBILITY_TIMEOUT_SECONDS = float(os.getenv("ELIGIBILITY_TIMEOUT_SECONDS", "30"))

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


def _eligibility_client():
    """Sync client for the separately deployed eligibility graph."""
    return get_sync_client(url=ELIGIBILITY_URL, timeout=ELIGIBILITY_TIMEOUT_SECONDS)


def _is_eligible(state: Case) -> tuple[bool | None, str | None]:
    """Look up member eligibility; returns (None, reason) when it cannot be verified."""
    member_id = state.get("SyntheticMemberID")
    if _is_missing(member_id):
        return None, "No SyntheticMemberID on the case; eligibility not checked."

    # Least-privilege: send member_id only, never clinical notes or free text.
    try:
        result = _eligibility_client().runs.wait(
            None,
            ELIGIBILITY_GRAPH,
            input={"member_id": member_id},
        )
    except (httpx.HTTPStatusError, httpx.TransportError) as exc:
        logger.warning("Eligibility lookup failed for %s: %r", member_id, exc)
        return None, f"Eligibility service call failed: {exc}"

    eligible = result.get("eligible") if isinstance(result, dict) else None
    if not isinstance(eligible, bool):
        logger.warning(
            "Eligibility lookup for %s returned no eligible flag: %r", member_id, result
        )
        return None, "Eligibility service returned no eligibility flag."
    return eligible, None


def validate_case(state: Case) -> dict:
    """Validate required fields; set missing_fields, error, high_cost, eligible."""
    high_cost = _is_high_cost(state)
    eligible, eligibility_error = _is_eligible(state)
    missing = [f for f in REQUIRED_FIELDS if _is_missing(state.get(f))]
    if missing:
        return {
            "missing_fields": missing,
            "error": f"Missing required fields: {', '.join(missing)}",
            "high_cost": high_cost,
            "eligible": eligible,
            "eligibility_error": eligibility_error,
        }
    return {
        "missing_fields": [],
        "error": None,
        "high_cost": high_cost,
        "eligible": eligible,
        "eligibility_error": eligibility_error,
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
    print("eligibility_error:", result.get("eligibility_error"))
    print("ssn_redacted:", "041-86-7735" not in note and "[REDACTED_SSN]" in note)
    print("ClinicalNoteFreeText:", note)
