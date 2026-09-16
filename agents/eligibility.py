# agents/eligibility.py — deployed `eligibility` graph
# Synthetic member eligibility lookup; mirrors agents/fa_4.py roster and fail-closed contract.

import re
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

MEMBER_ID_RE = re.compile(r"^SMBR-\d{6}$")

INELIGIBLE_MEMBER_ID = "SMBR-000018"  # PA-1018 planted termination

ELIGIBILITY = {
    f"SMBR-{n:06d}": {
        "eligible": True,
        "plan_status": "active",
        "network": "in-network",
        "auth_required": True,
        "reason": "Active member; prior auth may still be required for this service.",
    }
    for n in range(1, 26)
}
ELIGIBILITY[INELIGIBLE_MEMBER_ID] = {
    "eligible": False,
    "plan_status": "terminated",
    "network": "unknown",
    "auth_required": False,
    "reason": "Member not eligible on service date (synthetic termination).",
}


class EligibilityState(TypedDict):
    member_id: str
    eligible: NotRequired[bool]
    plan_status: NotRequired[str]
    network: NotRequired[str]
    auth_required: NotRequired[bool]
    reason: NotRequired[str]


class EligibilityInput(TypedDict):
    member_id: str


def lookup_eligibility(state: EligibilityState) -> dict:
    """Look up synthetic member eligibility, failing closed on malformed or unknown ids."""
    key = (state.get("member_id") or "").strip()
    if not MEMBER_ID_RE.match(key):
        return {
            "member_id": key,
            "eligible": False,
            "plan_status": "invalid_member_id",
            "network": "unknown",
            "auth_required": False,
            "reason": "member_id is not in the expected SMBR-###### format",
        }
    row = ELIGIBILITY.get(key)
    if row is None:
        return {
            "member_id": key,
            "eligible": False,
            "plan_status": "not_found",
            "network": "unknown",
            "auth_required": False,
            "reason": f"Unknown member_id: {key}",
        }
    return {"member_id": key, **row}


builder = StateGraph(EligibilityState, input_schema=EligibilityInput)
builder.add_node("lookup_eligibility", lookup_eligibility)

builder.add_edge(START, "lookup_eligibility")
builder.add_edge("lookup_eligibility", END)

eligibility = builder.compile(name="eligibility")


if __name__ == "__main__":
    for member_id in ("SMBR-000001", INELIGIBLE_MEMBER_ID, "PA-1001", "SMBR-999999"):
        print(member_id, "→", eligibility.invoke({"member_id": member_id}))
