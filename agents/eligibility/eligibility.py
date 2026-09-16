"""FA-4 member eligibility — LangGraph graph.

Input:  {"member_id": "SMBR-000001"}  (SyntheticMemberID only; no PHI / clinical text)
Output: eligibility payload for FA-1 / FA-2.

POC: always returns eligible=True (same payload shape). Planted deny can return later.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NotRequired, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")
os.environ.setdefault("LANGSMITH_PROJECT", "bcbs-eligibility-agent")


class EligibilityInput(TypedDict):
    member_id: str


class EligibilityState(TypedDict):
    member_id: str
    eligible: NotRequired[bool]
    plan_status: NotRequired[str]
    network: NotRequired[str]
    auth_required: NotRequired[bool]
    reason: NotRequired[str]


def lookup_eligibility(state: EligibilityState) -> dict:
    """Resolve member eligibility. Least-privilege: uses member_id only."""
    key = (state.get("member_id") or "").strip()
    # Planted deny: SMBR-000024 == CaseID PA-1024 in the synthetic CSV
    eligible = key != "SMBR-000024"
    return {
        "member_id": key,
        "eligible": eligible,
        "plan_status": "active" if eligible else "inactive",
        "network": "in-network" if eligible else "unknown",
        "auth_required": eligible,
        "reason": (
            "Active member; prior auth may still be required for this service."
            if eligible
            else "Member not eligible (planted deny for PA-1024 / SMBR-000024)."
        ),
    }


builder = StateGraph(EligibilityState, input_schema=EligibilityInput)
builder.add_node("lookup_eligibility", lookup_eligibility)
builder.add_edge(START, "lookup_eligibility")
builder.add_edge("lookup_eligibility", END)

eligibility_agent = builder.compile(name="eligibility")


if __name__ == "__main__":
    for mid in ("SMBR-000001", "SMBR-000024", "SMBR-999999"):
        out = eligibility_agent.invoke({"member_id": mid})
        print(mid, "→", out)
