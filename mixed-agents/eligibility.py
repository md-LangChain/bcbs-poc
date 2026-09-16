"""Eligibility — LangGraph graph (mixed-agents copy).

Input:  {"member_id": "SMBR-000001"}
Output: eligibility payload.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import NotRequired, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from utils.tracing import ELIGIBILITY_PROJECT, ensure_mixed_tracing_project  # noqa: E402

ensure_mixed_tracing_project(
    ELIGIBILITY_PROJECT,
    description="Mixed-agents eligibility (LangGraph) traces.",
)


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

eligibility_agent = builder.compile(name="fa-4-eligibility")


if __name__ == "__main__":
    print(eligibility_agent.invoke({"member_id": "SMBR-000001"}))
