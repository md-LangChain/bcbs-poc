"""FA-4 member eligibility — LangGraph graph (mixed-agents copy).

Input:  {"member_id": "SMBR-000001"}
Output: eligibility payload. POC always returns eligible=True.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NotRequired, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
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
    key = (state.get("member_id") or "").strip()
    return {
        "member_id": key,
        "eligible": True,
        "plan_status": "active",
        "network": "in-network",
        "auth_required": True,
        "reason": "Active member; prior auth may still be required for this service.",
    }


builder = StateGraph(EligibilityState, input_schema=EligibilityInput)
builder.add_node("lookup_eligibility", lookup_eligibility)
builder.add_edge(START, "lookup_eligibility")
builder.add_edge("lookup_eligibility", END)

eligibility_agent = builder.compile(name="fa-4-eligibility")


if __name__ == "__main__":
    print(eligibility_agent.invoke({"member_id": "SMBR-000001"}))
