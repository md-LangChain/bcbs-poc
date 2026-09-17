"""FA-1 Intake via Microsoft Agent Framework Agent + tools.

case_id → read CSV → flag missing fields → Case dict for FA-2.

No PHI redaction in this mixed-agents path (intentional contrast with agents/).

Tracing (LangSmith project bcbs-mixed-intake) via MAF OpenTelemetry:
  https://docs.langchain.com/langsmith/trace-with-microsoft-agent-framework

Expect traces like: invoke_agent → chat → read_case / validate_case → chat
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from dotenv import load_dotenv
from pydantic import Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from utils.tracing import (  # noqa: E402
    INTAKE_PROJECT,
    configure_maf_langsmith_otel,
    ensure_mixed_tracing_project,
    flush_otel,
)

# Import Case from validation first (may set LANGSMITH_PROJECT), then pin intake.
from validation import Case  # noqa: E402

ensure_mixed_tracing_project(
    INTAKE_PROJECT,
    description="Mixed-agents intake (Microsoft Agent Framework) traces.",
)
configure_maf_langsmith_otel(INTAKE_PROJECT)

from agent_framework import Agent, tool  # noqa: E402
from agent_framework.openai import OpenAIChatCompletionClient  # noqa: E402

CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "BCBSRI_Synthetic_PriorAuth_Datasetv1.csv"
)

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

# Latest validated case from tools — used so invoke() still returns a Case dict for FA-2.
_LAST_CASE: dict[str, Any] = {}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _load_case(case_id: str) -> dict[str, Any]:
    df = pd.read_csv(CSV_PATH)
    match = df[df["CaseID"] == case_id]
    if match.empty:
        return {"case_id": case_id, "error": f"CaseID not found: {case_id}"}

    row = {
        field: None if pd.isna(value) else value
        for field, value in match.iloc[0].to_dict().items()
    }
    if row.get("EstimatedCost") is not None:
        row["EstimatedCost"] = float(row["EstimatedCost"])
    row["case_id"] = case_id
    row["error"] = None
    return row


def _validate_case(state: dict[str, Any]) -> dict[str, Any]:
    missing = [f for f in REQUIRED_FIELDS if _is_missing(state.get(f))]
    return {
        **state,
        "missing_fields": missing,
        "error": (
            f"Missing required fields: {', '.join(missing)}" if missing else None
        ),
    }


@tool(name="read_case", description="Load a prior-auth case from the CSV by CaseID.")
def read_case(
    case_id: Annotated[str, Field(description="Prior-auth CaseID, e.g. PA-1021")],
) -> str:
    case = _load_case(case_id)
    _LAST_CASE.clear()
    _LAST_CASE.update(case)
    return json.dumps(case, default=str)


@tool(
    name="validate_case",
    description="Validate a prior-auth case JSON for missing required fields.",
)
def validate_case(
    case_json: Annotated[str, Field(description="Case object as JSON from read_case")],
) -> str:
    state = json.loads(case_json)
    validated = _validate_case(state)
    _LAST_CASE.clear()
    _LAST_CASE.update(validated)
    return json.dumps(validated, default=str)


def create_intake_agent() -> Agent[Any]:
    model = (
        os.getenv("OPENAI_MODEL")
        or os.getenv("OPENAI_CHAT_COMPLETION_MODEL")
        or "gpt-4.1-mini"
    )
    client = OpenAIChatCompletionClient(model=model)
    return Agent(
        client=client,
        name="fa-1-intake",
        instructions=(
            "You are the prior-auth intake agent. "
            "For the given CaseID: call read_case, then validate_case with that JSON. "
            "Reply briefly with CaseID, whether validation passed, and any missing_fields."
        ),
        tools=[read_case, validate_case],
    )


class _IntakeAgentShim:
    """Sync/async invoke API expected by FA-2 and experiment scripts."""

    def invoke(
        self,
        inputs: Case | dict[str, Any],
        config: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        try:
            return asyncio.run(self.ainvoke({"case_id": inputs["case_id"]}))
        finally:
            flush_otel()

    async def ainvoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        case_id = str(inputs["case_id"])
        _LAST_CASE.clear()
        agent = create_intake_agent()
        session = agent.create_session()
        try:
            await agent.run(
                f"Intake prior-auth case {case_id}. Use your tools.",
                session=session,
            )
            if _LAST_CASE:
                return dict(_LAST_CASE)
            # Fallback if the model skipped tools
            return _validate_case(_load_case(case_id))
        finally:
            flush_otel()


intake_agent = _IntakeAgentShim()


if __name__ == "__main__":
    result = intake_agent.invoke({"case_id": "PA-1021"})
    print("CaseID:", result.get("CaseID"))
    print("error:", result.get("error"))
    print("missing_fields:", result.get("missing_fields"))
    note = result.get("ClinicalNoteFreeText") or ""
    print("ClinicalNoteFreeText:", note[:200], "..." if len(note) > 200 else "")
    print(f"→ LangSmith project {INTAKE_PROJECT}: open 'invoke_agent fa-1-intake'")
