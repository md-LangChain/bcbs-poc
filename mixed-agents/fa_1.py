"""FA-1 Intake via Microsoft Agent Framework workflows.

case_id → read CSV → flag missing fields → Case dict for FA-2.

No PHI redaction in this mixed-agents path (intentional contrast with agents/).

Tracing (LangSmith project bcbs-mixed):
  - ``@traceable`` parent + child steps so Inputs/Outputs show full case data
  - MAF OTEL spans are OFF by default (empty I/O; set ENABLE_MAF_OTEL=true to opt in)
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
from pathlib import Path
from typing import Any, Never

import pandas as pd
from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler
from dotenv import load_dotenv
from langsmith import traceable, uuid7
from langsmith.run_helpers import get_current_run_tree

sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from tracing import TRACING_PROJECT, ensure_mixed_tracing_project  # noqa: E402

ensure_mixed_tracing_project()

from fa_2 import Case  # noqa: E402


def _configure_langsmith_otel() -> None:
    """Optional MAF→LangSmith OTEL. Off by default — those spans have empty I/O."""
    if os.getenv("ENABLE_MAF_OTEL", "").lower() not in {"1", "true", "yes"}:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        return

    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        return

    os.environ.setdefault("ENABLE_INSTRUMENTATION", "true")
    os.environ.setdefault("ENABLE_SENSITIVE_DATA", "true")
    os.environ.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    os.environ.setdefault(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "https://api.smith.langchain.com/otel/v1/traces",
    )
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = (
        f"x-api-key={api_key},Langsmith-Project={TRACING_PROJECT}"
    )
    os.environ.setdefault("LANGSMITH_TRACING", "true")

    from agent_framework.observability import configure_otel_providers

    configure_otel_providers(enable_sensitive_data=True)


def _flush_otel() -> None:
    if os.getenv("ENABLE_MAF_OTEL", "").lower() not in {"1", "true", "yes"}:
        return
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        force_flush = getattr(provider, "force_flush", None)
        if callable(force_flush):
            force_flush(timeout_millis=10_000)
    except Exception:  # noqa: BLE001
        pass


_configure_langsmith_otel()

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


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


# --- Business logic (traced so LangSmith shows full Inputs/Outputs) ---


@traceable(name="read_case", run_type="tool")
def read_case_logic(case_id: str) -> dict[str, Any]:
    """Load one CSV row. Return value = LangSmith Outputs."""
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


@traceable(name="validate_case", run_type="tool")
def validate_case_logic(state: dict[str, Any]) -> dict[str, Any]:
    """Flag missing required fields. Return value = LangSmith Outputs."""
    missing = [f for f in REQUIRED_FIELDS if _is_missing(state.get(f))]
    return {
        **state,
        "missing_fields": missing,
        "error": (
            f"Missing required fields: {', '.join(missing)}" if missing else None
        ),
    }


# --- MAF executors (orchestration); they call the traced helpers above ---


class ReadCase(Executor):
    @handler(input=dict, output=dict)
    async def run(self, state: dict[str, Any], ctx: WorkflowContext[dict]) -> None:
        await ctx.send_message(read_case_logic(str(state.get("case_id"))))


class ValidateCase(Executor):
    @handler(input=dict, workflow_output=dict)
    async def run(
        self, state: dict[str, Any], ctx: WorkflowContext[Never, dict]
    ) -> None:
        await ctx.yield_output(validate_case_logic(state))


def create_intake_workflow():
    read_case = ReadCase(id="read_case")
    validate = ValidateCase(id="validate_case")
    return (
        WorkflowBuilder(start_executor=read_case, name="fa-1-intake")
        .add_edge(read_case, validate)
        .build()
    )


@traceable(name="fa-1-intake", run_type="chain")
async def _run_intake(case_id: str) -> dict[str, Any]:
    """Parent trace: Inputs = {case_id}, Outputs = full case dict.

    Each invoke gets a unique LangSmith thread_id (not tied to case_id).
    """
    thread_id = str(uuid7())
    run_tree = get_current_run_tree()
    if run_tree is not None:
        run_tree.metadata["thread_id"] = thread_id
        run_tree.extra.setdefault("metadata", {})["thread_id"] = thread_id

    workflow = create_intake_workflow()
    events = await workflow.run({"case_id": case_id})
    outputs = events.get_outputs()
    if not outputs:
        return {"case_id": case_id}
    result = outputs[-1]
    return result if isinstance(result, dict) else {"case_id": case_id}


class _IntakeAgentShim:
    def invoke(
        self,
        inputs: Case | dict[str, Any],
        config: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        try:
            return asyncio.run(self.ainvoke({"case_id": inputs["case_id"]}))
        finally:
            _flush_otel()

    async def ainvoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            return await _run_intake(str(inputs["case_id"]))
        finally:
            _flush_otel()


intake_agent = _IntakeAgentShim()


if __name__ == "__main__":
    result = intake_agent.invoke({"case_id": "PA-1021"})
    print("CaseID:", result.get("CaseID"))
    print("error:", result.get("error"))
    print("missing_fields:", result.get("missing_fields"))
    note = result.get("ClinicalNoteFreeText") or ""
    print("ClinicalNoteFreeText:", note)
    print("→ LangSmith project bcbs-mixed, open run 'fa-1-intake' (not workflow.* spans)")
