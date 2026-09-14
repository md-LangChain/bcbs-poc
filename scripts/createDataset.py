"""Create LangSmith dataset: every FA-2 use case wired so far (one condition each)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DATASET_NAME = "PriorAuth Dataset"
# Match tracing project / Application resource tag (LANGSMITH_PROJECT=bcbs-poc)
APPLICATION_NAME = os.getenv("LANGSMITH_PROJECT", "bcbs-poc")


def _application_tag_value_id(client: Client, application: str) -> str:
    """Resolve LangSmith Application resource-tag value id (e.g. bcbs-poc)."""
    tags = client.request_with_retries(
        "GET",
        "/workspaces/current/tags",
    ).json()
    for tag in tags:
        if tag.get("key") != "Application":
            continue
        for value in tag.get("values") or []:
            if value.get("value") == application:
                return value["id"]
    raise ValueError(
        f"Application tag value '{application}' not found. "
        "Create it under LangSmith Settings → Resource tags, "
        "or set LANGSMITH_PROJECT to an existing Application value."
    )

# interrupt_type = first FA-2 interrupt type, or "none" if run finishes without HITL.
#
# Wired in FA-2 today:
#   none                      → clean intake + criteria (no interrupt)
#   high_cost_hitl            → EstimatedCost > 10k (run_intake)
#   intake_validation_failed  → missing/invalid fields (run_intake)
#   borderline_hitl           → criteria inconclusive (evaluate_criteria)
#
# Not scored here yet (built elsewhere / not wired into FA-2):
#   FA-4 eligibility deny (PA-1018 / SMBR-000018)
#   PHI redaction (PA-1021, PA-1017)
EXAMPLES = [
    {
        "inputs": {"case_id": "PA-1001"},
        "outputs": {"interrupt_type": "none"},
        "metadata": {
            "use_case": "clean_baseline",
            "label": "Clean path: intake ok, criteria clear, no HITL",
        },
    },
    {
        "inputs": {"case_id": "PA-1004"},
        "outputs": {"interrupt_type": "none"},
        "metadata": {
            "use_case": "clean_approve",
            "label": "Second clean approve path",
        },
    },
    {
        "inputs": {"case_id": "PA-1002"},
        "outputs": {"interrupt_type": "high_cost_hitl"},
        "metadata": {
            "use_case": "high_cost_hitl",
            "label": "High-cost surgical → intake HITL",
        },
    },
    {
        "inputs": {"case_id": "PA-1006"},
        "outputs": {"interrupt_type": "high_cost_hitl"},
        "metadata": {
            "use_case": "denial_high_cost_gate",
            "label": "Denial case still hits high-cost HITL first",
        },
    },
    {
        "inputs": {"case_id": "PA-1003"},
        "outputs": {"interrupt_type": "intake_validation_failed"},
        "metadata": {
            "use_case": "missing_criteria",
            "label": "Missing InterQualCriteriaSet → validation HITL",
        },
    },
    {
        "inputs": {"case_id": "PA-1025"},
        "outputs": {"interrupt_type": "intake_validation_failed"},
        "metadata": {
            "use_case": "malformed_record",
            "label": "Malformed / incomplete record → validation HITL",
        },
    },
    {
        "inputs": {"case_id": "PA-1005"},
        "outputs": {"interrupt_type": "borderline_hitl"},
        "metadata": {
            "use_case": "borderline_criteria",
            "label": "Borderline clinical criteria → criteria HITL",
        },
    },
]


def main() -> None:
    client = Client()
    app_tag_id = _application_tag_value_id(client, APPLICATION_NAME)
    dataset = client.create_dataset(
        DATASET_NAME,
        description=(
            "FA-2 use cases wired so far: clean, high-cost HITL, "
            "intake validation HITL, borderline HITL."
        ),
        metadata={"application": APPLICATION_NAME},
        tag_value_ids=[app_tag_id],
    )
    client.create_examples(dataset_id=dataset.id, examples=EXAMPLES)
    print(
        dataset.name,
        dataset.id,
        "n=",
        len(EXAMPLES),
        "application=",
        APPLICATION_NAME,
    )


if __name__ == "__main__":
    main()
