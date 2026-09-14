"""CI: run target on PriorAuth Dataset; pick up scores from dataset-attached eval.

Uses workspace evaluator \"PHI Detection\" (id 1a040e21-…) attached to the
dataset. Feedback key is \"phi\" (1 = PHI detected). We fail CI if avg > 0.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client, evaluate

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agents"))
load_dotenv(ROOT / ".env")

from fa_1 import intake_agent  # noqa: E402

DATASET_NAME = "PriorAuth Dataset"
FEEDBACK_KEY = "phi"  # PHI Detection — not phi_safe
POLL_SECONDS = 90
POLL_INTERVAL = 5


def target(inputs: dict) -> dict:
    """FA-1 intake (redact_phi) → model-bound text as evaluator \"output\"."""
    case_id = inputs["case_id"]
    case = intake_agent.invoke(
        {"case_id": case_id},
        {"configurable": {"thread_id": f"phi-ci-{case_id}"}},
    )
    # PHI Detection variable_mapping uses output → graded text
    output = f"""ServiceRequested: {case.get('ServiceRequested')}
CPTCode: {case.get('CPTCode')}
ICD10Code: {case.get('ICD10Code')}
ClinicalIndication: {case.get('ClinicalIndication')}
InterQualCriteriaSet: {case.get('InterQualCriteriaSet')}
RequestedUrgency: {case.get('RequestedUrgency')}
ClinicalNoteFreeText: {case.get('ClinicalNoteFreeText')}
"""
    return {"output": output, "case_id": case_id}


def _feedback_stats(results) -> dict:
    """get_experiment_results may return a dict or an object."""
    if isinstance(results, dict):
        return results.get("feedback_stats") or {}
    return getattr(results, "feedback_stats", None) or {}


def _wait_for_scores(client: Client, experiment_name: str) -> dict:
    deadline = time.time() + POLL_SECONDS
    while time.time() < deadline:
        results = client.get_experiment_results(name=experiment_name)
        stats = _feedback_stats(results)
        if FEEDBACK_KEY in stats:
            return stats[FEEDBACK_KEY]
        print(f"waiting for {FEEDBACK_KEY} on experiment={experiment_name}…")
        time.sleep(POLL_INTERVAL)
    raise SystemExit(
        f"Timed out after {POLL_SECONDS}s waiting for '{FEEDBACK_KEY}'. "
        f"Confirm PHI Detection is attached to '{DATASET_NAME}'."
    )


if __name__ == "__main__":
    client = Client()
    experiment = evaluate(
        target,
        data=DATASET_NAME,
        experiment_prefix="phi-ci",
        max_concurrency=1,
    )
    name = experiment.experiment_name
    print(f"experiment={name}")

    stats = _wait_for_scores(client, name)
    n = stats.get("n") or stats.get("count") or 0
    avg = stats.get("avg")
    print(f"{FEEDBACK_KEY} stats={stats}")

    if avg is None:
        raise SystemExit(f"No average in {FEEDBACK_KEY} stats: {stats}")

    # phi = 1 means PHI detected → any detection fails CI
    if avg > 0:
        raise SystemExit(
            f"PHI detected: {FEEDBACK_KEY} avg={avg} over n={n} (want avg=0)"
        )

    print(f"No PHI detected ({FEEDBACK_KEY} avg={avg}, n={n}).")
