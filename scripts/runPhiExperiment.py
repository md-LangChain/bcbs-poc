"""CI experiment: fail if PHI leaks into model-bound text.

Pulls existing LangSmith resources from your workspace (does not create them):

  - Dataset:   DATASET_NAME (default: \"PriorAuth Dataset\")
  - Evaluator: EVALUATOR_ID (default: PHI Safe UUID below)

The workspace evaluator's llm_evaluator config supplies the hub prompt +
variable_mapping (see scripts/createPhiEvaluator.py). We only define the
target app under test and a thin wrapper that runs that pulled judge.

Exit code 1 if any example scores phi_safe=0 (GitHub Actions fails).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langsmith import Client, evaluate
from langsmith.schemas import Example, Run
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agents"))
load_dotenv(ROOT / ".env")

from fa_1 import intake_agent  # noqa: E402

# --- Workspace resources (override via env without code changes) ---
DATASET_NAME = os.getenv("PHI_DATASET_NAME", "PriorAuth Dataset")
# Workspace "PHI Safe" evaluator id (LangSmith → Evaluators)
EVALUATOR_ID = os.getenv(
    "PHI_EVALUATOR_ID", "1a040e21-c6c8-4137-aba4-07dad4ea638a"
)
EXPERIMENT_PREFIX = "phi-ci"
FEEDBACK_KEY = "phi_safe"

# Filled in main() after we pull the evaluator from the workspace.
_PROMPT_REPO: str = ""
_VARIABLE_MAPPING: dict[str, str] = {}

# Collect scores during evaluate() for a reliable CI exit check.
_PHI_SCORES: list[tuple[str, int, str]] = []


class PhiJudgeResult(BaseModel):
    """Structured judge output — matches createPhiEvaluator.py / PHI Safe."""

    phi_safe: bool = Field(
        description="True if model_bound_text has no cleartext PHI; false if any PHI leaked"
    )
    comment: str = Field(
        description="Short explanation; if false, list leaked PHI categories and snippets"
    )


def _criteria_prompt(case: dict) -> str:
    """Reproduce the text FA-2 sends to the criteria LLM (PHI leak surface)."""
    return f"""You are a prior-auth clinical criteria reviewer (POC).
Decide meets_criteria and borderline using only these fields. Do not invent facts.

ServiceRequested: {case.get('ServiceRequested')}
CPTCode: {case.get('CPTCode')}
ICD10Code: {case.get('ICD10Code')}
ClinicalIndication: {case.get('ClinicalIndication')}
InterQualCriteriaSet: {case.get('InterQualCriteriaSet')}
RequestedUrgency: {case.get('RequestedUrgency')}
ClinicalNoteFreeText: {case.get('ClinicalNoteFreeText')}
"""


def target(inputs: dict) -> dict:
    """App under test: FA-1 intake (includes redact_phi) → model-bound text.

    Must emit model_bound_text so it matches PHI Safe's mapping:
    outputs.model_bound_text
    """
    case_id = inputs["case_id"]
    case = intake_agent.invoke(
        {"case_id": case_id},
        config={"configurable": {"thread_id": f"phi-ci-{case_id}"}},
    )
    return {
        "case_id": case_id,
        "model_bound_text": _criteria_prompt(case),
    }


def _resolve_path(path: str, *, run: Run, example: Example) -> Any:
    """Resolve a LangSmith variable_mapping path like 'outputs.model_bound_text'.

    Prefixes used by PHI Safe (createPhiEvaluator.py):
      inputs.*     → example.inputs
      outputs.*    → run.outputs  (target return value)
      reference.*  → example.outputs  (dataset reference outputs)
    """
    root, _, rest = path.partition(".")
    if root == "inputs":
        obj: Any = example.inputs or {}
    elif root == "outputs":
        obj = run.outputs or {}
    elif root == "reference":
        obj = example.outputs or {}
    else:
        return ""

    if not rest:
        return obj
    cur: Any = obj
    for key in rest.split("."):
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(key, "")
    return cur if cur is not None else ""


def _require_dataset(client: Client, name: str) -> None:
    """Fail fast if the workspace dataset is missing (we do not create it)."""
    datasets = list(client.list_datasets(dataset_name=name))
    if not datasets:
        raise SystemExit(
            f"Dataset '{name}' not found in LangSmith. "
            f"Create it (e.g. scripts/createDataset.py) or set PHI_DATASET_NAME."
        )
    print(f"dataset=pulled name={name} id={datasets[0].id}")


async def _pull_evaluator(
    client: Client, evaluator_id: str
) -> tuple[str, str, dict[str, str]]:
    """Retrieve workspace evaluator by id; return (name, prompt_repo, variable_mapping)."""
    try:
        full = await client.evaluators.retrieve(evaluator_id)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Evaluator id '{evaluator_id}' not found in LangSmith: {exc}"
        ) from exc

    llm = getattr(full, "llm_evaluator", None) or {}
    if hasattr(llm, "model_dump"):
        llm = llm.model_dump()
    prompt_repo = llm.get("prompt_repo_handle") or ""
    mapping = dict(llm.get("variable_mapping") or {})
    name = getattr(full, "name", "") or evaluator_id
    if not prompt_repo:
        raise SystemExit(
            f"Evaluator '{name}' ({evaluator_id}) has no prompt_repo_handle."
        )
    print(
        f"evaluator=pulled name={name} id={evaluator_id} "
        f"prompt_repo={prompt_repo} mapping={mapping}"
    )
    return name, prompt_repo, mapping


def phi_safe_evaluator(run: Run, example: Example) -> dict:
    """Run the pulled workspace PHI Safe judge against this experiment row."""
    client = Client()
    inputs = example.inputs or {}
    case_id = str(inputs.get("case_id", ""))

    # Build judge inputs from the evaluator's variable_mapping (not hard-coded keys).
    judge_vars = {
        var: _resolve_path(path, run=run, example=example)
        for var, path in _VARIABLE_MAPPING.items()
    }
    # Defaults if mapping omitted a key the hub prompt expects
    judge_vars.setdefault("case_id", case_id)
    judge_vars.setdefault("model_bound_text", (run.outputs or {}).get("model_bound_text", ""))
    judge_vars.setdefault("forbidden_strings", (example.outputs or {}).get("forbidden_strings", ""))

    # Pull hub prompt referenced by the workspace evaluator, then structured-judge.
    prompt = client.pull_prompt(_PROMPT_REPO)
    judge = prompt | init_chat_model("openai:gpt-4.1-mini").with_structured_output(
        PhiJudgeResult
    )
    judged: PhiJudgeResult = judge.invoke(judge_vars)

    score = int(judged.phi_safe)
    _PHI_SCORES.append((case_id, score, judged.comment))

    return {
        "key": FEEDBACK_KEY,
        "score": score,
        "value": "pass" if judged.phi_safe else "fail",
        "comment": judged.comment,
    }


if __name__ == "__main__":
    _PHI_SCORES.clear()
    client = Client()

    # 1) Pull existing dataset + evaluator from the workspace
    _require_dataset(client, DATASET_NAME)
    evaluator_name, _PROMPT_REPO, _VARIABLE_MAPPING = asyncio.run(
        _pull_evaluator(client, EVALUATOR_ID)
    )

    # 2) Run experiment: target on each example, then workspace-backed judge
    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[phi_safe_evaluator],
        experiment_prefix=EXPERIMENT_PREFIX,
        max_concurrency=1,
        metadata={
            "ci": "phi-leak",
            "dataset": DATASET_NAME,
            "evaluator_id": EVALUATOR_ID,
            "evaluator": evaluator_name,
            "prompt_repo": _PROMPT_REPO,
        },
    )
    print(results)

    # 3) Fail CI on any PHI leak
    failed = [(cid, comment) for cid, score, comment in _PHI_SCORES if not score]
    if failed:
        print("PHI leakage detected — failing CI:")
        for case_id, comment in failed:
            print(f"  case_id={case_id} comment={comment}")
        raise SystemExit(1)

    print(
        f"All {len(_PHI_SCORES)} examples PHI-safe "
        f"(dataset={DATASET_NAME}, evaluator={evaluator_name} id={EVALUATOR_ID})."
    )
