"""Evaluate validation use cases: assert first interrupt type from graph state."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langsmith import evaluate

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agents" / "validation"))
load_dotenv(ROOT / ".env")

from validation import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    SYSTEM_PROMPT,
    Fa2State,
    evaluate_criteria,
    run_intake,
)

DATASET_NAME = "PriorAuth Golden Dataset"


def get_interrupt_type(result: dict) -> str:
    """Return the first interrupt type from an agent.invoke result, or 'none'."""
    for item in result.get("__interrupt__") or ():
        value = getattr(item, "value", item)
        if isinstance(value, dict) and value.get("type"):
            return value["type"]
    return "none"


def target(inputs: dict) -> dict: 
    """Invoke validation once; return first interrupt type (no resume)."""
    case_id = inputs["case_id"]
    agent = create_agent(
        model="openai:gpt-4.1-mini",
        tools=[run_intake, evaluate_criteria],
        system_prompt=SYSTEM_PROMPT,
        name="validation-eval",
        checkpointer=MemorySaver(),
        state_schema=Fa2State,
    )

    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": f"Validate case {case_id}"},
            ],
        },
        {"configurable": {"thread_id": f"eval-{case_id}-{uuid.uuid4().hex[:8]}"}},
    )

    # Get interrupt type from the paused invoke result
    return {
        "case_id": case_id,
        "interrupt_type": get_interrupt_type(result),
    }


#Create Evaluator
def interrupt_type_correct( inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    expected = reference_outputs["interrupt_type"]
    actual = outputs.get("interrupt_type", "none")

    #does actual = expected 
    ok = actual == expected
    return {
        "key": "correct_path",
        "score": int(ok),
        "value": "pass" if ok else "fail",
        "comment": f"expected={expected} actual={actual}",
    }


if __name__ == "__main__":
    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[interrupt_type_correct],
        experiment_prefix="Functional-Experiment",
        max_concurrency=1,
    )
    print(results)
