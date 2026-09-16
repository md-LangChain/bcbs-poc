"""LangSmith instrumentation helpers: root-run metadata + HITL reviewer feedback."""

import logging
import os
from typing import Any

from langchain_core.runnables.config import ensure_config
from langsmith import Client
from langsmith.run_helpers import get_current_run_tree
from langsmith.run_trees import RunTree

logger = logging.getLogger(__name__)


def app_environment() -> str:
    """Origin of the current run: local_dev unless APP_ENV says otherwise."""
    return os.getenv("APP_ENV", "local_dev")


def _root_run_tree() -> RunTree | None:
    """Return the RunTree of the root run of the current trace, if traced."""
    run_tree = get_current_run_tree()
    if run_tree is None:
        return None
    root = run_tree
    while root.parent_run is not None:
        root = root.parent_run
    if str(root.id) == str(run_tree.trace_id):
        return root
    # Runs created through LangChain callbacks leave parent_run unset, so the root
    # has to come from the tracer that owns it: mutating that live object is what
    # makes the metadata survive the root run's own end-of-run patch.
    handlers = getattr(ensure_config().get("callbacks"), "handlers", None) or []
    for handler in handlers:
        run_map = getattr(handler, "run_map", None)
        if run_map:
            tracked = run_map.get(str(run_tree.trace_id))
            if tracked is not None:
                return tracked
    return None


def set_root_run_metadata(**metadata: Any) -> None:
    """Attach filterable keys to the root run of the current trace."""
    values = {key: value for key, value in metadata.items() if value is not None}
    if not values:
        return
    try:
        root = _root_run_tree()
        if root is not None:
            root.add_metadata(values)
    except Exception:
        logger.debug("Could not set root run metadata", exc_info=True)


def record_reviewer_agreement(decision: Any) -> None:
    """Log a resumed HITL decision as reviewer_agreement feedback on the root run."""
    text = str(decision).strip() if decision is not None else ""
    if not text:
        return
    run_tree = get_current_run_tree()
    if run_tree is None:
        return
    try:
        Client().create_feedback(
            run_id=run_tree.trace_id or run_tree.id,
            key="reviewer_agreement",
            score=1 if text.lower().startswith("approve") else 0,
            comment=text,
            session_id=run_tree.session_id,
        )
    except Exception:
        logger.debug("Could not record reviewer_agreement feedback", exc_info=True)
