"""LangSmith project setup for mixed-agents.

Tracing project: bcbs-mixed
Application resource tag: bcbs-poc (same Application as agents/)
Threads: enabled via thread_idle_seconds (same as bcbs-poc)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

TRACING_PROJECT = os.getenv("LANGSMITH_MIXED_PROJECT", "bcbs-mixed")
APPLICATION_NAME = os.getenv("LANGSMITH_APPLICATION", "bcbs-poc")
# Match bcbs-poc: idle timeout before a thread is considered closed (seconds)
THREAD_IDLE_SECONDS = int(os.getenv("LANGSMITH_THREAD_IDLE_SECONDS", "600"))


def _application_tag_value_id(client: Client, application: str) -> str:
    tags = client.request_with_retries("GET", "/workspaces/current/tags").json()
    for tag in tags:
        if tag.get("key") != "Application":
            continue
        for value in tag.get("values") or []:
            if value.get("value") == application:
                return value["id"]
    raise ValueError(
        f"Application tag value '{application}' not found. "
        "Create it under LangSmith Settings → Resource tags."
    )


def ensure_mixed_tracing_project() -> str:
    """Create/upsert bcbs-mixed under Application=bcbs-poc with threads enabled.

    Also sets LANGSMITH_PROJECT so @traceable / LangChain / OTEL land here.
    """
    os.environ["LANGSMITH_PROJECT"] = TRACING_PROJECT
    os.environ.setdefault("LANGSMITH_TRACING", "true")

    if not os.getenv("LANGSMITH_API_KEY"):
        return TRACING_PROJECT

    client = Client()
    app_tag_id = _application_tag_value_id(client, APPLICATION_NAME)
    project = client.create_project(
        project_name=TRACING_PROJECT,
        description=(
            "Mixed-agents FA-1 (Microsoft Agent Framework) + FA-2 traces. "
            f"Same Application resource tag as {APPLICATION_NAME}."
        ),
        upsert=True,
        tag_value_ids=[app_tag_id],
        metadata={"application": APPLICATION_NAME, "stack": "mixed-agents"},
    )

    # Enable Threads tab (bcbs-poc uses thread_idle_seconds=600 in session.extra)
    client.request_with_retries(
        "PATCH",
        f"/sessions/{project.id}",
        json={
            "extra": {
                "metadata": {
                    "application": APPLICATION_NAME,
                    "stack": "mixed-agents",
                },
                "thread_idle_seconds": THREAD_IDLE_SECONDS,
            },
            "tag_value_ids": [app_tag_id],
        },
    )
    return TRACING_PROJECT
