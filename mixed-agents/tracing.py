"""LangSmith project setup for mixed-agents.

Projects (same Application=bcbs-poc):
  - bcbs-mixed-fa1  — FA-1 (Microsoft Agent Framework)
  - bcbs-mixed-fa2  — FA-2 (LangChain)
  - bcbs-mixed      — misc demos (e.g. weather_agent)

MAF → LangSmith OTEL:
  https://docs.langchain.com/langsmith/trace-with-microsoft-agent-framework
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

FA1_PROJECT = os.getenv("LANGSMITH_MIXED_FA1_PROJECT", "bcbs-mixed-fa1")
FA2_PROJECT = os.getenv("LANGSMITH_MIXED_FA2_PROJECT", "bcbs-mixed-fa2")
# Default / demo project (weather_agent)
TRACING_PROJECT = os.getenv("LANGSMITH_MIXED_PROJECT", "bcbs-mixed")

APPLICATION_NAME = os.getenv("LANGSMITH_APPLICATION", "bcbs-poc")
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


def ensure_mixed_tracing_project(
    project_name: str | None = None,
    *,
    description: str | None = None,
    set_active: bool = True,
) -> str:
    """Create/upsert a mixed-agents LangSmith project under Application=bcbs-poc.

    If ``set_active`` is True, sets LANGSMITH_PROJECT so LangChain / @traceable
    land in this project.
    """
    name = project_name or TRACING_PROJECT
    if set_active:
        os.environ["LANGSMITH_PROJECT"] = name
    os.environ.setdefault("LANGSMITH_TRACING", "true")

    if not os.getenv("LANGSMITH_API_KEY"):
        return name

    client = Client()
    app_tag_id = _application_tag_value_id(client, APPLICATION_NAME)
    project = client.create_project(
        project_name=name,
        description=description
        or (
            f"Mixed-agents traces ({name}). "
            f"Same Application resource tag as {APPLICATION_NAME}."
        ),
        upsert=True,
        tag_value_ids=[app_tag_id],
        metadata={"application": APPLICATION_NAME, "stack": "mixed-agents"},
    )

    client.request_with_retries(
        "PATCH",
        f"/sessions/{project.id}",
        json={
            "extra": {
                "metadata": {
                    "application": APPLICATION_NAME,
                    "stack": "mixed-agents",
                    "langsmith_project": name,
                },
                "thread_idle_seconds": THREAD_IDLE_SECONDS,
            },
            "tag_value_ids": [app_tag_id],
        },
    )
    return name


def configure_maf_langsmith_otel(
    project_name: str | None = None,
    *,
    enable_sensitive_data: bool = True,
) -> None:
    """Wire Microsoft Agent Framework OTEL → LangSmith OTLP endpoint.

    ``project_name`` is sent as the Langsmith-Project OTEL header (defaults to
    FA1_PROJECT). No-op if LANGSMITH_API_KEY is missing.
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        return

    name = project_name or FA1_PROJECT
    os.environ["ENABLE_INSTRUMENTATION"] = "true"
    os.environ.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    os.environ.setdefault(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "https://api.smith.langchain.com/otel/v1/traces",
    )
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = (
        f"x-api-key={api_key},Langsmith-Project={name}"
    )
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    if enable_sensitive_data:
        os.environ.setdefault("ENABLE_SENSITIVE_DATA", "true")

    from agent_framework.observability import configure_otel_providers

    configure_otel_providers(enable_sensitive_data=enable_sensitive_data)


def flush_otel(timeout_millis: int = 10_000) -> None:
    """Flush pending OTEL spans so they show up in LangSmith promptly."""
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        force_flush = getattr(provider, "force_flush", None)
        if callable(force_flush):
            force_flush(timeout_millis=timeout_millis)
    except Exception:  # noqa: BLE001
        pass
