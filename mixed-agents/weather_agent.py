"""Simple Microsoft Agent Framework chat loop (weather demo).

Traces via MAF OpenTelemetry → LangSmith (bcbs-mixed), per:
https://docs.langchain.com/langsmith/trace-with-microsoft-agent-framework

Requires OPENAI_API_KEY (and optionally OPENAI_BASE_URL / OPENAI_MODEL)
and LANGSMITH_API_KEY.

Run:
  uv run python mixed-agents/weather_agent.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from random import randint
from typing import Annotated, Any

from dotenv import load_dotenv
from pydantic import Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from tracing import (  # noqa: E402
    TRACING_PROJECT,
    configure_maf_langsmith_otel,
    ensure_mixed_tracing_project,
    flush_otel,
)

ensure_mixed_tracing_project()
configure_maf_langsmith_otel()

from agent_framework import Agent, tool  # noqa: E402
from agent_framework.openai import OpenAIChatCompletionClient  # noqa: E402

_CONDITIONS = ("sunny", "cloudy", "rainy", "windy", "partly cloudy")


@tool(name="get_weather", description="Get the current weather for a city or location.")
def get_weather(
    location: Annotated[str, Field(description="City or location to look up.")],
) -> str:
    """Fake weather lookup — random condition/temp for the demo."""
    temp_c = randint(5, 32)
    condition = _CONDITIONS[randint(0, len(_CONDITIONS) - 1)]
    return f"The weather in {location} is {condition} with a high of {temp_c}°C."


def create_weather_agent() -> Agent[Any]:
    model = (
        os.getenv("OPENAI_MODEL")
        or os.getenv("OPENAI_CHAT_COMPLETION_MODEL")
        or "gpt-4.1-mini"
    )
    client = OpenAIChatCompletionClient(model=model)
    return Agent(
        client=client,
        name="weather-agent",
        instructions=(
            "You are a friendly weather assistant. "
            "Use the get_weather tool whenever the user asks about weather. "
            "Keep answers short and conversational."
        ),
        tools=[get_weather],
    )


async def chat_loop() -> None:
    agent = create_weather_agent()
    session = agent.create_session()

    print(f"Weather agent ready  ·  LangSmith project: {TRACING_PROJECT}")
    print("Ask about the weather (quit / exit / q to stop).\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"quit", "exit", "q"}:
            break

        try:
            response = await agent.run(user, session=session)
            text = (response.text or "").strip() or str(response)
            print(f"agent> {text}\n")
        finally:
            flush_otel()


if __name__ == "__main__":
    asyncio.run(chat_loop())
