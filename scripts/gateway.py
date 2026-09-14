"""Smoke-test LangSmith LLM Gateway via model configuration name.

Working route for config "gpt-4.1-nano" (OpenAI + Responses in UI):
  POST /models/gpt-4.1-nano/chat/completions
Do NOT append /v1 before /chat/completions — that becomes /openai/v1/v1/... (501).

Docs: https://docs.langchain.com/langsmith/llm-gateway-custom-providers
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

CONFIG_NAME = os.getenv("LANGSMITH_GATEWAY_MODEL_CONFIG", "gpt-4.1-nano")

client = OpenAI(
    api_key=os.environ["LANGSMITH_API_KEY"],
    # No trailing /v1 — config Base URL already includes /openai/v1
    base_url=(
        f"https://gateway.smith.langchain.com/models/"
        f"{quote(CONFIG_NAME, safe='')}"
    ),
)

response = client.chat.completions.create(
    model=CONFIG_NAME,
    messages=[
        {
            "role": "user",
            "content": (
                "Member Sarah M. Thompson (DOB 04/01/1986), MRN 4471982, "
                "Member ID SMBR-000021, SSN 041-86-7735, phone (401) 555-0192, "
                "email sarah.thompson86@examplemail.com, address 27 Maple Avenue, "
                "Warwick, RI 02886. Summarize this prior-auth note in one sentence "
                "without repeating identifiers."
            ),
        }
    ],
)
print(response.choices[0].message.content)
