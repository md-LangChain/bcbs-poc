"""Intake must finish validating a case even when the eligibility lookup fails."""

import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# fa_1 imports fa_2, which builds its chat model at import time; no model is called here.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")

from agents import fa_1

CASE_ID = "PA-1001"


class _Failing422Runs:
    def wait(self, thread_id, assistant_id, **kwargs):
        request = httpx.Request("POST", "https://eligibility.example/runs/wait")
        raise httpx.HTTPStatusError(
            "Client error '422 Unprocessable Entity' for url "
            "'https://eligibility.example/runs/wait'",
            request=request,
            response=httpx.Response(422, request=request),
        )


class _Failing422Client:
    runs = _Failing422Runs()


def test_intake_completes_when_eligibility_returns_422(monkeypatch):
    monkeypatch.setattr(fa_1, "_eligibility_client", lambda: _Failing422Client())

    result = fa_1.intake_agent.invoke({"case_id": CASE_ID})

    assert result["CaseID"] == CASE_ID
    assert result["ServiceRequested"]
    assert result["ClinicalNoteFreeText"]
    assert result["eligible"] is None
    assert "422" in result["eligibility_error"]
    assert result["missing_fields"] == []
    assert result["error"] is None
