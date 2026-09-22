"""Unit tests for validation system-prompt resolution."""

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_MODULE_PATH = Path(__file__).resolve().parent / "validation.py"


class _RaisingClient:
    def pull_agent(self, *args, **kwargs):
        raise RuntimeError("hub unreachable")


def _load_module():
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    os.environ["VALIDATION_PROMPT_REQUIRE_HUB"] = "false"
    spec = importlib.util.spec_from_file_location("validation_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with patch("langsmith.Client", _RaisingClient):
        spec.loader.exec_module(module)
    return module


validation = _load_module()


def test_import_records_fallback_source_in_permissive_mode():
    assert validation.PROMPT_SOURCE == "fallback_literal"
    assert validation.SYSTEM_PROMPT == validation.FALLBACK_SYSTEM_PROMPT
    assert len(validation.PROMPT_SHA) == 12


def test_fallback_when_hub_unavailable(monkeypatch):
    monkeypatch.setattr(validation, "Client", _RaisingClient)
    monkeypatch.setenv("VALIDATION_PROMPT_REQUIRE_HUB", "false")
    prompt, source = validation._load_system_prompt()
    assert source == "fallback_literal"
    assert prompt == validation.FALLBACK_SYSTEM_PROMPT


def test_strict_mode_raises_when_hub_unavailable(monkeypatch):
    monkeypatch.setattr(validation, "Client", _RaisingClient)
    monkeypatch.setenv("VALIDATION_PROMPT_REQUIRE_HUB", "true")
    with pytest.raises(RuntimeError, match="Context Hub prompt unavailable"):
        validation._load_system_prompt()


def test_hub_required_by_default_outside_local_dev(monkeypatch):
    monkeypatch.delenv("VALIDATION_PROMPT_REQUIRE_HUB", raising=False)
    monkeypatch.setenv("LANGSMITH_LANGGRAPH_API_VARIANT", "cloud")
    assert validation._require_hub_prompt() is True
    monkeypatch.setenv("LANGSMITH_LANGGRAPH_API_VARIANT", "local_dev")
    assert validation._require_hub_prompt() is False


def test_agent_records_prompt_provenance_metadata():
    agent = validation.create_validation_agent()
    metadata = agent.config["metadata"]
    assert metadata["prompt_source"] == validation.PROMPT_SOURCE
    assert metadata["prompt_sha"] == validation.PROMPT_SHA
