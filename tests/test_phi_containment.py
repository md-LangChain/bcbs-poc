"""PHI containment: member/provider identifiers must not leave the FA-1 intake subgraph."""

import json
import os
import re
import sys
from pathlib import Path

import pytest

# create_agent builds an OpenAI client at import time; no network call is made here.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agents"))

from fa_2 import project_intake_case, run_intake  # noqa: E402
from redact_phi import redact_case_identifiers  # noqa: E402

CASE_ID = "PA-1001"

DOB_SHAPED = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")
MEMBER_ID_SHAPED = re.compile(r"SMBR-\d{6}")
NPI_SHAPED = re.compile(r"\b\d{10}\b")


def _assert_no_identifiers(serialized: str) -> None:
    assert DOB_SHAPED.search(serialized) is None, serialized
    assert MEMBER_ID_SHAPED.search(serialized) is None, serialized
    assert NPI_SHAPED.search(serialized) is None, serialized


@pytest.fixture(scope="module")
def raw_case() -> dict:
    from fa_1 import intake_agent

    return intake_agent.invoke({"case_id": CASE_ID})


def test_raw_intake_state_still_holds_identifiers(raw_case):
    assert raw_case["SyntheticMemberID"] == "SMBR-000001"
    assert raw_case["SyntheticDOB"]
    assert raw_case["RequestingProviderNPI"]


def test_intake_projection_has_no_identifiers(raw_case):
    _assert_no_identifiers(json.dumps(project_intake_case(raw_case), default=str))


def test_intake_projection_keeps_decision_fields(raw_case):
    projection = project_intake_case(raw_case)
    assert projection["case_id"] == CASE_ID
    assert projection["CaseID"] == CASE_ID
    assert projection["ServiceRequested"] == "MRI Lumbar Spine w/o contrast"
    assert projection["error"] is None
    assert projection["missing_fields"] == []
    assert projection["high_cost"] is False
    assert projection["eligible"] is True


def test_run_intake_payload_has_no_identifiers():
    _assert_no_identifiers(run_intake.invoke({"case_id": CASE_ID}))


def test_new_identifier_columns_fail_closed():
    redacted = redact_case_identifiers(
        {
            "SubscriberDOB": "3/14/1968",
            "SecondaryMemberID": "SMBR-000042",
            "ServicingProviderNPI": 1000000017,
        }
    )
    _assert_no_identifiers(json.dumps(redacted, default=str))
