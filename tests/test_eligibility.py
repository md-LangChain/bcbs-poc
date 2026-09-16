"""Eligibility lookup must fail closed on malformed or unknown member ids."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agents"))

from eligibility import lookup_eligibility

FAIL_CLOSED_IDS = ("PA-1001", "PA-10024", "", "SMBR-999999")


def test_unresolvable_ids_are_not_eligible():
    for member_id in FAIL_CLOSED_IDS:
        result = lookup_eligibility({"member_id": member_id})
        assert result["eligible"] is False, member_id
        assert result["plan_status"] != "active", member_id
        assert result["network"] == "unknown", member_id
        assert result["auth_required"] is False, member_id


def test_malformed_ids_report_invalid_member_id():
    for member_id in ("PA-1001", "PA-10024", "", "  PA-1001\n"):
        result = lookup_eligibility({"member_id": member_id})
        assert result["plan_status"] == "invalid_member_id", member_id
        assert result["member_id"] == member_id.strip(), member_id


def test_well_formed_id_missing_from_roster_is_not_found():
    result = lookup_eligibility({"member_id": "SMBR-999999"})
    assert result["plan_status"] == "not_found"


def test_roster_ids_keep_their_real_determination():
    active = lookup_eligibility({"member_id": " SMBR-000001 "})
    assert active["member_id"] == "SMBR-000001"
    assert active["eligible"] is True
    assert active["plan_status"] == "active"
    assert active["network"] == "in-network"

    terminated = lookup_eligibility({"member_id": "SMBR-000018"})
    assert terminated["eligible"] is False
    assert terminated["plan_status"] == "terminated"


if __name__ == "__main__":
    test_unresolvable_ids_are_not_eligible()
    test_malformed_ids_report_invalid_member_id()
    test_well_formed_id_missing_from_roster_is_not_found()
    test_roster_ids_keep_their_real_determination()
    print("ok")
