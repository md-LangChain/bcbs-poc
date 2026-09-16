"""Regex-based PHI redaction for clinical free-text notes (POC)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_SSN_LABELED_NODASH = re.compile(r"\bSSN\s+\d{9}\b", re.IGNORECASE)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_DELIMITED = re.compile(
    r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)
_PHONE_DIGITS = re.compile(r"\b\d{10}\b")
_MEMBER_ID = re.compile(r"\bSMBR-\d{6}\b")
_POLICY = re.compile(r"\bBCBSRI-[A-Z0-9-]+\b")
_MRN = re.compile(r"\bMRN\s+\d+\b", re.IGNORECASE)
_NPI = re.compile(r"\bNPI\s+\d{10}\b", re.IGNORECASE)
_DOB_NUMERIC = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")
_DOB_SPELLED = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)
_ZIP = re.compile(r"\b\d{5}(?:-\d{4})?\b")
_STREET = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9.'\-]+\s+"
    r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Way)\b"
    r"(?:\s*,\s*[A-Za-z .'-]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)?",
    re.IGNORECASE,
)
_APT = re.compile(r"\bApt\.?\s+[A-Za-z0-9-]+\b", re.IGNORECASE)

# POC-only: planted demo names from PA-1021 / PA-1017
_DEMO_NAMES = re.compile(
    r"\b(?:"
    r"Sarah M\. Thompson|"
    r"Michael O'Brien-Reyes|"
    r"Dr\. Priya Nandakumar|"
    r"Dr\. Alan Whitfield"
    r")\b",
    re.IGNORECASE,
)


def _redact_ssn(text: str) -> str:
    """Replace ###-##-#### and labeled 9-digit SSN."""
    text = _SSN.sub("[REDACTED_SSN]", text)
    return _SSN_LABELED_NODASH.sub("[REDACTED_SSN]", text)


def _redact_email(text: str) -> str:
    return _EMAIL.sub("[REDACTED_EMAIL]", text)


def _redact_phone(text: str) -> str:
    """Delimited / dashed / dotted phones first, then bare 10-digit."""
    text = _PHONE_DELIMITED.sub("[REDACTED_PHONE]", text)
    return _PHONE_DIGITS.sub("[REDACTED_PHONE]", text)


def _redact_member_id(text: str) -> str:
    return _MEMBER_ID.sub("[REDACTED_MEMBER_ID]", text)


def _redact_policy(text: str) -> str:
    return _POLICY.sub("[REDACTED_POLICY]", text)


def _redact_mrn(text: str) -> str:
    return _MRN.sub("[REDACTED_MRN]", text)


def _redact_npi(text: str) -> str:
    return _NPI.sub("[REDACTED_NPI]", text)


def _redact_dob(text: str) -> str:
    text = _DOB_NUMERIC.sub("[REDACTED_DOB]", text)
    return _DOB_SPELLED.sub("[REDACTED_DOB]", text)


def _redact_address(text: str) -> str:
    text = _APT.sub("[REDACTED_APT]", text)
    text = _STREET.sub("[REDACTED_ADDRESS]", text)
    # ZIP last so street pattern can consume ZIP when present
    return _ZIP.sub("[REDACTED_ZIP]", text)


def _redact_demo_names(text: str) -> str:
    return _DEMO_NAMES.sub("[REDACTED_NAME]", text)


def redact_clinical_note(text: str | None) -> str | None:
    """Run enabled redactors on free text."""
    if not text:
        return text
    text = _redact_email(text)
    text = _redact_phone(text)
    text = _redact_ssn(text)
    text = _redact_member_id(text)
    text = _redact_policy(text)
    text = _redact_mrn(text)
    text = _redact_npi(text)
    text = _redact_dob(text)
    text = _redact_demo_names(text)
    text = _redact_address(text)
    return text


# Dataset columns that carry member / provider identifiers.
IDENTIFIER_FIELDS = (
    "SyntheticMemberID",
    "SyntheticDOB",
    "RequestingProviderNPI",
)

_BARE_NPI = re.compile(r"\b\d{10}\b")

_IDENTIFIER_VALUE_REDACTORS = (
    (_MEMBER_ID, "[REDACTED_MEMBER_ID]"),
    (_POLICY, "[REDACTED_POLICY]"),
    (_SSN, "[REDACTED_SSN]"),
    (_DOB_NUMERIC, "[REDACTED_DOB]"),
    (_DOB_SPELLED, "[REDACTED_DOB]"),
    (_BARE_NPI, "[REDACTED_NPI]"),
)


def _redact_identifier_value(value: Any) -> Any:
    """Scrub identifier-shaped content from a single case field value."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        # NPI columns arrive from the CSV as numbers, not strings.
        return "[REDACTED_NPI]" if _BARE_NPI.fullmatch(str(value)) else value
    if not isinstance(value, str):
        return value
    for pattern, placeholder in _IDENTIFIER_VALUE_REDACTORS:
        value = pattern.sub(placeholder, value)
    return value


def redact_case_identifiers(case: Mapping[str, Any] | None) -> dict[str, Any]:
    """Drop identifier columns from a case record and scrub identifier-shaped values."""
    if not case:
        return {}
    return {
        field: _redact_identifier_value(value)
        for field, value in case.items()
        if field not in IDENTIFIER_FIELDS
    }
