# agents/fa_4.py — FA-4 MCP Tool Server
# Demo: discovery of MCP endpoints + least-privilege eligibility lookup.
# Call with SyntheticMemberID only (e.g. SMBR-000018 → ineligible).

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "bcbsri-member-eligibility",
    instructions=(
        "Synthetic BCBSRI member eligibility lookup. No PHI. "
        "Least-privilege: accept member_id only; never send clinical notes or free text."
    ),
)

INELIGIBLE_MEMBER_ID = "SMBR-000018"  # PA-1018 planted termination

ELIGIBILITY = {
    f"SMBR-{n:06d}": {
        "eligible": True,
        "plan_status": "active",
        "network": "in-network",
        "auth_required": True,
        "reason": "Active member; prior auth may still be required for this service.",
    }
    for n in range(1, 26)
}
ELIGIBILITY[INELIGIBLE_MEMBER_ID] = {
    "eligible": False,
    "plan_status": "terminated",
    "network": "unknown",
    "auth_required": False,
    "reason": "Member not eligible on service date (synthetic termination).",
}


@mcp.tool()
def lookup_member_eligibility(member_id: str) -> dict:
    """Look up synthetic member eligibility for a prior-auth request.

    Args:
        member_id: SyntheticMemberID like SMBR-000001. No PHI.
    """
    key = (member_id or "").strip()
    row = ELIGIBILITY.get(key)
    if row is None:
        return {
            "member_id": key,
            "eligible": False,
            "plan_status": "not_found",
            "network": "unknown",
            "auth_required": False,
            "reason": f"Unknown member_id: {key}",
        }
    return {"member_id": key, **row}


if __name__ == "__main__":
    mcp.run()  # stdio