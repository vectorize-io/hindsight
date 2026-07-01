"""Write-gate policy integration with quarantine (GOV-003)."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.execution.ledger import record_execution
from app.governance.quarantine import create_quarantine_item


async def policy_check(
    session: AsyncSession,
    *,
    content: dict | str,
    tenant_id: str,
    actor_id: str | None = None,
    classification: str | None = None,
) -> dict[str, any]:
    """Apply write-gate policy: classify → decide.
    
    Returns:
        {
            "allowed": bool,
            "reason": str,
            "quarantine_id": str | None
        }
    """
    
    # If classification not provided, use default
    if not classification:
        classification = "internal"
    
    # Reject secret content — never store even redacted
    if classification == "secret":
        # Record rejection in execution ledger
        await record_execution(
            session,
            tenant_id=tenant_id,
            action_type="content_write",
            target="memory",
            agent_id=actor_id or "system",
            agent_role="policy-engine",
            risk_level="high",
            params={"classification": "secret"},
            status="completed",
        )
        return {
            "allowed": False,
            "reason": "secret_content_rejected",
            "quarantine_id": None,
        }
    
    # Quarantine restricted/private content
    if classification in ("restricted", "private"):
        quarantine_id = await create_quarantine_item(
            session,
            tenant_id=tenant_id,
            content=content,
            classification=classification,
            created_by=actor_id,
        )
        # Record quarantine in execution ledger
        await record_execution(
            session,
            tenant_id=tenant_id,
            action_type="content_quarantine",
            target="memory",
            agent_id=actor_id or "system",
            agent_role="policy-engine",
            risk_level="medium",
            params={"classification": classification, "quarantine_id": quarantine_id},
            status="staged",
        )
        return {
            "allowed": False,
            "reason": "awaiting_approval",
            "quarantine_id": quarantine_id,
        }
    
    # Allow public/internal content
    return {
        "allowed": True,
        "reason": "approved_by_policy",
        "quarantine_id": None,
    }
