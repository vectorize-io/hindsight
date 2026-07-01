"""Quarantine service — Postgres-backed storage (GOV-003).

Replaced in-memory _QUARANTINE/_APPROVALS with DB queries.
All operations use SQLAlchemy against quarantine_items and approval_decisions tables.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.logger import log_event
from app.db.ids import utcnow
from app.db.tables import quarantine_items, approval_decisions


async def create_quarantine_item(
    session: AsyncSession,
    *,
    tenant_id: str,
    content: dict | str,
    classification: str,
    created_by: str | None = None,
    reason: str | None = None,
) -> str:
    """Create and store quarantine item. Return item ID."""
    item_id = str(uuid.uuid4())
    content_str = content if isinstance(content, str) else json.dumps(content)
    content_hash = hashlib.sha256(content_str.encode()).hexdigest()
    
    stmt = quarantine_items.insert().values(
        id=item_id,
        tenant_id=tenant_id,
        content=content_str,
        content_hash=content_hash,
        classification=classification,
        status="pending",
        created_by=created_by,
        reason=reason,
        created_at=utcnow(),
    )
    await session.execute(stmt)
    await session.commit()
    
    # Audit
    log_event(
        tenant_id=tenant_id,
        operation="quarantine_create",
        actor_id=created_by,
        resource_type="quarantine_item",
        resource_id=item_id,
        metadata={"classification": classification, "content_hash": content_hash},
    )
    
    return item_id


async def get_quarantine_item(
    session: AsyncSession,
    *,
    item_id: str,
    tenant_id: str,
) -> dict[str, Any] | None:
    """Retrieve quarantine item (tenant-scoped)."""
    stmt = sa.select(quarantine_items).where(
        (quarantine_items.c.id == item_id) & (quarantine_items.c.tenant_id == tenant_id)
    )
    result = await session.execute(stmt)
    row = result.one_or_none()
    
    if not row:
        return None
    
    return {
        "id": row.id,
        "tenant_id": row.tenant_id,
        "content": row.content,
        "content_hash": row.content_hash,
        "classification": row.classification,
        "status": row.status,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "created_by": row.created_by,
        "reason": row.reason,
        "approved_by": row.approved_by,
        "approved_at": row.approved_at.isoformat() if row.approved_at else None,
    }


async def list_quarantine_items(
    session: AsyncSession,
    *,
    tenant_id: str,
    status: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List quarantine items for tenant."""
    stmt = sa.select(quarantine_items).where(
        quarantine_items.c.tenant_id == tenant_id
    )
    
    if status:
        stmt = stmt.where(quarantine_items.c.status == status)
    
    stmt = stmt.order_by(quarantine_items.c.created_at.desc()).limit(limit)
    result = await session.execute(stmt)
    rows = result.fetchall()
    
    return [
        {
            "id": row.id,
            "tenant_id": row.tenant_id,
            "content_hash": row.content_hash,
            "classification": row.classification,
            "status": row.status,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "created_by": row.created_by,
            "reason": row.reason,
        }
        for row in rows
    ]


async def approve_quarantine_item(
    session: AsyncSession,
    *,
    item_id: str,
    tenant_id: str,
    approver_id: str,
    reason: str | None = None,
) -> bool:
    """Approve quarantine item, move to approved state."""
    # Verify item exists and belongs to tenant
    stmt = sa.select(quarantine_items).where(
        (quarantine_items.c.id == item_id) & (quarantine_items.c.tenant_id == tenant_id)
    )
    result = await session.execute(stmt)
    item = result.one_or_none()
    
    if not item:
        return False
    
    # Update status
    update_stmt = (
        quarantine_items.update()
        .where(quarantine_items.c.id == item_id)
        .values(status="approved", approved_by=approver_id, approved_at=utcnow())
    )
    await session.execute(update_stmt)
    
    # Record approval decision
    decision_id = str(uuid.uuid4())
    insert_stmt = approval_decisions.insert().values(
        id=decision_id,
        quarantine_item_id=item_id,
        approver_id=approver_id,
        decision="approved",
        reason=reason,
        decided_at=utcnow(),
    )
    await session.execute(insert_stmt)
    await session.commit()
    
    # Audit
    log_event(
        tenant_id=tenant_id,
        operation="approval_approve",
        actor_id=approver_id,
        resource_type="quarantine_item",
        resource_id=item_id,
        metadata={"reason": reason or ""},
    )
    
    return True


async def reject_quarantine_item(
    session: AsyncSession,
    *,
    item_id: str,
    tenant_id: str,
    rejector_id: str,
    reason: str | None = None,
) -> bool:
    """Reject quarantine item, mark as rejected."""
    # Verify item exists and belongs to tenant
    stmt = sa.select(quarantine_items).where(
        (quarantine_items.c.id == item_id) & (quarantine_items.c.tenant_id == tenant_id)
    )
    result = await session.execute(stmt)
    item = result.one_or_none()
    
    if not item:
        return False
    
    # Update status
    update_stmt = (
        quarantine_items.update()
        .where(quarantine_items.c.id == item_id)
        .values(status="rejected")
    )
    await session.execute(update_stmt)
    
    # Record rejection decision
    decision_id = str(uuid.uuid4())
    insert_stmt = approval_decisions.insert().values(
        id=decision_id,
        quarantine_item_id=item_id,
        approver_id=rejector_id,
        decision="rejected",
        reason=reason,
        decided_at=utcnow(),
    )
    await session.execute(insert_stmt)
    await session.commit()
    
    # Audit
    log_event(
        tenant_id=tenant_id,
        operation="approval_reject",
        actor_id=rejector_id,
        resource_type="quarantine_item",
        resource_id=item_id,
        metadata={"reason": reason or ""},
    )
    
    return True


async def get_approval_history(
    session: AsyncSession,
    *,
    item_id: str,
    tenant_id: str,
) -> list[dict[str, Any]]:
    """Get approval history for quarantine item."""
    # Verify item belongs to tenant
    stmt = sa.select(quarantine_items).where(
        (quarantine_items.c.id == item_id) & (quarantine_items.c.tenant_id == tenant_id)
    )
    result = await session.execute(stmt)
    if not result.one_or_none():
        return []
    
    # Get decisions
    decision_stmt = (
        sa.select(approval_decisions)
        .where(approval_decisions.c.quarantine_item_id == item_id)
        .order_by(approval_decisions.c.decided_at.desc())
    )
    results = await session.execute(decision_stmt)
    decisions = results.fetchall()
    
    return [
        {
            "id": d.id,
            "quarantine_item_id": d.quarantine_item_id,
            "approver_id": d.approver_id,
            "decision": d.decision,
            "reason": d.reason,
            "decided_at": d.decided_at.isoformat() if d.decided_at else None,
        }
        for d in decisions
    ]
