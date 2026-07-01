"""Approval/quarantine endpoints (GOV-003 — Postgres-backed)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.context import ContextDep
from app.db.engine import get_session
from app.execution.ledger import approve_execution, record_execution
from app.governance.write_gate import policy_check as run_policy_check
from app.governance.quarantine import (
    approve_quarantine_item,
    get_approval_history,
    get_quarantine_item,
    list_quarantine_items,
    reject_quarantine_item,
)

router = APIRouter(prefix="/api/gov", tags=["governance"])


class QuarantineItemResponse(BaseModel):
    id: str
    content_hash: str
    classification: str
    status: str
    created_at: str
    created_by: str | None = None
    reason: str | None = None


class QuarantineDetailResponse(BaseModel):
    id: str
    content: str
    classification: str
    status: str
    created_at: str
    created_by: str | None = None
    reason: str | None = None
    approved_by: str | None = None
    approved_at: str | None = None


class ApprovalDecisionRequest(BaseModel):
    reason: str | None = None


class ApprovalDecisionResponse(BaseModel):
    status: str
    decided_by: str
    decided_at: str


class PolicyCheckRequest(BaseModel):
    content: str | dict
    classification: str | None = None


class PolicyCheckResponse(BaseModel):
    allowed: bool
    reason: str
    quarantine_id: str | None = None


@router.get("/approval-queue")
async def list_approval_queue(
    context: ContextDep,
    session: Annotated[AsyncSession, Depends(get_session)],
    status: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> dict:
    """List quarantine items pending review. Requires approver role."""
    if "approver" not in context.roles and "operator" not in context.roles:
        raise HTTPException(status_code=403, detail="requires approver role")

    items = await list_quarantine_items(session, tenant_id=context.tenant_id, status=status, limit=limit)

    return {
        "total": len(items),
        "items": [
            QuarantineItemResponse(
                id=item["id"],
                content_hash=item["content_hash"],
                classification=item["classification"],
                status=item["status"],
                created_at=item["created_at"],
                created_by=item["created_by"],
                reason=item["reason"],
            )
            for item in items
        ],
    }


@router.post("/policy-check")
async def policy_check(
    context: ContextDep,
    session: Annotated[AsyncSession, Depends(get_session)],
    req: PolicyCheckRequest,
) -> PolicyCheckResponse:
    """Run the governed write gate without storing content."""
    result = await run_policy_check(
        session,
        content=req.content,
        tenant_id=context.tenant_id,
        actor_id=context.actor_id,
        classification=req.classification,
    )
    return PolicyCheckResponse(
        allowed=result["allowed"],
        reason=result["reason"],
        quarantine_id=result["quarantine_id"],
    )


@router.get("/approval-queue/{item_id}")
async def get_approval_item(
    context: ContextDep,
    session: Annotated[AsyncSession, Depends(get_session)],
    item_id: str,
) -> QuarantineDetailResponse:
    """Get quarantine item detail. Requires approver role."""
    if "approver" not in context.roles and "operator" not in context.roles:
        raise HTTPException(status_code=403, detail="requires approver role")

    item = await get_quarantine_item(session, item_id=item_id, tenant_id=context.tenant_id)
    if not item:
        raise HTTPException(status_code=404, detail="not found")

    return QuarantineDetailResponse(
        id=item["id"],
        content=item["content"],
        classification=item["classification"],
        status=item["status"],
        created_at=item["created_at"],
        created_by=item["created_by"],
        reason=item["reason"],
        approved_by=item["approved_by"],
        approved_at=item["approved_at"],
    )


@router.post("/approval-queue/{item_id}/approve")
async def approve_item(
    context: ContextDep,
    session: Annotated[AsyncSession, Depends(get_session)],
    item_id: str,
    req: ApprovalDecisionRequest,
) -> ApprovalDecisionResponse:
    """Approve quarantine item. Requires approver role."""
    if "approver" not in context.roles and "operator" not in context.roles:
        raise HTTPException(status_code=403, detail="requires approver role")

    success = await approve_quarantine_item(
        session, item_id=item_id, tenant_id=context.tenant_id, approver_id=context.actor_id, reason=req.reason
    )
    if not success:
        raise HTTPException(status_code=404, detail="not found or not quarantined")

    # Record in execution ledger
    await approve_execution(
        session,
        execution_id=item_id,
        tenant_id=context.tenant_id,
        approver_id=context.actor_id,
        approval_note=req.reason,
    )

    item = await get_quarantine_item(session, item_id=item_id, tenant_id=context.tenant_id)
    return ApprovalDecisionResponse(
        status=item["status"],
        decided_by=item["approved_by"],
        decided_at=item["approved_at"],
    )


@router.post("/approval-queue/{item_id}/reject")
async def reject_item(
    context: ContextDep,
    session: Annotated[AsyncSession, Depends(get_session)],
    item_id: str,
    req: ApprovalDecisionRequest,
) -> ApprovalDecisionResponse:
    """Reject quarantine item. Requires approver role."""
    if "approver" not in context.roles and "operator" not in context.roles:
        raise HTTPException(status_code=403, detail="requires approver role")

    success = await reject_quarantine_item(
        session, item_id=item_id, tenant_id=context.tenant_id, rejector_id=context.actor_id, reason=req.reason
    )
    if not success:
        raise HTTPException(status_code=404, detail="not found or not quarantined")

    # Record in execution ledger
    await approve_execution(
        session,
        execution_id=item_id,
        tenant_id=context.tenant_id,
        approver_id=context.actor_id,
        approval_note=f"Rejected: {req.reason}" if req.reason else "Rejected",
    )

    item = await get_quarantine_item(session, item_id=item_id, tenant_id=context.tenant_id)
    return ApprovalDecisionResponse(
        status=item["status"],
        decided_by=context.actor_id,
        decided_at=item.get("approved_at"),
    )


@router.get("/approval-queue/{item_id}/history")
async def get_item_history(
    context: ContextDep,
    session: Annotated[AsyncSession, Depends(get_session)],
    item_id: str,
) -> dict:
    """Get approval history for item. Requires approver role."""
    if "approver" not in context.roles and "operator" not in context.roles:
        raise HTTPException(status_code=403, detail="requires approver role")

    # Verify item exists and belongs to tenant
    item = await get_quarantine_item(session, item_id=item_id, tenant_id=context.tenant_id)
    if not item:
        raise HTTPException(status_code=404, detail="not found")

    history = await get_approval_history(session, item_id=item_id, tenant_id=context.tenant_id)
    return {
        "item_id": item_id,
        "decisions": history,
    }
