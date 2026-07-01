"""Operator routes — /api/operator/*.

Scaffold: the review queue is empty (no DB). Approve/reject log audit events and
return the resulting status. Approve/reject require operator scopes.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter

from app.audit.logger import log_event, new_trace_id
from app.auth.context import ContextDep, RequestContext
from app.auth.permissions import require_scope
from app.operator.schemas import (
    ApproveRequest,
    OperatorActionResponse,
    RejectRequest,
    ReviewResponse,
)
from app.policy.rules import Status

router = APIRouter(prefix="/api/operator", tags=["operator"])

# Module-level dependency aliases (avoids calling require_scope in arg defaults).
OperatorApprove = Annotated[RequestContext, require_scope("operator:approve")]


@router.get("/review", response_model=ReviewResponse)
async def review(ctx: ContextDep) -> ReviewResponse:
    # Stub: target = list quarantined/draft items for the workspace.
    return ReviewResponse(count=0, items=[])


@router.post("/approve", response_model=OperatorActionResponse)
async def approve(req: ApproveRequest, ctx: OperatorApprove) -> OperatorActionResponse:
    trace_id = new_trace_id()
    log_event(
        tenant_id=ctx.tenant_id, actor_id=ctx.actor_id, operation="approve",
        resource_id=req.memory_id, trace_id=trace_id, metadata={"note": req.note},
    )
    return OperatorActionResponse(
        memory_id=req.memory_id, status=Status.verified, trace_id=trace_id
    )


@router.post("/reject", response_model=OperatorActionResponse)
async def reject(req: RejectRequest, ctx: OperatorApprove) -> OperatorActionResponse:
    trace_id = new_trace_id()
    log_event(
        tenant_id=ctx.tenant_id, actor_id=ctx.actor_id, operation="reject",
        resource_id=req.memory_id, trace_id=trace_id, metadata={"reason": req.reason},
    )
    return OperatorActionResponse(
        memory_id=req.memory_id, status=Status.deleted, trace_id=trace_id
    )
