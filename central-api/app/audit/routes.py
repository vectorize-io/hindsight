"""Audit routes."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.audit.logger import get_events, get_all_events
from app.audit.schemas import AuditQueryResponse
from app.auth.context import ContextDep

router = APIRouter(tags=["audit"])


@router.get("/api/audit/events", response_model=AuditQueryResponse)
async def list_events(ctx: ContextDep, limit: int = Query(100, le=500)) -> AuditQueryResponse:
    events = get_events(ctx.tenant_id, limit=limit)
    return AuditQueryResponse(tenant_id=ctx.tenant_id, count=len(events), events=events)


@router.get("/api/audit/events-simple")
async def list_events_simple(limit: int = Query(100, le=500)) -> dict:
    """Dashboard-compatible audit events (no auth, cross-tenant).
    
    Format: { events: [...] } with flat structure for dashboard display.
    """
    events = get_all_events(limit=limit)
    return {
        "events": [
            {
                "actor_id": e.actor_id,
                "source_app_id": e.source_app_id,
                "operation": e.operation,
                "resource_type": e.resource_type,
                "resource_id": e.resource_id,
                "outcome": e.outcome,
                "created_at": str(e.timestamp),
            }
            for e in events
        ]
    }
