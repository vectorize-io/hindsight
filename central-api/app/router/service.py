"""Router decision service."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import select

from app.auth.context import RequestContext
from app.db.engine import session_scope
from app.db.ids import new_id
from app.db.tables import router_decisions
from app.router.schemas import RouterDecision, RouterDecisionsResponse


async def list_decisions(
    ctx: RequestContext,
    limit: int = 100,
    offset: int = 0,
) -> RouterDecisionsResponse:
    """List router decisions for the current tenant, ordered newest first.
    
    Args:
        ctx: Request context (tenant_id from authenticated context)
        limit: Maximum number of decisions to return
        offset: Number of decisions to skip
        
    Returns:
        RouterDecisionsResponse with decisions and count
        
    Raises:
        HTTPException: 503 if database is unavailable
    """
    try:
        async with session_scope() as session:
            # Build query: filter by tenant, order newest first
            query = (
                select(router_decisions)
                .where(router_decisions.c.tenant_id == ctx.tenant_id)
                .order_by(router_decisions.c.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            
            result = await session.execute(query)
            rows = result.mappings().all()
            
            # Convert rows to RouterDecision objects
            decisions = [
                RouterDecision(
                    id=row["id"],
                    timestamp=row["created_at"],
                    tenant_id=row["tenant_id"],
                    actor_id=row["actor_id"],
                    request_type=row["request_type"],
                    selected_model=row["selected_model"],
                    candidate_models=row["candidate_models"] or [],
                    selection_reason=row["selection_reason"],
                    latency_ms=row["latency_ms"],
                    estimated_cost=row["estimated_cost"],
                    fallback_chain=row["fallback_chain"] or [],
                    status=row["status"],
                    trace_id=row["trace_id"],
                )
                for row in rows
            ]
            
            return RouterDecisionsResponse(
                decisions=decisions,
                count=len(decisions),
            )
        
    except Exception as e:
        # Fail-closed: DB error returns 503 with degraded flag
        raise HTTPException(
            status_code=503,
            detail={
                "error": "database_unavailable",
                "message": "Router decisions service is temporarily unavailable",
                "degraded": True,
            },
        ) from e


async def write_decision(
    *,
    tenant_id: str,
    actor_id: str,
    request_type: str,
    selected_model: Optional[str] = None,
    selected_provider: Optional[str] = None,
    candidate_models: Optional[list] = None,
    selection_reason: Optional[str] = None,
    latency_ms: Optional[int] = None,
    estimated_cost: Optional[float] = None,
    fallback_chain: Optional[list] = None,
    status: str = "selected",
    trace_id: Optional[str] = None,
) -> str:
    """Write a router decision to the DB. Returns the new decision ID.

    Used by route_preview (status=preview) and future real model calls.
    Health/model listing must NOT call this.
    """
    now = datetime.now(timezone.utc)
    decision_id = new_id()

    async with session_scope() as session:
        await session.execute(
            router_decisions.insert().values(
                id=decision_id,
                tenant_id=tenant_id,
                actor_id=actor_id,
                request_type=request_type,
                selected_model=selected_model,
                candidate_models=candidate_models or [],
                selection_reason=selection_reason,
                latency_ms=latency_ms,
                estimated_cost=estimated_cost,
                fallback_chain=fallback_chain or [],
                status=status,
                trace_id=trace_id,
                created_at=now,
            )
        )

    return decision_id
