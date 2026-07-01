"""Router routes — /api/router/*."""
from __future__ import annotations

from fastapi import APIRouter, Query

from app.auth.context import ContextDep
from app.router import service
from app.router.schemas import RouterDecisionsResponse

router = APIRouter(prefix="/api/router", tags=["router"])


@router.get("/decisions", response_model=RouterDecisionsResponse)
async def get_decisions(
    ctx: ContextDep,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> RouterDecisionsResponse:
    """List router decisions for the current tenant.
    
    Returns decisions ordered newest first. Empty list if no decisions exist.
    Fails with 503 if database is unavailable.
    """
    return await service.list_decisions(ctx, limit=limit, offset=offset)
