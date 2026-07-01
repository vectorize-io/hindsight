"""Dashboard endpoints (no workspace scoping, no auth, for platform overview).

These provide aggregated cross-workspace data for the operator dashboard.
All endpoints are public (no authentication required) for dev/operator convenience.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import repositories as repo
from app.db.engine import get_session

router = APIRouter(tags=["dashboard"])


@router.get("/api/dashboard/connectors")
async def dashboard_connectors(
    session: Annotated[AsyncSession, Depends(get_session)]
) -> dict:
    """List all connectors across workspaces (dashboard overview)."""
    connectors = await repo.list_all_connectors(session)
    return {"connectors": connectors}


@router.get("/api/dashboard/workspaces")
async def dashboard_workspaces(
    session: Annotated[AsyncSession, Depends(get_session)]
) -> dict:
    """List all workspaces (dashboard overview)."""
    workspaces = await repo.list_all_workspaces(session)
    return {"workspaces": workspaces}


@router.get("/api/dashboard/agent-activity")
async def dashboard_agent_activity(
    session: Annotated[AsyncSession, Depends(get_session)],
    limit: int = Query(100, le=500)
) -> dict:
    """Agent activity feed (dashboard overview)."""
    # For now, return empty - will implement when agent activity tracking exists
    return {"activity": []}


@router.get("/api/dashboard/ingestion-jobs")
async def dashboard_ingestion_jobs(
    session: Annotated[AsyncSession, Depends(get_session)],
    limit: int = Query(100, le=500)
) -> dict:
    """List ingestion jobs across workspaces (dashboard overview)."""
    jobs = await repo.list_all_jobs(session, limit=limit)
    return {"jobs": jobs}


@router.post("/api/context/build")
async def context_build() -> dict:
    """Context build stub (dashboard panel)."""
    return {
        "status": "not_implemented",
        "message": "/api/context/build is not implemented"
    }
