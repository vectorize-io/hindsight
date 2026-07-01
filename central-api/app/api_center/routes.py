"""API Center routes — API-CENTER-001.

GET /api/api-center/routes
GET /api/api-center/services
GET /api/api-center/health
"""
from __future__ import annotations

from fastapi import APIRouter, Query

from app.api_center import service as api_center_service
from app.api_center.schemas import (
    ApiCenterHealthResponse,
    RoutesResponse,
    ServicesResponse,
)
from app.auth.context import ContextDep

router = APIRouter(prefix="/api/api-center", tags=["api-center"])


@router.get("/routes", response_model=RoutesResponse)
async def get_routes(
    ctx: ContextDep,
    service: str | None = Query(None, description="Filter by service"),
    status: str | None = Query(None, description="Filter by status"),
    auth_type: str | None = Query(None, description="Filter by auth type"),
    method: str | None = Query(None, description="Filter by HTTP method"),
) -> RoutesResponse:
    """List all API routes with optional filters.
    
    Returns route catalog for operator visibility. Does not expose secrets.
    """
    return await api_center_service.list_routes(
        service_filter=service,
        status_filter=status,
        auth_filter=auth_type,
        method_filter=method,
    )


@router.get("/services", response_model=ServicesResponse)
async def get_services(ctx: ContextDep) -> ServicesResponse:
    """List all services in the CollabMind stack.
    
    Returns service catalog with route counts and health status.
    """
    return await api_center_service.list_services()


@router.get("/health", response_model=ApiCenterHealthResponse)
async def get_health(ctx: ContextDep) -> ApiCenterHealthResponse:
    """Get API Center health status.
    
    Returns health status with route and service counts.
    """
    return await api_center_service.get_health()
