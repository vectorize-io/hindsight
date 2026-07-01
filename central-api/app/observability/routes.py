"""Observability API routes — runtime health + metrics."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.auth.context import ContextDep
from app.observability.health import (
    HealthStatus,
    RuntimeHealth,
    ServiceHealth,
    ProviderHealth,
    get_runtime_health,
)


router = APIRouter(prefix="/api/observability", tags=["observability"])
public_router = APIRouter(tags=["observability"])


class ServiceHealthResponse(BaseModel):
    name: str
    port: int
    status: str
    latency_ms: int
    last_checked_at: int
    error_summary: str | None = None


class ProviderHealthResponse(BaseModel):
    name: str
    type: str
    status: str
    latency_ms: int
    model_count: int = 0
    error_summary: str | None = None


class RuntimeHealthResponse(BaseModel):
    timestamp_ms: int
    services: dict[str, ServiceHealthResponse]
    providers: list[ProviderHealthResponse]
    governance_healthy: bool
    audit_events_count: int
    quarantine_items_count: int


@router.get("/runtime-health")
async def runtime_health(context: ContextDep) -> RuntimeHealthResponse:
    """Get aggregated runtime health across all services."""
    health = await get_runtime_health(tenant_id=context.tenant_id)
    
    return RuntimeHealthResponse(
        timestamp_ms=health.timestamp_ms,
        services={
            name: ServiceHealthResponse(
                name=svc.name,
                port=svc.port,
                status=svc.status.value,
                latency_ms=svc.latency_ms,
                last_checked_at=svc.last_checked_at,
                error_summary=svc.error_summary,
            )
            for name, svc in health.services.items()
        },
        providers=[
            ProviderHealthResponse(
                name=p.name,
                type=p.type,
                status=p.status.value,
                latency_ms=p.latency_ms,
                model_count=p.model_count,
                error_summary=p.error_summary,
            )
            for p in health.providers
        ],
        governance_healthy=health.governance_healthy,
        audit_events_count=health.audit_events_count,
        quarantine_items_count=health.quarantine_items_count,
    )


@router.get("/service-status")
async def service_status(context: ContextDep) -> dict:
    """Quick service status summary."""
    health = await get_runtime_health(tenant_id=context.tenant_id)
    
    all_healthy = all(
        svc.status == HealthStatus.healthy
        for svc in health.services.values()
    )
    any_down = any(
        svc.status == HealthStatus.down
        for svc in health.services.values()
    )
    
    return {
        "overall_status": "healthy" if all_healthy else "degraded" if any_down else "unknown",
        "service_count": len(health.services),
        "healthy_services": sum(
            1 for svc in health.services.values()
            if svc.status == HealthStatus.healthy
        ),
        "degraded_services": sum(
            1 for svc in health.services.values()
            if svc.status == HealthStatus.degraded
        ),
        "down_services": sum(
            1 for svc in health.services.values()
            if svc.status == HealthStatus.down
        ),
        "provider_count": len(health.providers),
        "governance_healthy": health.governance_healthy,
    }


@public_router.get("/api/health")
async def health() -> dict:
    """Public health check (no auth required)."""
    return {
        "status": "ok",
        "service": "central-api",
        "version": "1.0.0",
    }
