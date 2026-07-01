"""Health routes — public liveness + per-engine status."""

from __future__ import annotations

import time

from fastapi import APIRouter

from app.adapters import get_all_adapters
from app.db.engine import get_health as get_db_health
from app.observability.health import get_runtime_health

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """Public liveness check (no auth)."""
    db_ok, db_latency = await get_db_health()
    return {
        "status": "ok",
        "service": "central-api",
        "version": "0.1.0",
        "db": {"ok": db_ok, "latency_ms": db_latency},
    }


@router.get("/api/health/engines")
async def engines() -> dict:
    """Aggregate each adapter's health (status, detail).
    
    Returns dashboard-compatible format:
    {
        "status": "ok" | "degraded",
        "services": [...],
        "checked_at": <epoch_ms>
    }
    """
    results: list[dict] = []
    any_degraded = False
    
    for adapter in get_all_adapters():
        try:
            h = await adapter.health()
            latency_ms = h.detail.get("latency_ms") if h.detail else None
            
            # Map adapter status to service status
            if h.status in ("ok", "stub"):
                svc_status = "ok"
            elif h.status == "degraded":
                svc_status = "ok"  # Still reachable but degraded
                any_degraded = True
            else:
                svc_status = "down"
                any_degraded = True
                
            results.append({
                "key": h.backend,
                "label": h.backend.replace("_", " ").title(),
                "url": h.detail.get("url") if h.detail else None,
                "status": svc_status,
                "latency_ms": latency_ms,
                "detail": h.detail.get("error") if h.detail and "error" in h.detail else None,
            })
        except Exception as exc:  # noqa: BLE001
            results.append({
                "key": adapter.backend,
                "label": adapter.backend.replace("_", " ").title(),
                "url": None,
                "status": "down",
                "latency_ms": None,
                "detail": str(exc),
            })
            any_degraded = True
    
    return {
        "status": "degraded" if any_degraded else "ok",
        "services": results,
        "checked_at": int(time.time() * 1000),
    }


@router.get("/health/dependencies")
async def dependencies() -> dict:
    """Public dependency snapshot for operator readiness."""
    runtime = await get_runtime_health()
    services = {
        name: {
            "name": svc.name,
            "port": svc.port,
            "status": svc.status.value,
            "latency_ms": svc.latency_ms,
            "last_checked_at": svc.last_checked_at,
            "error_summary": svc.error_summary,
            "service_type": svc.service_type.value if svc.service_type else None,
            "is_critical": svc.is_critical,
        }
        for name, svc in runtime.services.items()
    }
    providers = [
        {
            "name": provider.name,
            "type": provider.type,
            "status": provider.status.value,
            "latency_ms": provider.latency_ms,
            "model_count": provider.model_count,
            "error_summary": provider.error_summary,
        }
        for provider in runtime.providers
    ]
    return {
        "status": "healthy" if all(svc["status"] == "healthy" for svc in services.values()) else "degraded",
        "timestamp_ms": runtime.timestamp_ms,
        "services": services,
        "providers": providers,
        "governance_healthy": runtime.governance_healthy,
        "audit_events_count": runtime.audit_events_count,
        "quarantine_items_count": runtime.quarantine_items_count,
        "is_quarantined": runtime.is_quarantined,
    }
