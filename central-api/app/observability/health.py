"""Runtime observability — service health aggregation, status tracking, and quarantine monitoring."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional

import httpx

from app.audit.logger import get_events
from app.governance.quarantine import list_quarantine_items
from app.config import settings


class HealthStatus(str, Enum):
    """Service health state."""
    healthy = "healthy"
    degraded = "degraded"
    down = "down"
    unknown = "unknown"


class ServiceStatusType(str, Enum):
    """Type of runtime service."""
    execution = "execution"  # execution/quarantine services
    auth = "auth"  # authentication/authorization services
    retrieval = "retrieval"  # data retrieval services
    ai = "ai"  # AI provider services
    infrastructure = "infrastructure"  # core infrastructure


@dataclass
class ServiceHealth:
    """Health check result for a single service."""
    name: str
    port: int
    status: HealthStatus
    latency_ms: int
    last_checked_at: int  # ms since epoch
    service_type: ServiceStatusType | None = None
    error_summary: str | None = None
    is_critical: bool = False  # whether service is critical to operation


@dataclass
class ProviderHealth:
    """Provider adapter health."""
    name: str
    type: str  # localai | ollama | litellm | openmemory
    status: HealthStatus
    latency_ms: int
    model_count: int = 0
    error_summary: str | None = None


@dataclass
class ServiceStatus:
    """Runtime service status tracking."""
    name: str
    service_type: ServiceStatusType
    status: HealthStatus
    is_quarantined: bool = False
    reason: str | None = None
    last_update: int = 0  # ms since epoch


@dataclass
class RuntimeHealth:
    """Full runtime observability snapshot."""
    timestamp_ms: int
    services: dict[str, ServiceHealth]  # keyed by service name
    providers: list[ProviderHealth]
    service_statuses: list[ServiceStatus]  # service status tracking
    governance_healthy: bool
    audit_events_count: int
    quarantine_items_count: int
    is_quarantined: bool = False  # true if any critical service is quarantined


async def check_service_health(
    url: str,
    name: str,
    port: int,
    timeout: float = 5.0,
) -> ServiceHealth:
    """Check if service is healthy via HTTP GET."""
    start = time.time()
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=timeout)
            latency = int((time.time() - start) * 1000)
            if 200 <= resp.status_code < 300:
                return ServiceHealth(
                    name=name,
                    port=port,
                    status=HealthStatus.healthy,
                    latency_ms=latency,
                    last_checked_at=int(time.time() * 1000),
                )
            else:
                return ServiceHealth(
                    name=name,
                    port=port,
                    status=HealthStatus.degraded,
                    latency_ms=latency,
                    last_checked_at=int(time.time() * 1000),
                    error_summary=f"HTTP {resp.status_code}",
                )
    except asyncio.TimeoutError:
        latency = int((time.time() - start) * 1000)
        return ServiceHealth(
            name=name,
            port=port,
            status=HealthStatus.down,
            latency_ms=latency,
            last_checked_at=int(time.time() * 1000),
            error_summary="timeout",
        )
    except Exception as e:
        latency = int((time.time() - start) * 1000)
        return ServiceHealth(
            name=name,
            port=port,
            status=HealthStatus.down,
            latency_ms=latency,
            last_checked_at=int(time.time() * 1000),
            error_summary=str(type(e).__name__),
        )


async def check_provider_health(
    provider_name: str,
    base_url: str,
    api_key: str | None = None,
) -> ProviderHealth:
    """Check provider health via /v1/models endpoint."""
    start = time.time()
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{base_url}/v1/models",
                headers=headers,
                timeout=5.0,
            )
            latency = int((time.time() - start) * 1000)
            
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    models = data.get("data", [])
                    return ProviderHealth(
                        name=provider_name,
                        type=provider_name.lower(),
                        status=HealthStatus.healthy,
                        latency_ms=latency,
                        model_count=len(models),
                    )
                except Exception:
                    return ProviderHealth(
                        name=provider_name,
                        type=provider_name.lower(),
                        status=HealthStatus.degraded,
                        latency_ms=latency,
                        error_summary="malformed_response",
                    )
            else:
                return ProviderHealth(
                    name=provider_name,
                    type=provider_name.lower(),
                    status=HealthStatus.degraded,
                    latency_ms=latency,
                    error_summary=f"HTTP {resp.status_code}",
                )
    except asyncio.TimeoutError:
        latency = int((time.time() - start) * 1000)
        return ProviderHealth(
            name=provider_name,
            type=provider_name.lower(),
            status=HealthStatus.down,
            latency_ms=latency,
            error_summary="timeout",
        )
    except Exception as e:
        latency = int((time.time() - start) * 1000)
        return ProviderHealth(
            name=provider_name,
            type=provider_name.lower(),
            status=HealthStatus.down,
            latency_ms=latency,
            error_summary=str(type(e).__name__),
        )


async def get_runtime_health(tenant_id: str | None = None) -> RuntimeHealth:
    """Aggregate runtime health across all services."""
    
    # Run all health checks in parallel
    service_checks = [
        check_service_health("http://localhost:3050/health", "collabmind-api", 3050),
        check_service_health("http://localhost:8000/health", "central-api", 8000),
        check_service_health("http://localhost:3020/health", "collabmind-memory", 3020),
        check_service_health("http://localhost:3000/", "console", 3000),
    ]
    
    provider_checks = [
        check_provider_health(
            "LocalAI",
            settings.localai_base_url or "http://localhost:8080",
            settings.localai_api_key or None,
        ),
        check_provider_health(
            "Ollama",
            settings.ollama_url or "http://localhost:11434",
        ),
        check_provider_health(
            "LiteLLM",
            settings.litellm_url or "http://localhost:4000",
            settings.litellm_api_key or None,
        ),
    ]
    
    # Execute all checks
    service_results = await asyncio.gather(*service_checks, return_exceptions=True)
    provider_results = await asyncio.gather(*provider_checks, return_exceptions=True)
    
    # Convert results to proper objects
    services_dict = {}
    for result in service_results:
        if isinstance(result, ServiceHealth):
            services_dict[result.name] = result
        else:
            # Exception during health check
            services_dict["unknown"] = ServiceHealth(
                name="unknown",
                port=0,
                status=HealthStatus.unknown,
                latency_ms=0,
                last_checked_at=int(time.time() * 1000),
                error_summary=str(result),
            )
    
    providers_list = []
    for result in provider_results:
        if isinstance(result, ProviderHealth):
            providers_list.append(result)
        else:
            providers_list.append(
                ProviderHealth(
                    name="unknown",
                    type="unknown",
                    status=HealthStatus.unknown,
                    latency_ms=0,
                    error_summary=str(result),
                )
            )
    
    # Check governance health (audit + quarantine reachable)
    governance_healthy = True
    try:
        if tenant_id:
            # Try to list audit events
            events = get_events(tenant_id, limit=1)
            # Try to list quarantine items
            _ = await list_quarantine_items(tenant_id=tenant_id, limit=1)
            governance_healthy = True
    except Exception:
        governance_healthy = False
    
    # Get audit event count and quarantine item count
    audit_count = 0
    quarantine_count = 0
    if tenant_id:
        try:
            audit_count = len(get_events(tenant_id, limit=1000))
            quarantine_count = len(
                await list_quarantine_items(tenant_id=tenant_id, limit=1000)
            )
        except Exception:
            pass
    
    return RuntimeHealth(
        timestamp_ms=int(time.time() * 1000),
        services=services_dict,
        providers=providers_list,
        service_statuses=[],
        governance_healthy=governance_healthy,
        audit_events_count=audit_count,
        quarantine_items_count=quarantine_count,
    )
