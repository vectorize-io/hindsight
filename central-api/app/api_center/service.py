"""API Center service — API-CENTER-001.

Provides route registry and service catalog for operator visibility.
Does NOT expose secrets, does NOT fake verification status.
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.api_center.schemas import (
    ApiRoute,
    ApiService,
    RoutesResponse,
    ServicesResponse,
    ApiCenterHealthResponse,
)


# ── Route Registry ────────────────────────────────────────────────────────────

_KNOWN_ROUTES: list[dict] = [
    # central-api routes
    {"path": "/api/ai/providers", "method": "GET", "service": "central-api", "auth_type": "bearer_jwt", "description": "List AI providers"},
    {"path": "/api/ai/providers/{provider_id}", "method": "GET", "service": "central-api", "auth_type": "bearer_jwt", "description": "Get provider details"},
    {"path": "/api/ai/providers/{provider_id}/health", "method": "GET", "service": "central-api", "auth_type": "bearer_jwt", "description": "Check provider health", "provider_example": "litellm"},
    {"path": "/api/ai/models", "method": "GET", "service": "central-api", "auth_type": "bearer_jwt", "description": "List all models"},
    {"path": "/api/ai/models/{provider_id}", "method": "GET", "service": "central-api", "auth_type": "bearer_jwt", "description": "List models for provider", "provider_example": "litellm"},
    {"path": "/api/ai/route/preview", "method": "POST", "service": "central-api", "auth_type": "bearer_jwt", "description": "Preview routing decision"},
    {"path": "/api/router/decisions", "method": "GET", "service": "central-api", "auth_type": "bearer_jwt", "description": "List router decisions"},
    
    # collabmind-api routes (proxied)
    {"path": "/api/cp/ai/*", "method": "GET", "service": "collabmind-api", "auth_type": "bearer_jwt", "description": "AI gateway proxy"},
    {"path": "/api/cp/router/*", "method": "GET", "service": "collabmind-api", "auth_type": "bearer_jwt", "description": "Router proxy"},
    {"path": "/api/cp/memory/search", "method": "POST", "service": "collabmind-api", "auth_type": "bearer_jwt", "description": "Memory search"},
    {"path": "/api/cp/memory/store", "method": "POST", "service": "collabmind-api", "auth_type": "bearer_jwt", "description": "Store memory"},
    {"path": "/api/gov/policy-check", "method": "POST", "service": "central-api", "auth_type": "bearer_jwt", "description": "Governance policy check"},
    {"path": "/api/executions/history", "method": "GET", "service": "central-api", "auth_type": "bearer_jwt", "description": "Execution ledger history"},
    {"path": "/api/memory/search/unified", "method": "POST", "service": "central-api", "auth_type": "bearer_jwt", "description": "Memory-controller unified search"},
    {"path": "/api/memory/governance/policy-check", "method": "POST", "service": "central-api", "auth_type": "bearer_jwt", "description": "Memory-controller policy check"},
    {"path": "/api/memory/stats", "method": "GET", "service": "central-api", "auth_type": "bearer_jwt", "description": "Memory-controller stats"},
    {"path": "/api/memory/audit", "method": "GET", "service": "central-api", "auth_type": "bearer_jwt", "description": "Memory-controller audit"},
    {"path": "/health/dependencies", "method": "GET", "service": "central-api", "auth_type": "none", "description": "Public dependency health snapshot"},
    
    # MCP routes
    {"path": "/api/mcp/tools/list", "method": "GET", "service": "central-api", "auth_type": "bearer_jwt", "description": "List MCP tools", "mcp_tool": "list_tools"},
    {"path": "/api/mcp/tools/call", "method": "POST", "service": "central-api", "auth_type": "bearer_jwt", "description": "Call MCP tool", "mcp_tool": "call_tool"},
]

_KNOWN_SERVICES: list[dict] = [
    {
        "service_id": "central-api",
        "display_name": "Central API",
        "base_url": "http://localhost:8000",
        "port": 8000,
        "description": "AI Gateway + governance control plane",
        "status": "unknown",
    },
    {
        "service_id": "collabmind-api",
        "display_name": "CollabMind API",
        "base_url": "http://localhost:3050",
        "port": 3050,
        "description": "Public-facing governance edge",
        "status": "unknown",
    },
    {
        "service_id": "collabmind-memory",
        "display_name": "CollabMind Memory",
        "base_url": "http://localhost:3020",
        "port": 3020,
        "description": "Memory backend (HSG, write-gate, RRF)",
        "status": "unknown",
    },
    {
        "service_id": "collabmind-console",
        "display_name": "CollabMind Console",
        "base_url": "http://localhost:3000",
        "port": 3000,
        "description": "Operator GUI",
        "status": "unknown",
    },
]


async def list_routes(
    service_filter: str | None = None,
    status_filter: str | None = None,
    auth_filter: str | None = None,
    method_filter: str | None = None,
) -> RoutesResponse:
    """List all known routes, optionally filtered.
    
    Args:
        service_filter: Filter by service name
        status_filter: Filter by route status
        auth_filter: Filter by auth type
        method_filter: Filter by HTTP method
        
    Returns:
        RoutesResponse with filtered routes
    """
    routes = []
    
    for r in _KNOWN_ROUTES:
        # Apply filters
        if service_filter and r["service"] != service_filter:
            continue
        if status_filter and r.get("status", "active") != status_filter:
            continue
        if auth_filter and r.get("auth_type", "unknown") != auth_filter:
            continue
        if method_filter and r["method"] != method_filter:
            continue
        
        routes.append(ApiRoute(**r))
    
    return RoutesResponse(routes=routes, count=len(routes))


async def list_services() -> ServicesResponse:
    """List all known services with route counts.
    
    Returns:
        ServicesResponse with services and route counts
    """
    services = []
    
    for s in _KNOWN_SERVICES:
        # Count routes for this service
        route_count = sum(1 for r in _KNOWN_ROUTES if r["service"] == s["service_id"])
        
        service = ApiService(
            **s,
            route_count=route_count,
            last_health_check=None,  # Health checks not implemented yet
        )
        services.append(service)
    
    return ServicesResponse(services=services, count=len(services))


async def get_health() -> ApiCenterHealthResponse:
    """Get API Center health status.
    
    Returns:
        ApiCenterHealthResponse with counts and status
    """
    return ApiCenterHealthResponse(
        status="healthy",
        routes_count=len(_KNOWN_ROUTES),
        services_count=len(_KNOWN_SERVICES),
        checked_at=datetime.now(timezone.utc),
    )
