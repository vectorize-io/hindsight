"""API Center schemas — API-CENTER-001."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# Type definitions
HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
AuthType = Literal["none", "bearer_jwt", "api_key", "internal_context", "unknown"]
RouteStatus = Literal["active", "deprecated", "experimental", "mocked", "unknown"]
ServiceStatus = Literal["healthy", "degraded", "down", "unknown"]


class ApiRoute(BaseModel):
    """A single API route/endpoint."""
    
    path: str
    method: HttpMethod
    service: str  # Which service owns this route
    description: Optional[str] = None
    auth_type: AuthType = "unknown"
    status: RouteStatus = "active"
    mcp_tool: Optional[str] = None  # MCP tool name if mapped
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApiService(BaseModel):
    """A service in the CollabMind stack."""
    
    service_id: str
    display_name: str
    base_url: str
    port: int
    status: ServiceStatus = "unknown"
    description: Optional[str] = None
    route_count: int = 0
    last_health_check: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutesResponse(BaseModel):
    """Response containing API routes."""
    
    routes: list[ApiRoute] = Field(default_factory=list)
    count: int = 0


class ServicesResponse(BaseModel):
    """Response containing services."""
    
    services: list[ApiService] = Field(default_factory=list)
    count: int = 0


class ApiCenterHealthResponse(BaseModel):
    """Health status of API Center."""
    
    status: ServiceStatus
    routes_count: int
    services_count: int
    checked_at: datetime = Field(default_factory=datetime.utcnow)
