"""Audit schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AuditEvent(BaseModel):
    trace_id: str
    tenant_id: str
    actor_id: str | None = None
    source_app_id: str | None = None
    operation: str  # store|search|update|delete|approve|reject|export|query
    resource_type: str | None = "memory"
    resource_id: str | None = None
    outcome: str = "success"  # success|error|forbidden
    metadata: dict = Field(default_factory=dict)
    timestamp: int  # ms since epoch


class AuditQueryResponse(BaseModel):
    tenant_id: str
    count: int
    events: list[AuditEvent]
