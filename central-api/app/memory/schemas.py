"""Memory request/response schemas (Pydantic v2)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.policy.rules import PolicyDecision, Sensitivity, Status


class MemoryHit(BaseModel):
    memory_id: str
    backend: str
    content: str
    memory_type: str = "conversation"
    score: float | None = None
    citation: dict | None = None
    metadata: dict = Field(default_factory=dict)


class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1)
    workspace_id: str | None = None
    memory_type: str | None = None
    k: int = Field(default=10, ge=1, le=100)


class MemorySearchResponse(BaseModel):
    results: list[MemoryHit]
    count: int
    degraded: list[str] = Field(default_factory=list)


class GovernanceInfo(BaseModel):
    sensitivity: Sensitivity
    decision: PolicyDecision
    redactions: int
    reasons: list[str] = Field(default_factory=list)


class MemoryStoreRequest(BaseModel):
    content: str = Field(min_length=1)
    memory_type: str = "conversation"
    workspace_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class MemoryStoreResponse(BaseModel):
    memory_id: str | None
    status: Status
    governance: GovernanceInfo
    trace_id: str


class MemoryUpdateRequest(BaseModel):
    memory_id: str
    content: str = Field(min_length=1)
    metadata: dict = Field(default_factory=dict)


class MemoryUpdateResponse(BaseModel):
    memory_id: str
    status: Status
    governance: GovernanceInfo
    trace_id: str


class MemoryDeleteRequest(BaseModel):
    memory_id: str
    propagate: bool = True


class MemoryDeleteResponse(BaseModel):
    memory_id: str
    deleted_from: list[str]
    trace_id: str


class MemoryExportRequest(BaseModel):
    workspace_id: str | None = None


class MemoryExportResponse(BaseModel):
    tenant_id: str
    count: int
    records: list[dict]
