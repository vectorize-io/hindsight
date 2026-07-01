"""Retrieval + context-pack schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CodeSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=100)
    repo: str | None = None


class DocSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=100)
    collection: str | None = None


class RetrievalHit(BaseModel):
    backend: str
    content: str
    citation: dict | None = None
    score: float | None = None
    metadata: dict = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    results: list[RetrievalHit]
    count: int
    degraded: list[str] = Field(default_factory=list)


class RetrievalRunRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    collection_aliases: list[str] = Field(default_factory=list)
    filters: dict = Field(default_factory=dict)


class RetrievalRunResult(BaseModel):
    rank: int
    vector_score: float | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    content_preview: str | None = None


class RetrievalContextPackage(BaseModel):
    id: str
    chunk_count: int
    token_count: int
    context_text: str


class RetrievalCitation(BaseModel):
    citation_key: str
    title: str | None = None
    excerpt: str | None = None
    document_id: str | None = None


class RetrievalRunResponse(BaseModel):
    """Contract of POST /api/retrieval/run (runtime: document-service Phase 5)."""

    ok: bool = True
    retrieval_run_id: str
    results: list[RetrievalRunResult]
    context_package: RetrievalContextPackage
    citations: list[RetrievalCitation]
    latency_ms: int


class RetrievalRunRecord(BaseModel):
    id: str
    query: str
    status: str
    retrieval_type: str = "semantic"


class RetrievalTraceResponse(BaseModel):
    """Contract of GET /api/retrieval/runs/{run_id}: full trace for one run."""

    run: RetrievalRunRecord
    results: list[RetrievalRunResult]
    context_package: RetrievalContextPackage | None = None
    citations: list[RetrievalCitation]


class ContextBuildRequest(BaseModel):
    query: str = Field(min_length=1)
    workspace_id: str | None = None
    k: int = Field(default=10, ge=1, le=100)
    include_memory: bool = True
    include_code: bool = True


class ContextItem(BaseModel):
    backend: str
    content: str
    score: float | None = None
    citation: dict | None = None


class ContextPack(BaseModel):
    """The product asset: a governed, cited, audited context bundle for an agent."""

    query: str
    selected_items: list[ContextItem]
    citations: list[dict]
    blocked_items_count: int
    policy_decisions: list[dict]
    audit_trace_id: str
    confidence: float | None = None
    preview: bool = False
