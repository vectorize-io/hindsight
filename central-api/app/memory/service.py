"""Memory service — orchestration: governance write-gate, routing, audit.

This is where the control plane enforces policy. Engine I/O is delegated to
adapters (stubs for now); the gate/audit logic is real.
"""

from __future__ import annotations

from fastapi import HTTPException

from app.adapters import get_memory_search_adapters, get_write_adapter
from app.audit.logger import log_event, new_trace_id
from app.auth.context import RequestContext
from app.memory.schemas import (
    GovernanceInfo,
    MemoryDeleteRequest,
    MemoryDeleteResponse,
    MemoryExportRequest,
    MemoryExportResponse,
    MemoryHit,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryStoreRequest,
    MemoryStoreResponse,
    MemoryUpdateRequest,
    MemoryUpdateResponse,
)
from app.policy.rules import PolicyDecision, Status, run_write_gate


def _to_hit(adapter_hit) -> MemoryHit:
    return MemoryHit(
        memory_id=adapter_hit.memory_id,
        backend=adapter_hit.backend,
        content=adapter_hit.content,
        memory_type=adapter_hit.memory_type,
        score=adapter_hit.score,
        citation=adapter_hit.citation,
        metadata=adapter_hit.metadata,
    )


async def search(ctx: RequestContext, req: MemorySearchRequest) -> MemorySearchResponse:
    """Scatter across memory backends; degrade (don't fail) on a backend error."""
    results: list[MemoryHit] = []
    degraded: list[str] = []
    for adapter in get_memory_search_adapters():
        try:
            hits = await adapter.search(
                req.query, workspace_id=req.workspace_id, memory_type=req.memory_type, k=req.k
            )
            results.extend(_to_hit(h) for h in hits)
        except Exception:  # noqa: BLE001 — degrade, don't fail
            degraded.append(adapter.backend)
    log_event(
        tenant_id=ctx.tenant_id, actor_id=ctx.actor_id, operation="search",
        metadata={"query_len": len(req.query), "results": len(results), "degraded": degraded},
    )
    return MemorySearchResponse(results=results, count=len(results), degraded=degraded)


async def store(ctx: RequestContext, req: MemoryStoreRequest) -> MemoryStoreResponse:
    """Governance write-gate → route → adapter.store → audit."""
    trace_id = new_trace_id()
    gate = run_write_gate(req.content)
    governance = GovernanceInfo(
        sensitivity=gate.sensitivity, decision=gate.decision,
        redactions=gate.redactions, reasons=gate.reasons,
    )

    if gate.decision is PolicyDecision.reject:
        log_event(
            tenant_id=ctx.tenant_id, actor_id=ctx.actor_id, operation="store",
            outcome="forbidden", trace_id=trace_id,
            metadata={"reasons": gate.reasons, "sensitivity": gate.sensitivity.value},
        )
        raise HTTPException(
            status_code=403,
            detail={"error": "policy_rejected", "reasons": gate.reasons, "trace_id": trace_id},
        )

    adapter = get_write_adapter(req.memory_type)
    hit = await adapter.store(
        gate.content, memory_type=req.memory_type, workspace_id=req.workspace_id,
        tags=req.tags, metadata={**req.metadata, "sensitivity": gate.sensitivity.value},
    )
    status = Status.quarantined if gate.decision is PolicyDecision.quarantine else Status.active

    log_event(
        tenant_id=ctx.tenant_id, actor_id=ctx.actor_id, operation="store",
        resource_id=hit.memory_id, trace_id=trace_id,
        metadata={"decision": gate.decision.value, "redactions": gate.redactions},
    )
    return MemoryStoreResponse(
        memory_id=hit.memory_id, status=status, governance=governance, trace_id=trace_id
    )


async def update(ctx: RequestContext, req: MemoryUpdateRequest) -> MemoryUpdateResponse:
    trace_id = new_trace_id()
    gate = run_write_gate(req.content)
    governance = GovernanceInfo(
        sensitivity=gate.sensitivity, decision=gate.decision,
        redactions=gate.redactions, reasons=gate.reasons,
    )
    if gate.decision is PolicyDecision.reject:
        raise HTTPException(
            status_code=403,
            detail={"error": "policy_rejected", "reasons": gate.reasons, "trace_id": trace_id},
        )
    adapter = get_write_adapter()
    await adapter.update(req.memory_id, gate.content, metadata=req.metadata)
    log_event(
        tenant_id=ctx.tenant_id, actor_id=ctx.actor_id, operation="update",
        resource_id=req.memory_id, trace_id=trace_id,
    )
    return MemoryUpdateResponse(
        memory_id=req.memory_id, status=Status.active, governance=governance, trace_id=trace_id
    )


async def delete(ctx: RequestContext, req: MemoryDeleteRequest) -> MemoryDeleteResponse:
    trace_id = new_trace_id()
    adapter = get_write_adapter()
    await adapter.delete(req.memory_id)
    log_event(
        tenant_id=ctx.tenant_id, actor_id=ctx.actor_id, operation="delete",
        resource_id=req.memory_id, trace_id=trace_id, metadata={"propagate": req.propagate},
    )
    return MemoryDeleteResponse(
        memory_id=req.memory_id, deleted_from=[adapter.backend], trace_id=trace_id
    )


async def export(ctx: RequestContext, req: MemoryExportRequest) -> MemoryExportResponse:
    adapter = get_write_adapter()
    records = await adapter.export(req.workspace_id)
    log_event(
        tenant_id=ctx.tenant_id, actor_id=ctx.actor_id, operation="export",
        metadata={"count": len(records)},
    )
    return MemoryExportResponse(tenant_id=ctx.tenant_id, count=len(records), records=records)
