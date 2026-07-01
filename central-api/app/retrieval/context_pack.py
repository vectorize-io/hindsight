"""Context-pack builder.

Combines memory + code/doc hits into one governed, cited bundle. Pure function so
it is unit-testable without engines. RRF ordering + dedup is applied here.
"""

from __future__ import annotations

from app.adapters.base import AdapterHit
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.schemas import ContextItem, ContextPack


def _content_key(hit: AdapterHit) -> str:
    return " ".join(hit.content.lower().split())[:200]


def build_context_pack(
    *,
    query: str,
    memory_hits: list[AdapterHit],
    code_hits: list[AdapterHit],
    trace_id: str,
    blocked_items_count: int = 0,
    policy_decisions: list[dict] | None = None,
    limit: int | None = None,
    preview: bool = False,
) -> ContextPack:
    fused = reciprocal_rank_fusion(
        [memory_hits, code_hits], key=_content_key, limit=limit
    )

    selected: list[ContextItem] = []
    citations: list[dict] = []
    for hit, score in fused:
        selected.append(
            ContextItem(
                backend=hit.backend, content=hit.content, score=score, citation=hit.citation
            )
        )
        if hit.citation:
            citations.append(hit.citation)
        elif hit.metadata.get("file"):
            citations.append({"backend": hit.backend, "file": hit.metadata["file"],
                              "line": hit.metadata.get("line")})

    confidence = round(sum(s for _, s in fused) / len(fused), 4) if fused else None

    return ContextPack(
        query=query,
        selected_items=selected,
        citations=citations,
        blocked_items_count=blocked_items_count,
        policy_decisions=policy_decisions or [],
        audit_trace_id=trace_id,
        confidence=confidence,
        preview=preview,
    )
