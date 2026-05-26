"""
Tool implementations for the reflect agent.

Implements hierarchical retrieval:
1. search_mental_models - User-curated stored reflect responses (highest quality)
2. search_observations - Consolidated knowledge with freshness
3. recall - Raw facts as ground truth
"""

import json
import logging
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from asyncpg import Connection

    from ...api.http import RequestContext
    from ..memory_engine import MemoryEngine

logger = logging.getLogger(__name__)


def _document_metadata_from_retain_params(retain_params: Any) -> dict[str, Any] | None:
    """Return document metadata stored under retain_params.metadata."""
    if isinstance(retain_params, str):
        try:
            retain_params = json.loads(retain_params)
        except json.JSONDecodeError:
            return None

    if not isinstance(retain_params, dict):
        return None

    metadata = retain_params.get("metadata")
    return metadata if isinstance(metadata, dict) else None


async def tool_search_mental_models(
    memory_engine: "MemoryEngine",
    conn: "Connection",
    bank_id: str,
    query: str,
    query_embedding: list[float],
    max_results: int = 5,
    tags: list[str] | None = None,
    tags_match: str = "any",
    tag_groups: "list | None" = None,
    exclude_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Search user-curated mental models by semantic similarity.

    Mental models are high-quality, manually created summaries about specific topics.
    They should be searched FIRST as they represent the most reliable synthesized knowledge.

    Args:
        conn: Database connection
        bank_id: Bank identifier
        query: Search query (for logging/tracing)
        query_embedding: Pre-computed embedding for semantic search
        max_results: Maximum number of mental models to return
        tags: Optional tags to filter mental models
        tags_match: How to match tags - "any" (OR), "all" (AND)
        exclude_ids: Optional list of mental model IDs to exclude (e.g., when refreshing a mental model)

    Returns:
        Dict with matching mental models including content and freshness info
    """
    from ..memory_engine import fq_table
    from ..search.tags import build_tag_groups_where_clause, build_tags_where_clause

    # Build filters dynamically
    filters = ""
    params: list[Any] = [bank_id, str(query_embedding), max_results]
    next_param = 4

    # Use the centralized tag filtering logic
    if tags:
        tag_clause, tag_params, next_param = build_tags_where_clause(tags, param_offset=next_param, match=tags_match)
        filters += f" {tag_clause}"
        params.extend(tag_params)

    if tag_groups:
        groups_clause, groups_params, next_param = build_tag_groups_where_clause(tag_groups, next_param)
        filters += f" {groups_clause}"
        params.extend(groups_params)

    if exclude_ids:
        filters += f" AND id != ALL(${next_param}::text[])"
        params.append(exclude_ids)
        next_param += 1

    # Search mental models by embedding similarity
    rows = await conn.fetch(
        f"""
        SELECT
            id, name, content,
            tags, created_at, last_refreshed_at, trigger,
            1 - (embedding <=> $2::vector) as relevance
        FROM {fq_table("mental_models")}
        WHERE bank_id = $1 AND embedding IS NOT NULL {filters}
        ORDER BY embedding <=> $2::vector
        LIMIT $3
        """,
        *params,
    )

    mental_models = []

    for row in rows:
        last_refreshed_at = row["last_refreshed_at"]
        if last_refreshed_at and last_refreshed_at.tzinfo is None:
            last_refreshed_at = last_refreshed_at.replace(tzinfo=timezone.utc)

        # Per-MM staleness: new in-scope memories since last refresh (includes pending).
        is_stale = await memory_engine.compute_mental_model_is_stale(conn, bank_id, row)
        staleness_reason = "new in-scope memories ingested since last refresh" if is_stale else None

        mental_models.append(
            {
                "id": str(row["id"]),
                "name": row["name"],
                "content": row["content"],
                "tags": row["tags"] or [],
                "relevance": round(row["relevance"], 4),
                "updated_at": last_refreshed_at.isoformat() if last_refreshed_at else None,
                "is_stale": is_stale,
                "staleness_reason": staleness_reason,
            }
        )

    return {
        "query": query,
        "count": len(mental_models),
        "mental_models": mental_models,
    }


async def tool_search_observations(
    memory_engine: "MemoryEngine",
    bank_id: str,
    query: str,
    request_context: "RequestContext",
    max_tokens: int = 5000,
    tags: list[str] | None = None,
    tags_match: str = "any",
    tag_groups: "list | None" = None,
    last_consolidated_at: datetime | None = None,
    pending_consolidation: int = 0,
    source_facts_max_tokens: int = -1,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
) -> dict[str, Any]:
    """
    Search consolidated observations using recall.

    Observations are auto-generated from memories. Returns freshness info
    so the agent knows if it should also verify with recall().

    Args:
        memory_engine: Memory engine instance
        bank_id: Bank identifier
        query: Search query
        request_context: Request context for authentication
        max_tokens: Maximum tokens for results (default 5000)
        tags: Optional tags to filter observations
        tags_match: How to match tags - "any" (OR), "all" (AND)
        last_consolidated_at: When consolidation last ran (for staleness check)
        pending_consolidation: Number of memories waiting to be consolidated
        source_facts_max_tokens: Token budget for source facts (-1 = disabled, 0+ = enabled with limit)

    Returns:
        Dict with matching observations including freshness info and source memories
    """
    include_source_facts = source_facts_max_tokens != -1
    recall_kwargs: dict[str, Any] = {}
    if include_source_facts and source_facts_max_tokens > 0:
        recall_kwargs["max_source_facts_tokens"] = source_facts_max_tokens

    # Use an internal request context so this recall is not billed as a
    # user-facing operation. The reflect caller is already billed for the
    # overall reflect operation; double-billing the sub-recalls would
    # overcharge the customer.
    internal_ctx = replace(request_context, internal=True)
    result = await memory_engine.recall_async(
        bank_id=bank_id,
        query=query,
        fact_type=["observation"],
        max_tokens=max_tokens,
        enable_trace=False,
        request_context=internal_ctx,
        tags=tags,
        tags_match=tags_match,
        tag_groups=tag_groups,
        include_source_facts=include_source_facts,
        created_after=created_after,
        created_before=created_before,
        _connection_budget=1,
        _quiet=True,
        **recall_kwargs,
    )

    is_stale = pending_consolidation > 0
    if pending_consolidation == 0:
        freshness = "up_to_date"
    elif pending_consolidation < 10:
        freshness = "slightly_stale"
    else:
        freshness = "stale"

    return {
        "query": query,
        "count": len(result.results),
        "observations": [m.model_dump() for m in result.results],
        "source_facts": {k: v.model_dump() for k, v in (result.source_facts or {}).items()},
        "is_stale": is_stale,
        "freshness": freshness,
    }


async def tool_recall(
    memory_engine: "MemoryEngine",
    bank_id: str,
    query: str,
    request_context: "RequestContext",
    max_tokens: int = 2048,
    tags: list[str] | None = None,
    tags_match: str = "any",
    tag_groups: "list | None" = None,
    connection_budget: int = 1,
    max_chunk_tokens: int = 1000,
    fact_types: list[str] | None = None,
    include_chunks: bool = True,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
) -> dict[str, Any]:
    """
    Search memories using TEMPR retrieval.

    This is the ground truth - raw facts and experiences.
    Use when mental models/observations don't exist, are stale, or need verification.

    Args:
        memory_engine: Memory engine instance
        bank_id: Bank identifier
        query: Search query
        request_context: Request context for authentication
        max_tokens: Maximum tokens for results (default 2048)
        tags: Filter by tags (includes untagged memories)
        tags_match: How to match tags - "any" (OR), "all" (AND), or "exact"
        connection_budget: Max DB connections for this recall (default 1 for internal ops)
        max_chunk_tokens: Maximum tokens for raw source chunk text (default 1000)
        fact_types: Optional filter for fact types to retrieve. Defaults to ["experience", "world"].
        include_chunks: Whether to fetch raw chunk text alongside facts (default True).

    Returns:
        Dict with list of matching memories including raw chunk text (when include_chunks)
    """
    # Only world/experience are valid for raw recall (observation is handled by search_observations)
    recall_fact_type = [ft for ft in (fact_types or ["experience", "world"]) if ft in ("world", "experience")]
    internal_ctx = replace(request_context, internal=True)
    result = await memory_engine.recall_async(
        bank_id=bank_id,
        query=query,
        fact_type=recall_fact_type,
        max_tokens=max_tokens,
        enable_trace=False,
        request_context=internal_ctx,
        tags=tags,
        tags_match=tags_match,
        tag_groups=tag_groups,
        created_after=created_after,
        created_before=created_before,
        _connection_budget=connection_budget,
        _quiet=True,  # Suppress logging for internal operations
        include_chunks=include_chunks,
        max_chunk_tokens=max_chunk_tokens,
    )

    return {
        "query": query,
        "memories": [m.model_dump() for m in result.results],
        "chunks": {k: v.model_dump() for k, v in (result.chunks or {}).items()},
    }


async def tool_expand(
    conn: "Connection",
    bank_id: str,
    memory_ids: list[str],
    depth: str,
) -> dict[str, Any]:
    """
    Expand multiple memories to get chunk or document context.

    Args:
        conn: Database connection
        bank_id: Bank identifier
        memory_ids: List of memory unit IDs
        depth: "chunk" or "document"

    Returns:
        Dict with results array, each containing memory, chunk, and optionally document data
    """
    from ..memory_engine import fq_table

    if not memory_ids:
        return {"error": "memory_ids is required and must not be empty"}

    # Validate and convert UUIDs
    valid_uuids: list[uuid.UUID] = []
    errors: dict[str, str] = {}
    for mid in memory_ids:
        try:
            valid_uuids.append(uuid.UUID(mid))
        except ValueError:
            errors[mid] = f"Invalid memory_id format: {mid}"

    if not valid_uuids:
        return {"error": "No valid memory IDs provided", "details": errors}

    # Batch fetch all memory units
    memories = await conn.fetch(
        f"""
        SELECT id, text, chunk_id, document_id, fact_type, context
        FROM {fq_table("memory_units")}
        WHERE id = ANY($1) AND bank_id = $2
        """,
        valid_uuids,
        bank_id,
    )
    memory_map = {row["id"]: row for row in memories}

    # Collect chunk_ids and document_ids for batch fetching
    chunk_ids = [m["chunk_id"] for m in memories if m["chunk_id"]]
    doc_ids_from_chunks: set[str] = set()
    doc_ids_direct: set[str] = set()

    # Batch fetch all chunks
    chunk_map: dict[str, Any] = {}
    if chunk_ids:
        chunks = await conn.fetch(
            f"""
            SELECT chunk_id, chunk_text, chunk_index, document_id
            FROM {fq_table("chunks")}
            WHERE chunk_id = ANY($1)
            """,
            chunk_ids,
        )
        chunk_map = {row["chunk_id"]: row for row in chunks}
        if depth == "document":
            doc_ids_from_chunks = {c["document_id"] for c in chunks if c["document_id"]}

    # Collect direct document IDs (memories without chunks)
    if depth == "document":
        for m in memories:
            if not m["chunk_id"] and m["document_id"]:
                doc_ids_direct.add(m["document_id"])

    # Batch fetch all documents
    doc_map: dict[str, Any] = {}
    all_doc_ids = list(doc_ids_from_chunks | doc_ids_direct)
    if all_doc_ids:
        docs = await conn.fetch(
            f"""
            SELECT id, original_text, retain_params
            FROM {fq_table("documents")}
            WHERE id = ANY($1) AND bank_id = $2
            """,
            all_doc_ids,
            bank_id,
        )
        doc_map = {row["id"]: row for row in docs}

    # Build results
    results: list[dict[str, Any]] = []
    for mid, mem_uuid in zip(memory_ids, valid_uuids):
        if mid in errors:
            results.append({"memory_id": mid, "error": errors[mid]})
            continue

        memory = memory_map.get(mem_uuid)
        if not memory:
            results.append({"memory_id": mid, "error": f"Memory not found: {mid}"})
            continue

        item: dict[str, Any] = {
            "memory_id": mid,
            "memory": {
                "id": str(memory["id"]),
                "text": memory["text"],
                "type": memory["fact_type"],
                "context": memory["context"],
            },
        }

        # Add chunk if available
        if memory["chunk_id"] and memory["chunk_id"] in chunk_map:
            chunk = chunk_map[memory["chunk_id"]]
            item["chunk"] = {
                "id": chunk["chunk_id"],
                "text": chunk["chunk_text"],
                "index": chunk["chunk_index"],
                "document_id": chunk["document_id"],
            }
            # Add document if depth=document
            if depth == "document" and chunk["document_id"] in doc_map:
                doc = doc_map[chunk["document_id"]]
                item["document"] = {
                    "id": doc["id"],
                    "full_text": doc["original_text"],
                    "metadata": _document_metadata_from_retain_params(doc["retain_params"]),
                    "retain_params": doc["retain_params"],
                }
        elif memory["document_id"] and depth == "document" and memory["document_id"] in doc_map:
            # No chunk, but has document_id
            doc = doc_map[memory["document_id"]]
            item["document"] = {
                "id": doc["id"],
                "full_text": doc["original_text"],
                "metadata": _document_metadata_from_retain_params(doc["retain_params"]),
                "retain_params": doc["retain_params"],
            }

        results.append(item)

    return {"results": results, "count": len(results)}
