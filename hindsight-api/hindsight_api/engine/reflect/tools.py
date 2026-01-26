"""
Tool implementations for the reflect agent.

Implements hierarchical retrieval:
1. search_mental_models - User-curated stored reflect responses (highest quality)
2. search_observations - Consolidated knowledge with freshness
3. recall - Raw facts as ground truth
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from asyncpg import Connection

    from ...api.http import RequestContext
    from ..memory_engine import MemoryEngine

logger = logging.getLogger(__name__)

# Observation is considered stale if not updated in this many days
STALE_THRESHOLD_DAYS = 7


async def tool_search_mental_models(
    conn: "Connection",
    bank_id: str,
    query: str,
    query_embedding: list[float],
    max_results: int = 5,
    tags: list[str] | None = None,
    tags_match: str = "any",
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

    # Build filters dynamically
    filters = ""
    params: list[Any] = [bank_id, str(query_embedding), max_results]
    next_param = 4

    if tags:
        if tags_match == "all":
            filters += f" AND tags @> ${next_param}::varchar[]"
        else:
            filters += f" AND (tags && ${next_param}::varchar[] OR tags IS NULL OR tags = '{{}}')"
        params.append(tags)
        next_param += 1

    if exclude_ids:
        filters += f" AND id != ALL(${next_param}::uuid[])"
        params.append(exclude_ids)
        next_param += 1

    # Search mental models by embedding similarity
    rows = await conn.fetch(
        f"""
        SELECT
            id, name, content, reflect_response,
            tags, created_at, last_refreshed_at,
            1 - (embedding <=> $2::vector) as relevance
        FROM {fq_table("mental_models")}
        WHERE bank_id = $1 AND embedding IS NOT NULL {filters}
        ORDER BY embedding <=> $2::vector
        LIMIT $3
        """,
        *params,
    )

    now = datetime.now(timezone.utc)
    mental_models = []

    for row in rows:
        last_refreshed_at = row["last_refreshed_at"]
        if last_refreshed_at and last_refreshed_at.tzinfo is None:
            last_refreshed_at = last_refreshed_at.replace(tzinfo=timezone.utc)

        # Calculate freshness
        is_stale = False
        if last_refreshed_at:
            age = now - last_refreshed_at
            is_stale = age > timedelta(days=STALE_THRESHOLD_DAYS)

        mental_models.append(
            {
                "id": str(row["id"]),
                "name": row["name"],
                "content": row["content"],
                "reflect_response": row["reflect_response"],
                "tags": row["tags"] or [],
                "relevance": round(row["relevance"], 4),
                "updated_at": last_refreshed_at.isoformat() if last_refreshed_at else None,
                "is_stale": is_stale,
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
    last_consolidated_at: datetime | None = None,
    pending_consolidation: int = 0,
) -> dict[str, Any]:
    """
    Search consolidated observations using recall with include_observations.

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

    Returns:
        Dict with matching observations including freshness info
    """
    from ..memory_engine import fq_table

    # Use recall to search observations (they come back in results field when fact_type=["observation"])
    result = await memory_engine.recall_async(
        bank_id=bank_id,
        query=query,
        fact_type=["observation"],  # Only retrieve observations
        max_tokens=max_tokens,  # Token budget controls how many observations are returned
        enable_trace=False,
        request_context=request_context,
        tags=tags,
        tags_match=tags_match,
        _connection_budget=1,
        _quiet=True,
    )

    observations = []

    # When fact_type=["observation"], results come back in `results` field as MemoryFact objects
    # We need to fetch additional fields (proof_count, source_memory_ids) from the database
    if result.results:
        obs_ids = [m.id for m in result.results]

        # Fetch proof_count and source_memory_ids for these observations
        pool = await memory_engine._get_pool()
        async with pool.acquire() as conn:
            obs_rows = await conn.fetch(
                f"""
                SELECT id, proof_count, source_memory_ids
                FROM {fq_table("memory_units")}
                WHERE id = ANY($1::uuid[])
                """,
                obs_ids,
            )
            obs_data = {str(row["id"]): row for row in obs_rows}

        for m in result.results:
            # Get additional data from DB lookup
            extra = obs_data.get(m.id, {})
            proof_count = extra.get("proof_count", 1) if extra else 1
            source_ids = extra.get("source_memory_ids", []) if extra else []
            # Convert UUIDs to strings
            source_memory_ids = [str(sid) for sid in (source_ids or [])]

            # Determine staleness
            is_stale = False
            staleness_reason = None
            if pending_consolidation > 0:
                is_stale = True
                staleness_reason = f"{pending_consolidation} memories pending consolidation"

            observations.append(
                {
                    "id": str(m.id),
                    "text": m.text,
                    "proof_count": proof_count,
                    "source_memory_ids": source_memory_ids,
                    "tags": m.tags or [],
                    "is_stale": is_stale,
                    "staleness_reason": staleness_reason,
                }
            )

    # Return freshness info (more understandable than raw pending_consolidation count)
    if pending_consolidation == 0:
        freshness = "up_to_date"
    elif pending_consolidation < 10:
        freshness = "slightly_stale"
    else:
        freshness = "stale"

    return {
        "query": query,
        "count": len(observations),
        "observations": observations,
        "freshness": freshness,
    }


async def tool_recall(
    memory_engine: "MemoryEngine",
    bank_id: str,
    query: str,
    request_context: "RequestContext",
    max_tokens: int = 2048,
    max_results: int = 50,
    tags: list[str] | None = None,
    tags_match: str = "any",
    connection_budget: int = 1,
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
        max_results: Maximum number of results
        tags: Filter by tags (includes untagged memories)
        tags_match: How to match tags - "any" (OR), "all" (AND), or "exact"
        connection_budget: Max DB connections for this recall (default 1 for internal ops)

    Returns:
        Dict with list of matching memories
    """
    result = await memory_engine.recall_async(
        bank_id=bank_id,
        query=query,
        fact_type=["experience", "world"],  # Exclude opinions and observations
        max_tokens=max_tokens,
        enable_trace=False,
        request_context=request_context,
        tags=tags,
        tags_match=tags_match,
        _connection_budget=connection_budget,
        _quiet=True,  # Suppress logging for internal operations
    )

    memories = []
    for m in result.results[:max_results]:
        memories.append(
            {
                "id": str(m.id),
                "text": m.text,
                "type": m.fact_type,
                "entities": m.entities or [],
                "occurred": m.occurred_start,  # Already ISO format string
            }
        )

    return {
        "query": query,
        "count": len(memories),
        "memories": memories,
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
            SELECT id, original_text, metadata, retain_params
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
                    "metadata": doc["metadata"],
                    "retain_params": doc["retain_params"],
                }
        elif memory["document_id"] and depth == "document" and memory["document_id"] in doc_map:
            # No chunk, but has document_id
            doc = doc_map[memory["document_id"]]
            item["document"] = {
                "id": doc["id"],
                "full_text": doc["original_text"],
                "metadata": doc["metadata"],
                "retain_params": doc["retain_params"],
            }

        results.append(item)

    return {"results": results, "count": len(results)}
