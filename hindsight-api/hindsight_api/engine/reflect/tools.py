"""
Tool implementations for the reflect agent.
"""

import logging
import re
import uuid
from typing import TYPE_CHECKING, Any

from .models import MentalModelInput

if TYPE_CHECKING:
    from asyncpg import Connection

    from ...api.http import RequestContext
    from ..memory_engine import MemoryEngine

logger = logging.getLogger(__name__)


def generate_model_id(name: str) -> str:
    """Generate a stable ID from mental model name."""
    # Normalize: lowercase, replace spaces/special chars with hyphens
    normalized = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    # Truncate to reasonable length
    return normalized[:50]


async def tool_lookup(
    conn: "Connection",
    bank_id: str,
    model_id: str | None = None,
    tags: list[str] | None = None,
    tags_match: str = "any",
) -> dict[str, Any]:
    """
    List or get mental models.

    Args:
        conn: Database connection
        bank_id: Bank identifier
        model_id: Optional specific model ID to get (if None, lists all)
        tags: Optional tags to filter models (when listing)
        tags_match: How to match tags - "any" (OR), "all" (AND)

    Returns:
        Dict with either a list of models or a single model's details
    """
    if model_id:
        # Get specific mental model with full details including observations
        row = await conn.fetchrow(
            """
            SELECT id, subtype, name, description, observations, entity_id, last_updated
            FROM mental_models
            WHERE id = $1 AND bank_id = $2
            """,
            model_id,
            bank_id,
        )
        if row:
            # Parse observations JSON
            obs_data = row["observations"] or {"observations": []}
            if isinstance(obs_data, str):
                import json

                obs_data = json.loads(obs_data)
            observations_raw = obs_data.get("observations", []) if isinstance(obs_data, dict) else obs_data

            # Normalize observation format: map memory_ids/fact_ids to based_on
            observations = []
            for obs in observations_raw:
                if isinstance(obs, dict):
                    based_on = obs.get("memory_ids") or obs.get("fact_ids") or []
                    observations.append(
                        {
                            "title": obs.get("title", ""),
                            "text": obs.get("text", ""),
                            "based_on": based_on,
                        }
                    )

            return {
                "found": True,
                "model": {
                    "id": row["id"],
                    "subtype": row["subtype"],
                    "name": row["name"],
                    "description": row["description"],
                    "observations": observations,  # [{title, text, based_on}, ...]
                    "entity_id": str(row["entity_id"]) if row["entity_id"] else None,
                    "last_updated": row["last_updated"].isoformat() if row["last_updated"] else None,
                },
            }
        return {"found": False, "model_id": model_id}
    else:
        # List mental models (compact: id, name, description only)
        # Full observations are retrieved via get_mental_model(model_id)
        # Filter by tags if provided
        if tags:
            if tags_match == "all":
                # All tags must match
                rows = await conn.fetch(
                    """
                    SELECT id, subtype, name, description
                    FROM mental_models
                    WHERE bank_id = $1 AND tags @> $2::varchar[]
                    ORDER BY last_updated DESC NULLS LAST, created_at DESC
                    """,
                    bank_id,
                    tags,
                )
            else:
                # Any tag matches (OR) - default
                rows = await conn.fetch(
                    """
                    SELECT id, subtype, name, description
                    FROM mental_models
                    WHERE bank_id = $1 AND tags && $2::varchar[]
                    ORDER BY last_updated DESC NULLS LAST, created_at DESC
                    """,
                    bank_id,
                    tags,
                )
        else:
            rows = await conn.fetch(
                """
                SELECT id, subtype, name, description
                FROM mental_models
                WHERE bank_id = $1
                ORDER BY last_updated DESC NULLS LAST, created_at DESC
                """,
                bank_id,
            )

        return {
            "count": len(rows),
            "models": [
                {
                    "id": row["id"],
                    "subtype": row["subtype"],
                    "name": row["name"],
                    "description": row["description"],
                }
                for row in rows
            ],
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
        fact_type=["experience", "world"],  # Exclude opinions
        max_tokens=max_tokens,
        enable_trace=False,
        request_context=request_context,
        tags=tags,
        tags_match=tags_match,
        _connection_budget=connection_budget,
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


async def tool_learn(
    conn: "Connection",
    bank_id: str,
    input: MentalModelInput,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Create a mental model placeholder with subtype='learned'.

    The agent only specifies name and description - actual observations are generated
    in the background via refresh, similar to pinned models.

    Args:
        conn: Database connection
        bank_id: Bank identifier
        input: Mental model input data (name, description, optional entity_id)
        tags: Tags to apply to new mental models (from reflect context)

    Returns:
        Dict with created model info including model_id for background generation
    """
    model_id = generate_model_id(input.name)

    # Parse entity_id if provided
    entity_uuid = None
    if input.entity_id:
        try:
            entity_uuid = uuid.UUID(input.entity_id)
        except ValueError:
            logger.warning(f"Invalid entity_id format: {input.entity_id}")

    # Check if model exists
    existing = await conn.fetchrow(
        "SELECT id FROM mental_models WHERE id = $1 AND bank_id = $2",
        model_id,
        bank_id,
    )

    if existing:
        # Update description only - observations will be regenerated
        await conn.execute(
            """
            UPDATE mental_models SET
                description = $3,
                entity_id = $4
            WHERE id = $1 AND bank_id = $2
            """,
            model_id,
            bank_id,
            input.description,
            entity_uuid,
        )
        status = "updated"
    else:
        # Insert new model placeholder - observations will be generated in background
        await conn.execute(
            """
            INSERT INTO mental_models (id, bank_id, subtype, name, description, observations, entity_id, tags, created_at)
            VALUES ($1, $2, 'learned', $3, $4, '{}'::jsonb, $5, $6, NOW())
            """,
            model_id,
            bank_id,
            input.name,
            input.description,
            entity_uuid,
            tags or [],
        )
        status = "created"

    logger.info(f"[REFLECT] Mental model '{model_id}' {status} in bank {bank_id} - pending background generation")

    return {
        "status": status,
        "model_id": model_id,
        "name": input.name,
        "pending_generation": True,
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
        """
        SELECT id, text, chunk_id, document_id, fact_type, context
        FROM memory_units
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
            """
            SELECT chunk_id, chunk_text, chunk_index, document_id
            FROM chunks
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
            """
            SELECT id, original_text, metadata, retain_params
            FROM documents
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
