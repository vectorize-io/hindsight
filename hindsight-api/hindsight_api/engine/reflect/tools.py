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
) -> dict[str, Any]:
    """
    List or get mental models.

    Args:
        conn: Database connection
        bank_id: Bank identifier
        model_id: Optional specific model ID to get (if None, lists all)

    Returns:
        Dict with either a list of models or a single model's details
    """
    if model_id:
        # Get specific mental model with full details
        row = await conn.fetchrow(
            """
            SELECT id, type, subtype, name, description, summary, entity_id, triggers, last_updated
            FROM mental_models
            WHERE id = $1 AND bank_id = $2
            """,
            model_id,
            bank_id,
        )
        if row:
            return {
                "found": True,
                "model": {
                    "id": row["id"],
                    "type": row["type"],
                    "subtype": row["subtype"],
                    "name": row["name"],
                    "description": row["description"],
                    "summary": row["summary"],
                    "entity_id": str(row["entity_id"]) if row["entity_id"] else None,
                    "triggers": row["triggers"] or [],
                    "last_updated": row["last_updated"].isoformat() if row["last_updated"] else None,
                },
            }
        return {"found": False, "model_id": model_id}
    else:
        # List all mental models (name + description only for efficiency)
        rows = await conn.fetch(
            """
            SELECT id, type, subtype, name, description
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
                    "type": row["type"],
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
    max_results: int = 20,
    tags: list[str] | None = None,
    tags_match: str = "any",
) -> dict[str, Any]:
    """
    Search facts using TEMPR retrieval.

    Args:
        memory_engine: Memory engine instance
        bank_id: Bank identifier
        query: Search query
        request_context: Request context for authentication
        max_results: Maximum number of results
        tags: Filter by tags (includes untagged memories)
        tags_match: How to match tags - "any" (OR), "all" (AND), or "exact"

    Returns:
        Dict with list of matching facts
    """
    result = await memory_engine.recall_async(
        bank_id=bank_id,
        query=query,
        fact_type=["experience", "world"],  # Exclude opinions
        max_tokens=4000,
        enable_trace=False,
        request_context=request_context,
        tags=tags,
        tags_match=tags_match,
    )

    facts = []
    for f in result.results[:max_results]:
        facts.append(
            {
                "id": str(f.id),
                "text": f.text,
                "type": f.fact_type,
                "entities": f.entities or [],
                "occurred": f.occurred_start,  # Already ISO format string
            }
        )

    return {
        "query": query,
        "count": len(facts),
        "facts": facts,
    }


async def tool_learn(
    conn: "Connection",
    bank_id: str,
    input: MentalModelInput,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Create or update a mental model with subtype='learned'.

    Args:
        conn: Database connection
        bank_id: Bank identifier
        input: Mental model input data
        tags: Tags to apply to new mental models (from reflect context)

    Returns:
        Dict with created/updated model info
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
        # Update existing model (keep existing tags)
        await conn.execute(
            """
            UPDATE mental_models SET
                description = $3,
                summary = $4,
                triggers = $5,
                entity_id = $6,
                last_updated = NOW()
            WHERE id = $1 AND bank_id = $2
            """,
            model_id,
            bank_id,
            input.description,
            input.summary,
            input.triggers,
            entity_uuid,
        )
        status = "updated"
    else:
        # Insert new model with tags from reflect context
        await conn.execute(
            """
            INSERT INTO mental_models (id, bank_id, type, subtype, name, description, summary, triggers, entity_id, tags, last_updated, created_at)
            VALUES ($1, $2, $3, 'learned', $4, $5, $6, $7, $8, $9, NOW(), NOW())
            """,
            model_id,
            bank_id,
            input.type.value,
            input.name,
            input.description,
            input.summary,
            input.triggers,
            entity_uuid,
            tags or [],
        )
        status = "created"

    logger.info(f"[REFLECT] Mental model '{model_id}' {status} in bank {bank_id}")

    return {
        "status": status,
        "model_id": model_id,
        "name": input.name,
        "type": input.type.value,
    }


async def tool_expand(
    conn: "Connection",
    bank_id: str,
    memory_id: str,
    depth: str,
) -> dict[str, Any]:
    """
    Expand a memory to get chunk or document context.

    Args:
        conn: Database connection
        bank_id: Bank identifier
        memory_id: Memory unit ID
        depth: "chunk" or "document"

    Returns:
        Dict with memory, chunk, and optionally document data
    """
    try:
        mem_uuid = uuid.UUID(memory_id)
    except ValueError:
        return {"error": f"Invalid memory_id format: {memory_id}"}

    # Get memory unit
    memory = await conn.fetchrow(
        """
        SELECT id, text, chunk_id, document_id, fact_type, context
        FROM memory_units
        WHERE id = $1 AND bank_id = $2
        """,
        mem_uuid,
        bank_id,
    )

    if not memory:
        return {"error": f"Memory not found: {memory_id}"}

    result: dict[str, Any] = {
        "memory": {
            "id": str(memory["id"]),
            "text": memory["text"],
            "type": memory["fact_type"],
            "context": memory["context"],
        }
    }

    # Get chunk if available
    if memory["chunk_id"]:
        chunk = await conn.fetchrow(
            """
            SELECT chunk_id, chunk_text, chunk_index, document_id
            FROM chunks
            WHERE chunk_id = $1
            """,
            memory["chunk_id"],
        )
        if chunk:
            result["chunk"] = {
                "id": chunk["chunk_id"],
                "text": chunk["chunk_text"],
                "index": chunk["chunk_index"],
                "document_id": chunk["document_id"],
            }

            # Get document if depth=document
            if depth == "document" and chunk["document_id"]:
                doc = await conn.fetchrow(
                    """
                    SELECT id, original_text, metadata, retain_params
                    FROM documents
                    WHERE id = $1 AND bank_id = $2
                    """,
                    chunk["document_id"],
                    bank_id,
                )
                if doc:
                    result["document"] = {
                        "id": doc["id"],
                        "full_text": doc["original_text"],
                        "metadata": doc["metadata"],
                        "retain_params": doc["retain_params"],
                    }
    elif memory["document_id"]:
        # No chunk, but has document_id
        if depth == "document":
            doc = await conn.fetchrow(
                """
                SELECT id, original_text, metadata, retain_params
                FROM documents
                WHERE id = $1 AND bank_id = $2
                """,
                memory["document_id"],
                bank_id,
            )
            if doc:
                result["document"] = {
                    "id": doc["id"],
                    "full_text": doc["original_text"],
                    "metadata": doc["metadata"],
                    "retain_params": doc["retain_params"],
                }

    return result
