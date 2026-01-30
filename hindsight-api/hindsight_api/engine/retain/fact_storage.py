"""
Fact storage for retain pipeline.

Handles insertion of facts into the database.
"""

import json
import logging

from ..memory_engine import fq_table
from .fact_extraction import _sanitize_text
from .types import ProcessedFact

logger = logging.getLogger(__name__)


async def insert_facts_batch(
    conn, bank_id: str, facts: list[ProcessedFact], document_id: str | None = None
) -> list[str]:
    """
    Insert facts into the database in batch.

    Args:
        conn: Database connection
        bank_id: Bank identifier
        facts: List of ProcessedFact objects to insert
        document_id: Optional document ID to associate with facts

    Returns:
        List of unit IDs (UUIDs as strings) for the inserted facts
    """
    if not facts:
        return []

    # Prepare data for batch insert
    fact_texts = []
    embeddings = []
    event_dates = []
    occurred_starts = []
    occurred_ends = []
    mentioned_ats = []
    contexts = []
    fact_types = []
    confidence_scores = []
    metadata_jsons = []
    chunk_ids = []
    document_ids = []
    tags_list = []

    for fact in facts:
        fact_texts.append(_sanitize_text(fact.fact_text))
        # Convert embedding to string for asyncpg vector type
        embeddings.append(str(fact.embedding))
        # event_date: Use occurred_start if available, otherwise use mentioned_at
        # This maintains backward compatibility while handling None occurred_start
        event_dates.append(fact.occurred_start if fact.occurred_start is not None else fact.mentioned_at)
        occurred_starts.append(fact.occurred_start)
        occurred_ends.append(fact.occurred_end)
        mentioned_ats.append(fact.mentioned_at)
        contexts.append(_sanitize_text(fact.context))
        fact_types.append(fact.fact_type)
        # confidence_score is only for opinion facts
        confidence_scores.append(1.0 if fact.fact_type == "opinion" else None)
        metadata_jsons.append(json.dumps(fact.metadata))
        chunk_ids.append(fact.chunk_id)
        # Use per-fact document_id if available, otherwise fallback to batch-level document_id
        document_ids.append(fact.document_id if fact.document_id else document_id)
        # Convert tags to JSON string for proper batch insertion (PostgreSQL unnest doesn't handle 2D arrays well)
        tags_list.append(json.dumps(fact.tags if fact.tags else []))

    # Batch insert all facts
    # Note: tags are passed as JSON strings and converted back to varchar[] via jsonb_array_elements_text + array_agg
    results = await conn.fetch(
        f"""
        WITH input_data AS (
            SELECT * FROM unnest(
                $2::text[], $3::vector[], $4::timestamptz[], $5::timestamptz[], $6::timestamptz[], $7::timestamptz[],
                $8::text[], $9::text[], $10::float[], $11::jsonb[], $12::text[], $13::text[], $14::jsonb[]
            ) AS t(text, embedding, event_date, occurred_start, occurred_end, mentioned_at,
                   context, fact_type, confidence_score, metadata, chunk_id, document_id, tags_json)
        )
        INSERT INTO {fq_table("memory_units")} (bank_id, text, embedding, event_date, occurred_start, occurred_end, mentioned_at,
                                 context, fact_type, confidence_score, metadata, chunk_id, document_id, tags)
        SELECT
            $1,
            text, embedding, event_date, occurred_start, occurred_end, mentioned_at,
            context, fact_type, confidence_score, metadata, chunk_id, document_id,
            COALESCE(
                (SELECT array_agg(elem) FROM jsonb_array_elements_text(tags_json) AS elem),
                '{{}}'::varchar[]
            )
        FROM input_data
        RETURNING id
        """,
        bank_id,
        fact_texts,
        embeddings,
        event_dates,  # event_date: occurred_start if available, else mentioned_at
        occurred_starts,
        occurred_ends,
        mentioned_ats,
        contexts,
        fact_types,
        confidence_scores,
        metadata_jsons,
        chunk_ids,
        document_ids,
        tags_list,
    )

    unit_ids = [str(row["id"]) for row in results]
    return unit_ids


async def ensure_bank_exists(conn, bank_id: str) -> None:
    """
    Ensure bank exists in the database.

    Creates bank with default values if it doesn't exist.

    Args:
        conn: Database connection
        bank_id: Bank identifier
    """
    await conn.execute(
        f"""
        INSERT INTO {fq_table("banks")} (bank_id, disposition, mission)
        VALUES ($1, $2::jsonb, $3)
        ON CONFLICT (bank_id) DO UPDATE
        SET updated_at = NOW()
        """,
        bank_id,
        '{"skepticism": 3, "literalism": 3, "empathy": 3}',
        "",
    )


async def handle_document_tracking(
    conn,
    bank_id: str,
    document_id: str,
    combined_content: str,
    is_first_batch: bool,
    retain_params: dict | None = None,
    document_tags: list[str] | None = None,
) -> None:
    """
    Handle document tracking in the database.

    Args:
        conn: Database connection
        bank_id: Bank identifier
        document_id: Document identifier
        combined_content: Combined content text from all content items
        is_first_batch: Whether this is the first batch (for chunked operations)
        retain_params: Optional parameters passed during retain (context, event_date, etc.)
        document_tags: Optional list of tags to associate with the document
    """
    import hashlib

    # Sanitize and calculate content hash
    combined_content = _sanitize_text(combined_content) or ""
    content_hash = hashlib.sha256(combined_content.encode()).hexdigest()

    # Always delete old document first if it exists (cascades to units and links)
    # Only delete on the first batch to avoid deleting data we just inserted
    if is_first_batch:
        await conn.fetchval(
            f"DELETE FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2 RETURNING id", document_id, bank_id
        )

    # Insert document (or update if exists from concurrent operations)
    await conn.execute(
        f"""
        INSERT INTO {fq_table("documents")} (id, bank_id, original_text, content_hash, metadata, retain_params, tags)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (id, bank_id) DO UPDATE
        SET original_text = EXCLUDED.original_text,
            content_hash = EXCLUDED.content_hash,
            metadata = EXCLUDED.metadata,
            retain_params = EXCLUDED.retain_params,
            tags = EXCLUDED.tags,
            updated_at = NOW()
        """,
        document_id,
        bank_id,
        combined_content,
        content_hash,
        json.dumps({}),  # Empty metadata dict
        json.dumps(retain_params) if retain_params else None,
        document_tags or [],
    )
