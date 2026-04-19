"""
Fact storage for retain pipeline.

Handles insertion of facts into the database.
"""

import json
import logging
import uuid

from ...config import get_config
from ..memory_engine import fq_table
from .bank_utils import DEFAULT_DISPOSITION, create_bank_vector_indexes
from .fact_extraction import _sanitize_text
from .types import ProcessedFact

logger = logging.getLogger(__name__)


async def get_document_content(
    conn,
    bank_id: str,
    document_id: str,
) -> str | None:
    """Fetch the original_text of an existing document.

    Returns None if the document does not exist.
    """
    row = await conn.fetchval(
        f"SELECT original_text FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
        document_id,
        bank_id,
    )
    return row


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
    metadata_jsons = []
    chunk_ids = []
    document_ids = []
    tags_list = []
    observation_scopes_list = []
    text_signals_list = []

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
        metadata_jsons.append(json.dumps(fact.metadata))
        chunk_ids.append(fact.chunk_id)
        # Use per-fact document_id if available, otherwise fallback to batch-level document_id
        document_ids.append(fact.document_id if fact.document_id else document_id)
        # Convert tags to JSON string for proper batch insertion (PostgreSQL unnest doesn't handle 2D arrays well)
        tags_list.append(json.dumps(fact.tags if fact.tags else []))
        # observation_scopes: stored as JSONB (string or 2D array), None if not provided
        observation_scopes_list.append(
            json.dumps(fact.observation_scopes) if fact.observation_scopes is not None else None
        )
        # Build text_signals: entity names + date tokens for enriched BM25 indexing
        signal_parts = []
        if fact.entities:
            signal_parts.extend(e.name for e in fact.entities)
        if fact.occurred_start:
            try:
                signal_parts.append(fact.occurred_start.strftime("%B %d %Y").lstrip("0").replace(" 0", " "))
            except (ValueError, AttributeError):
                pass
        if fact.occurred_end and fact.occurred_end != fact.occurred_start:
            try:
                signal_parts.append(fact.occurred_end.strftime("%B %d %Y").lstrip("0").replace(" 0", " "))
            except (ValueError, AttributeError):
                pass
        text_signals_list.append(" ".join(signal_parts) if signal_parts else None)

    # Batch insert all facts
    # Note: tags are passed as JSON strings and converted back to varchar[] via jsonb_array_elements_text + array_agg
    # Query varies based on text search backend
    config = get_config()
    if config.text_search_extension == "vchord":
        # VectorChord: manually tokenize and insert search_vector
        # text_signals (entity names etc.) are included in the tokenize input for enriched BM25
        query = f"""
            WITH input_data AS (
                SELECT * FROM unnest(
                    $2::text[], $3::vector[], $4::timestamptz[], $5::timestamptz[], $6::timestamptz[], $7::timestamptz[],
                    $8::text[], $9::text[], $10::jsonb[], $11::text[], $12::text[], $13::jsonb[], $14::jsonb[], $15::text[]
                ) AS t(text, embedding, event_date, occurred_start, occurred_end, mentioned_at,
                       context, fact_type, metadata, chunk_id, document_id, tags_json,
                       observation_scopes_json, text_signals)
            )
            INSERT INTO {fq_table("memory_units")} (bank_id, text, embedding, event_date, occurred_start, occurred_end, mentioned_at,
                                     context, fact_type, metadata, chunk_id, document_id, tags,
                                     observation_scopes, text_signals, search_vector)
            SELECT
                $1,
                text, embedding, event_date, occurred_start, occurred_end, mentioned_at,
                context, fact_type, metadata, chunk_id, document_id,
                COALESCE(
                    (SELECT array_agg(elem) FROM jsonb_array_elements_text(tags_json) AS elem),
                    '{{}}'::varchar[]
                ),
                observation_scopes_json,
                text_signals,
                tokenize(
                    COALESCE(text, '') || ' ' || COALESCE(context, '') || ' ' || COALESCE(text_signals, ''),
                    'llmlingua2'
                )::bm25_catalog.bm25vector
            FROM input_data
            RETURNING id
        """
    else:  # native or pg_textsearch
        # Native PostgreSQL: search_vector is GENERATED ALWAYS (expression includes text_signals), don't include it
        # pg_textsearch: indexes operate on base columns directly, don't populate search_vector
        query = f"""
            WITH input_data AS (
                SELECT * FROM unnest(
                    $2::text[], $3::vector[], $4::timestamptz[], $5::timestamptz[], $6::timestamptz[], $7::timestamptz[],
                    $8::text[], $9::text[], $10::jsonb[], $11::text[], $12::text[], $13::jsonb[], $14::jsonb[], $15::text[]
                ) AS t(text, embedding, event_date, occurred_start, occurred_end, mentioned_at,
                       context, fact_type, metadata, chunk_id, document_id, tags_json,
                       observation_scopes_json, text_signals)
            )
            INSERT INTO {fq_table("memory_units")} (bank_id, text, embedding, event_date, occurred_start, occurred_end, mentioned_at,
                                     context, fact_type, metadata, chunk_id, document_id, tags,
                                     observation_scopes, text_signals)
            SELECT
                $1,
                text, embedding, event_date, occurred_start, occurred_end, mentioned_at,
                context, fact_type, metadata, chunk_id, document_id,
                COALESCE(
                    (SELECT array_agg(elem) FROM jsonb_array_elements_text(tags_json) AS elem),
                    '{{}}'::varchar[]
                ),
                observation_scopes_json,
                text_signals
            FROM input_data
            RETURNING id
        """

    results = await conn.fetch(
        query,
        bank_id,
        fact_texts,
        embeddings,
        event_dates,  # event_date: occurred_start if available, else mentioned_at
        occurred_starts,
        occurred_ends,
        mentioned_ats,
        contexts,
        fact_types,
        metadata_jsons,
        chunk_ids,
        document_ids,
        tags_list,
        observation_scopes_list,
        text_signals_list,
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
    # Generate internal_id here so we control the value and can use it
    # immediately for HNSW index creation without a RETURNING round-trip.
    internal_id = uuid.uuid4()
    inserted = await conn.fetchval(
        f"""
        INSERT INTO {fq_table("banks")} (bank_id, disposition, mission, internal_id)
        VALUES ($1, $2::jsonb, $3, $4)
        ON CONFLICT (bank_id) DO NOTHING
        RETURNING bank_id
        """,
        bank_id,
        json.dumps(DEFAULT_DISPOSITION),
        "",
        internal_id,
    )
    if inserted:
        # Fresh insert — create per-bank vector indexes
        await create_bank_vector_indexes(conn, bank_id, str(internal_id))


async def delete_stale_observations_for_memories(
    conn,
    bank_id: str,
    fact_ids: "list[str | uuid.UUID]",
) -> int:
    """Delete observations whose source memories are about to be removed.

    Mirrors the cleanup performed by ``MemoryEngine.delete_document`` so that
    every code path that removes ``memory_units`` also removes the
    observations derived from them. Without this, ingesting a fresh version
    of a document via the retain pipeline (which does a full-replace
    ``DELETE FROM documents`` cascade) used to leave orphan observations
    pointing at memory IDs that no longer existed.

    For each observation referencing any of ``fact_ids``:
    1. Delete the observation row (its text is stale once even one source
       memory disappears).
    2. Reset ``consolidated_at = NULL`` on the surviving source memories so
       they get re-consolidated under fresh observations on the next run.

    Must be called within an active transaction, before the source memories
    are deleted.

    Returns the number of observations deleted.
    """
    if not fact_ids:
        return 0

    fact_uuids = [uuid.UUID(str(fid)) if not isinstance(fid, uuid.UUID) else fid for fid in fact_ids]

    affected_obs = await conn.fetch(
        f"""
        SELECT id, source_memory_ids
        FROM {fq_table("memory_units")}
        WHERE bank_id = $1
          AND fact_type = 'observation'
          AND source_memory_ids && $2::uuid[]
        """,
        bank_id,
        fact_uuids,
    )

    if not affected_obs:
        return 0

    deleted_set = {str(uid) for uid in fact_uuids}
    obs_ids = [obs["id"] for obs in affected_obs]
    seen_remaining: set[str] = set()
    remaining_source_ids: list[uuid.UUID] = []
    for obs in affected_obs:
        for src_id in obs["source_memory_ids"] or []:
            src_str = str(src_id)
            if src_str not in deleted_set and src_str not in seen_remaining:
                remaining_source_ids.append(src_id)
                seen_remaining.add(src_str)

    await conn.execute(
        f"DELETE FROM {fq_table('memory_units')} WHERE id = ANY($1::uuid[])",
        obs_ids,
    )

    if remaining_source_ids:
        await conn.execute(
            f"""
            UPDATE {fq_table("memory_units")}
            SET consolidated_at = NULL
            WHERE id = ANY($1::uuid[])
              AND fact_type IN ('experience', 'world')
            """,
            remaining_source_ids,
        )

    logger.info(
        f"[OBSERVATIONS] Deleted {len(obs_ids)} observations, reset {len(remaining_source_ids)} "
        f"source memories for re-consolidation in bank {bank_id}"
    )
    return len(obs_ids)


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
    Handle document tracking in the database (full-replace mode).

    Deletes the existing document (cascading to all units and links) on the
    first batch, then inserts the new document record.

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

    # Delete old document first (cascades to units and links).
    # Only delete on the first batch to avoid deleting data we just inserted.
    # Before the cascade, fan out to delete observations derived from the
    # outgoing memory_units — otherwise the FK ON DELETE CASCADE removes the
    # source memory_units but leaves observation rows pointing at IDs that
    # no longer exist (consolidated_at on co-source memories also stays
    # frozen). Same cleanup the explicit ``delete_document`` API performs.
    if is_first_batch:
        existing_unit_rows = await conn.fetch(
            f"""
            SELECT id FROM {fq_table("memory_units")}
            WHERE document_id = $1 AND fact_type IN ('experience', 'world')
            """,
            document_id,
        )
        existing_unit_ids = [row["id"] for row in existing_unit_rows]
        if existing_unit_ids:
            invalidated = await delete_stale_observations_for_memories(conn, bank_id, existing_unit_ids)
            if invalidated:
                logger.info(
                    f"[RETAIN] Document {document_id} re-ingested: invalidated "
                    f"{invalidated} observation(s) derived from {len(existing_unit_ids)} outgoing memory_units"
                )
        await conn.fetchval(
            f"DELETE FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2 RETURNING id",
            document_id,
            bank_id,
        )

    # Insert document (or update if exists from concurrent operations)
    await _upsert_document_row(conn, bank_id, document_id, combined_content, content_hash, retain_params, document_tags)


async def upsert_document_metadata(
    conn,
    bank_id: str,
    document_id: str,
    combined_content: str,
    retain_params: dict | None = None,
    document_tags: list[str] | None = None,
) -> None:
    """
    Update document metadata without deleting existing facts/chunks.

    Used by delta retain: the document row is upserted but chunks and
    memory_units are managed separately at the chunk level.
    """
    import hashlib

    combined_content = _sanitize_text(combined_content) or ""
    content_hash = hashlib.sha256(combined_content.encode()).hexdigest()

    await _upsert_document_row(conn, bank_id, document_id, combined_content, content_hash, retain_params, document_tags)


async def _upsert_document_row(
    conn,
    bank_id: str,
    document_id: str,
    combined_content: str,
    content_hash: str,
    retain_params: dict | None = None,
    document_tags: list[str] | None = None,
) -> None:
    """Insert or update a document row."""
    await conn.execute(
        f"""
        INSERT INTO {fq_table("documents")} (id, bank_id, original_text, content_hash, retain_params, tags)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (id, bank_id) DO UPDATE
        SET original_text = EXCLUDED.original_text,
            content_hash = EXCLUDED.content_hash,
            retain_params = EXCLUDED.retain_params,
            tags = EXCLUDED.tags,
            updated_at = NOW()
        """,
        document_id,
        bank_id,
        combined_content,
        content_hash,
        json.dumps(retain_params) if retain_params else None,
        document_tags or [],
    )


async def update_memory_units_tags(
    conn,
    bank_id: str,
    document_id: str,
    tags: list[str],
) -> int:
    """
    Update tags on all memory_units belonging to a document.

    Used during delta retain to propagate tag changes to unchanged facts.

    Returns:
        Number of memory units updated.
    """
    result = await conn.execute(
        f"""
        UPDATE {fq_table("memory_units")}
        SET tags = $3, updated_at = NOW()
        WHERE bank_id = $1 AND document_id = $2
        """,
        bank_id,
        document_id,
        tags or [],
    )
    # result is a status string like "UPDATE 5"
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0
