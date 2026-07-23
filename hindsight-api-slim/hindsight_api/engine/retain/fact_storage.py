"""
Fact storage for retain pipeline.

Handles insertion of facts into the database.
"""

import json
import logging
import uuid
from datetime import datetime

from ...config import get_config
from ..memory_engine import fq_table
from .bank_utils import DEFAULT_DISPOSITION, create_bank_vector_indexes
from .fact_extraction import _sanitize_text
from .types import ProcessedFact

logger = logging.getLogger(__name__)

#: Page size for walking a replaced document's outgoing memories. Large enough
#: that one page covers any ordinary document, small enough that a pathological
#: one does not arrive as a single result set.
_OUTGOING_PAGE = 500


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
    conn,
    bank_id: str,
    facts: list[ProcessedFact],
    document_id: str | None = None,
    ops=None,
    defer_index: bool = False,
) -> list[str]:
    """
    Store facts and return their unit ids, in order.

    Args:
        conn: Database connection
        bank_id: Bank identifier
        facts: List of ProcessedFact objects to insert
        document_id: Optional document ID to associate with facts
        defer_index: Ask for ids without the write. The retain orchestrator needs
            this because it can only supply entity ids and causal edges after
            Phase-1 placeholders have been remapped onto real unit ids; it then
            calls `index_facts` with the complete picture. The Postgres store,
            whose write *is* the insert that mints the ids, ignores it.

    Returns:
        List of unit IDs (UUIDs as strings) for the inserted facts
    """
    if not facts:
        return []

    from ..memories import get_memories

    return await get_memories().insert_facts(
        conn=conn,
        ops=ops,
        bank_id=bank_id,
        facts=facts,
        document_id=document_id,
        defer_index=defer_index,
    )


async def index_facts(
    bank_id: str,
    unit_ids: list[str],
    facts: list[ProcessedFact],
    document_id: str | None = None,
    unit_entity_ids: dict[str, list[str]] | None = None,
) -> None:
    """Complete a deferred `insert_facts_batch`, now that the edges are known.

    ``unit_entity_ids`` is the unit→entity posting and each fact's causal
    relations are its edges; both travel with the memory for a store that owns
    them. A no-op for the Postgres store, which wrote all of it already.
    """
    from ..memories import get_memories

    await get_memories().index_facts(bank_id, unit_ids, facts, document_id, unit_entity_ids)


async def ensure_bank_exists(conn, bank_id: str, ops=None) -> None:
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
        INSERT INTO {fq_table("banks")} (bank_id, name, disposition, mission, internal_id)
        VALUES ($1, $2, $3::jsonb, $4, $5)
        ON CONFLICT (bank_id) DO NOTHING
        RETURNING bank_id
        """,
        bank_id,
        bank_id,  # Default name is the bank_id (matches get_or_create_bank_profile)
        json.dumps(DEFAULT_DISPOSITION),
        "",
        internal_id,
    )
    if inserted:
        # Fresh insert — create per-bank vector indexes
        await create_bank_vector_indexes(conn, bank_id, str(internal_id), ops=ops)


async def delete_stale_observations_for_memories(
    conn,
    bank_id: str,
    fact_ids: "list[str | uuid.UUID]",
    ops=None,
) -> int:
    """Delete observations whose source memories are about to be removed.

    Mirrors the cleanup performed by ``MemoryEngine.delete_document`` so that
    every code path that removes memories also removes the observations derived
    from them. Without this, ingesting a fresh version of a document via the
    retain pipeline (which does a full-replace ``DELETE FROM documents``
    cascade) used to leave orphan observations pointing at memory IDs that no
    longer existed.

    For each observation referencing any of ``fact_ids``:
    1. Delete the observation (its text is stale once even one source memory
       disappears).
    2. Reset the consolidated marker on the surviving source memories so they
       get re-consolidated under fresh observations on the next run.

    Must be called within an active transaction, before the source memories are
    deleted.

    Returns:
        Number of observations deleted.
    """
    if not fact_ids:
        return 0

    from ..memories import get_memories

    return await get_memories().delete_stale_observations(
        conn=conn,
        ops=ops,
        fq_table=fq_table,
        bank_id=bank_id,
        fact_ids=fact_ids,
    )


async def handle_document_tracking(
    conn,
    bank_id: str,
    document_id: str,
    combined_content: str,
    is_first_batch: bool,
    retain_params: dict | None = None,
    document_tags: list[str] | None = None,
    ops=None,
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
        ops: Backend-specific DataAccessOps. Required by the inner
            ``delete_stale_observations_for_memories`` call to choose the PG
            (native array) vs Oracle (junction table) read path. Defaults to
            None so older callers don't break, but the PG branch is only
            taken when ops is non-None — pass ``pool.ops`` from the caller.
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
    preserved_created_at = None
    if is_first_batch:
        from ..memories import get_memories

        store = get_memories()
        # Which memories the outgoing version left behind. Asked of the store
        # rather than queried here, because it is the store that knows where they
        # are. Paged to exhaustion: every one of them is about to be deleted, and
        # a document whose facts overflow one page must not keep half of them.
        existing_unit_ids: list[str] = []
        page_token = ""
        while True:
            page = await store.scan_memories(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                fact_types=["experience", "world"],
                document_id=document_id,
                limit=_OUTGOING_PAGE,
                page_token=page_token,
            )
            existing_unit_ids.extend(m.unit_id for m in page.memories)
            page_token = page.next_page_token
            if not page_token:
                break
        if existing_unit_ids:
            invalidated = await delete_stale_observations_for_memories(conn, bank_id, existing_unit_ids, ops=ops)
            if invalidated:
                logger.info(
                    f"[RETAIN] Document {document_id} re-ingested: invalidated "
                    f"{invalidated} observation(s) derived from {len(existing_unit_ids)} outgoing memory_units"
                )
            # Capture link-recompute victims BEFORE the cascade. Same staleness
            # applies on upsert as on explicit delete: surviving units in OTHER
            # documents that linked to these doomed units are about to lose
            # those links. ``ops`` may be None for older callers that haven't
            # been wired up — skip enqueue in that case rather than crash.
            if ops is not None:
                from ..graph_maintenance import enqueue_relink_victims

                await enqueue_relink_victims(conn, bank_id, [str(uid) for uid in existing_unit_ids], ops=ops)

        # Explicitly delete memory_units by document_id BEFORE deleting the
        # document row. The CASCADE from documents→chunks→memory_units only
        # catches units that have a non-NULL chunk_id FK. Units with chunk_id=NULL
        # (e.g. from partial writes or edge cases) would survive the cascade.
        # This explicit delete ensures complete cleanup.
        await store.delete_document(conn=conn, fq_table=fq_table, bank_id=bank_id, document_id=document_id)
        # Capture created_at before deletion so re-ingestion preserves it.
        preserved_created_at = await conn.fetchval(
            f"DELETE FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2 RETURNING created_at",
            document_id,
            bank_id,
        )

    # Insert document (or update if exists from concurrent operations)
    await _upsert_document_row(
        conn,
        bank_id,
        document_id,
        combined_content,
        content_hash,
        retain_params,
        document_tags,
        preserved_created_at=preserved_created_at,
    )


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
    preserved_created_at: datetime | None = None,
) -> None:
    """Insert or update a document row.

    When ``preserved_created_at`` is provided, it is used for ``created_at`` on
    INSERT so that re-ingesting a document (which deletes + inserts the row)
    keeps the original creation timestamp. ``updated_at`` is always set to
    ``NOW()`` on both INSERT and the ON CONFLICT UPDATE branch.

    When ``store_document_text`` is disabled, the raw source text
    is dropped and ``original_text`` is stored as NULL. The ``content_hash`` is
    still computed from the real content so delta-retain dedup is unaffected.
    """
    original_text = combined_content if get_config().store_document_text else None
    await conn.execute(
        f"""
        INSERT INTO {fq_table("documents")} (id, bank_id, original_text, content_hash, retain_params, tags, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, COALESCE($7, NOW()), NOW())
        ON CONFLICT (id, bank_id) DO UPDATE
        SET original_text = EXCLUDED.original_text,
            content_hash = EXCLUDED.content_hash,
            retain_params = EXCLUDED.retain_params,
            tags = EXCLUDED.tags,
            updated_at = NOW()
        """,
        document_id,
        bank_id,
        original_text,
        content_hash,
        json.dumps(retain_params) if retain_params else None,
        document_tags or [],
        preserved_created_at,
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
