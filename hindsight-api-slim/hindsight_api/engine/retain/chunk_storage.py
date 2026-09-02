"""
Chunk storage for retain pipeline.

Handles storage of document chunks in the database.
"""

import hashlib
import logging
from dataclasses import dataclass

from ...config import _get_raw_config
from ..memory_engine import fq_table
from .types import ChunkMetadata

logger = logging.getLogger(__name__)

# Page size for walking the facts a chunk owns out of a store that keeps memories outside SQL.
_OUTGOING_PAGE = 500


def compute_chunk_hash(chunk_text: str) -> str:
    """Compute SHA256 hash of chunk text for delta comparison."""
    return hashlib.sha256(chunk_text.encode()).hexdigest()


@dataclass
class ExistingChunk:
    """Represents a chunk already stored in the database."""

    chunk_id: str
    chunk_index: int
    content_hash: str | None


async def load_existing_chunks(conn, bank_id: str, document_id: str) -> list[ExistingChunk]:
    """
    Load existing chunk metadata for a document.

    Returns list of ExistingChunk with chunk_id, chunk_index, and content_hash.
    """
    rows = await conn.fetch(
        f"""
        SELECT chunk_id, chunk_index, content_hash
        FROM {fq_table("chunks")}
        WHERE document_id = $1 AND bank_id = $2
        ORDER BY chunk_index
        """,
        document_id,
        bank_id,
    )
    return [
        ExistingChunk(
            chunk_id=row["chunk_id"],
            chunk_index=row["chunk_index"],
            content_hash=row["content_hash"],
        )
        for row in rows
    ]


async def memory_ids_for_chunks(conn, bank_id: str, chunk_ids: list[str], *, store=None) -> list[str]:
    """Ids of the facts these chunks own, asked of whichever store holds them.

    The SQL store keeps ``chunk_id`` as a column; a store that keeps memories outside SQL
    carries it in the metadata bag (the same key its ``delete_where`` predicate matches on),
    so the two are read differently. Only ``experience``/``world`` units are returned:
    observations are not chunk-scoped, and feeding one back as a *source* id would be
    meaningless. Paged to exhaustion — every id is about to be deleted, and a chunk whose
    facts overflow one page must not keep half of them.

    ``store`` is the provider the caller already resolved for this bank. A caller inside a
    store-owned path has it in hand, and passing it is what keeps this from re-deciding ownership
    against the global registry: that registry answers for the PROCESS, and a router serving both
    kinds of bank would send a store-owned delta down the SQL branch -- reading with the connection
    that path deliberately never acquired.
    """
    from ..memories import META_CHUNK_ID, get_memories

    store = get_memories() if store is None else store
    if not store.store_owned_for(bank_id):
        rows = await conn.fetch(
            f"""
            SELECT id
            FROM {fq_table("memory_units")}
            WHERE bank_id = $1
              AND chunk_id = ANY($2::text[])
              AND fact_type IN ('experience', 'world')
            """,
            bank_id,
            chunk_ids,
        )
        return [str(row["id"]) for row in rows]

    unit_ids: list[str] = []
    for chunk_id in chunk_ids:
        page_token = ""
        while True:
            page = await store.scan_memories(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                fact_types=["experience", "world"],
                metadata_equals={META_CHUNK_ID: chunk_id},
                limit=_OUTGOING_PAGE,
                page_token=page_token,
            )
            unit_ids.extend(m.unit_id for m in page.memories)
            page_token = page.next_page_token
            if not page_token:
                break
    return unit_ids


async def delete_chunks_by_ids(conn, chunk_ids: list[str], bank_id: str | None = None, ops=None) -> int:
    """
    Delete specific chunks by their IDs.

    This cascades to memory_units (via FK with CASCADE delete)
    and their links.

    ``ops`` is the backend-specific DataAccessOps the observation sweep below needs to choose
    the PG (native array) vs Oracle (junction table) read path — pass ``pool.ops``.

    Returns the number of observations invalidated by the sweep, so the caller can log it.
    """
    if not chunk_ids:
        return 0

    # Delete the observations derived from the facts these chunks own, BEFORE the facts
    # themselves go. Nothing can reach those observations afterwards: consolidation batches
    # are built from facts, so an observation whose sources are all deleted is never selected
    # into a batch again, and it stays valid and recallable — stale knowledge from the previous
    # version of the document surviving the replace (issue #3294). The full-replace path does
    # this in ``handle_document_tracking``; the delta path deletes facts through this cascade
    # instead, which is why the sweep has to live here rather than at one of the call sites.
    invalidated = 0
    if bank_id:
        outgoing_unit_ids = await memory_ids_for_chunks(conn, bank_id, chunk_ids)
        if outgoing_unit_ids:
            from ..graph_maintenance import enqueue_entity_prune_candidates
            from .fact_storage import delete_stale_observations_for_memories

            invalidated = await delete_stale_observations_for_memories(conn, bank_id, outgoing_unit_ids, ops=ops)
            # Queue the entities these facts reference BEFORE the cascade takes
            # their unit_entities rows: afterwards an entity whose last posting
            # was here is unreachable garbage. Delta retain deletes facts only
            # through this cascade, so this is the one place that can catch them
            # (the full-replace path enqueues in ``handle_document_tracking``).
            await enqueue_entity_prune_candidates(conn, bank_id, outgoing_unit_ids)

            # Capture surviving units whose temporal/semantic links point at
            # the outgoing facts before the link/chunk cascade below removes
            # the evidence needed to find them. Full document replacement does
            # the same in ``handle_document_tracking``; without it, a delta
            # edit leaves survivors permanently below their configured link
            # caps even though retain submits graph maintenance afterwards.
            from ..graph_maintenance import enqueue_relink_victims

            await enqueue_relink_victims(conn, bank_id, outgoing_unit_ids)

    # The chunks->memory_units FK cascade below does not reach a store that keeps memories
    # outside SQL (its memory_units is empty), so drop the memories carrying each deleted
    # chunk_id through the store — otherwise a delta re-ingest leaves the old ones as duplicates.
    from ..memories import META_CHUNK_ID, DeletePredicate, get_memories

    _store = get_memories()
    if bank_id and _store.store_owned_for(bank_id):
        for _cid in chunk_ids:
            await _store.delete_where(bank_id, DeletePredicate(metadata_equals={META_CHUNK_ID: _cid}))

    # PostgreSQL's FK cascade deletes child memory_links in executor-chosen
    # order. Concurrent chunk deletes for the same bank can then lock overlapping
    # memory_links in opposite orders and deadlock. Delete links explicitly in a
    # total order before deleting chunks so every writer takes row locks the same
    # way; the FK cascade still handles anything inserted later in this transaction.
    #
    # ``matched_links`` collects the endpoints as a UNION of two single-column joins
    # rather than the one ``tu.id = ml.from_unit_id OR tu.id = ml.to_unit_id`` predicate
    # it replaces. An OR spanning two columns of ``ml`` is not indexable: the planner
    # cannot drive it from either endpoint index, so it made memory_links the outer
    # relation of a nested-loop semi join and sequentially scanned the whole table once
    # per delete — O(rows_in_memory_links x target_units). Past a few million links that
    # exceeded the asyncpg command timeout and delta retain failed with a bare
    # TimeoutError (issue #3387). Split in two, each half is an index scan on
    # idx_memory_links_from_type_weight / idx_memory_links_to_type_weight.
    # The UNION yields the identical row set; the deterministic ORDER BY and
    # FOR UPDATE that #2570 added stay in ``ordered_links``, which locks the rows in
    # that order after the endpoints have been found.
    await conn.execute(
        f"""
        WITH target_units AS MATERIALIZED (
            SELECT id
            FROM {fq_table("memory_units")}
            WHERE chunk_id = ANY($1::text[])
        ),
        matched_links AS MATERIALIZED (
            SELECT ml.ctid AS link_ctid
            FROM {fq_table("memory_links")} ml
            JOIN target_units tu ON tu.id = ml.from_unit_id
            UNION
            SELECT ml.ctid AS link_ctid
            FROM {fq_table("memory_links")} ml
            JOIN target_units tu ON tu.id = ml.to_unit_id
        ),
        ordered_links AS MATERIALIZED (
            SELECT ml.ctid
            FROM {fq_table("memory_links")} ml
            JOIN matched_links ON ml.ctid = matched_links.link_ctid
            ORDER BY
                LEAST(ml.from_unit_id, ml.to_unit_id),
                GREATEST(ml.from_unit_id, ml.to_unit_id),
                ml.link_type,
                COALESCE(ml.entity_id, '00000000-0000-0000-0000-000000000000'::uuid)
            FOR UPDATE OF ml
        )
        DELETE FROM {fq_table("memory_links")} ml
        USING ordered_links ol
        WHERE ml.ctid = ol.ctid
        """,
        chunk_ids,
    )
    await conn.execute(
        f"""
        WITH ordered_chunks AS MATERIALIZED (
            SELECT chunk_id
            FROM {fq_table("chunks")}
            WHERE chunk_id = ANY($1::text[])
            ORDER BY chunk_id
            FOR UPDATE
        )
        DELETE FROM {fq_table("chunks")} c
        USING ordered_chunks oc
        WHERE c.chunk_id = oc.chunk_id
        """,
        chunk_ids,
    )
    return invalidated


async def store_chunks_batch(
    conn,
    bank_id: str,
    document_id: str,
    chunks: list[ChunkMetadata],
    ops=None,
    store_document_text: bool | None = None,
) -> dict[int, str]:
    """
    Store document chunks in the database.

    Args:
        conn: Database connection
        bank_id: Bank identifier
        document_id: Document identifier
        chunks: List of ChunkMetadata objects
        ops: DataAccessOps instance (from backend.ops)
        store_document_text: Whether to persist raw chunk text. When ``None``,
            falls back to the server-level default; callers on the retain path
            pass the per-bank resolved value.

    Returns:
        Dictionary mapping global chunk index to chunk_id
    """
    if not chunks:
        return {}

    # When document text storage is disabled, persist empty chunk_text (the
    # column is NOT NULL) while still computing content_hash from the real text
    # so delta-retain dedup is unaffected.
    # Fallback to the raw global default (not get_config(), which guards
    # bank-configurable fields); the retain path always passes the resolved value.
    store_text = store_document_text if store_document_text is not None else _get_raw_config().store_document_text
    # A store that owns a dedicated document store keeps the chunk TEXT there, so the
    # SQL chunks row carries only its metadata (chunk_id, index, content_hash) with empty text —
    # same shape as store_document_text=False, and idempotency is unaffected (content_hash stays).
    from ..memories import get_memories

    if get_memories().store_owned_for(bank_id):
        store_text = False

    # Prepare chunk data for batch insert
    chunk_ids = []
    chunk_texts = []
    chunk_indices = []
    content_hashes = []
    chunk_id_map = {}

    for chunk in chunks:
        chunk_id = f"{bank_id}_{document_id}_{chunk.chunk_index}"
        chunk_ids.append(chunk_id)
        chunk_texts.append(chunk.chunk_text if store_text else "")
        chunk_indices.append(chunk.chunk_index)
        content_hashes.append(compute_chunk_hash(chunk.chunk_text))
        chunk_id_map[chunk.chunk_index] = chunk_id

    # Batch upsert all chunks. ON CONFLICT makes this idempotent: re-submitting
    # a retain under the same document_id may produce chunk_ids that already exist.
    # Overwriting is the correct behavior per document_id grouping semantics.
    await ops.bulk_upsert_chunks(
        conn,
        fq_table("chunks"),
        chunk_ids,
        [document_id] * len(chunk_texts),
        [bank_id] * len(chunk_texts),
        chunk_texts,
        chunk_indices,
        content_hashes,
    )

    return chunk_id_map
