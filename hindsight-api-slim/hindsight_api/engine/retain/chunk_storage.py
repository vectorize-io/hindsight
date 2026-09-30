"""
Chunk storage for retain pipeline.

Handles storage of document chunks in the database.
"""

import hashlib
import logging

from ...config import _get_raw_config
from ..chunk_ids import build_chunk_id
from ..memory_engine import fq_table
from .types import ChunkMetadata

logger = logging.getLogger(__name__)


def compute_chunk_hash(chunk_text: str) -> str:
    """Compute SHA256 hash of chunk text for delta comparison."""
    return hashlib.sha256(chunk_text.encode()).hexdigest()


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
    from ..memories import get_memories

    store = get_memories() if store is None else store
    return await store.memory_ids_for_chunks(conn=conn, fq_table=fq_table, bank_id=bank_id, chunk_ids=chunk_ids)


async def delete_chunks_by_ids(conn, chunk_ids: list[str], bank_id: str, ops=None) -> int:
    """
    Delete the named chunks, which must all belong to ``bank_id``.

    This cascades to memory_units (via FK with CASCADE delete)
    and their links.

    ``bank_id`` is required, not optional: `chunks` is keyed on chunk_id alone, so an id
    that collides with another bank's row (possible for rows written before the escaping
    in `chunk_ids.py` -- see #4244) would otherwise let a delta retain here cascade that
    bank's facts away. Every statement below carries it.

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

    # Then the chunks themselves, with the facts and links they own. Postgres takes the row locks
    # in a total order; a store that keeps memories outside SQL (where no FK cascade reaches them)
    # drops the memories carrying each chunk_id — otherwise a delta re-ingest leaves the old ones
    # as duplicates.
    from ..memories import get_memories

    await get_memories().delete_chunks(conn=conn, fq_table=fq_table, bank_id=bank_id, chunk_ids=chunk_ids)
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

    # Prepare chunk data for batch insert
    chunk_ids = []
    chunk_texts = []
    chunk_indices = []
    content_hashes = []
    chunk_id_map = {}

    for chunk in chunks:
        chunk_id = build_chunk_id(bank_id, document_id, chunk.chunk_index)
        chunk_ids.append(chunk_id)
        chunk_texts.append(chunk.chunk_text if store_text else "")
        chunk_indices.append(chunk.chunk_index)
        content_hashes.append(compute_chunk_hash(chunk.chunk_text))
        chunk_id_map[chunk.chunk_index] = chunk_id

    # Batch upsert all chunks. A store that keeps the chunk texts in its own document store has
    # no chunk rows to write (the texts travel with the document record).
    from ..memories import get_memories

    await get_memories().upsert_chunks(
        conn=conn,
        ops=ops,
        fq_table=fq_table,
        bank_id=bank_id,
        document_id=document_id,
        chunk_ids=chunk_ids,
        chunk_texts=chunk_texts,
        chunk_indices=chunk_indices,
        content_hashes=content_hashes,
    )

    return chunk_id_map
