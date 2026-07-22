"""Writes against `memory_units`: the fact insert, the deletes, and observation invalidation.

Everything here mutates the memories slice and nothing else. The document row,
the chunks, the entity registry and the link tables stay with their own callers —
what lands in this module is only the statements that touch `memory_units` (and,
on backends that keep one, the `observation_sources` junction that hangs off it).

Each function takes the live connection and Hindsight's ``fq_table`` resolver, so
it runs inside whatever transaction the caller already holds; ``ops`` is the
dialect ops object, which is what lets the same code serve the PG (native array)
and Oracle (junction table) shapes of the observation→source relation.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING

from ....config import get_config
from ..base import StoredMemory

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...retain.types import ProcessedFact

logger = logging.getLogger(__name__)


async def insert_facts(
    *,
    conn,
    ops,
    bank_id: str,
    facts: list[ProcessedFact],
    document_id: str | None = None,
) -> list[str]:
    """Insert facts into the database in batch.

    Args:
        conn: Database connection
        bank_id: Bank identifier
        facts: List of ProcessedFact objects to insert
        document_id: Optional document ID to associate with facts

    Returns:
        List of unit IDs (UUIDs as strings) for the inserted facts, in the same
        order as ``facts``.
    """
    if not facts:
        return []

    # Imported here: `retain` reaches back into the engine for `fq_table`, so a
    # module-level import would close the cycle once the engine imports this store.
    from ...retain.fact_extraction import _sanitize_text

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

    # Batch insert all facts — delegates to DataAccessOps which handles
    # unnest (PG) vs row-by-row (Oracle) transparently.
    config = get_config()

    return await ops.insert_facts_batch(
        conn,
        bank_id,
        fact_texts,
        embeddings,
        event_dates,
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
        text_search_extension=config.text_search_extension,
    )


async def delete_document(*, conn, fq_table: Callable[[str], str], bank_id: str, document_id: str) -> None:
    """Delete every memory unit belonging to ``document_id``.

    Explicitly delete memory_units by document_id BEFORE deleting the
    document row. The CASCADE from documents→chunks→memory_units only
    catches units that have a non-NULL chunk_id FK. Units with chunk_id=NULL
    (e.g. from partial writes or edge cases) would survive the cascade.
    This explicit delete ensures complete cleanup.

    Called when a document is replaced, so it races the replacement's writes: it
    must remove only what was written *before* this call, never the facts
    arriving moments later — which the ``document_id``/``bank_id`` predicate
    gives for free inside the caller's transaction.
    """
    await conn.execute(
        f"DELETE FROM {fq_table('memory_units')} WHERE document_id = $1 AND bank_id = $2",
        document_id,
        bank_id,
    )


async def delete_observations(*, conn, fq_table: Callable[[str], str], bank_id: str) -> None:
    """Delete all observations in a bank, leaving the facts behind them.

    Only the observation rows: requeuing the surviving sources (clearing
    ``consolidated_at``) and resetting the bank's consolidation timestamp belong
    to the caller, which owns the bank row.
    """
    await conn.execute(
        f"DELETE FROM {fq_table('memory_units')} WHERE bank_id = $1 AND fact_type = 'observation'",
        bank_id,
    )


async def observations_for_sources(
    *,
    conn,
    ops,
    fq_table: Callable[[str], str],
    bank_id: str,
    unit_ids: list[str | uuid.UUID],
) -> list[StoredMemory]:
    """Observations consolidated from any of ``unit_ids``.

    Only ``unit_id`` and ``source_memory_ids`` are populated — the caller uses
    them to delete the observations and to work out which sources survive, and
    the rest of the row is about to be deleted anyway.
    """
    if not unit_ids:
        return []

    fact_uuids = [uuid.UUID(str(fid)) if not isinstance(fid, uuid.UUID) else fid for fid in unit_ids]

    if ops is not None and not ops.uses_observation_sources_table:
        # PG: use native array overlap operator
        rows = await conn.fetch(
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
    else:
        # Oracle / default: use observation_sources junction table
        rows = await conn.fetch(
            f"""
            SELECT mu.id, mu.source_memory_ids
            FROM {fq_table("memory_units")} mu
            WHERE mu.bank_id = $1
              AND mu.fact_type = 'observation'
              AND EXISTS (
                  SELECT 1 FROM {fq_table("observation_sources")} os
                  WHERE os.observation_id = mu.id
                    AND os.source_id = ANY($2::uuid[])
              )
            """,
            bank_id,
            fact_uuids,
        )

    return [
        StoredMemory(
            unit_id=str(row["id"]),
            text="",
            fact_type="observation",
            source_memory_ids=[str(src_id) for src_id in (row["source_memory_ids"] or [])],
        )
        for row in rows
    ]


async def delete_stale_observations(
    *,
    conn,
    ops,
    fq_table: Callable[[str], str],
    bank_id: str,
    fact_ids: list[str | uuid.UUID],
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

    affected_obs = await observations_for_sources(
        conn=conn, ops=ops, fq_table=fq_table, bank_id=bank_id, unit_ids=fact_uuids
    )
    if not affected_obs:
        return 0

    deleted_set = {str(uid) for uid in fact_uuids}
    obs_ids = [uuid.UUID(obs.unit_id) for obs in affected_obs]
    seen_remaining: set[str] = set()
    remaining_source_ids: list[uuid.UUID] = []
    for obs in affected_obs:
        for src_str in obs.source_memory_ids:
            if src_str not in deleted_set and src_str not in seen_remaining:
                remaining_source_ids.append(uuid.UUID(src_str))
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


__all__ = [
    "delete_document",
    "delete_observations",
    "delete_stale_observations",
    "insert_facts",
    "observations_for_sources",
]
