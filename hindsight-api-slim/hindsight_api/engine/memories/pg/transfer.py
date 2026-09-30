"""Bank transfer (export / import) against the Postgres store's tables.

The SQL half of :mod:`hindsight_api.engine.transfer`: reading a bank's documents,
facts, observations and curation archive out for an archive, and writing the
store-side rows an import restores. Lifted verbatim from ``transfer/export.py`` and
``transfer/importer.py``; only the ``fq_table`` resolver is now a parameter. The
archive assembly (ordinals, typed transfer records) stays in the transfer package
and is imported from there.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from typing import Any

from ...causal_links import CAUSAL_LINK_TYPES
from ...chunk_ids import build_chunk_id, parse_chunk_id
from ...metadata_utils import as_string_metadata
from ...transfer.export import (
    _DERIVED_COLUMNS,
    _EXPORTED_FACT_TYPES,
    _as_connection,
    _as_jsonb,
    _chunk_index_from_chunk_id,
    _iter_id_batches,
    _LoadedExport,
    _LoadedFacts,
    _UnitLocation,
)
from ...transfer.importer import (
    FactLifecycle,
    _ObservationOutcome,
    _remap_causal_link_snapshot,
    _restore_rows,
)
from ...transfer.schema import (
    BankRowsJSONEncoding,
    TransferCausalRelation,
    TransferChunk,
    TransferDocument,
    TransferFact,
    TransferObservation,
    TransferObservationSource,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ export


async def iter_documents(
    *,
    backend: Any,
    fq_table: Callable[[str], str],
    bank_id: str,
    document_ids: list[str] | None,
    include_lifecycle: bool,
    batch_size: int,
) -> AsyncIterator[_LoadedExport]:
    """The bank's documents ``batch_size`` at a time, each batch read on its own connection.

    The connection is returned before a batch is yielded, so the caller streaming it
    into the archive holds none.
    """
    doc_filter = "AND id = ANY($2)" if document_ids else ""
    params: list[Any] = [bank_id, document_ids] if document_ids else [bank_id]
    async with _as_connection(backend) as conn:
        id_rows = await conn.fetch(
            f"SELECT id FROM {fq_table('documents')} WHERE bank_id = $1 {doc_filter} ORDER BY created_at, id",
            *params,
        )
    all_ids = [row["id"] for row in id_rows]
    for start in range(0, len(all_ids), batch_size):
        async with _as_connection(backend) as conn:
            loaded = await load_documents(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                document_ids=all_ids[start : start + batch_size],
                include_lifecycle=include_lifecycle,
            )
        yield loaded


async def load_documents(
    *,
    conn: Any,
    fq_table: Callable[[str], str],
    bank_id: str,
    document_ids: list[str] | None,
    include_lifecycle: bool = False,
) -> _LoadedExport:
    """Load and assemble TransferDocument payloads for the requested documents."""
    doc_filter = "AND id = ANY($2)" if document_ids else ""
    params: list[Any] = [bank_id]
    if document_ids:
        params.append(document_ids)
    doc_rows = await conn.fetch(
        f"""
        SELECT id, original_text, retain_params, tags, created_at
        FROM {fq_table("documents")}
        WHERE bank_id = $1 {doc_filter}
        ORDER BY created_at, id
        """,
        *params,
    )
    if not doc_rows:
        return _LoadedExport()

    selected_ids = [row["id"] for row in doc_rows]

    chunks_by_doc = await _load_chunks(conn, fq_table, bank_id, selected_ids)
    loaded = await _load_facts(conn, fq_table, bank_id, selected_ids, include_lifecycle=include_lifecycle)
    await _attach_entities(conn, fq_table, loaded)
    await _attach_causal_relations(conn, fq_table, loaded)

    documents: list[TransferDocument] = []
    for row in doc_rows:
        doc_id = row["id"]
        documents.append(
            TransferDocument(
                id=doc_id,
                original_text=row["original_text"],
                retain_params=_as_jsonb(row["retain_params"]),
                tags=list(row["tags"] or []),
                created_at=row["created_at"],
                chunks=chunks_by_doc.get(doc_id, []),
                facts=loaded.facts_by_doc.get(doc_id, []),
            )
        )
    return _LoadedExport(documents=documents, unit_index=loaded.unit_index)


async def load_observations(
    *,
    conn: Any,
    fq_table: Callable[[str], str],
    bank_id: str,
    unit_index: dict[Any, _UnitLocation],
) -> list[TransferObservation]:
    """Load observations whose source facts are all present in the exported set.

    Each source unit id is rewritten to its (document_id, fact_index) reference
    via ``unit_index``. Only called for a whole-bank export, so every live source
    fact is present; an observation is skipped only if a source no longer exists
    (stale reference) — that keeps every exported observation resolvable on import.
    """
    rows = await conn.fetch(
        f"""
        SELECT id, text, tags, created_at, event_date, occurred_start, occurred_end,
               mentioned_at, observation_scopes, proof_count, source_memory_ids
        FROM {fq_table("memory_units")}
        WHERE bank_id = $1 AND fact_type = 'observation'
        ORDER BY created_at, id
        """,
        bank_id,
    )

    observations: list[TransferObservation] = []
    skipped = 0
    for row in rows:
        source_ids = list(row["source_memory_ids"] or [])
        locations = [unit_index.get(sid) for sid in source_ids]
        if not source_ids or any(loc is None for loc in locations):
            # An observation with sources outside the exported documents would be
            # incoherent on import — skip it rather than emit dangling refs.
            skipped += 1
            continue
        observations.append(
            TransferObservation(
                source_id=str(row["id"]),
                text=row["text"],
                created_at=row["created_at"],
                tags=list(row["tags"] or []),
                event_date=row["event_date"],
                occurred_start=row["occurred_start"],
                occurred_end=row["occurred_end"],
                mentioned_at=row["mentioned_at"],
                observation_scopes=_as_jsonb(row["observation_scopes"]),
                proof_count=row["proof_count"] or len(source_ids),
                sources=[
                    TransferObservationSource(document_id=loc.document_id, fact_index=loc.ordinal)
                    for loc in locations
                    if loc is not None
                ],
            )
        )
    if skipped:
        logger.info("[transfer] Skipped %d observation(s) with sources outside the exported documents", skipped)
    return observations


async def _load_chunks(
    conn: Any, fq_table: Callable[[str], str], bank_id: str, doc_ids: list[str]
) -> dict[str, list[TransferChunk]]:
    rows = await conn.fetch(
        f"""
        SELECT document_id, chunk_index, chunk_text
        FROM {fq_table("chunks")}
        WHERE bank_id = $1 AND document_id = ANY($2)
        ORDER BY document_id, chunk_index
        """,
        bank_id,
        doc_ids,
    )
    chunks_by_doc: dict[str, list[TransferChunk]] = {}
    for row in rows:
        chunks_by_doc.setdefault(row["document_id"], []).append(
            TransferChunk(chunk_index=row["chunk_index"], chunk_text=row["chunk_text"])
        )
    return chunks_by_doc


async def _load_facts(
    conn: Any, fq_table: Callable[[str], str], bank_id: str, doc_ids: list[str], include_lifecycle: bool = False
) -> _LoadedFacts:
    """Load non-observation facts grouped by document, with a unit-id location index.

    The ordering is fixed (created_at, id) so that
    ``causal_relations.target_fact_index`` ordinals stay consistent.

    ``include_lifecycle`` carries each fact's ``created_at`` / ``consolidated_at`` /
    ``consolidation_failed_at`` (whole-bank / with-observations export). It is left
    off for the plain document export so the target re-consolidates from scratch.
    """
    rows = await conn.fetch(
        f"""
        SELECT id, document_id, text, fact_type, context, event_date,
               occurred_start, occurred_end, mentioned_at, metadata,
               chunk_id, tags, observation_scopes,
               created_at, consolidated_at, consolidation_failed_at
        FROM {fq_table("memory_units")}
        WHERE bank_id = $1
          AND document_id = ANY($2)
          AND fact_type = ANY($3)
        ORDER BY document_id, created_at, id
        """,
        bank_id,
        doc_ids,
        list(_EXPORTED_FACT_TYPES),
    )

    loaded = _LoadedFacts()
    for row in rows:
        doc_id = row["document_id"]
        bucket = loaded.facts_by_doc.setdefault(doc_id, [])
        ordinal = len(bucket)
        fact = TransferFact(
            source_id=str(row["id"]),
            text=row["text"],
            fact_type=row["fact_type"],
            context=row["context"],
            event_date=row["event_date"],
            occurred_start=row["occurred_start"],
            occurred_end=row["occurred_end"],
            mentioned_at=row["mentioned_at"],
            # Same dict[str, str] contract as the recall path: a legacy row
            # holding a JSON null or a raw integer must not fail the export
            # that would let an operator move the bank (issue #3209).
            metadata=as_string_metadata(_as_jsonb(row["metadata"])),
            tags=list(row["tags"] or []),
            observation_scopes=_as_jsonb(row["observation_scopes"]),
            chunk_index=_chunk_index_from_chunk_id(row["chunk_id"]),
            created_at=row["created_at"] if include_lifecycle else None,
            consolidated_at=row["consolidated_at"] if include_lifecycle else None,
            consolidation_failed_at=row["consolidation_failed_at"] if include_lifecycle else None,
        )
        bucket.append(fact)
        loaded.unit_index[row["id"]] = _UnitLocation(document_id=doc_id, ordinal=ordinal)
    return loaded


async def _attach_entities(conn: Any, fq_table: Callable[[str], str], loaded: _LoadedFacts) -> None:
    """Populate each fact's ``entities`` list with its entities' canonical names."""
    if not loaded.unit_index:
        return
    for batch in _iter_id_batches(list(loaded.unit_index.keys())):
        rows = await conn.fetch(
            f"""
            SELECT ue.unit_id, e.canonical_name
            FROM {fq_table("unit_entities")} ue
            JOIN {fq_table("entities")} e ON e.id = ue.entity_id
            WHERE ue.unit_id = ANY($1)
            ORDER BY e.canonical_name
            """,
            batch,
        )
        for row in rows:
            location = loaded.unit_index.get(row["unit_id"])
            if location is None:
                continue
            loaded.facts_by_doc[location.document_id][location.ordinal].entities.append(row["canonical_name"])


async def _attach_causal_relations(conn: Any, fq_table: Callable[[str], str], loaded: _LoadedFacts) -> None:
    """Reconstruct causal edges as fact ordinals within each document.

    A memory_link (from_unit -> to_unit, link_type) means ``from_unit`` carries
    the relation pointing at ``to_unit``, so the edge is attached to the source
    fact with the target's ordinal. Edges spanning two documents are skipped
    (causal links are created within a single retain batch in practice).
    """
    if not loaded.unit_index:
        return
    # Batch on ``from_unit_id`` only. The old query also constrained
    # ``to_unit_id = ANY(<full set>)``, but splitting one list across both bounds
    # would drop edges whose endpoints fall in different batches. Instead we
    # filter the target endpoint in Python (``target is None``) exactly as before,
    # which keeps every in-set edge while bounding the parameter size.
    for batch in _iter_id_batches(list(loaded.unit_index.keys())):
        rows = await conn.fetch(
            f"""
            SELECT from_unit_id, to_unit_id, link_type
            FROM {fq_table("memory_links")}
            WHERE link_type = ANY($1)
              AND from_unit_id = ANY($2)
            """,
            list(CAUSAL_LINK_TYPES),
            batch,
        )
        for row in rows:
            source = loaded.unit_index.get(row["from_unit_id"])
            target = loaded.unit_index.get(row["to_unit_id"])
            if source is None or target is None:
                continue
            if source.document_id != target.document_id:
                continue
            loaded.facts_by_doc[source.document_id][source.ordinal].causal_relations.append(
                TransferCausalRelation(
                    relation_type=row["link_type"],
                    target_fact_index=target.ordinal,
                )
            )


async def dump_entity_maintenance_queue(*, conn: Any, fq_table: Callable[[str], str], bank_id: str) -> list[dict]:
    """The entity maintenance queue, each entity named rather than numbered.

    Entities are re-resolved by canonical name on the target, so a row whose entity
    has no name here cannot be restored and is left out.
    """
    entity_queue = await conn.fetch(
        f"""
        SELECT q.bank_id, q.entity_id, q.enqueued_at, e.canonical_name
        FROM {fq_table("entity_maintenance_queue")} q
        LEFT JOIN {fq_table("entities")} e ON e.id = q.entity_id
        WHERE q.bank_id = $1
        ORDER BY q.enqueued_at
        """,
        bank_id,
    )
    return [{k: v for k, v in dict(row).items() if k != "entity_id"} for row in entity_queue if row["canonical_name"]]


async def dump_archived_memories(*, conn: Any, fq_table: Callable[[str], str], bank_id: str) -> list[dict]:
    """Dump the curation archive (facts a user invalidated but can still revert).

    Two columns cannot travel as they are. ``entity_ids`` point at source-bank
    entity rows, so they are carried as canonical names for the target to
    re-resolve; ``chunk_id`` embeds the bank id (see ``chunk_ids``), so only its
    ordinal is carried and the target rebuilds the id. The surrogate ``id`` is
    dropped — the replay mints fresh unit ids and nothing references an archived
    one across a transfer.
    """
    rows = await conn.fetch(
        f"SELECT * FROM {fq_table('invalidated_memory_units')} WHERE bank_id = $1 ORDER BY invalidated_at, id",
        bank_id,
    )
    if not rows:
        return []
    every_entity = {e for row in rows for e in (row["entity_ids"] or [])}
    names: dict[Any, str] = {}
    if every_entity:
        name_rows = await conn.fetch(
            f"SELECT id, canonical_name FROM {fq_table('entities')} WHERE id = ANY($1)", list(every_entity)
        )
        names = {r["id"]: r["canonical_name"] for r in name_rows}
    dumped: list[dict] = []
    for row in rows:
        record = {k: v for k, v in dict(row).items() if k not in _DERIVED_COLUMNS and k not in ("id", "entity_ids")}
        # parse_chunk_id, not a naive rsplit: ids written since #4244 escape the
        # separator inside the bank and document components, so splitting on the
        # last underscore recovers the wrong ordinal for an escaped id.
        parsed_chunk = parse_chunk_id(record.pop("chunk_id", None))
        record["chunk_index"] = parsed_chunk.chunk_index if parsed_chunk else None
        record["entity_names"] = sorted(n for n in (names.get(e) for e in (row["entity_ids"] or [])) if n)
        dumped.append(record)
    return dumped


# ------------------------------------------------------------------ import


async def document_exists(*, backend: Any, fq_table: Callable[[str], str], bank_id: str, document_id: str) -> bool:
    """Whether the bank already has a ``documents`` row with this id."""
    from ...db_utils import acquire_with_retry

    async with acquire_with_retry(backend) as conn:
        exists = await conn.fetchval(
            f"SELECT 1 FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
            document_id,
            bank_id,
        )
    return bool(exists)


async def restore_document_created_at(
    *, conn: Any, fq_table: Callable[[str], str], bank_id: str, document_id: str, created_at: datetime
) -> None:
    """Stamp the source's creation time on an imported document row."""
    await conn.execute(
        f"UPDATE {fq_table('documents')} SET created_at = $1 WHERE id = $2 AND bank_id = $3",
        created_at,
        document_id,
        bank_id,
    )


async def restore_fact_lifecycle(
    *, conn: Any, fq_table: Callable[[str], str], bank_id: str, rows: list[FactLifecycle]
) -> None:
    """Apply each imported fact's source consolidation timestamps to its new row, in one batch."""
    await conn.executemany(
        f"UPDATE {fq_table('memory_units')} "
        f"SET created_at = COALESCE($2, created_at), consolidated_at = $3, consolidation_failed_at = $4 "
        f"WHERE id = $1 AND bank_id = $5",
        [
            (uuid.UUID(row.unit_id), row.created_at, row.consolidated_at, row.consolidation_failed_at, bank_id)
            for row in rows
        ],
    )


async def resolve_entity_ids_by_name(
    *, conn: Any, fq_table: Callable[[str], str], bank_id: str, names: set[str]
) -> dict[str, Any]:
    """Map canonical entity names to the target bank's entity ids.

    Entities are re-resolved by name during the replay, so a name is the only
    handle on an entity that survives a transfer.
    """
    if not names:
        return {}
    rows = await conn.fetch(
        f"SELECT id, canonical_name FROM {fq_table('entities')} WHERE bank_id = $1 AND canonical_name = ANY($2::text[])",
        bank_id,
        sorted(names),
    )
    return {row["canonical_name"]: row["id"] for row in rows}


async def restore_archived_memories(
    *,
    conn: Any,
    fq_table: Callable[[str], str],
    bank_id: str,
    rows: list[dict],
    unit_id_map: dict[str, str],
    document_id_map: dict[str, str],
    bank_rows_json_encoding: BankRowsJSONEncoding,
) -> int:
    """Restore the curation archive so invalidated facts stay revertable.

    Three columns are rebuilt for the target: ``entity_names`` back into entity
    ids, the chunk ordinal back into a chunk id (chunk ids embed the bank id), and
    the causal-link snapshot onto the replayed units — edges whose other endpoint
    did not come back are dropped, which is what revert already tolerates. The
    unit id itself is minted fresh: nothing outside this row references it, and
    keeping the source's would collide with the source bank on a clone.
    """
    if not rows:
        return 0
    names = {name for row in rows for name in (row.get("entity_names") or [])}
    entity_ids = await resolve_entity_ids_by_name(conn=conn, fq_table=fq_table, bank_id=bank_id, names=names)

    prepared: list[dict] = []
    for row in rows:
        record = dict(row)
        document_id = record.get("document_id")
        if document_id is not None:
            document_id = document_id_map.get(document_id, document_id)
            record["document_id"] = document_id
        chunk_index = record.pop("chunk_index", None)
        record["chunk_id"] = (
            build_chunk_id(bank_id, document_id, chunk_index)
            if chunk_index is not None and document_id is not None
            else None
        )
        record["entity_ids"] = [entity_ids[n] for n in (record.pop("entity_names", None) or []) if n in entity_ids]
        record["causal_links"] = _remap_causal_link_snapshot(record.get("causal_links"), unit_id_map)
        record["id"] = str(uuid.uuid4())
        prepared.append(record)
    return await _restore_rows(
        conn,
        "invalidated_memory_units",
        prepared,
        bank_rows_json_encoding=bank_rows_json_encoding,
        fq=fq_table,
    )


async def import_observations(
    *,
    backend: Any,
    ops: Any,
    fq_table: Callable[[str], str],
    bank_id: str,
    resolved: list[tuple[TransferObservation, list[str]]],
    processed: list,
    outcome: _ObservationOutcome,
) -> _ObservationOutcome:
    """Insert the resolved observations and link them to their (freshly imported) sources.

    One transaction: re-check the sources are live, insert the observation rows,
    restore their carried ``created_at`` / ``event_date``, attach sources + proof
    count, and mark the sources consolidated.
    """
    from ...db_utils import acquire_with_retry
    from ...retain import fact_storage

    async with acquire_with_retry(backend) as conn:
        async with conn.transaction():
            # ``source_memory_ids`` is a bare uuid[] with no foreign key, so a
            # ref that resolved above but whose unit is no longer in this bank
            # would be written as a dangling reference and silently corrupt the
            # observation graph (an observation citing units that exist nowhere).
            # Re-check liveness inside the write transaction and treat a missing
            # source exactly like an unresolved one: skip the observation.
            live = {
                r["id"]
                for r in await conn.fetch(
                    f"SELECT id FROM {fq_table('memory_units')} WHERE bank_id = $1 AND id = ANY($2)",
                    bank_id,
                    [uuid.UUID(s) for _obs, sources in resolved for s in sources],
                )
            }
            kept_resolved: list[tuple[TransferObservation, list[str]]] = []
            kept_processed: list = []
            for (obs, sources), fact in zip(resolved, processed):
                missing = [s for s in sources if uuid.UUID(s) not in live]
                if missing:
                    logger.warning(
                        "[transfer] Skipping observation for bank %s: %d of %d source units are missing (%s)",
                        bank_id,
                        len(missing),
                        len(sources),
                        ", ".join(sorted(missing)),
                    )
                    outcome.skipped += 1
                    continue
                kept_resolved.append((obs, sources))
                kept_processed.append(fact)
            if not kept_resolved:
                return outcome
            resolved = kept_resolved
            processed = kept_processed

            obs_unit_ids = await fact_storage.insert_facts_batch(conn, bank_id, processed, ops=ops)

            all_source_ids: set[uuid.UUID] = set()
            for (obs, sources), obs_unit_id in zip(resolved, obs_unit_ids):
                observation_uuid = uuid.UUID(obs_unit_id)
                if obs.created_at is not None:
                    await conn.execute(
                        f"UPDATE {fq_table('memory_units')} SET created_at = $1 WHERE id = $2 AND bank_id = $3",
                        obs.created_at,
                        observation_uuid,
                        bank_id,
                    )
                if obs.event_date is not None:
                    # insert_facts_batch derives event_date for normal writes;
                    # transfer restores the source value carried by the archive.
                    await conn.execute(
                        f"UPDATE {fq_table('memory_units')} SET event_date = $1 WHERE id = $2 AND bank_id = $3",
                        obs.event_date,
                        observation_uuid,
                        bank_id,
                    )
                source_uuids = [uuid.UUID(s) for s in sources]
                all_source_ids.update(source_uuids)
                await _link_observation_sources(
                    conn, ops, fq_table, bank_id, observation_uuid, source_uuids, obs.proof_count
                )
                if obs.source_id is not None:
                    outcome.remapped_unit_ids[obs.source_id] = str(observation_uuid)

            # Mark source facts consolidated so the target consolidator skips
            # them. COALESCE keeps the exact source timestamp already restored by
            # restore_fact_lifecycle (new archives); now() is the fallback only
            # for legacy archives that carry no per-fact lifecycle state.
            if all_source_ids:
                await conn.execute(
                    f"UPDATE {fq_table('memory_units')} SET consolidated_at = COALESCE(consolidated_at, now()) "
                    f"WHERE bank_id = $1 AND id = ANY($2)",
                    bank_id,
                    list(all_source_ids),
                )

    outcome.imported = len(resolved)
    return outcome


async def _link_observation_sources(
    conn: Any,
    ops: Any,
    fq_table: Callable[[str], str],
    bank_id: str,
    observation_id: uuid.UUID,
    source_ids: list[uuid.UUID],
    proof_count: int,
) -> None:
    """Attach source ids + proof_count to a freshly inserted observation row.

    PG stores the sources in the ``source_memory_ids`` array column; Oracle uses
    the ``observation_sources`` junction table (same split as consolidation).
    """
    if ops.uses_observation_sources_table:
        await conn.executemany(
            f"INSERT INTO {fq_table('observation_sources')} (observation_id, source_id) "
            f"VALUES ($1, $2) ON CONFLICT (observation_id, source_id) DO NOTHING",
            [(observation_id, sid) for sid in dict.fromkeys(source_ids)],
        )
        await conn.execute(
            f"UPDATE {fq_table('memory_units')} SET proof_count = $1 WHERE id = $2 AND bank_id = $3",
            proof_count,
            observation_id,
            bank_id,
        )
    else:
        await conn.execute(
            f"UPDATE {fq_table('memory_units')} SET source_memory_ids = $1, proof_count = $2 "
            f"WHERE id = $3 AND bank_id = $4",
            source_ids,
            proof_count,
            observation_id,
            bank_id,
        )
