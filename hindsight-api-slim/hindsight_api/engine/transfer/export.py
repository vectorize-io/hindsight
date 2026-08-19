"""Export documents (with extracted facts, entities, causal links, chunks) to a ZIP archive.

Reads directly from the database via the backend connection. Embeddings and
database ids are deliberately omitted — they are regenerated/re-resolved on
import. Consolidated observations are excluded unless ``include_observations``
is set, in which case they are written to ``observations.json``.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

import anyio.to_thread

from ..causal_links import CAUSAL_LINK_TYPES
from ..db_utils import acquire_with_retry
from ..metadata_utils import as_string_metadata
from ..schema import fq_table
from .schema import (
    CARRIED_HISTORY_TABLES,
    HISTORY_TABLES,
    SCHEMA_VERSION,
    BankRowsJSONEncoding,
    TransferCausalRelation,
    TransferChunk,
    TransferDocument,
    TransferFact,
    TransferKnowledgePage,
    TransferManifest,
    TransferObservation,
    TransferObservationSource,
)

logger = logging.getLogger(__name__)

# Whole-bank export classification. Every bank-scoped table (admin.cli.BACKUP_TABLES)
# must fall into exactly one bucket below; tests/test_document_transfer.py's
# test_export_bank_covers_schema enforces this so a table added by a future
# migration can't be silently dropped from a migration archive.

# NOT written to the archive — rebuilt on import by replaying the document/fact/
# observation payload through the import pipeline:
#   * documents / chunks / memory_units carry their *text* in the logical document
#     payload (TransferDocument) and are re-embedded with the target model;
#   * entities / unit_entities / memory_links / entity_cooccurrences are derived
#     data — the pipeline re-resolves entities and rebuilds links/cooccurrence
#     stats against the target bank, so they are never exported.
# Listed here only so the coverage guard can assert every table is classified.
_REPLAYED_TABLES = frozenset(
    {
        "documents",
        "chunks",
        "memory_units",
        "entities",
        "unit_entities",
        "memory_links",
        "entity_cooccurrences",
        # observation_history FKs to a memory_units observation, but observations
        # are derived: they're regenerated with FRESH ids when consolidation is
        # replayed on import (see _EXPORTED_FACT_TYPES — observations are excluded).
        # There is no stable observation id to re-attach history to, so it is not
        # carried; the target rebuilds observation history as it re-consolidates.
        "observation_history",
    }
)
# Carried verbatim as JSON rows (bank config + synthesized state). Embedding-bearing
# rows have their vector stripped (see _DERIVED_COLUMNS) and are re-embedded on import.
_BANK_ROW_TABLES = ("banks", "mental_models", "directives", "webhooks")
# Bank-scoped child-history carried verbatim. Unlike observations, mental models
# keep their (id, bank_id) across export/import, so their refresh history can be
# re-attached. The surrogate ``id`` is dropped on dump so the target reassigns it
# (see _dump_history_rows); restored after its parent table (mental_models).
# Operational history — only carried with include_history=True.
# Intentionally never exported.
_SKIP_TABLES = frozenset(
    {
        "async_operations",  # in-flight ops; drain on the source before migrating
        "graph_maintenance_queue",  # transient work queue; regenerated on import
        "entity_maintenance_queue",  # transient work queue; regenerated on import
        "file_storage",  # raw uploads; documents.original_text is already carried
        # Curation archive of retired facts — local operational state, not part of
        # the live knowledge the export replays. Its rows mirror memory_units (stale
        # embedding) and snapshot source-bank entity ids that the import re-resolves
        # to fresh ids, so carrying them would only produce dangling associations.
        # Revert anything worth keeping on the source before migrating.
        "invalidated_memory_units",
    }
)
# Derived columns dropped from carried rows so the target regenerates them with
# its own embedding model / text-search backend.
_DERIVED_COLUMNS = ("embedding", "search_vector")


@dataclass
class _UnitLocation:
    """Where a memory unit's fact lives in the assembled export (document + ordinal)."""

    document_id: str
    ordinal: int


@dataclass
class _LoadedFacts:
    """Facts grouped by document plus an index from unit id to its location.

    ``facts_by_doc`` and ``unit_index`` share the same fixed ordering so that
    causal ``target_fact_index`` ordinals stay consistent across both.
    """

    facts_by_doc: dict[str, list[TransferFact]] = field(default_factory=dict)
    unit_index: dict[Any, _UnitLocation] = field(default_factory=dict)


@dataclass
class _LoadedExport:
    """Assembled documents plus the unit-id → location index.

    ``unit_index`` is retained so observation source unit ids can be resolved to
    (document_id, fact_index) references when observations are exported.
    """

    documents: list[TransferDocument] = field(default_factory=list)
    unit_index: dict[Any, _UnitLocation] = field(default_factory=dict)


# Retain currently writes only ``caused_by``. The legacy types stay in archives
# so importing a historical bank preserves its graph; temporal/semantic/entity
# links are regenerated against the target bank.
# Facts of these types are exported; observations are derived and excluded.
_EXPORTED_FACT_TYPES = ("world", "experience")


def _as_jsonb(value: Any) -> Any:
    """Coerce an asyncpg JSONB column (str or already-decoded) to a Python object."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # Admin connections register a JSONB decoder, so a valid scalar such
            # as `"combined"` arrives here as the already-decoded `combined`.
            return value
    return value


def _chunk_index_from_chunk_id(chunk_id: str | None) -> int | None:
    """Recover the chunk ordinal from a ``{bank_id}_{document_id}_{index}`` chunk_id.

    The index is always the final underscore-delimited segment, so rsplit is
    correct even when bank/document ids themselves contain underscores.
    """
    if not chunk_id:
        return None
    try:
        return int(chunk_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


async def export_documents(
    backend: Any,
    bank_id: str,
    document_ids: list[str] | None = None,
    *,
    include_observations: bool = False,
) -> bytes:
    """Export documents from ``bank_id`` into an in-memory ZIP archive.

    Args:
        backend: Database backend (provides ``acquire()``).
        bank_id: Source bank.
        document_ids: Specific document ids to export. ``None`` exports every
            document in the bank.
        include_observations: Also export consolidated observations (written to
            ``observations.json``). Only valid for a whole-bank export.

    Returns:
        The ZIP archive as bytes.

    Raises:
        ValueError: if ``include_observations`` is combined with ``document_ids``.
    """
    # Observations are bank-level and can be derived from facts spanning several
    # documents, so they're only coherent when the whole bank is exported. For a
    # document subset we'd have to silently drop every cross-document observation
    # — reject the combination instead so the caller isn't surprised.
    if include_observations and document_ids is not None:
        raise ValueError("include_observations is only supported when exporting the whole bank (omit document_id)")

    async with acquire_with_retry(backend) as conn:
        # Carry per-fact consolidation lifecycle exactly when observations are
        # carried: with observations in the archive the target must NOT re-derive
        # them, so imported facts keep their consolidated/failed state. Without
        # observations (the default document export) the target re-consolidates
        # from scratch, so lifecycle is deliberately dropped.
        loaded = await _load_documents(conn, bank_id, document_ids, include_lifecycle=include_observations)
        documents = loaded.documents
        observations = await _load_observations(conn, bank_id, loaded.unit_index) if include_observations else []

    fact_total = sum(len(document.facts) for document in documents)
    manifest = TransferManifest(
        schema_version=SCHEMA_VERSION,
        source_bank_id=bank_id,
        exported_at=datetime.now(UTC),
        document_count=len(documents),
        fact_count=fact_total,
        observation_count=len(observations),
    )

    # ZIP compression and per-document JSON serialisation are CPU-bound and, on a
    # large bank, would block the event loop for seconds (issue #3321). Run the
    # assembly in a worker thread so unrelated requests/tasks keep progressing.
    archive_bytes = await anyio.to_thread.run_sync(_build_archive_bytes, documents, observations, manifest)

    logger.info(
        "[transfer] Exported %d document(s), %d fact(s), %d observation(s) from bank %s",
        len(documents),
        fact_total,
        len(observations),
        bank_id,
    )
    return archive_bytes


def _build_archive_bytes(
    documents: list[TransferDocument],
    observations: list[TransferObservation],
    manifest: TransferManifest,
) -> bytes:
    """Serialise the loaded documents/observations/manifest into a ZIP archive.

    Pure CPU work (DEFLATE + Pydantic JSON dumps) with no I/O, so it runs off the
    event loop via :func:`anyio.to_thread.run_sync`.
    """
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for index, document in enumerate(documents):
            zf.writestr(
                f"documents/{index:06d}.json",
                document.model_dump_json(indent=2, exclude_none=False),
            )

        if observations:
            payload = "[\n" + ",\n".join(o.model_dump_json(indent=2) for o in observations) + "\n]\n"
            zf.writestr("observations.json", payload)

        zf.writestr("manifest.json", manifest.model_dump_json(indent=2))

    return archive.getvalue()


def _row_json_default(obj: Any) -> Any:
    """JSON serializer for the value types asyncpg returns from bank rows."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, Decimal):
        # str preserves precision; import casts back to numeric.
        return str(obj)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(obj)).decode("ascii")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


async def _dump_bank_rows(conn: Any, table: str, bank_id: str) -> list[dict]:
    """Dump all rows of a bank-scoped table as JSON-ready dicts (derived columns stripped).

    Embedding/search-vector columns are omitted so the target instance
    regenerates them with its own model/backend on import.
    """
    rows = await conn.fetch(f"SELECT * FROM {fq_table(table)} WHERE bank_id = $1", bank_id)
    return [{k: v for k, v in dict(row).items() if k not in _DERIVED_COLUMNS} for row in rows]


async def _load_knowledge_pages(conn: Any, bank_id: str) -> list[TransferKnowledgePage]:
    """Load the knowledge-base tree (folders + pages) as typed rows.

    Ordered parents-before-children (root folders first) via a recursive walk of
    ``parent_id`` so the archive is deterministic and import can insert in list
    order; import re-derives a safe order regardless. IDs, ``parent_id``,
    ``mental_model_id``, ``managed`` and ``sort_order`` are all preserved.
    """
    rows = await conn.fetch(
        f"""
        WITH RECURSIVE tree AS (
            SELECT kp.*, 0 AS depth
            FROM {fq_table("knowledge_pages")} kp
            WHERE kp.bank_id = $1 AND kp.parent_id IS NULL
            UNION ALL
            SELECT kp.*, t.depth + 1
            FROM {fq_table("knowledge_pages")} kp
            JOIN tree t ON kp.parent_id = t.id AND kp.bank_id = $1
        )
        SELECT id, parent_id, kind, name, mental_model_id, sort_order, managed, created_at, updated_at
        FROM tree
        ORDER BY depth, sort_order, id
        """,
        bank_id,
    )
    return [
        TransferKnowledgePage(
            id=row["id"],
            parent_id=row["parent_id"],
            kind=row["kind"],
            name=row["name"],
            mental_model_id=row["mental_model_id"],
            sort_order=row["sort_order"],
            managed=row["managed"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        for row in rows
    ]


async def _dump_history_rows(conn: Any, table: str, bank_id: str) -> list[dict]:
    """Dump a bank-scoped child-history table for carrying across instances.

    Drops the surrogate ``id`` so the target reassigns it from its own IDENTITY
    sequence (carrying explicit ids would leave the sequence un-advanced and
    collide with later writes). Ordered oldest-first so the reassigned ids keep
    the same chronological tie-break order the read path relies on.
    """
    rows = await conn.fetch(
        f"SELECT * FROM {fq_table(table)} WHERE bank_id = $1 ORDER BY changed_at, id",
        bank_id,
    )
    return [{k: v for k, v in dict(row).items() if k not in _DERIVED_COLUMNS and k != "id"} for row in rows]


async def export_bank(
    conn: Any,
    bank_id: str,
    *,
    include_history: bool = False,
    bank_rows_json_encoding: BankRowsJSONEncoding = "serialized",
) -> bytes:
    """Export an entire bank into a portable ZIP archive (no embeddings).

    Produces a superset of the documents archive: the logical
    document/fact/observation export (replayed and re-embedded on import) plus
    the bank's config, mental models, directives and webhooks as JSON rows. With
    ``include_history`` the operational tails (audit_log, llm_requests) are also
    carried. Intended for migrating a bank to a new instance configured with a
    different embedding model / vector / text-search backend — every vector is
    regenerated on the target, so nothing here is encoder-specific.

    ``conn`` is a live connection scoped to the bank's schema (the admin CLI sets
    ``_current_schema`` and passes its raw connection; the engine acquires one
    after tenant auth).
    """
    # Whole-bank export always carries observations (they're bank-level state)
    # and, with them, the per-fact consolidation lifecycle so the target restores
    # exact eligibility instead of re-consolidating historical facts (#2965).
    loaded = await _load_documents(conn, bank_id, None, include_lifecycle=True)
    documents = loaded.documents
    observations = await _load_observations(conn, bank_id, loaded.unit_index)

    bank_rows = {table: await _dump_bank_rows(conn, table, bank_id) for table in _BANK_ROW_TABLES}
    for table in CARRIED_HISTORY_TABLES:
        bank_rows[table] = await _dump_history_rows(conn, table, bank_id)
    knowledge_pages = await _load_knowledge_pages(conn, bank_id)
    history_rows: dict[str, list[dict]] = {}
    if include_history:
        history_rows = {table: await _dump_bank_rows(conn, table, bank_id) for table in HISTORY_TABLES}

    archive = io.BytesIO()
    fact_total = 0
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for index, document in enumerate(documents):
            fact_total += len(document.facts)
            zf.writestr(f"documents/{index:06d}.json", document.model_dump_json(indent=2, exclude_none=False))

        if observations:
            payload = "[\n" + ",\n".join(o.model_dump_json(indent=2) for o in observations) + "\n]\n"
            zf.writestr("observations.json", payload)

        for table, rows in bank_rows.items():
            zf.writestr(f"{table}.json", json.dumps(rows, indent=2, default=_row_json_default))
        # Typed knowledge-page tree (parent-first). Written even when empty so the
        # importer can distinguish "no pages" from a pre-tree archive.
        zf.writestr(
            "knowledge_pages.json",
            "[\n" + ",\n".join(p.model_dump_json(indent=2) for p in knowledge_pages) + "\n]\n"
            if knowledge_pages
            else "[]\n",
        )
        for table, rows in history_rows.items():
            zf.writestr(f"history/{table}.json", json.dumps(rows, indent=2, default=_row_json_default))

        manifest = TransferManifest(
            schema_version=SCHEMA_VERSION,
            source_bank_id=bank_id,
            exported_at=datetime.now(UTC),
            document_count=len(documents),
            fact_count=fact_total,
            observation_count=len(observations),
            archive_type="bank",
            mental_model_count=len(bank_rows.get("mental_models", [])),
            knowledge_page_count=len(knowledge_pages),
            directive_count=len(bank_rows.get("directives", [])),
            webhook_count=len(bank_rows.get("webhooks", [])),
            includes_history=include_history,
            bank_rows_json_encoding=bank_rows_json_encoding,
        )
        zf.writestr("manifest.json", manifest.model_dump_json(indent=2))

    logger.info(
        "[transfer] Exported bank %s: %d document(s), %d fact(s), %d observation(s), "
        "%d mental model(s), %d knowledge page(s), %d directive(s), %d webhook(s)%s",
        bank_id,
        len(documents),
        fact_total,
        len(observations),
        len(bank_rows.get("mental_models", [])),
        len(knowledge_pages),
        len(bank_rows.get("directives", [])),
        len(bank_rows.get("webhooks", [])),
        " (with history)" if include_history else "",
    )
    return archive.getvalue()


async def _load_documents(
    conn: Any,
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

    chunks_by_doc = await _load_chunks(conn, bank_id, selected_ids)
    loaded = await _load_facts(conn, bank_id, selected_ids, include_lifecycle=include_lifecycle)
    await _attach_entities(conn, loaded)
    await _attach_causal_relations(conn, loaded)

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


async def _load_observations(
    conn: Any,
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
        SELECT id, text, tags, event_date, occurred_start, occurred_end,
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
                text=row["text"],
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


async def _load_chunks(conn: Any, bank_id: str, doc_ids: list[str]) -> dict[str, list[TransferChunk]]:
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


async def _load_facts(conn: Any, bank_id: str, doc_ids: list[str], include_lifecycle: bool = False) -> _LoadedFacts:
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


# A whole-bank export can index hundreds of thousands of memory units. Passing
# every unit id as a single ``ANY($1)`` parameter (issue #3321) inflates the
# query, spikes memory, and pins the connection while one enormous scan runs.
# Fetch the attach queries in bounded batches instead: each unit id belongs to
# exactly one batch, so per-fact ordering is preserved, and awaiting between
# batches yields the event loop.
_ATTACH_BATCH_SIZE = 5000


def _iter_id_batches(ids: list[Any], batch_size: int = _ATTACH_BATCH_SIZE) -> Iterator[list[Any]]:
    """Yield ``ids`` in fixed-size chunks (bounds SQL ``ANY`` parameter size)."""
    for start in range(0, len(ids), batch_size):
        yield ids[start : start + batch_size]


async def _attach_entities(conn: Any, loaded: _LoadedFacts) -> None:
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


async def _attach_causal_relations(conn: Any, loaded: _LoadedFacts) -> None:
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
