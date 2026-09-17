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
from ..chunk_ids import parse_chunk_id
from ..db_utils import acquire_with_retry
from ..metadata_utils import as_string_metadata
from ..schema import fq_table
from .schema import (
    ATTACHMENT_TABLES,
    CARRIED_HISTORY_TABLES,
    HISTORY_TABLES,
    OPERATIONAL_TABLES,
    SCHEMA_VERSION,
    BankRowsJSONEncoding,
    TransferAttachment,
    TransferCausalRelation,
    TransferChunk,
    TransferDocument,
    TransferFact,
    TransferKnowledgePage,
    TransferManifest,
    TransferObservation,
    TransferObservationSource,
    TransferScope,
    TransferScopeManifest,
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
#: How the bank is set up: its config overrides and the rules/endpoints it was
#: given. Carried with ``bank_config``.
_CONFIG_ROW_TABLES = ("banks", "directives", "webhooks")
#: What the bank has *synthesized from its own memories* — mental models, and
#: (as typed rows elsewhere) the knowledge-page tree over them. These are carried
#: with ``data``, not with the configuration: a mental model is a reading of the
#: bank's facts, and its ``based_on`` evidence points straight at them, so it
#: belongs with the memories it was derived from rather than with the settings.
_SYNTHESIZED_ROW_TABLES = ("mental_models",)
_BANK_ROW_TABLES = (*_CONFIG_ROW_TABLES, *_SYNTHESIZED_ROW_TABLES)
# Bank-scoped child-history carried verbatim. Unlike observations, mental models
# keep their (id, bank_id) across export/import, so their refresh history can be
# re-attached. The surrogate ``id`` is dropped on dump so the target reassigns it
# (see _dump_history_rows); restored after its parent table (mental_models).
# Operational history — only carried with scope.history.
# Carried under ``scope.data`` as verbatim rows under ``data/``, alongside the
# attachment rows (whose bytes ride in ``blobs/``). These are keyed by globally
# unique ids, or reference ids the replay regenerates, so the importer remaps
# them rather than inserting them as they stand.
_DATA_ROW_TABLES = (*OPERATIONAL_TABLES, *ATTACHMENT_TABLES, "invalidated_memory_units")
# Intentionally never exported.
_SKIP_TABLES = frozenset(
    {
        # The native file-storage backend's own table. Blobs travel as archive
        # entries (``blobs/``) keyed to the attachment rows that name them, not as
        # rows of whatever backend happens to hold them on the source — an S3
        # deployment has no rows here at all. Raw document uploads are not carried:
        # documents.original_text already holds the extracted text, which is what
        # the replay needs.
        "file_storage",
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
    # Populated only by the store-owned loader, where the entity postings and causal edges ride on
    # the memory instead of living in `unit_entities` / `memory_links` for a later join.
    entity_ids_by_unit: dict[str, list[str]] = field(default_factory=dict)
    causal_by_unit: dict[str, list[Any]] = field(default_factory=dict)


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


def _resolve_memories(memories: Any) -> Any:
    """The memories store to export through, asking for it when the caller did not pass one.

    The loaders read `memory_units` / `documents` directly, and for a store-owned bank those tables
    are empty — so treating "no store supplied" as "SQL-backed" produces a silently EMPTY archive
    with a success status. That has already happened once (see the call in
    `MemoryEngine.export_documents`), and it was fixed at the call sites while the default that
    causes it stayed. An export is the last thing that should fail quietly, so the default now asks
    rather than assumes.
    """
    if memories is not None:
        return memories
    try:
        from ..memories import get_memories

        return get_memories()
    except Exception:  # noqa: BLE001 - no store configured at all is genuinely SQL-backed
        return None


def _is_store_owned(memories: Any, bank_id: str) -> bool:
    """Whether this bank's memories live outside SQL, so the loaders must go through the store."""
    if memories is None:
        return False
    try:
        return memories.store_owned_for(bank_id)
    except Exception:  # noqa: BLE001 - a store that cannot answer is treated as SQL-backed
        return False


async def export_documents(
    backend: Any,
    bank_id: str,
    document_ids: list[str] | None = None,
    *,
    include_observations: bool = False,
    include_knowledge_base: bool = False,
    memories: Any = None,
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
        ValueError: if a bank-level section is combined with ``document_ids``.
    """
    # Observations are bank-level and can be derived from facts spanning several
    # documents, so they're only coherent when the whole bank is exported. For a
    # document subset we'd have to silently drop every cross-document observation
    # — reject the combination instead so the caller isn't surprised.
    if (include_observations or include_knowledge_base) and document_ids is not None:
        raise ValueError(
            "include_observations and include_knowledge_base are only supported when exporting the whole bank (omit document_id)"
        )

    memories = _resolve_memories(memories)

    # Carry per-fact consolidation lifecycle exactly when observations are
    # carried: with observations in the archive the target must NOT re-derive
    # them, so imported facts keep their consolidated/failed state. Without
    # observations (the default document export) the target re-consolidates
    # from scratch, so lifecycle is deliberately dropped.
    mental_models: list[dict] = []
    knowledge_pages: list[TransferKnowledgePage] = []
    if _is_store_owned(memories, bank_id):
        # No connection is taken at all: for this bank every table the SQL loaders read is empty,
        # so holding one would only make the empty result look better-founded than it is.
        loaded = await _load_documents_from_store(
            memories, bank_id, document_ids, include_lifecycle=include_observations
        )
        documents = loaded.documents
        observations = (
            await _load_observations_from_store(memories, bank_id, loaded.unit_index) if include_observations else []
        )
        if include_knowledge_base:
            async with acquire_with_retry(backend) as conn:
                mental_models = await _dump_bank_rows(conn, "mental_models", bank_id)
                knowledge_pages = await _load_knowledge_pages(conn, bank_id)
    else:
        async with acquire_with_retry(backend) as conn:
            loaded = await _load_documents(conn, bank_id, document_ids, include_lifecycle=include_observations)
            documents = loaded.documents
            observations = await _load_observations(conn, bank_id, loaded.unit_index) if include_observations else []
            if include_knowledge_base:
                mental_models = await _dump_bank_rows(conn, "mental_models", bank_id)
                knowledge_pages = await _load_knowledge_pages(conn, bank_id)

    fact_total = sum(len(document.facts) for document in documents)
    manifest = TransferManifest(
        schema_version=SCHEMA_VERSION,
        source_bank_id=bank_id,
        exported_at=datetime.now(UTC),
        document_count=len(documents),
        fact_count=fact_total,
        observation_count=len(observations),
        mental_model_count=len(mental_models),
        knowledge_page_count=len(knowledge_pages),
    )

    # ZIP compression and per-document JSON serialisation are CPU-bound and, on a
    # large bank, would block the event loop for seconds (issue #3321). Run the
    # assembly in a worker thread so unrelated requests/tasks keep progressing.
    archive_bytes = await anyio.to_thread.run_sync(
        _build_archive_bytes, documents, observations, mental_models, knowledge_pages, manifest
    )

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
    mental_models: list[dict],
    knowledge_pages: list[TransferKnowledgePage],
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

        if mental_models:
            zf.writestr("mental_models.json", json.dumps(mental_models, indent=2, default=_row_json_default))
        if knowledge_pages:
            payload = "[\n" + ",\n".join(p.model_dump_json(indent=2) for p in knowledge_pages) + "\n]\n"
            zf.writestr("knowledge_pages.json", payload)

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
    scope: TransferScope | None = None,
    bank_rows_json_encoding: BankRowsJSONEncoding = "serialized",
    memories: Any = None,
    file_storage: Any = None,
) -> bytes:
    """Export an entire bank into a portable ZIP archive (no embeddings).

    ``scope`` decides what travels — see :class:`TransferScope` for the mapping
    from its three booleans to tables. The default carries data and bank config
    but not the operational history tails. Intended for migrating a bank to a new
    instance configured with a different embedding model / vector / text-search
    backend — every vector is regenerated on the target, so nothing here is
    encoder-specific.

    ``conn`` is a live connection scoped to the bank's schema (the admin CLI sets
    ``_current_schema`` and passes its raw connection; the engine acquires one
    after tenant auth). ``file_storage`` is needed only for attachment bytes; a
    bank with attachments exported without one raises rather than silently
    producing rows that point at blobs the archive does not carry.

    This convenience form keeps ``conn`` for the whole call, including the
    compression — fine for the admin CLI, which owns a connection of its own and
    has no event loop to share. A server path wants
    :func:`load_bank_export` and :func:`build_bank_archive` instead, so the
    connection returns to the pool as soon as the reads are done.
    """
    payload = await load_bank_export(
        conn,
        bank_id,
        scope=scope,
        bank_rows_json_encoding=bank_rows_json_encoding,
        memories=memories,
        file_storage=file_storage,
    )
    return await build_bank_archive(payload)


@dataclass
class BankExportPayload:
    """Everything a whole-bank archive is built from, already read out of the bank.

    Loading and building are separate so the CPU-bound half — Pydantic JSON for
    every document plus DEFLATE over the whole bank — can run off the event loop
    *and* outside the read transaction. Before the split, ``export_bank`` did both
    while the caller held an open transaction, so a large bank's compression both
    stalled the loop and pinned a pooled connection for its duration. That was
    harmless while the only caller was the admin CLI (its own short-lived
    process); it stopped being harmless when the transfer and clone endpoints
    started calling it inside the API/worker process.
    """

    bank_id: str
    scope: TransferScope
    bank_rows_json_encoding: BankRowsJSONEncoding = "serialized"
    documents: list[TransferDocument] = field(default_factory=list)
    observations: list[TransferObservation] = field(default_factory=list)
    attachments: list[TransferAttachment] = field(default_factory=list)
    #: archive entry name -> attachment bytes
    blobs: dict[str, bytes] = field(default_factory=dict)
    data_rows: dict[str, list[dict]] = field(default_factory=dict)
    bank_rows: dict[str, list[dict]] = field(default_factory=dict)
    knowledge_pages: list[TransferKnowledgePage] = field(default_factory=list)
    history_rows: dict[str, list[dict]] = field(default_factory=dict)


async def load_bank_export(
    conn: Any,
    bank_id: str,
    *,
    scope: TransferScope | None = None,
    bank_rows_json_encoding: BankRowsJSONEncoding = "serialized",
    memories: Any = None,
    file_storage: Any = None,
) -> BankExportPayload:
    """Read everything ``scope`` asks for out of the bank, without building anything.

    Every query lives here, so a caller that wants a point-in-time archive wraps
    only this call in a transaction and lets the connection go before handing the
    payload to :func:`build_bank_archive`.
    """
    scope = scope or TransferScope()
    memories = _resolve_memories(memories)
    documents: list[TransferDocument] = []
    observations: list[TransferObservation] = []
    attachments: list[TransferAttachment] = []
    blobs: dict[str, bytes] = {}
    data_rows: dict[str, list[dict]] = {}

    if scope.data:
        # Whole-bank export always carries observations (they're bank-level state)
        # and, with them, the per-fact consolidation lifecycle so the target restores
        # exact eligibility instead of re-consolidating historical facts (#2965).
        #
        # Only the memories move to the store. Everything below — bank config, mental models,
        # directives, webhooks, knowledge pages, the history tails — lives in Postgres for every
        # deployment, so `conn` stays the source for all of it.
        if _is_store_owned(memories, bank_id):
            loaded = await _load_documents_from_store(memories, bank_id, None, include_lifecycle=True)
            documents = loaded.documents
            observations = await _load_observations_from_store(memories, bank_id, loaded.unit_index)
        else:
            loaded = await _load_documents(conn, bank_id, None, include_lifecycle=True)
            documents = loaded.documents
            observations = await _load_observations(conn, bank_id, loaded.unit_index)
        exported_attachments = await _dump_attachments(conn, bank_id, file_storage)
        attachments, blobs = exported_attachments.rows, exported_attachments.blobs
        data_rows["document_attachments"] = await _dump_bank_rows(conn, "document_attachments", bank_id)
        data_rows.update(await _dump_operational_rows(conn, bank_id))
        data_rows["invalidated_memory_units"] = await _dump_invalidated_units(conn, bank_id)

    bank_rows: dict[str, list[dict]] = {}
    knowledge_pages: list[TransferKnowledgePage] = []
    if scope.data:
        # Synthesized knowledge travels with the memories it was synthesized from:
        # a mental model reads the bank's facts and cites them by id, and a
        # knowledge page is a view over a mental model.
        for table in _SYNTHESIZED_ROW_TABLES:
            bank_rows[table] = await _dump_bank_rows(conn, table, bank_id)
        for table in CARRIED_HISTORY_TABLES:
            bank_rows[table] = await _dump_history_rows(conn, table, bank_id)
        knowledge_pages = await _load_knowledge_pages(conn, bank_id)
    if scope.bank_config:
        for table in _CONFIG_ROW_TABLES:
            bank_rows[table] = await _dump_bank_rows(conn, table, bank_id)

    history_rows: dict[str, list[dict]] = {}
    if scope.history:
        history_rows = {table: await _dump_bank_rows(conn, table, bank_id) for table in HISTORY_TABLES}

    return BankExportPayload(
        bank_id=bank_id,
        scope=scope,
        bank_rows_json_encoding=bank_rows_json_encoding,
        documents=documents,
        observations=observations,
        attachments=attachments,
        blobs=blobs,
        data_rows=data_rows,
        bank_rows=bank_rows,
        knowledge_pages=knowledge_pages,
        history_rows=history_rows,
    )


async def build_bank_archive(payload: BankExportPayload) -> bytes:
    """Serialise a loaded bank into its ZIP archive, off the event loop.

    Needs no database connection — everything it writes is already in ``payload``
    — which is the point: the caller's read transaction is closed by the time the
    compression starts. Mirrors what the document export does for the same reason
    (issue #3321).
    """
    archive_bytes = await anyio.to_thread.run_sync(_build_bank_archive_bytes, payload)
    scope = payload.scope
    logger.info(
        "[transfer] Exported bank %s (data=%s, bank_config=%s, history=%s): %d document(s), "
        "%d observation(s), %d attachment(s), %d operation(s), %d mental model(s), %d knowledge page(s), "
        "%d directive(s), %d webhook(s), %d byte(s)",
        payload.bank_id,
        scope.data,
        scope.bank_config,
        scope.history,
        len(payload.documents),
        len(payload.observations),
        len(payload.attachments),
        len(payload.data_rows.get("async_operations", [])),
        len(payload.bank_rows.get("mental_models", [])),
        len(payload.knowledge_pages),
        len(payload.bank_rows.get("directives", [])),
        len(payload.bank_rows.get("webhooks", [])),
        len(archive_bytes),
    )
    return archive_bytes


def _build_bank_archive_bytes(payload: BankExportPayload) -> bytes:
    """Assemble the whole-bank ZIP. Pure CPU, no I/O — runs in a worker thread."""
    bank_id = payload.bank_id
    scope = payload.scope
    documents = payload.documents
    observations = payload.observations
    attachments = payload.attachments
    blobs = payload.blobs
    data_rows = payload.data_rows
    bank_rows = payload.bank_rows
    knowledge_pages = payload.knowledge_pages
    history_rows = payload.history_rows
    bank_rows_json_encoding = payload.bank_rows_json_encoding

    archive = io.BytesIO()
    fact_total = 0
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for index, document in enumerate(documents):
            fact_total += len(document.facts)
            zf.writestr(f"documents/{index:06d}.json", document.model_dump_json(indent=2, exclude_none=False))

        if observations:
            section = "[\n" + ",\n".join(o.model_dump_json(indent=2) for o in observations) + "\n]\n"
            zf.writestr("observations.json", section)

        if attachments:
            section = "[\n" + ",\n".join(a.model_dump_json(indent=2) for a in attachments) + "\n]\n"
            zf.writestr("attachments.json", section)
            for entry, data in blobs.items():
                # Stored, not deflated: attachment bytes are images/PDFs that are
                # already compressed, so DEFLATE would burn CPU for nothing.
                zf.writestr(zipfile.ZipInfo(entry), data, compress_type=zipfile.ZIP_STORED)

        for table, rows in data_rows.items():
            zf.writestr(f"data/{table}.json", json.dumps(rows, indent=2, default=_row_json_default))

        for table, rows in bank_rows.items():
            zf.writestr(f"{table}.json", json.dumps(rows, indent=2, default=_row_json_default))
        # Typed knowledge-page tree (parent-first). Written even when empty so the
        # importer can distinguish "no pages" from a pre-tree archive.
        if scope.data:
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
            includes_history=scope.history,
            scope=TransferScopeManifest(data=scope.data, bank_config=scope.bank_config, history=scope.history),
            attachment_count=len(attachments),
            operation_count=len(data_rows.get("async_operations", [])),
            invalidated_memory_count=len(data_rows.get("invalidated_memory_units", [])),
            bank_rows_json_encoding=bank_rows_json_encoding,
        )
        zf.writestr("manifest.json", manifest.model_dump_json(indent=2))

    return archive.getvalue()


@dataclass
class _ExportedAttachments:
    """Attachment rows paired with the archive entries carrying their bytes."""

    rows: list[TransferAttachment] = field(default_factory=list)
    #: archive entry name -> the bytes written under it
    blobs: dict[str, bytes] = field(default_factory=dict)


async def _dump_attachments(conn: Any, bank_id: str, file_storage: Any) -> _ExportedAttachments:
    """Attachment rows plus their bytes, read out of file storage.

    The bytes are the point: ``attachments`` rows only name a storage key, and a
    target instance has neither that key nor the blob behind it. Without the
    bytes an imported bank keeps facts extracted *from* an attachment while the
    attachment itself resolves to nothing.

    Raises when a bank has attachments and no ``file_storage`` was supplied — the
    alternative is an archive that looks complete and silently is not.
    """
    rows = await conn.fetch(
        f"SELECT bank_id, attachment_hash, short_id, media_type, byte_size, storage_key, kind, created_at "
        f"FROM {fq_table('attachments')} WHERE bank_id = $1 ORDER BY attachment_hash",
        bank_id,
    )
    if not rows:
        return _ExportedAttachments()
    if file_storage is None:
        raise ValueError(
            f"Bank '{bank_id}' has {len(rows)} attachment(s) but no file storage was supplied to the export; "
            f"their bytes cannot be carried."
        )
    attachments: list[TransferAttachment] = []
    blobs: dict[str, bytes] = {}
    for index, row in enumerate(rows):
        entry = f"blobs/{index:06d}.bin"
        data = await file_storage.retrieve(row["storage_key"])
        if data is None:
            # The row outlived its blob (an interrupted reclaim, a storage
            # migration). Carrying the row alone reproduces the dangling state on
            # the target, so drop it: the facts keep their text either way.
            logger.warning(
                "[transfer] Attachment %s of bank %s has no bytes at %s; not carried",
                row["short_id"],
                bank_id,
                row["storage_key"],
            )
            continue
        blobs[entry] = bytes(data)
        attachments.append(
            TransferAttachment(
                bank_id=row["bank_id"],
                attachment_hash=row["attachment_hash"],
                short_id=row["short_id"],
                media_type=row["media_type"],
                byte_size=row["byte_size"],
                kind=row["kind"],
                created_at=row["created_at"],
                entry=entry,
            )
        )
    return _ExportedAttachments(rows=attachments, blobs=blobs)


async def _dump_operational_rows(conn: Any, bank_id: str) -> dict[str, list[dict]]:
    """Dump the operations log and the maintenance queues.

    The queues are keyed by ids the import regenerates, so each row is annotated
    with what the target can resolve it by: a unit's ``unit_id`` survives as the
    source id the importer maps, and an entity is named rather than numbered
    (entities are re-resolved by canonical name on the target).
    """
    rows: dict[str, list[dict]] = {}
    rows["async_operations"] = await _dump_bank_rows(conn, "async_operations", bank_id)
    rows["graph_maintenance_queue"] = await _dump_bank_rows(conn, "graph_maintenance_queue", bank_id)
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
    rows["entity_maintenance_queue"] = [
        {k: v for k, v in dict(row).items() if k != "entity_id"} for row in entity_queue if row["canonical_name"]
    ]
    return rows


async def _dump_invalidated_units(conn: Any, bank_id: str) -> list[dict]:
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


# One page of a store scan/listing. Export is a bulk operation and the store pages server-side,
# so this trades round trips against peak memory rather than against latency.
_STORE_PAGE = 500


async def _load_documents_from_store(
    memories: Any,
    bank_id: str,
    document_ids: list[str] | None,
    include_lifecycle: bool = False,
) -> _LoadedExport:
    """The SQL loader's counterpart for a bank whose memories are not in SQL.

    Same archive, assembled through the memories interface: `list_documents` + `get_document_record`
    for the document and its text, `list_chunk_texts` for the chunks, `scan_memories` for the facts,
    and each memory's own `entity_ids` / `causal_edges` for what SQL reads out of `unit_entities` and
    `memory_links`.

    Fact order is re-established here rather than inherited. The SQL query orders by
    `(document_id, created_at, id)` and `causal_relations.target_fact_index` is an ordinal into that
    order, so a scan returning a different one would silently repoint every causal edge. Sorting
    explicitly makes the archive independent of how a store happens to walk.

    One field cannot be carried: `consolidation_failed_at` has no interface field — it is written
    into the store's metadata bag and nothing reads it back — so it exports as unset. That loses the
    record of a consolidation that gave up, not any memory.
    """
    listing = await memories.list_documents(bank_id=bank_id, limit=_STORE_PAGE, offset=0)
    items = list(listing.get("items", []))
    total = int(listing.get("total") or len(items))
    while len(items) < total:
        page = await memories.list_documents(bank_id=bank_id, limit=_STORE_PAGE, offset=len(items))
        page_items = list(page.get("items", []))
        if not page_items:
            break
        items.extend(page_items)

    wanted = set(document_ids) if document_ids else None
    doc_items = [it for it in items if wanted is None or it.get("id") in wanted]
    doc_items.sort(key=lambda it: (it.get("created_at") or datetime.min.replace(tzinfo=UTC), str(it.get("id"))))
    if not doc_items:
        return _LoadedExport()

    selected_ids = [it["id"] for it in doc_items]
    loaded = await _load_facts_from_store(memories, bank_id, selected_ids, include_lifecycle=include_lifecycle)
    await _attach_entities_from_store(memories, bank_id, loaded)
    _attach_causal_relations_from_store(loaded)

    documents: list[TransferDocument] = []
    for item in doc_items:
        doc_id = item["id"]
        record = await memories.get_document_record(bank_id=bank_id, document_id=doc_id, include_text=True)
        texts = await memories.list_chunk_texts(bank_id=bank_id, document_id=doc_id) or []
        documents.append(
            TransferDocument(
                id=doc_id,
                original_text=(record or {}).get("original_text") or (record or {}).get("text") or "",
                retain_params=_as_jsonb(item.get("retain_params")),
                tags=list(item.get("tags") or []),
                created_at=item.get("created_at"),
                chunks=[TransferChunk(chunk_index=i, chunk_text=t) for i, t in enumerate(texts)],
                facts=loaded.facts_by_doc.get(doc_id, []),
            )
        )
    return _LoadedExport(documents=documents, unit_index=loaded.unit_index)


async def _scan_all_memories(memories: Any, bank_id: str, fact_types: list[str] | None) -> list[Any]:
    """Every memory of the given fact types, walked to exhaustion."""
    out: list[Any] = []
    token = ""
    while True:
        page = await memories.scan_memories(
            conn=None,
            fq_table=None,
            bank_id=bank_id,
            fact_types=fact_types,
            limit=_STORE_PAGE,
            page_token=token,
            include_edges=True,
        )
        out.extend(page.memories)
        token = page.next_page_token
        if not token:
            return out


async def _load_facts_from_store(
    memories: Any, bank_id: str, doc_ids: list[str], include_lifecycle: bool = False
) -> _LoadedFacts:
    """Non-observation facts grouped by document, ordered the way the SQL loader orders them."""
    wanted = set(doc_ids)
    stored = [
        m for m in await _scan_all_memories(memories, bank_id, list(_EXPORTED_FACT_TYPES)) if m.document_id in wanted
    ]
    stored.sort(key=lambda m: (m.document_id, m.created_at or datetime.min.replace(tzinfo=UTC), m.unit_id))

    loaded = _LoadedFacts()
    for memory in stored:
        bucket = loaded.facts_by_doc.setdefault(memory.document_id, [])
        ordinal = len(bucket)
        bucket.append(
            TransferFact(
                text=memory.text,
                fact_type=memory.fact_type,
                context=memory.context,
                event_date=memory.event_date,
                occurred_start=memory.occurred_start,
                occurred_end=memory.occurred_end,
                mentioned_at=memory.mentioned_at,
                metadata=as_string_metadata(memory.metadata),
                tags=list(memory.tags or []),
                observation_scopes=memory.observation_scopes,
                chunk_index=_chunk_index_from_chunk_id(memory.chunk_id),
                created_at=memory.created_at if include_lifecycle else None,
                consolidated_at=memory.consolidated_at if include_lifecycle else None,
                consolidation_failed_at=None,
            )
        )
        loaded.unit_index[memory.unit_id] = _UnitLocation(document_id=memory.document_id, ordinal=ordinal)
        loaded.causal_by_unit[memory.unit_id] = list(memory.causal_edges or [])
        loaded.entity_ids_by_unit[memory.unit_id] = list(memory.entity_ids or [])
    return loaded


async def _attach_entities_from_store(memories: Any, bank_id: str, loaded: _LoadedFacts) -> None:
    """Entity canonical names, from the ids each memory carries."""
    if not loaded.entity_ids_by_unit:
        return
    every_id = {e for ids in loaded.entity_ids_by_unit.values() for e in ids}
    names = await memories.resolve_entity_names(conn=None, fq_table=None, bank_id=bank_id, entity_ids=sorted(every_id))
    for unit_id, ids in loaded.entity_ids_by_unit.items():
        location = loaded.unit_index.get(unit_id)
        if location is None:
            continue
        fact = loaded.facts_by_doc[location.document_id][location.ordinal]
        fact.entities.extend(sorted(n for n in (names.get(e) for e in ids) if n))


def _attach_causal_relations_from_store(loaded: _LoadedFacts) -> None:
    """Causal edges as ordinals, from the edges the memories carry.

    Same rule as the SQL path: an edge whose endpoints land in different documents is skipped,
    because the ordinal is only meaningful within one document's fact list.
    """
    for unit_id, edges in loaded.causal_by_unit.items():
        source = loaded.unit_index.get(unit_id)
        if source is None:
            continue
        for edge in edges:
            target = loaded.unit_index.get(edge.target_unit_id)
            if target is None or target.document_id != source.document_id:
                continue
            loaded.facts_by_doc[source.document_id][source.ordinal].causal_relations.append(
                TransferCausalRelation(relation_type=edge.relation_type, target_fact_index=target.ordinal)
            )


async def _load_observations_from_store(
    memories: Any, bank_id: str, unit_index: dict[Any, _UnitLocation]
) -> list[TransferObservation]:
    """Observations whose every source fact is present in the exported set.

    Same rule as the SQL loader: an observation referencing a fact outside the export would import
    as a dangling reference, so it is skipped rather than emitted. Sources come off the memory's own
    `source_memory_ids` here instead of a column, which is also how a store-owned path already
    resolves them — an observation's sources are denormalised onto it at write time.
    """
    stored = await _scan_all_memories(memories, bank_id, ["observation"])
    stored.sort(key=lambda m: (m.created_at or datetime.min.replace(tzinfo=UTC), m.unit_id))

    observations: list[TransferObservation] = []
    skipped = 0
    for memory in stored:
        source_ids = list(memory.source_memory_ids or [])
        locations = [unit_index.get(sid) for sid in source_ids]
        if not source_ids or any(loc is None for loc in locations):
            skipped += 1
            continue
        observations.append(
            TransferObservation(
                source_id=str(memory.unit_id),
                text=memory.text,
                created_at=memory.created_at,
                tags=list(memory.tags or []),
                event_date=memory.event_date,
                occurred_start=memory.occurred_start,
                occurred_end=memory.occurred_end,
                mentioned_at=memory.mentioned_at,
                observation_scopes=memory.observation_scopes,
                proof_count=memory.proof_count or len(source_ids),
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
