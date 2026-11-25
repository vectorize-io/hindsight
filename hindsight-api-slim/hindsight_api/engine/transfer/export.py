"""Export documents (with extracted facts, entities, causal links, chunks) to a ZIP archive.

The memories (documents, chunks, facts, observations, the curation archive) are read
through the memories store — the Postgres store reads its tables on the backend
connection; everything else about the bank is read here directly. Embeddings and
database ids are deliberately omitted — they are regenerated/re-resolved on
import. Consolidated observations are excluded unless ``include_observations``
is set, in which case they are written to ``observations.json``.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

import pydantic_core

from ..db_utils import as_connection, iter_keyset_pages
from ..metadata_utils import as_string_metadata
from ..retain.bank_utils import bank_exists_on_conn
from ..schema import fq_table
from .schema import (
    ATTACHMENT_TABLES,
    CARRIED_HISTORY_TABLES,
    HISTORY_TABLES,
    OPERATIONAL_TABLES,
    SCHEMA_VERSION,
    SESSION_TABLES,
    BankRowsJSONEncoding,
    TransferAttachment,
    TransferCausalRelation,
    TransferChunk,
    TransferDocument,
    TransferFact,
    TransferManifest,
    TransferObservation,
    TransferObservationSource,
    TransferPage,
    TransferScope,
    TransferScopeManifest,
    standalone_mental_models,
)
from .stream_archive import ZipStreamer

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
#: (as typed rows elsewhere) the page tree over them. These are carried
#: with ``data``, not with the configuration: a mental model is a reading of the
#: bank's facts, and its ``based_on`` evidence points straight at them, so it
#: belongs with the memories it was derived from rather than with the settings.
_SYNTHESIZED_ROW_TABLES = ("mental_models",)
_BANK_ROW_TABLES = (*_CONFIG_ROW_TABLES, *_SYNTHESIZED_ROW_TABLES)
# Bank-scoped child-history carried verbatim. Unlike observations, mental models
# keep their (id, bank_id) across export/import, so their refresh history can be
# re-attached. The surrogate ``id`` is dropped on dump so the target reassigns it
# (see _iter_history_row_pages); restored after its parent table (mental_models).
# Operational history — only carried with scope.history.
# Carried under ``scope.data`` as verbatim rows under ``data/``, alongside the
# attachment rows (whose bytes ride in ``blobs/``). These are keyed by globally
# unique ids, or reference ids the replay regenerates, so the importer remaps
# them rather than inserting them as they stand.
_DATA_ROW_TABLES = (*OPERATIONAL_TABLES, *ATTACHMENT_TABLES, "invalidated_memory_units", *SESSION_TABLES)
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
        # Aliases are routing identity, not bank content: they say which ids reach
        # THIS bank on THIS deployment. Carrying them would make an import or clone
        # claim the source's ids — silently stealing its traffic on the same
        # instance, and failing the alias-vs-bank check on another. Add them to the
        # target deliberately, once it is the bank you mean to route to.
        "bank_aliases",
        # Knowledge-bank documents, passages and vectors belong to banks of kind 'knowledge',
        # never to a memory bank being exported; a memory bank only names the knowledge banks
        # it links (bank config `knowledge_banks`), and that travels with its config.
        "kb_documents",
        "kb_passages",
        "kb_passage_vectors",
        "kb_staged_passages",
        "kb_staged_passage_vectors",
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

    For a store-owned bank the SQL tables are empty, so treating "no store supplied" as
    "SQL-backed" produced a silently EMPTY archive with a success status. That has already
    happened once (see the call in `MemoryEngine.export_documents`), and it was fixed at the call
    sites while the default that causes it stayed. An export is the last thing that should fail
    quietly, so the default now asks rather than assumes; only when no store can be built at all
    is the bank genuinely SQL-backed, and then the Postgres store reads it.
    """
    if memories is not None:
        return memories
    try:
        from ..memories import get_memories

        return get_memories()
    except Exception:  # noqa: BLE001 - no store configured at all is genuinely SQL-backed
        from ..memories.postgres import PostgresMemories

        return PostgresMemories({})


@dataclass
class _DocumentTally:
    """What streaming the documents learned, for the sections and manifest after them."""

    count: int = 0
    facts: int = 0
    unit_index: dict[Any, _UnitLocation] = field(default_factory=dict)


async def _stream_documents(
    streamer: ZipStreamer,
    conn_or_backend: Any,
    bank_id: str,
    document_ids: list[str] | None,
    *,
    include_lifecycle: bool,
    memories: Any,
    batch_size: int,
    tally: _DocumentTally,
) -> AsyncIterator[bytes]:
    """Write ``documents/NNNNNN.json`` entries, reading SQL-backed banks ``batch_size`` documents at a time.

    Only one batch of documents is in memory at once; ``tally.unit_index`` still
    grows with every fact, because observations are resolved against it after the
    last batch.
    """

    async def emit(loaded: _LoadedExport) -> AsyncIterator[bytes]:
        for document in loaded.documents:
            tally.facts += len(document.facts)
            entry = f"documents/{tally.count:06d}.json"
            async for chunk in streamer.write_file_bytes(entry, pydantic_core.to_json(document)):
                yield chunk
            tally.count += 1
        tally.unit_index.update(loaded.unit_index)

    # One call either way: the Postgres store reads the bank `batch_size` documents at a time, each
    # batch on a connection it returns before the batch is yielded; a store that owns its memories
    # takes no connection at all.
    async for loaded in memories.iter_transfer_documents(
        backend=conn_or_backend,
        fq_table=fq_table,
        bank_id=bank_id,
        document_ids=document_ids,
        include_lifecycle=include_lifecycle,
        batch_size=batch_size,
    ):
        async for chunk in emit(loaded):
            yield chunk


async def stream_export_documents(
    backend: Any,
    bank_id: str,
    document_ids: list[str] | None = None,
    *,
    include_observations: bool = False,
    include_knowledge_base: bool = False,
    memories: Any = None,
    batch_size: int = 100,
) -> AsyncIterator[bytes]:
    """Stream documents from ``bank_id`` as the chunks of a ZIP archive.

    Args:
        backend: Database backend (provides ``acquire()``) or a single connection.
        bank_id: Source bank.
        document_ids: Specific document ids to export. ``None`` exports every
            document in the bank.
        include_observations: Also export consolidated observations (written to
            ``observations.json``). Only valid for a whole-bank export.

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
    streamer = ZipStreamer()
    tally = _DocumentTally()
    # Carry per-fact consolidation lifecycle exactly when observations are
    # carried: with observations in the archive the target must NOT re-derive
    # them, so imported facts keep their consolidated/failed state. Without
    # observations (the default document export) the target re-consolidates
    # from scratch, so lifecycle is deliberately dropped.
    async for chunk in _stream_documents(
        streamer,
        backend,
        bank_id,
        document_ids,
        include_lifecycle=include_observations,
        memories=memories,
        batch_size=batch_size,
        tally=tally,
    ):
        yield chunk

    observations: list[TransferObservation] = []
    if include_observations:
        async with as_connection(backend) as conn:
            observations = await memories.load_transfer_observations(
                conn=conn, fq_table=fq_table, bank_id=bank_id, unit_index=tally.unit_index
            )
    mental_models: list[dict] = []
    knowledge_pages: list[TransferPage] = []
    if include_knowledge_base:
        async with as_connection(backend) as conn:
            mental_models = await _dump_bank_rows(conn, "mental_models", bank_id)
            knowledge_pages = await _load_pages(conn, bank_id)

    if observations:
        async for chunk in streamer.write_file_bytes("observations.json", pydantic_core.to_json(observations)):
            yield chunk
    if mental_models:
        mm_bytes = json.dumps(mental_models, default=_row_json_default).encode("utf-8")
        async for chunk in streamer.write_file_bytes("mental_models.json", mm_bytes):
            yield chunk
    if knowledge_pages:
        async for chunk in streamer.write_file_bytes("knowledge_pages.json", pydantic_core.to_json(knowledge_pages)):
            yield chunk

    manifest = TransferManifest(
        schema_version=SCHEMA_VERSION,
        source_bank_id=bank_id,
        exported_at=datetime.now(UTC),
        document_count=tally.count,
        fact_count=tally.facts,
        observation_count=len(observations),
        mental_model_count=len(standalone_mental_models(mental_models, knowledge_pages)),
        knowledge_page_count=len(knowledge_pages),
    )
    async for chunk in streamer.write_file_bytes("manifest.json", pydantic_core.to_json(manifest, indent=2)):
        yield chunk
    yield streamer.finish()

    logger.info(
        "[transfer] Exported %d document(s), %d fact(s), %d observation(s) from bank %s",
        tally.count,
        tally.facts,
        len(observations),
        bank_id,
    )


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

    Convenience wrapper buffering ``stream_export_documents`` into memory.
    The async export task streams ``stream_export_documents`` into file storage instead.
    """
    chunks = []
    async for chunk in stream_export_documents(
        backend,
        bank_id,
        document_ids,
        include_observations=include_observations,
        include_knowledge_base=include_knowledge_base,
        memories=memories,
    ):
        chunks.append(chunk)
    return b"".join(chunks)


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


# The key each table dumped by _dump_bank_rows is ordered by (its primary key
# within a bank). Without an ORDER BY the rows come back in heap order, which
# moves whenever a row is updated (e.g. a mental model refresh), so two exports
# of an unchanged bank could differ (#5998). The paged history tables are
# already read in key order (_PAGED_ROW_KEYS). A table missing here fails
# loudly rather than silently exporting in arbitrary order.
_DUMP_ORDER_BY = {
    "banks": "bank_id",
    "directives": "id",
    "webhooks": "id",
    "mental_models": "id",
}


async def _dump_bank_rows(conn: Any, table: str, bank_id: str) -> list[dict]:
    """Dump all rows of a bank-scoped table as JSON-ready dicts (derived columns stripped).

    Embedding/search-vector columns are omitted so the target instance
    regenerates them with its own model/backend on import.
    """
    rows = await conn.fetch(
        f"SELECT * FROM {fq_table(table)} WHERE bank_id = $1 ORDER BY {_DUMP_ORDER_BY[table]}", bank_id
    )
    return [{k: v for k, v in dict(row).items() if k not in _DERIVED_COLUMNS} for row in rows]


# The tables that grow with a bank's activity rather than its configuration, and the key — unique
# within a bank — each is paged by on export. The operation log alone can hold millions of rows,
# each with its task payload; reading one whole OOM-killed the API on a large bank (#5737).
_PAGED_ROW_KEYS = {
    "async_operations": "operation_id",
    "graph_maintenance_queue": "unit_id",
    "audit_log": "id",
    "llm_requests": "id",
    # Every refresh stores a full snapshot of the model: megabytes a row, gigabytes a bank (#6259).
    "mental_model_history": "id",
    "sessions": "session_id",
}
#: Rows per page. Small, because one operation row carries its whole task payload.
_ROW_PAGE = 100


async def _iter_bank_row_pages(conn_or_backend: Any, table: str, bank_id: str) -> AsyncIterator[list[dict]]:
    """A paged table's rows, derived columns stripped, ``_ROW_PAGE`` at a time in key order."""
    async for rows in iter_keyset_pages(
        conn_or_backend,
        f"SELECT * FROM {fq_table(table)} WHERE bank_id = $1",
        [bank_id],
        key=_PAGED_ROW_KEYS[table],
        page_size=_ROW_PAGE,
    ):
        yield [{k: v for k, v in dict(row).items() if k not in _DERIVED_COLUMNS} for row in rows]


async def _iter_session_event_pages(conn_or_backend: Any, bank_id: str) -> AsyncIterator[list[dict]]:
    """Every session's log, session by session in id order, each one paged by ``seq``.

    The key is ``(session_id, seq)``, which a single-column keyset can't page, so the sessions are
    paged by id and each one's events by ``seq`` within it; neither list is held whole.
    """
    async for sessions in iter_keyset_pages(
        conn_or_backend,
        f"SELECT session_id FROM {fq_table('sessions')} WHERE bank_id = $1",
        [bank_id],
        key="session_id",
        page_size=_ROW_PAGE,
    ):
        for session in sessions:
            async for rows in iter_keyset_pages(
                conn_or_backend,
                f"SELECT * FROM {fq_table('session_events')} WHERE bank_id = $1 AND session_id = $2",
                [bank_id, session["session_id"]],
                key="seq",
                page_size=_ROW_PAGE,
            ):
                yield [dict(row) for row in rows]


async def _load_pages(conn: Any, bank_id: str) -> list[TransferPage]:
    """Load the page tree (folders + pages) as typed rows.

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
        TransferPage(
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


async def _iter_history_row_pages(conn_or_backend: Any, table: str, bank_id: str) -> AsyncIterator[list[dict]]:
    """A carried child-history table's rows, a page at a time, oldest first.

    Drops the surrogate ``id`` so the target reassigns it from its own IDENTITY
    sequence (carrying explicit ids would leave the sequence un-advanced and
    collide with later writes). Paged in ``id`` order — the order rows were
    written — so the reassigned ids keep the chronological tie-break order the
    read path relies on.
    """
    async for rows in _iter_bank_row_pages(conn_or_backend, table, bank_id):
        yield [{k: v for k, v in row.items() if k != "id"} for row in rows]


class BankNotFoundError(LookupError):
    """The bank to export has no row: a typo'd id or the wrong instance, not an empty bank."""


async def stream_export_bank(
    conn_or_backend: Any,
    bank_id: str,
    *,
    scope: TransferScope | None = None,
    bank_rows_json_encoding: BankRowsJSONEncoding = "serialized",
    memories: Any = None,
    file_storage: Any = None,
    batch_size: int = 100,
) -> AsyncIterator[bytes]:
    """Stream a whole bank as the chunks of a portable ZIP archive (no embeddings).

    ``scope`` decides what travels — see :class:`TransferScope`. The one way a
    whole bank is read: the export endpoint, the admin CLI and the clone all go
    through here.

    Memory: documents are read ``batch_size`` at a time, attachment bytes are
    streamed from file storage, and the operation log, the maintenance queues, the
    curation archive, the history tables and the mental-model history are read and
    written a page at a time, so none is ever held whole (#5737, #6259). The index
    of every fact, the observations and the bank's own rows (config, mental models,
    pages) are still loaded in full before they are written.

    Consistency: ``conn_or_backend`` is either one connection (the admin CLI, which
    owns it) or a backend that each read borrows a connection from and returns —
    the export can run for as long as its attachments take to copy, and holding a
    pooled connection and transaction open that long would starve everyone else.
    The price is that a bank written to during the export is read at several
    points in time: observations are filtered to the facts that were exported,
    but a mental model can cite a document that arrived after the documents were
    read. The clone takes the same reads: it once read under one transaction
    instead, which on a large bank stayed open until a pooler cut it (#6666) and,
    at READ COMMITTED, never gave a single snapshot anyway.

    Raises :class:`BankNotFoundError` before the first chunk when the bank has no
    row, so a caller writing the stream to a file never gets an archive for it.
    """
    # A missing bank would otherwise export as a well-formed, empty archive that an
    # operator only finds out is useless at restore time (#6776).
    async with as_connection(conn_or_backend) as conn:
        if not await bank_exists_on_conn(conn, bank_id):
            raise BankNotFoundError(f"Bank '{bank_id}' not found")
    scope = scope or TransferScope()
    memories = _resolve_memories(memories)
    streamer = ZipStreamer()
    tally = _DocumentTally()
    observations: list[TransferObservation] = []
    attachments: list[TransferAttachment] = []
    # Rows written per entry by `write_row_pages`, for the manifest.
    row_counts: dict[str, int] = {}
    knowledge_pages: list[TransferPage] = []
    bank_rows: dict[str, list[dict]] = {}

    async def write_rows(entry: str, rows: list[dict]) -> AsyncIterator[bytes]:
        async for chunk in streamer.write_file_bytes(
            entry, json.dumps(rows, default=_row_json_default).encode("utf-8")
        ):
            yield chunk

    async def write_row_pages(entry: str, pages: AsyncIterator[list[dict]]) -> AsyncIterator[bytes]:
        """Write a JSON array one page of rows at a time, so the entry is never encoded whole."""

        async def encoded() -> AsyncIterator[bytes]:
            yield b"["
            separator = b""
            async for rows in pages:
                if not rows:  # a page can filter down to nothing, and "[, …]" is not JSON
                    continue
                yield separator + json.dumps(rows, default=_row_json_default)[1:-1].encode("utf-8")
                separator = b", "
                row_counts[entry] = row_counts.get(entry, 0) + len(rows)
            yield b"]"

        # Row tables grow without bound (history/llm_requests passes 4 GB on busy banks, #6745).
        async for chunk in streamer.write_file_chunks(entry, encoded(), zip64=True):
            yield chunk

    if scope.data:
        # Whole-bank export always carries observations (they're bank-level state)
        # and, with them, the per-fact consolidation lifecycle so the target restores
        # exact eligibility instead of re-consolidating historical facts (#2965).
        async for chunk in _stream_documents(
            streamer,
            conn_or_backend,
            bank_id,
            None,
            include_lifecycle=True,
            memories=memories,
            batch_size=batch_size,
            tally=tally,
        ):
            yield chunk
        async with as_connection(conn_or_backend) as conn:
            observations = await memories.load_transfer_observations(
                conn=conn, fq_table=fq_table, bank_id=bank_id, unit_index=tally.unit_index
            )
        if observations:
            async for chunk in streamer.write_file_bytes("observations.json", pydantic_core.to_json(observations)):
                yield chunk

        async for chunk in _stream_attachments(streamer, conn_or_backend, bank_id, file_storage, attachments):
            yield chunk

        # Lazy: each table is read only as its entry is written.
        data_sections = {
            "data/async_operations.json": _iter_bank_row_pages(conn_or_backend, "async_operations", bank_id),
            # Keyed by units the import regenerates: each row's `unit_id` stays the source id,
            # which the importer maps to the replayed unit (a row whose unit did not come back
            # is dropped there).
            "data/graph_maintenance_queue.json": _iter_bank_row_pages(
                conn_or_backend, "graph_maintenance_queue", bank_id
            ),
            "data/entity_maintenance_queue.json": memories.iter_entity_maintenance_queue(
                backend=conn_or_backend, fq_table=fq_table, bank_id=bank_id, page_size=_ROW_PAGE
            ),
            "data/invalidated_memory_units.json": _iter_invalidated_units(conn_or_backend, bank_id, memories),
            "data/sessions.json": _iter_bank_row_pages(conn_or_backend, "sessions", bank_id),
            "data/session_events.json": _iter_session_event_pages(conn_or_backend, bank_id),
        }
        for entry, row_pages in data_sections.items():
            async for chunk in write_row_pages(entry, row_pages):
                yield chunk

        # Synthesized knowledge travels with the memories it was synthesized from:
        # a mental model reads the bank's facts and cites them by id, and a
        # page is a view over a mental model.
        async with as_connection(conn_or_backend) as conn:
            for table in _SYNTHESIZED_ROW_TABLES:
                bank_rows[table] = await _dump_bank_rows(conn, table, bank_id)
            knowledge_pages = await _load_pages(conn, bank_id)
        for table in CARRIED_HISTORY_TABLES:
            async for chunk in write_row_pages(
                f"{table}.json", _iter_history_row_pages(conn_or_backend, table, bank_id)
            ):
                yield chunk

    if scope.bank_config:
        async with as_connection(conn_or_backend) as conn:
            for table in _CONFIG_ROW_TABLES:
                bank_rows[table] = await _dump_bank_rows(conn, table, bank_id)
    for table, rows in bank_rows.items():
        async for chunk in write_rows(f"{table}.json", rows):
            yield chunk
    if scope.data:
        # Typed page tree (parent-first). Written even when empty so the
        # importer can distinguish "no pages" from a pre-tree archive.
        pages = pydantic_core.to_json(knowledge_pages) if knowledge_pages else b"[]\n"
        async for chunk in streamer.write_file_bytes("knowledge_pages.json", pages):
            yield chunk

    if scope.history:
        for table in HISTORY_TABLES:
            async for chunk in write_row_pages(
                f"history/{table}.json", _iter_bank_row_pages(conn_or_backend, table, bank_id)
            ):
                yield chunk

    manifest = TransferManifest(
        schema_version=SCHEMA_VERSION,
        source_bank_id=bank_id,
        exported_at=datetime.now(UTC),
        document_count=tally.count,
        fact_count=tally.facts,
        observation_count=len(observations),
        archive_type="bank",
        mental_model_count=len(standalone_mental_models(bank_rows.get("mental_models", []), knowledge_pages)),
        knowledge_page_count=len(knowledge_pages),
        directive_count=len(bank_rows.get("directives", [])),
        webhook_count=len(bank_rows.get("webhooks", [])),
        includes_history=scope.history,
        scope=TransferScopeManifest(data=scope.data, bank_config=scope.bank_config, history=scope.history),
        attachment_count=len(attachments),
        operation_count=row_counts.get("data/async_operations.json", 0),
        invalidated_memory_count=row_counts.get("data/invalidated_memory_units.json", 0),
        bank_rows_json_encoding=bank_rows_json_encoding,
    )
    async for chunk in streamer.write_file_bytes("manifest.json", pydantic_core.to_json(manifest, indent=2)):
        yield chunk
    yield streamer.finish()

    logger.info(
        "[transfer] Exported bank %s (data=%s, bank_config=%s, history=%s): %d document(s), "
        "%d observation(s), %d attachment(s), %d operation(s), %d mental model(s), %d page(s), "
        "%d directive(s), %d webhook(s)",
        bank_id,
        scope.data,
        scope.bank_config,
        scope.history,
        tally.count,
        len(observations),
        len(attachments),
        row_counts.get("data/async_operations.json", 0),
        len(standalone_mental_models(bank_rows.get("mental_models", []), knowledge_pages)),
        len(knowledge_pages),
        len(bank_rows.get("directives", [])),
        len(bank_rows.get("webhooks", [])),
    )


async def _stream_attachments(
    streamer: ZipStreamer,
    conn_or_backend: Any,
    bank_id: str,
    file_storage: Any,
    exported: list[TransferAttachment],
) -> AsyncIterator[bytes]:
    """Stream each attachment's bytes into ``blobs/``, then ``attachments.json`` naming them.

    Appends every carried attachment to ``exported``. A bank with attachments and
    no ``file_storage`` raises — the alternative is an archive that looks complete
    and silently is not — and an attachment whose bytes are gone is left out with
    a warning.
    """
    async with as_connection(conn_or_backend) as conn:
        rows = await conn.fetch(
            f"SELECT bank_id, document_id, attachment_hash, short_id, media_type, byte_size, storage_key, kind, "
            f"filename, created_at "
            f"FROM {fq_table('attachments')} WHERE bank_id = $1 ORDER BY document_id, attachment_hash",
            bank_id,
        )
    if not rows:
        return
    if file_storage is None:
        raise ValueError(
            f"Bank '{bank_id}' has {len(rows)} attachment(s) but no file storage was supplied to the export; "
            f"their bytes cannot be carried."
        )
    for index, row in enumerate(rows):
        blob = file_storage.retrieve_stream(row["storage_key"])
        # Pull the first piece before the entry is opened: a missing blob surfaces
        # here as FileNotFoundError, and once the entry header is written there is
        # no taking it back. (An existence probe first would cost a round trip per
        # attachment and still race a concurrent delete.)
        try:
            first = await anext(blob)
        except FileNotFoundError:
            logger.warning(
                "[transfer] Attachment %s of bank %s has no bytes at %s; not carried",
                row["short_id"],
                bank_id,
                row["storage_key"],
            )
            continue
        except StopAsyncIteration:
            first = b""

        async def rest(first: bytes = first, blob: AsyncIterator[bytes] = blob) -> AsyncIterator[bytes]:
            yield first
            async for piece in blob:
                yield piece

        entry = f"blobs/{index:06d}.bin"
        # DEFLATE at level 0: attachments are images/PDFs that are already
        # compressed, but a deflate stream (unlike STORED) is self-terminating,
        # which sequential ZIP readers need when sizes only follow the data.
        # Zip64 is announced in the entry header, before any byte is streamed, so it is
        # decided from the recorded size. Attachments over 4 GB used to fail the export.
        zip64 = row["byte_size"] >= 0xFFFFFFFF
        async for chunk in streamer.write_file_chunks(entry, rest(), compression_level=0, zip64=zip64):
            yield chunk
        exported.append(
            TransferAttachment(
                bank_id=row["bank_id"],
                document_id=row["document_id"],
                attachment_hash=row["attachment_hash"],
                short_id=row["short_id"],
                media_type=row["media_type"],
                byte_size=row["byte_size"],
                kind=row["kind"],
                filename=row["filename"],
                created_at=row["created_at"],
                entry=entry,
            )
        )
    if exported:
        async for chunk in streamer.write_file_bytes("attachments.json", pydantic_core.to_json(exported)):
            yield chunk


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

    Convenience wrapper buffering ``stream_export_bank`` into memory.
    Production export endpoints and CLI use ``stream_export_bank`` directly
    so the archive is never held whole.
    """
    chunks = []
    async for chunk in stream_export_bank(
        conn,
        bank_id,
        scope=scope,
        bank_rows_json_encoding=bank_rows_json_encoding,
        memories=memories,
        file_storage=file_storage,
    ):
        chunks.append(chunk)
    return b"".join(chunks)


async def _iter_invalidated_units(conn_or_backend: Any, bank_id: str, memories: Any) -> AsyncIterator[list[dict]]:
    """The curation archive (facts a user invalidated but can still revert), by page, from the store that holds it.

    A store that cannot list its archive exports without it, loudly: the archive used to be read
    from a SQL table a store-owned bank leaves empty, so the export silently carried none (#4969).
    """
    try:
        async for page in memories.iter_archived_memories(
            backend=conn_or_backend, fq_table=fq_table, bank_id=bank_id, page_size=_ROW_PAGE
        ):
            yield page
    except NotImplementedError:
        logger.warning(
            "[transfer] The memories store cannot list archived memories; bank %s is exported without "
            "its curation archive",
            bank_id,
        )


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
    loaded = await _load_facts_from_store(
        memories, bank_id, selected_ids, include_lifecycle=include_lifecycle, whole_bank=wanted is None
    )
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


async def _scan_all_memories(
    memories: Any, bank_id: str, fact_types: list[str] | None, document_id: str | None = None
) -> list[Any]:
    """Every memory of the given fact types (of one document, when named), walked to exhaustion."""
    out: list[Any] = []
    token = ""
    while True:
        page = await memories.scan_memories(
            conn=None,
            fq_table=None,
            bank_id=bank_id,
            fact_types=fact_types,
            document_id=document_id,
            limit=_STORE_PAGE,
            page_token=token,
            include_edges=True,
        )
        out.extend(page.memories)
        token = page.next_page_token
        if not token:
            return out


async def _load_facts_from_store(
    memories: Any, bank_id: str, doc_ids: list[str], include_lifecycle: bool = False, whole_bank: bool = True
) -> _LoadedFacts:
    """Non-observation facts grouped by document, ordered the way the SQL loader orders them.

    ``whole_bank`` is False when the caller picked the documents: those are then read one document
    at a time, a scan a store can answer from the document alone, rather than by walking every
    memory of the bank to keep the few that belong to them."""
    fact_types = list(_EXPORTED_FACT_TYPES)
    if whole_bank:
        wanted = set(doc_ids)
        stored = [m for m in await _scan_all_memories(memories, bank_id, fact_types) if m.document_id in wanted]
    else:
        stored = [m for d in doc_ids for m in await _scan_all_memories(memories, bank_id, fact_types, document_id=d)]
    stored.sort(key=lambda m: (m.document_id, m.created_at or datetime.min.replace(tzinfo=UTC), m.unit_id))

    loaded = _LoadedFacts()
    for memory in stored:
        bucket = loaded.facts_by_doc.setdefault(memory.document_id, [])
        ordinal = len(bucket)
        bucket.append(
            TransferFact(
                text=memory.text,
                # Carried so an import can point mental-model evidence at the fact's new id.
                source_id=str(memory.unit_id),
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
