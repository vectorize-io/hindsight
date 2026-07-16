"""Export documents (with extracted facts, entities, causal links, chunks) to a ZIP archive.

Reads directly from the database via the backend connection. Embeddings and
database ids are deliberately omitted — they are regenerated/re-resolved on
import. Consolidated observations are excluded unless ``include_observations``
is set, in which case they are written to ``observations.json``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from ..causal_links import CAUSAL_LINK_TYPES
from ..db_utils import acquire_with_retry
from ..schema import fq_table
from ..storage.base import FileStorage
from .archive import TransferArchive
from .schema import (
    CARRIED_HISTORY_TABLES,
    HISTORY_TABLES,
    SCHEMA_VERSION,
    BankRowsJSONEncoding,
    TransferCausalRelation,
    TransferChunk,
    TransferDocument,
    TransferFact,
    TransferImageAsset,
    TransferImageLink,
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
        # Logical image rows are carried by TransferDocument in document-transfer
        # v2; they are not dumped as raw database rows or source storage keys.
        "image_assets",
        "document_image_links",
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
        "transfer_staging",  # transient upload locator; archives are cleaned after import
        "file_storage",  # raw uploads; documents.original_text is already carried
        "file_storage_chunks",  # physical chunks owned by skipped file_storage rows
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
_DOCUMENT_EXPORT_PAGE_SIZE = 100


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


@dataclass(frozen=True)
class _ArchiveDocumentStats:
    """Counts produced while writing the shared document portion of an archive."""

    documents: int
    facts: int
    observations: int
    image_assets: int


async def _write_document_entries(
    zf: zipfile.ZipFile,
    conn: Any,
    bank_id: str,
    *,
    document_ids: list[str] | None,
    include_observations: bool,
    file_storage: FileStorage | None,
) -> _ArchiveDocumentStats:
    """Write paged documents, images, and observations into an open ZIP."""
    fact_total = 0
    document_total = 0
    observation_total = 0
    image_entries: dict[str, str] = {}
    unit_index_path: str | None = None
    unit_index_db: sqlite3.Connection | None = None
    if include_observations:
        index_file = tempfile.NamedTemporaryFile(prefix="hindsight-unit-index-", suffix=".sqlite", delete=False)
        unit_index_path = index_file.name
        index_file.close()
        unit_index_db = sqlite3.connect(unit_index_path)
        unit_index_db.execute(
            "CREATE TABLE unit_locations (unit_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, ordinal INTEGER NOT NULL)"
        )
    try:
        offset = 0
        while True:
            loaded = await _load_documents(
                conn,
                bank_id,
                document_ids,
                offset=offset,
                limit=_DOCUMENT_EXPORT_PAGE_SIZE,
            )
            documents = loaded.documents
            if not documents:
                break
            if unit_index_db is not None:
                unit_index_db.executemany(
                    "INSERT INTO unit_locations (unit_id, document_id, ordinal) VALUES (?, ?, ?)",
                    [
                        (str(unit_id), location.document_id, location.ordinal)
                        for unit_id, location in loaded.unit_index.items()
                    ],
                )
            image_storage_keys = await _attach_images(conn, bank_id, documents)
            if image_storage_keys and file_storage is None:
                raise ValueError("FileStorage is required to export documents containing managed images")
            for document in documents:
                for asset in document.image_assets:
                    if asset.archive_entry is None:
                        continue
                    if asset.asset_id not in image_entries:
                        assert file_storage is not None
                        entry = f"assets/{len(image_entries):06d}"
                        image_entries[asset.asset_id] = entry
                        storage_key = image_storage_keys[asset.asset_id]
                        with zf.open(entry, "w", force_zip64=True) as target:
                            async for chunk in file_storage.iter_bytes(storage_key):
                                target.write(chunk)
                    asset.archive_entry = image_entries[asset.asset_id]
                fact_total += len(document.facts)
                zf.writestr(
                    f"documents/{document_total:06d}.json",
                    document.model_dump_json(indent=2, exclude_none=False),
                )
                document_total += 1
            offset += len(documents)
            if len(documents) < _DOCUMENT_EXPORT_PAGE_SIZE:
                break
        if unit_index_db is not None:
            unit_index_db.commit()
            observation_total = await _write_observations_stream(zf, conn, bank_id, unit_index_db)
    finally:
        if unit_index_db is not None:
            unit_index_db.close()
        if unit_index_path is not None:
            try:
                os.unlink(unit_index_path)
            except FileNotFoundError:
                pass
    return _ArchiveDocumentStats(
        documents=document_total,
        facts=fact_total,
        observations=observation_total,
        image_assets=len(image_entries),
    )


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
    file_storage: Any | None = None,
) -> TransferArchive:
    """Export documents from ``bank_id`` into a disk-backed ZIP archive.

    Args:
        backend: Database backend (provides ``acquire()``).
        bank_id: Source bank.
        document_ids: Specific document ids to export. ``None`` exports every
            document in the bank.
        include_observations: Also export consolidated observations (written to
            ``observations.json``). Only valid for a whole-bank export.

    Returns:
        A one-shot streamable archive handle.

    Raises:
        ValueError: if ``include_observations`` is combined with ``document_ids``.
    """
    # Observations are bank-level and can be derived from facts spanning several
    # documents, so they're only coherent when the whole bank is exported. For a
    # document subset we'd have to silently drop every cross-document observation
    # — reject the combination instead so the caller isn't surprised.
    if include_observations and document_ids is not None:
        raise ValueError("include_observations is only supported when exporting the whole bank (omit document_id)")

    archive_file = tempfile.NamedTemporaryFile(prefix="hindsight-transfer-", suffix=".zip", delete=False)
    archive_path = archive_file.name
    archive_file.close()
    try:
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            async with acquire_with_retry(backend) as conn:
                async with conn.transaction():
                    stats = await _write_document_entries(
                        zf,
                        conn,
                        bank_id,
                        document_ids=document_ids,
                        include_observations=include_observations,
                        file_storage=file_storage,
                    )

            manifest = TransferManifest(
                schema_version=SCHEMA_VERSION,
                source_bank_id=bank_id,
                exported_at=datetime.now(UTC),
                document_count=stats.documents,
                fact_count=stats.facts,
                observation_count=stats.observations,
                image_asset_count=stats.image_assets,
            )
            zf.writestr("manifest.json", manifest.model_dump_json(indent=2))
    except Exception:
        try:
            os.unlink(archive_path)
        except FileNotFoundError:
            pass
        raise
    logger.info(
        "[transfer] Exported %d document(s), %d fact(s), %d observation(s) from bank %s",
        stats.documents,
        stats.facts,
        stats.observations,
        bank_id,
    )
    return TransferArchive(path=archive_path, size_bytes=os.path.getsize(archive_path))


async def _attach_images(conn: Any, bank_id: str, documents: list[TransferDocument]) -> dict[str, str]:
    """Attach active managed-image metadata without exposing storage keys."""
    document_ids = [document.id for document in documents]
    if not document_ids:
        return {}
    links = await conn.fetch(
        f"SELECT * FROM {fq_table('document_image_links')} "
        "WHERE bank_id = $1 AND document_id = ANY($2) ORDER BY document_id, ordinal",
        bank_id,
        document_ids,
    )
    if not links:
        return {}
    asset_ids = list(dict.fromkeys(str(row["asset_id"]) for row in links))
    assets = await conn.fetch(
        f"SELECT * FROM {fq_table('image_assets')} WHERE bank_id = $1 AND asset_id = ANY($2)",
        bank_id,
        asset_ids,
    )
    assets_by_id = {str(row["asset_id"]): row for row in assets}
    links_by_document: dict[str, list[Any]] = {}
    for row in links:
        links_by_document.setdefault(str(row["document_id"]), []).append(row)
    storage_keys: dict[str, str] = {}
    for document in documents:
        document_links = links_by_document.get(document.id, [])
        document.image_links = [
            TransferImageLink(
                asset_id=str(row["asset_id"]),
                ordinal=int(row["ordinal"]),
                image_context=row["image_context"],
                created_at=row["created_at"],
            )
            for row in document_links
        ]
        seen: set[str] = set()
        for link in document_links:
            asset_id = str(link["asset_id"])
            if asset_id in seen:
                continue
            seen.add(asset_id)
            row = assets_by_id[asset_id]
            ready = str(row["status"]) == "ready"
            document.image_assets.append(
                TransferImageAsset(
                    asset_id=asset_id,
                    mime_type=str(row["mime_type"]),
                    size_bytes=int(row["size_bytes"]),
                    sha256=str(row["sha256"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                    status=str(row["status"]),
                    archive_entry="pending" if ready else None,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )
            if ready:
                storage_keys[asset_id] = str(row["storage_key"])
    return storage_keys


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


_BANK_TABLE_ORDER = {
    "banks": "bank_id",
    "mental_models": "id",
    "directives": "id",
    "webhooks": "id",
    "mental_model_history": "changed_at, id",
    "audit_log": "started_at, id",
    "llm_requests": "started_at, id",
}


async def _write_bank_table_stream(
    zf: zipfile.ZipFile,
    conn: Any,
    table: str,
    bank_id: str,
    *,
    archive_entry: str,
    drop_surrogate_id: bool = False,
) -> int:
    """Write a bank table as a paged JSON array without retaining all rows."""
    target = zf.open(archive_entry, "w", force_zip64=True)
    offset = 0
    written = 0
    try:
        target.write(b"[\n")
        while True:
            rows = await conn.fetch(
                f"SELECT * FROM {fq_table(table)} WHERE bank_id = $1 "
                f"ORDER BY {_BANK_TABLE_ORDER[table]} "
                "OFFSET $2 ROWS FETCH NEXT $3 ROWS ONLY",
                bank_id,
                offset,
                _DOCUMENT_EXPORT_PAGE_SIZE,
            )
            for source_row in rows:
                row = {
                    key: value
                    for key, value in dict(source_row).items()
                    if key not in _DERIVED_COLUMNS and not (drop_surrogate_id and key == "id")
                }
                if written:
                    target.write(b",\n")
                target.write(json.dumps(row, indent=2, default=_row_json_default).encode())
                written += 1
            offset += len(rows)
            if len(rows) < _DOCUMENT_EXPORT_PAGE_SIZE:
                break
        target.write(b"\n]\n")
    finally:
        target.close()
    return written


async def export_bank(
    conn: Any,
    bank_id: str,
    *,
    include_history: bool = False,
    bank_rows_json_encoding: BankRowsJSONEncoding = "serialized",
    file_storage: FileStorage | None = None,
) -> TransferArchive:
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
    archive_file = tempfile.NamedTemporaryFile(prefix="hindsight-bank-export-", suffix=".zip", delete=False)
    archive_path = archive_file.name
    archive_file.close()
    try:
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            async with conn.transaction():
                stats = await _write_document_entries(
                    zf,
                    conn,
                    bank_id,
                    document_ids=None,
                    include_observations=True,
                    file_storage=file_storage,
                )
                table_counts: dict[str, int] = {}
                for table in _BANK_ROW_TABLES:
                    table_counts[table] = await _write_bank_table_stream(
                        zf,
                        conn,
                        table,
                        bank_id,
                        archive_entry=f"{table}.json",
                    )
                for table in CARRIED_HISTORY_TABLES:
                    table_counts[table] = await _write_bank_table_stream(
                        zf,
                        conn,
                        table,
                        bank_id,
                        archive_entry=f"{table}.json",
                        drop_surrogate_id=True,
                    )
                if include_history:
                    for table in HISTORY_TABLES:
                        await _write_bank_table_stream(
                            zf,
                            conn,
                            table,
                            bank_id,
                            archive_entry=f"history/{table}.json",
                        )

            manifest = TransferManifest(
                schema_version=SCHEMA_VERSION,
                source_bank_id=bank_id,
                exported_at=datetime.now(UTC),
                document_count=stats.documents,
                fact_count=stats.facts,
                observation_count=stats.observations,
                image_asset_count=stats.image_assets,
                archive_type="bank",
                mental_model_count=table_counts.get("mental_models", 0),
                directive_count=table_counts.get("directives", 0),
                webhook_count=table_counts.get("webhooks", 0),
                includes_history=include_history,
                bank_rows_json_encoding=bank_rows_json_encoding,
            )
            zf.writestr("manifest.json", manifest.model_dump_json(indent=2))
    except Exception:
        try:
            os.unlink(archive_path)
        except FileNotFoundError:
            pass
        raise

    logger.info(
        "[transfer] Exported bank %s: %d document(s), %d fact(s), %d observation(s), "
        "%d mental model(s), %d directive(s), %d webhook(s)%s",
        bank_id,
        stats.documents,
        stats.facts,
        stats.observations,
        table_counts.get("mental_models", 0),
        table_counts.get("directives", 0),
        table_counts.get("webhooks", 0),
        " (with history)" if include_history else "",
    )
    return TransferArchive(path=archive_path, size_bytes=os.path.getsize(archive_path))


async def _load_documents(
    conn: Any,
    bank_id: str,
    document_ids: list[str] | None,
    *,
    offset: int | None = None,
    limit: int | None = None,
) -> _LoadedExport:
    """Load and assemble TransferDocument payloads for the requested documents."""
    doc_filter = "AND id = ANY($2)" if document_ids else ""
    params: list[Any] = [bank_id]
    if document_ids:
        params.append(document_ids)
    page_clause = ""
    if offset is not None or limit is not None:
        offset = offset or 0
        limit = limit or _DOCUMENT_EXPORT_PAGE_SIZE
        offset_placeholder = len(params) + 1
        limit_placeholder = len(params) + 2
        params.extend((offset, limit))
        page_clause = f"OFFSET ${offset_placeholder} ROWS FETCH NEXT ${limit_placeholder} ROWS ONLY"
    doc_rows = await conn.fetch(
        f"""
        SELECT id, original_text, retain_params, tags, created_at
        FROM {fq_table("documents")}
        WHERE bank_id = $1 {doc_filter}
        ORDER BY created_at, id
        {page_clause}
        """,
        *params,
    )
    if not doc_rows:
        return _LoadedExport()

    selected_ids = [row["id"] for row in doc_rows]

    chunks_by_doc = await _load_chunks(conn, bank_id, selected_ids)
    loaded = await _load_facts(conn, bank_id, selected_ids)
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


async def _write_observations_stream(
    zf: zipfile.ZipFile,
    conn: Any,
    bank_id: str,
    unit_index_db: sqlite3.Connection,
) -> int:
    """Write observations incrementally using a disk-backed source-unit index."""
    offset = 0
    written = 0
    skipped = 0
    target = None
    try:
        while True:
            rows = await conn.fetch(
                f"""
                SELECT id, text, tags, event_date, occurred_start, occurred_end,
                       mentioned_at, observation_scopes, proof_count, source_memory_ids
                FROM {fq_table("memory_units")}
                WHERE bank_id = $1 AND fact_type = 'observation'
                ORDER BY created_at, id
                OFFSET $2 ROWS FETCH NEXT $3 ROWS ONLY
                """,
                bank_id,
                offset,
                _DOCUMENT_EXPORT_PAGE_SIZE,
            )
            if not rows:
                break
            for row in rows:
                source_ids = list(row["source_memory_ids"] or [])
                locations: list[_UnitLocation] = []
                for source_id in source_ids:
                    location_row = unit_index_db.execute(
                        "SELECT document_id, ordinal FROM unit_locations WHERE unit_id = ?",
                        (str(source_id),),
                    ).fetchone()
                    if location_row is None:
                        locations = []
                        break
                    locations.append(_UnitLocation(document_id=str(location_row[0]), ordinal=int(location_row[1])))
                if not source_ids or len(locations) != len(source_ids):
                    skipped += 1
                    continue
                observation = TransferObservation(
                    text=row["text"],
                    tags=list(row["tags"] or []),
                    event_date=row["event_date"],
                    occurred_start=row["occurred_start"],
                    occurred_end=row["occurred_end"],
                    mentioned_at=row["mentioned_at"],
                    observation_scopes=_as_jsonb(row["observation_scopes"]),
                    proof_count=row["proof_count"] or len(source_ids),
                    sources=[
                        TransferObservationSource(document_id=location.document_id, fact_index=location.ordinal)
                        for location in locations
                    ],
                )
                if target is None:
                    target = zf.open("observations.json", "w", force_zip64=True)
                    target.write(b"[\n")
                elif written:
                    target.write(b",\n")
                target.write(observation.model_dump_json(indent=2).encode())
                written += 1
            offset += len(rows)
            if len(rows) < _DOCUMENT_EXPORT_PAGE_SIZE:
                break
        if target is not None:
            target.write(b"\n]\n")
    finally:
        if target is not None:
            target.close()
    if skipped:
        logger.info("[transfer] Skipped %d observation(s) with sources outside the exported documents", skipped)
    return written


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


async def _load_facts(conn: Any, bank_id: str, doc_ids: list[str]) -> _LoadedFacts:
    """Load non-observation facts grouped by document, with a unit-id location index.

    The ordering is fixed (created_at, id) so that
    ``causal_relations.target_fact_index`` ordinals stay consistent.
    """
    rows = await conn.fetch(
        f"""
        SELECT id, document_id, text, fact_type, context, event_date,
               occurred_start, occurred_end, mentioned_at, metadata,
               chunk_id, tags, observation_scopes
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
            metadata=_as_jsonb(row["metadata"]) or {},
            tags=list(row["tags"] or []),
            observation_scopes=_as_jsonb(row["observation_scopes"]),
            chunk_index=_chunk_index_from_chunk_id(row["chunk_id"]),
        )
        bucket.append(fact)
        loaded.unit_index[row["id"]] = _UnitLocation(document_id=doc_id, ordinal=ordinal)
    return loaded


async def _attach_entities(conn: Any, loaded: _LoadedFacts) -> None:
    """Populate each fact's ``entities`` list with its entities' canonical names."""
    if not loaded.unit_index:
        return
    rows = await conn.fetch(
        f"""
        SELECT ue.unit_id, e.canonical_name
        FROM {fq_table("unit_entities")} ue
        JOIN {fq_table("entities")} e ON e.id = ue.entity_id
        WHERE ue.unit_id = ANY($1)
        ORDER BY e.canonical_name
        """,
        list(loaded.unit_index.keys()),
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
    rows = await conn.fetch(
        f"""
        SELECT from_unit_id, to_unit_id, link_type
        FROM {fq_table("memory_links")}
        WHERE link_type = ANY($1)
          AND from_unit_id = ANY($2)
          AND to_unit_id = ANY($2)
        """,
        list(CAUSAL_LINK_TYPES),
        list(loaded.unit_index.keys()),
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
