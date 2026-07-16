"""Import documents from a transfer archive by replaying the deterministic retain pipeline.

For each document the importer rebuilds the extracted facts, re-embeds them with
the *target* bank's embedding model, then runs entity resolution (Phase 1) and
the fact/link insert (Phase 2) — exactly the steps retain runs after LLM
extraction. No LLM is called. Temporal/semantic/causal links and entity merges
are therefore computed relative to the target bank's existing memories.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import tempfile
import uuid
import zipfile
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Literal

from ..causal_links import CANONICAL_CAUSAL_LINK_TYPE, LEGACY_CAUSAL_LINK_TYPES
from ..db_utils import acquire_with_retry
from ..image import validate_image
from ..retain import bank_utils, chunk_storage, embedding_processing, fact_storage, link_utils, orchestrator
from ..retain.types import (
    CausalRelation,
    ChunkMetadata,
    ExtractedFact,
    ProcessedFact,
    RetainContent,
)
from ..schema import fq_table
from .archive import TransferArchive
from .schema import (
    CARRIED_HISTORY_TABLES,
    HISTORY_TABLES,
    SCHEMA_VERSION,
    BankRowsJSONEncoding,
    TransferDocument,
    TransferFact,
    TransferManifest,
    TransferObservation,
)

logger = logging.getLogger(__name__)

OnConflict = Literal["skip", "replace", "new-id"]
_VALID_CONFLICT_MODES: tuple[OnConflict, ...] = ("skip", "replace", "new-id")
MAX_TRANSFER_ARCHIVE_BYTES = 1024 * 1024 * 1024
MAX_TRANSFER_ENTRIES = 100_000
MAX_TRANSFER_ENTRY_BYTES = 64 * 1024 * 1024
MAX_TRANSFER_EXPANDED_BYTES = 4 * 1024 * 1024 * 1024
MAX_TRANSFER_COMPRESSION_RATIO = 200


@dataclass
class ImportedDocument:
    """A single document successfully imported, with the units it produced.

    Carried back so the engine can fire the post-retain extension hook
    (usage tracking / metrics / notifications) once per imported document,
    mirroring how retain reports each completed document.
    """

    document_id: str
    unit_ids: list[str]
    content: str
    tags: list[str]


@dataclass
class ImportResult:
    """Outcome of importing a transfer archive into a bank."""

    documents_imported: int = 0
    documents_skipped: int = 0
    facts_imported: int = 0
    observations_imported: int = 0
    # Observations dropped because some source fact was not imported in this run.
    observations_skipped: int = 0
    skipped_document_ids: list[str] = field(default_factory=list)
    # Original id -> freshly generated id, for documents imported under "new-id".
    remapped_document_ids: dict[str, str] = field(default_factory=dict)
    # Per-document outcomes, for the engine's post-retain hook. Not serialized
    # into operation result_metadata (the worker handler writes counts only).
    imported_documents: list[ImportedDocument] = field(default_factory=list)


@dataclass
class _ObservationOutcome:
    """Counts from the observation import pass."""

    imported: int = 0
    skipped: int = 0


@dataclass
class _ImportedFactBatch:
    """Inserted fact IDs paired with their ordinals in the source archive."""

    unit_ids: list[str]
    original_ordinals: list[int]


@dataclass
class ParsedArchive:
    """A transfer archive after parsing/validation."""

    manifest: TransferManifest
    documents: list[TransferDocument]
    observations: list[TransferObservation] = field(default_factory=list)


async def spool_archive_stream(chunks: AsyncIterator[bytes]) -> TransferArchive:
    """Copy an upload/object stream to a seekable file with an archive-size guard."""
    path = ""
    size_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(prefix="hindsight-import-", suffix=".zip", delete=False) as output:
            path = output.name
            async for chunk in chunks:
                size_bytes += len(chunk)
                if size_bytes > MAX_TRANSFER_ARCHIVE_BYTES:
                    raise ValueError("Transfer archive exceeds the 1 GiB compressed-size limit")
                output.write(chunk)
        return TransferArchive(path=path, size_bytes=size_bytes)
    except Exception:
        if path:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        raise


def _validate_zip_limits(zf: zipfile.ZipFile) -> set[str]:
    entries = zf.infolist()
    if len(entries) > MAX_TRANSFER_ENTRIES:
        raise ValueError(f"Transfer archive contains more than {MAX_TRANSFER_ENTRIES} entries")
    expanded_bytes = 0
    for info in entries:
        if info.is_dir():
            continue
        if info.file_size > MAX_TRANSFER_ENTRY_BYTES and not info.filename.startswith("assets/"):
            raise ValueError(f"Transfer archive entry is too large: {info.filename}")
        expanded_bytes += info.file_size
        if expanded_bytes > MAX_TRANSFER_EXPANDED_BYTES:
            raise ValueError("Transfer archive exceeds the 4 GiB expanded-size limit")
        if info.file_size and info.compress_size == 0:
            raise ValueError(f"Transfer archive entry has an invalid compression size: {info.filename}")
        if info.compress_size and info.file_size / info.compress_size > MAX_TRANSFER_COMPRESSION_RATIO:
            raise ValueError(f"Transfer archive entry exceeds the compression-ratio limit: {info.filename}")
        parts = info.filename.split("/")
        if info.filename.startswith("/") or ".." in parts or "" in parts:
            raise ValueError(f"Invalid transfer archive entry: {info.filename}")
    return {info.filename for info in entries}


def parse_archive(archive_bytes: bytes | TransferArchive) -> ParsedArchive:
    """Parse and validate a transfer ZIP archive produced by ``export_documents``."""
    source = io.BytesIO(archive_bytes) if isinstance(archive_bytes, bytes) else archive_bytes.path
    with zipfile.ZipFile(source, "r") as zf:
        names = _validate_zip_limits(zf)
        if "manifest.json" not in names:
            raise ValueError("Invalid transfer archive: manifest.json is missing")
        manifest = TransferManifest.model_validate_json(zf.read("manifest.json"))
        if manifest.schema_version not in (1, SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported transfer archive schema version {manifest.schema_version} "
                f"(this build supports {SCHEMA_VERSION})"
            )
        doc_names = sorted(n for n in names if n.startswith("documents/") and n.endswith(".json"))
        documents = [TransferDocument.model_validate_json(zf.read(name)) for name in doc_names]
        observations: list[TransferObservation] = []
        if "observations.json" in names:
            observations = [TransferObservation.model_validate(o) for o in json.loads(zf.read("observations.json"))]
    return ParsedArchive(
        manifest=manifest,
        documents=documents,
        observations=observations,
    )


def validate_archive_file(path: str, *, image_max_file_size_bytes: int) -> TransferManifest:
    """Validate ZIP structure and bounded JSON metadata without loading image entries."""
    try:
        archive = zipfile.ZipFile(path, "r")
    except zipfile.BadZipFile as exc:
        raise ValueError("Invalid transfer archive: file is not a ZIP archive") from exc
    with archive as zf:
        names = _validate_zip_limits(zf)
        if "manifest.json" not in names:
            raise ValueError("Invalid transfer archive: manifest.json is missing")
        manifest = TransferManifest.model_validate_json(zf.read("manifest.json"))
        if manifest.schema_version not in (1, SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported transfer archive schema version {manifest.schema_version} "
                f"(this build supports {SCHEMA_VERSION})"
            )
        document_names = sorted(name for name in names if name.startswith("documents/") and name.endswith(".json"))
        if len(document_names) != manifest.document_count:
            raise ValueError("Transfer manifest document_count does not match document entries")
        referenced_assets: set[str] = set()
        for name in document_names:
            document = TransferDocument.model_validate_json(zf.read(name))
            for asset in document.image_assets:
                entry = asset.archive_entry
                if entry is None:
                    continue
                if not entry.startswith("assets/") or entry not in names:
                    raise ValueError(f"Image archive entry is missing or invalid: {entry}")
                info = zf.getinfo(entry)
                if info.file_size != asset.size_bytes:
                    raise ValueError(f"Image archive entry size mismatch: {entry}")
                if info.file_size > image_max_file_size_bytes:
                    raise ValueError(f"Image archive entry exceeds the configured image file limit: {entry}")
                referenced_assets.add(entry)
        if len(referenced_assets) != manifest.image_asset_count:
            raise ValueError("Transfer manifest image_asset_count does not match referenced image entries")
        return manifest


async def import_documents(
    *,
    backend: Any,
    embeddings_model: Any,
    entity_resolver: Any,
    config: Any,
    format_date_fn: Any,
    bank_id: str,
    archive_path: str,
    on_conflict: OnConflict = "skip",
    ops: Any = None,
    outbox_callback_factory: Any = None,
    file_storage: Any | None = None,
    transfer_id: str = "direct-import",
    image_max_file_size_bytes: int = 10 * 1024 * 1024,
) -> ImportResult:
    """Import every document in a seekable, disk-backed archive into ``bank_id``.

    Args:
        backend: Database backend (provides ``acquire()`` and ``ops``).
        embeddings_model: Target bank's embedding model (used to re-embed facts).
        entity_resolver: Shared entity resolver for the target bank.
        config: Resolved bank config for the target bank.
        format_date_fn: Date formatter used when augmenting fact text for embedding
            (must match retain so embeddings are consistent).
        bank_id: Target bank.
        archive_path: A ZIP archive produced by ``export_documents``.
        on_conflict: How to handle a document id that already exists in the target
            bank — ``skip`` (default), ``replace`` (delete old data and re-import),
            or ``new-id`` (import under a freshly generated id).
        ops: Backend ``DataAccessOps``. Defaults to ``backend.ops``.

    Returns:
        An :class:`ImportResult` with per-document counts.
    """
    if on_conflict not in _VALID_CONFLICT_MODES:
        raise ValueError(f"Invalid on_conflict '{on_conflict}'; expected one of {_VALID_CONFLICT_MODES}")
    if ops is None:
        ops = backend.ops

    result = ImportResult()

    # (original document_id, fact ordinal) -> freshly inserted unit id. Used to
    # resolve observation source references after all facts exist.
    ref_map: dict[tuple[str, int], str] = {}

    with zipfile.ZipFile(archive_path, "r") as archive:
        names = _validate_zip_limits(archive)
        TransferManifest.model_validate_json(archive.read("manifest.json"))
        document_names = sorted(name for name in names if name.startswith("documents/") and name.endswith(".json"))
        for document_name in document_names:
            document = TransferDocument.model_validate_json(archive.read(document_name))
            asset_id_map = await _resolve_image_asset_ids(
                backend=backend,
                file_storage=file_storage,
                bank_id=bank_id,
                document=document,
                on_conflict=on_conflict,
            )
            if asset_id_map is None:
                result.documents_skipped += 1
                result.skipped_document_ids.append(document.id)
                continue
            target_id = await _resolve_target_id(backend, bank_id, document.id, on_conflict)
            if target_id is None:
                result.documents_skipped += 1
                result.skipped_document_ids.append(document.id)
                continue
            if target_id != document.id:
                result.remapped_document_ids[document.id] = target_id

            prepared_images = None
            if document.image_assets:
                if file_storage is None:
                    raise ValueError("This archive contains images but no FileStorage is configured")
                prepared_images = await _prepare_document_images(
                    backend=backend,
                    file_storage=file_storage,
                    bank_id=bank_id,
                    document=document,
                    target_document_id=target_id,
                    asset_id_map=asset_id_map,
                    archive=archive,
                    transfer_id=transfer_id,
                    image_max_file_size_bytes=image_max_file_size_bytes,
                )
            try:
                imported_facts = await _import_one_document(
                    backend=backend,
                    embeddings_model=embeddings_model,
                    entity_resolver=entity_resolver,
                    config=config,
                    format_date_fn=format_date_fn,
                    bank_id=bank_id,
                    document=document,
                    target_id=target_id,
                    ops=ops,
                    outbox_callback_factory=outbox_callback_factory,
                    prepared_images=prepared_images,
                )
            except Exception:
                if prepared_images is not None:
                    await prepared_images.compensate(file_storage)
                raise
            result.documents_imported += 1
            result.facts_imported += len(imported_facts.unit_ids)
            result.imported_documents.append(
                ImportedDocument(
                    document_id=target_id,
                    unit_ids=imported_facts.unit_ids,
                    content=document.original_text or "",
                    tags=list(document.tags),
                )
            )
            for ordinal, unit_id in zip(imported_facts.original_ordinals, imported_facts.unit_ids, strict=True):
                ref_map[(document.id, ordinal)] = unit_id

        observations: list[TransferObservation] = []
        if "observations.json" in names:
            observations = [
                TransferObservation.model_validate(item) for item in json.loads(archive.read("observations.json"))
            ]

    if observations:
        outcome = await _import_observations(
            backend=backend,
            embeddings_model=embeddings_model,
            bank_id=bank_id,
            observations=observations,
            ref_map=ref_map,
            ops=ops,
        )
        result.observations_imported = outcome.imported
        result.observations_skipped = outcome.skipped

    logger.info(
        "[transfer] Imported %d document(s), %d fact(s), %d observation(s) into bank %s "
        "(%d docs skipped, %d observations skipped)",
        result.documents_imported,
        result.facts_imported,
        result.observations_imported,
        bank_id,
        result.documents_skipped,
        result.observations_skipped,
    )
    return result


async def _resolve_image_asset_ids(
    backend: Any,
    file_storage: Any | None,
    bank_id: str,
    document: TransferDocument,
    on_conflict: OnConflict,
) -> dict[str, str] | None:
    """Resolve bank-scoped asset IDs before mutating the target document."""
    if not document.image_assets:
        return {}
    asset_ids = [asset.asset_id for asset in document.image_assets]
    async with acquire_with_retry(backend) as conn:
        rows = await conn.fetch(
            f"SELECT asset_id, sha256, status, storage_key, size_bytes FROM {fq_table('image_assets')} "
            "WHERE bank_id = $1 AND asset_id = ANY($2)",
            bank_id,
            asset_ids,
        )
    existing = {str(row["asset_id"]): row for row in rows}
    mapping: dict[str, str] = {}
    for asset in document.image_assets:
        current = existing.get(asset.asset_id)
        if current is None:
            mapping[asset.asset_id] = asset.asset_id
            continue
        reusable = (
            str(current["sha256"]) == asset.sha256 and str(current["status"]) == "ready" and file_storage is not None
        )
        if reusable:
            try:
                reusable = (await file_storage.stat(str(current["storage_key"]))).size_bytes == int(
                    current["size_bytes"]
                )
            except FileNotFoundError:
                reusable = False
        if reusable:
            mapping[asset.asset_id] = asset.asset_id
            continue
        if on_conflict == "skip":
            return None
        if on_conflict == "new-id":
            mapping[asset.asset_id] = str(uuid.uuid4())
            continue
        raise ValueError(
            f"Image asset {asset.asset_id!r} already exists with different content; "
            "replace never overwrites managed image bytes"
        )
    return mapping


@dataclass
class _PreparedDocumentImages:
    bank_id: str
    document: TransferDocument
    target_document_id: str
    asset_id_map: dict[str, str]
    existing_ids: set[str]
    storage_keys: dict[str, str]
    stored_keys: list[str]

    async def compensate(self, file_storage: Any) -> None:
        for storage_key in self.stored_keys:
            try:
                await file_storage.delete(storage_key)
            except Exception:
                logger.warning("Failed to compensate imported image blob %s", storage_key, exc_info=True)


async def _prepare_document_images(
    *,
    backend: Any,
    file_storage: Any,
    bank_id: str,
    document: TransferDocument,
    target_document_id: str,
    asset_id_map: dict[str, str],
    archive: zipfile.ZipFile,
    transfer_id: str,
    image_max_file_size_bytes: int,
) -> _PreparedDocumentImages:
    """Validate and store image blobs before the document's atomic DB transaction."""
    target_asset_ids = list(asset_id_map.values())
    async with acquire_with_retry(backend) as conn:
        existing_ids = {
            str(row["asset_id"])
            for row in await conn.fetch(
                f"SELECT asset_id FROM {fq_table('image_assets')} "
                "WHERE bank_id = $1 AND asset_id = ANY($2) AND status = 'ready'",
                bank_id,
                target_asset_ids,
            )
        }

    storage_keys: dict[str, str] = {}
    stored_keys: list[str] = []
    for asset in document.image_assets:
        target_asset_id = asset_id_map[asset.asset_id]
        if target_asset_id in existing_ids:
            continue
        storage_key = (
            "image-assets/import/" + hashlib.sha256(f"{bank_id}\0{transfer_id}\0{target_asset_id}".encode()).hexdigest()
        )
        storage_keys[target_asset_id] = storage_key
        if asset.archive_entry is not None:
            info = archive.getinfo(asset.archive_entry)
            if info.file_size > image_max_file_size_bytes:
                raise ValueError(f"Image archive entry exceeds the configured image file limit: {asset.archive_entry}")
            with archive.open(asset.archive_entry, "r") as source:
                data = source.read(image_max_file_size_bytes + 1)
            if len(data) != info.file_size:
                raise ValueError(f"Image archive entry size mismatch: {asset.archive_entry}")
            if hashlib.sha256(data).hexdigest() != asset.sha256:
                raise ValueError(f"Image archive entry hash mismatch: {asset.archive_entry}")
            validated = validate_image(
                data,
                asset.mime_type,
                max_size_bytes=image_max_file_size_bytes,
            )
            if (validated.width, validated.height) != (asset.width, asset.height):
                raise ValueError(f"Image archive entry metadata mismatch: {asset.archive_entry}")
            await file_storage.store(data, storage_key)
            stored_keys.append(storage_key)
    return _PreparedDocumentImages(
        bank_id=bank_id,
        document=document,
        target_document_id=target_document_id,
        asset_id_map=asset_id_map,
        existing_ids=existing_ids,
        storage_keys=storage_keys,
        stored_keys=stored_keys,
    )


async def _publish_document_images(conn: Any, prepared: _PreparedDocumentImages) -> None:
    """Publish assets and links in the same transaction as Document/Facts."""
    document = prepared.document
    if prepared.existing_ids:
        # Serialize reuse with synchronous asset deletion. Resolution happens
        # before image bytes are prepared, so an asset can enter deleting before
        # this final publication transaction acquires its row lock.
        reusable_rows = await conn.fetch(
            f"SELECT asset_id, status FROM {fq_table('image_assets')} "
            "WHERE bank_id = $1 AND asset_id = ANY($2) FOR UPDATE",
            prepared.bank_id,
            list(prepared.existing_ids),
        )
        reusable_statuses = {str(row["asset_id"]): str(row["status"]) for row in reusable_rows}
        if any(reusable_statuses.get(asset_id) != "ready" for asset_id in prepared.existing_ids):
            raise ValueError("one or more reusable image assets became unavailable during import")
    for asset in document.image_assets:
        target_asset_id = prepared.asset_id_map[asset.asset_id]
        if target_asset_id in prepared.existing_ids:
            continue
        await conn.execute(
            f"""
            INSERT INTO {fq_table("image_assets")}
                (bank_id, asset_id, storage_key, mime_type, size_bytes, sha256,
                 width, height, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            prepared.bank_id,
            target_asset_id,
            prepared.storage_keys[target_asset_id],
            asset.mime_type,
            asset.size_bytes,
            asset.sha256,
            asset.width,
            asset.height,
            "ready" if asset.archive_entry is not None else "failed",
        )
    for link in document.image_links:
        await conn.execute(
            f"""
            INSERT INTO {fq_table("document_image_links")}
                (bank_id, document_id, asset_id, ordinal, image_context)
            VALUES ($1, $2, $3, $4, $5)
            """,
            prepared.bank_id,
            prepared.target_document_id,
            prepared.asset_id_map[link.asset_id],
            link.ordinal,
            link.image_context,
        )


# Bank-level config/state tables restored verbatim from a whole-bank archive.
# Order matters for foreign keys: banks (parent) is restored before any child.
_BANK_CHILD_TABLES = ("mental_models", "directives", "webhooks")
# Child-history carried verbatim; restored after its parent (mental_models) so the
# foreign key resolves. Surrogate ids were dropped on export (the target reassigns
# them), so these restore via fresh IDENTITY values.


@dataclass
class BankImportResult:
    """Outcome of importing a whole-bank archive."""

    bank_id: str
    documents_imported: int = 0
    facts_imported: int = 0
    observations_imported: int = 0
    mental_models_imported: int = 0
    mental_model_history_imported: int = 0
    directives_imported: int = 0
    webhooks_imported: int = 0
    history_rows_imported: int = 0


@dataclass
class ParsedBankArchive:
    """The bank-level sections of a whole-bank archive (documents read separately)."""

    manifest: TransferManifest
    # table name -> list of verbatim row dicts (banks, mental_models, directives, webhooks)
    bank_rows: dict[str, list[dict]] = field(default_factory=dict)
    # table name -> rows (audit_log, llm_requests), present only with --include-history
    history_rows: dict[str, list[dict]] = field(default_factory=dict)


def parse_bank_archive(archive_bytes: bytes | _ArchivePath) -> ParsedBankArchive:
    """Parse the bank-level sections of a whole-bank archive (``archive_type='bank'``)."""
    source = io.BytesIO(archive_bytes) if isinstance(archive_bytes, bytes) else archive_bytes.path
    with zipfile.ZipFile(source, "r") as zf:
        names = set(zf.namelist())
        if "manifest.json" not in names:
            raise ValueError("Invalid transfer archive: manifest.json is missing")
        manifest = TransferManifest.model_validate_json(zf.read("manifest.json"))
        if manifest.archive_type != "bank":
            raise ValueError(
                f"Not a whole-bank archive (archive_type={manifest.archive_type!r}); use import_documents instead"
            )
        bank_rows: dict[str, list[dict]] = {}
        for table in ("banks", *_BANK_CHILD_TABLES, *CARRIED_HISTORY_TABLES):
            fname = f"{table}.json"
            bank_rows[table] = json.loads(zf.read(fname)) if fname in names else []
        history_rows: dict[str, list[dict]] = {}
        for table in HISTORY_TABLES:
            fname = f"history/{table}.json"
            if fname in names:
                history_rows[table] = json.loads(zf.read(fname))
    return ParsedBankArchive(manifest=manifest, bank_rows=bank_rows, history_rows=history_rows)


def _resolve_bank_rows_json_encoding(manifest: TransferManifest) -> BankRowsJSONEncoding:
    """Resolve row JSON provenance, including the released v1 archive contract."""
    return manifest.bank_rows_json_encoding or "decoded"


async def _restore_rows(
    conn: Any,
    table: str,
    rows: list[dict],
    *,
    bank_rows_json_encoding: BankRowsJSONEncoding = "decoded",
) -> int:
    """Insert verbatim rows into a bank-scoped table, coercing JSON-encoded values
    back to the column's type (timestamps, uuids, jsonb). ``ON CONFLICT DO NOTHING``
    keeps an import idempotent and safe to re-run against a partially-filled target."""
    if not rows:
        return 0
    from ..memory_engine import get_current_schema

    schema = get_current_schema()
    col_types = {
        r["column_name"]: r["data_type"]
        for r in await conn.fetch(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = $1 AND table_name = $2",
            schema,
            table,
        )
    }
    inserted = 0
    for row in rows:
        cols = [c for c in row if c in col_types]
        placeholders: list[str] = []
        values: list[Any] = []
        for position, col in enumerate(cols, start=1):
            data_type = col_types[col]
            value = row[col]
            if data_type in ("jsonb", "json"):
                # asyncpg has no JSON codec on these raw connections; pass JSON
                # text and cast. Provenance is required because a decoded JSON
                # scalar containing JSON text is indistinguishable from a raw
                # serialized object after the outer archive JSON is parsed.
                if value is not None and (bank_rows_json_encoding == "decoded" or not isinstance(value, str)):
                    value = json.dumps(value)
                values.append(value)
                placeholders.append(f"${position}::jsonb")
                continue
            if value is not None and isinstance(value, str):
                if data_type in ("timestamp with time zone", "timestamp without time zone"):
                    value = datetime.fromisoformat(value)
                elif data_type == "date":
                    value = date.fromisoformat(value)
                elif data_type == "uuid":
                    value = uuid.UUID(value)
            placeholders.append(f"${position}")
            values.append(value)
        col_list = ", ".join(f'"{c}"' for c in cols)
        await conn.execute(
            f"INSERT INTO {fq_table(table)} ({col_list}) VALUES ({', '.join(placeholders)}) ON CONFLICT DO NOTHING",
            *values,
        )
        inserted += 1
    return inserted


async def import_bank(
    *,
    backend: Any,
    embeddings_model: Any,
    entity_resolver: Any,
    config: Any,
    format_date_fn: Any,
    archive_bytes: bytes | _ArchivePath,
    target_bank_id: str | None = None,
    include_history: bool = False,
    ops: Any = None,
    file_storage: Any | None = None,
    image_max_file_size_bytes: int = 10 * 1024 * 1024,
) -> BankImportResult:
    """Restore a whole bank from a ``export_bank`` archive into the target instance.

    Re-embeds facts with the *target* instance's embedding model and rebuilds links,
    entities and search/vector indexes — the path for migrating a bank to an instance
    configured with a different embedding model / vector / text-search backend.

    The **target bank must not already exist**: import restores a complete bank
    (config + facts + mental models + …) and is not a merge. If a bank with the
    target id is present, this raises — delete it first or pass ``target_bank_id``
    for a fresh id. A migration restores *exact* state, so unlike the document
    import it fires no retain webhooks and triggers no consolidation/graph
    maintenance: observations and mental models are restored as exported.
    """
    if ops is None:
        ops = backend.ops
    parsed = parse_bank_archive(archive_bytes)
    bank_rows_json_encoding = _resolve_bank_rows_json_encoding(parsed.manifest)
    source_bank_id = parsed.manifest.source_bank_id
    bank_id = target_bank_id or source_bank_id

    # Remapping to a different id: rewrite the carried bank_id on every row so FKs
    # and PKs line up with the (also-remapped) documents/facts.
    if bank_id != source_bank_id:
        for rows in (*parsed.bank_rows.values(), *parsed.history_rows.values()):
            for row in rows:
                if "bank_id" in row:
                    row["bank_id"] = bank_id

    async with acquire_with_retry(backend) as conn:
        # Refuse to import into an existing bank — this restores a whole bank, it
        # does not merge. Merging would silently mix the archive's config/mental
        # models/webhooks with whatever is already there (and global-unique ids
        # like webhooks/directives would collide).
        if await conn.fetchval(f"SELECT 1 FROM {fq_table('banks')} WHERE bank_id = $1", bank_id):
            raise ValueError(
                f"Target bank '{bank_id}' already exists; import-bank restores into a fresh bank "
                f"(it is not a merge). Delete the bank first, or pass a different target bank id."
            )
        # Bank row first — children (documents, mental_models, …) FK to it.
        await _restore_rows(
            conn,
            "banks",
            parsed.bank_rows.get("banks", []),
            bank_rows_json_encoding=bank_rows_json_encoding,
        )
        # The restored banks row bypasses the fresh-INSERT gate that normally
        # creates per-bank vector indexes, so create them explicitly here while
        # the bank is still empty (facts are imported below, so the build is
        # instant). get_or_create_bank_profile would NOT do this: the row now
        # exists, so it takes the SELECT branch and skips index creation —
        # leaving the restored bank falling back to the global index +
        # post-filter (slower, under-returning recall). See #2645.
        internal_id = await conn.fetchval(f"SELECT internal_id FROM {fq_table('banks')} WHERE bank_id = $1", bank_id)
        if internal_id is not None:
            await bank_utils.create_bank_vector_indexes(conn, bank_id, str(internal_id), ops=ops)

    archive_path = ""
    owns_archive_path = False
    try:
        if isinstance(archive_bytes, bytes):
            with tempfile.NamedTemporaryFile(prefix="hindsight-bank-import-", suffix=".zip", delete=False) as output:
                archive_path = output.name
                output.write(archive_bytes)
            owns_archive_path = True
        else:
            archive_path = archive_bytes.path
        doc_result = await import_documents(
            backend=backend,
            embeddings_model=embeddings_model,
            entity_resolver=entity_resolver,
            config=config,
            format_date_fn=format_date_fn,
            bank_id=bank_id,
            archive_path=archive_path,
            ops=ops,
            outbox_callback_factory=None,
            file_storage=file_storage,
            image_max_file_size_bytes=image_max_file_size_bytes,
        )
    finally:
        if owns_archive_path:
            try:
                os.unlink(archive_path)
            except FileNotFoundError:
                pass

    result = BankImportResult(
        bank_id=bank_id,
        documents_imported=doc_result.documents_imported,
        facts_imported=doc_result.facts_imported,
        observations_imported=doc_result.observations_imported,
    )
    async with acquire_with_retry(backend) as conn:
        result.mental_models_imported = await _restore_rows(
            conn,
            "mental_models",
            parsed.bank_rows.get("mental_models", []),
            bank_rows_json_encoding=bank_rows_json_encoding,
        )
        # Restored after mental_models so the (mental_model_id, bank_id) FK resolves.
        result.mental_model_history_imported = await _restore_rows(
            conn,
            "mental_model_history",
            parsed.bank_rows.get("mental_model_history", []),
            bank_rows_json_encoding=bank_rows_json_encoding,
        )
        result.directives_imported = await _restore_rows(
            conn,
            "directives",
            parsed.bank_rows.get("directives", []),
            bank_rows_json_encoding=bank_rows_json_encoding,
        )
        result.webhooks_imported = await _restore_rows(
            conn,
            "webhooks",
            parsed.bank_rows.get("webhooks", []),
            bank_rows_json_encoding=bank_rows_json_encoding,
        )
        if include_history:
            for table in HISTORY_TABLES:
                result.history_rows_imported += await _restore_rows(
                    conn,
                    table,
                    parsed.history_rows.get(table, []),
                    bank_rows_json_encoding=bank_rows_json_encoding,
                )

    logger.info(
        "[transfer] Imported bank %s: %d doc(s), %d fact(s), %d observation(s), "
        "%d mental model(s), %d mm-history row(s), %d directive(s), %d webhook(s), %d history row(s)",
        bank_id,
        result.documents_imported,
        result.facts_imported,
        result.observations_imported,
        result.mental_models_imported,
        result.mental_model_history_imported,
        result.directives_imported,
        result.webhooks_imported,
        result.history_rows_imported,
    )
    return result


async def _resolve_target_id(backend: Any, bank_id: str, document_id: str, on_conflict: OnConflict) -> str | None:
    """Decide the document id to write under, or ``None`` to skip.

    Returns the original id when there is no conflict, a fresh id under
    ``new-id``, the original id under ``replace`` (the insert path cascades the
    old data away), or ``None`` under ``skip`` when the document already exists.
    """
    async with acquire_with_retry(backend) as conn:
        exists = await conn.fetchval(
            f"SELECT 1 FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
            document_id,
            bank_id,
        )
    if not exists:
        return document_id
    if on_conflict == "skip":
        return None
    if on_conflict == "new-id":
        return str(uuid.uuid4())
    return document_id  # replace


async def _import_one_document(
    *,
    backend: Any,
    embeddings_model: Any,
    entity_resolver: Any,
    config: Any,
    format_date_fn: Any,
    bank_id: str,
    document: TransferDocument,
    target_id: str,
    ops: Any,
    outbox_callback_factory: Any = None,
    prepared_images: _PreparedDocumentImages | None = None,
) -> _ImportedFactBatch:
    """Re-embed and insert a document; map original fact ordinals to new unit ids."""
    log_buffer: list[str] = []

    # Fire the same retain.completed webhook retain emits, transactionally inside
    # this document's insert. Factory returns None when no webhook manager exists.
    outbox_callback = (
        outbox_callback_factory([{"document_id": target_id, "tags": list(document.tags)}])
        if outbox_callback_factory
        else None
    )

    extracted_facts = [_to_extracted_fact(fact) for fact in document.facts]
    legacy_causal_relations = _legacy_causal_relations(document)

    processed_facts: list[ProcessedFact] = []
    retained_index_by_original: list[int | None] = []
    if extracted_facts:
        augmented = embedding_processing.augment_texts_with_dates(extracted_facts, format_date_fn)
        embeddings = await embedding_processing.generate_embeddings_batch(embeddings_model, augmented)
        fact_batch = orchestrator._process_extracted_facts(extracted_facts, embeddings)
        extracted_facts = fact_batch.extracted_facts
        processed_facts = fact_batch.processed_facts
        retained_index_by_original = fact_batch.retained_index_by_original
        legacy_causal_relations = orchestrator._remap_causal_relations(
            legacy_causal_relations,
            retained_index_by_original,
        )

    contents = [RetainContent(content=document.original_text or "")]
    chunk_meta = [
        ChunkMetadata(chunk_text=chunk.chunk_text, fact_count=0, content_index=0, chunk_index=chunk.chunk_index)
        for chunk in document.chunks
    ]

    # Phase 1 (entity resolution + semantic ANN) on its own connection, outside
    # the write transaction — mirrors the retain pipeline.
    entity_resolver.discard_pending_stats()
    phase1 = await orchestrator._pre_resolve_phase1(
        backend,
        entity_resolver,
        bank_id,
        contents,
        processed_facts,
        config,
        log_buffer,
        skip_semantic_ann=False,
    )

    async with acquire_with_retry(backend) as conn:
        async with conn.transaction():
            # is_first_batch=True: cascade-delete any existing data for this id
            # (the "replace" path) and (re)insert the document row.
            await fact_storage.handle_document_tracking(
                conn,
                bank_id,
                target_id,
                document.original_text or "",
                True,
                document.retain_params,
                document.tags,
                ops=ops,
            )
            if document.created_at is not None:
                # Transfer archives carry source provenance. Apply it here,
                # without changing normal retain/upsert timestamp semantics.
                await conn.execute(
                    f"UPDATE {fq_table('documents')} SET created_at = $1 WHERE id = $2 AND bank_id = $3",
                    document.created_at,
                    target_id,
                    bank_id,
                )

            chunk_id_map: dict[int, str] = {}
            if chunk_meta:
                chunk_id_map = await chunk_storage.store_chunks_batch(conn, bank_id, target_id, chunk_meta, ops=ops)

            for extracted, processed in zip(extracted_facts, processed_facts):
                processed.document_id = target_id
                if chunk_id_map and extracted.chunk_index is not None:
                    chunk_id = chunk_id_map.get(extracted.chunk_index)
                    if chunk_id:
                        processed.chunk_id = chunk_id

            result_unit_ids = await orchestrator._insert_facts_and_links(
                conn,
                entity_resolver,
                bank_id,
                contents,
                extracted_facts,
                processed_facts,
                config,
                log_buffer,
                resolved_entities=phase1.entities.resolved_entities,
                entity_to_unit=phase1.entities.entity_to_unit,
                unit_to_entity_ids=phase1.entities.unit_to_entity_ids,
                semantic_ann_links=phase1.semantic_ann_links,
                skip_semantic_links=False,
                outbox_callback=outbox_callback,
                ops=ops,
            )

            # Retain writes only ``caused_by``. Restore legacy archive edges
            # separately so their distinct direction and semantics survive a
            # transfer without broadening the normal retain write contract.
            if result_unit_ids and legacy_causal_relations:
                await link_utils.restore_legacy_causal_links_batch(
                    conn,
                    bank_id,
                    result_unit_ids[0],
                    legacy_causal_relations,
                    ops=ops,
                )

            if prepared_images is not None:
                await _publish_document_images(conn, prepared_images)

        try:
            await entity_resolver.flush_pending_stats()
        except Exception:
            logger.warning("[transfer] Entity stats flush failed for document %s", target_id, exc_info=True)

    logger.debug("[transfer] Imported document %s:\n%s", target_id, "\n".join(log_buffer))
    # Single content item -> result_unit_ids[0] follows the retained fact order.
    retained_unit_ids = list(result_unit_ids[0]) if result_unit_ids else []
    return _ImportedFactBatch(
        unit_ids=retained_unit_ids,
        original_ordinals=[
            original_index
            for original_index, retained_index in enumerate(retained_index_by_original)
            if retained_index is not None
        ],
    )


async def _import_observations(
    *,
    backend: Any,
    embeddings_model: Any,
    bank_id: str,
    observations: list[TransferObservation],
    ref_map: dict[tuple[str, int], str],
    ops: Any,
) -> _ObservationOutcome:
    """Insert observations whose source facts were all imported in this run.

    Observations carry no embedding, links, or entity rows — only the unit row
    plus ``source_memory_ids`` (remapped to the freshly inserted source units)
    and ``proof_count``. Their source facts are marked ``consolidated_at`` so the
    target bank's consolidator won't re-process them. Mirrors what consolidation
    writes, but driven from the archive instead of the LLM.

    Inserted as-is: imported observations are NOT merged or deduplicated against
    observations that already exist in the target bank (unlike consolidation,
    which merges related observations). Importing into a bank that already has
    observations — or importing the same archive twice — can therefore produce
    overlapping observations over the same facts.
    """
    outcome = _ObservationOutcome()

    # Resolve each observation's sources to new unit ids; drop any whose sources
    # weren't all imported (e.g. a subset/skip import).
    resolved: list[tuple[TransferObservation, list[str]]] = []
    for obs in observations:
        source_ids = [ref_map.get((s.document_id, s.fact_index)) for s in obs.sources]
        if not source_ids or any(sid is None for sid in source_ids):
            outcome.skipped += 1
            continue
        resolved.append((obs, [sid for sid in source_ids if sid is not None]))

    if not resolved:
        return outcome

    # Observations embed the raw text (matching consolidation), not the
    # date-augmented text used for facts.
    embeddings = await embedding_processing.generate_embeddings_batch(
        embeddings_model, [obs.text for obs, _ in resolved]
    )
    processed = [
        ProcessedFact(
            fact_text=obs.text,
            fact_type="observation",
            embedding=embedding,
            occurred_start=obs.occurred_start,
            occurred_end=obs.occurred_end,
            mentioned_at=_observation_mentioned_at(obs),
            context="",
            metadata={},
            tags=list(obs.tags),
            observation_scopes=obs.observation_scopes,
            document_id=None,
            chunk_id=None,
        )
        for (obs, _sources), embedding in zip(resolved, embeddings)
    ]

    async with acquire_with_retry(backend) as conn:
        async with conn.transaction():
            obs_unit_ids = await fact_storage.insert_facts_batch(conn, bank_id, processed, ops=ops)

            all_source_ids: set[uuid.UUID] = set()
            for (obs, sources), obs_unit_id in zip(resolved, obs_unit_ids):
                observation_uuid = uuid.UUID(obs_unit_id)
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
                await _link_observation_sources(conn, ops, bank_id, observation_uuid, source_uuids, obs.proof_count)

            # Mark source facts consolidated so the target consolidator skips them.
            if all_source_ids:
                await conn.execute(
                    f"UPDATE {fq_table('memory_units')} SET consolidated_at = now() "
                    f"WHERE bank_id = $1 AND id = ANY($2)",
                    bank_id,
                    list(all_source_ids),
                )

    outcome.imported = len(resolved)
    return outcome


async def _link_observation_sources(
    conn: Any,
    ops: Any,
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


def _observation_mentioned_at(obs: TransferObservation) -> datetime | None:
    """event_date (NOT NULL) is derived from occurred_start or mentioned_at on
    insert; fall back so the column stays populated for observations too."""
    mentioned_at = obs.mentioned_at
    if obs.occurred_start is None and mentioned_at is None:
        mentioned_at = obs.event_date or datetime.now(UTC)
    return mentioned_at


def _to_extracted_fact(fact: TransferFact) -> ExtractedFact:
    """Rebuild the retain pipeline's ExtractedFact from a serialized transfer fact."""
    # event_date is NOT NULL in the schema and is derived from occurred_start or
    # mentioned_at on insert. When neither is present, fall back to the carried
    # event_date (or now) via mentioned_at so the column stays populated.
    mentioned_at = fact.mentioned_at
    if fact.occurred_start is None and mentioned_at is None:
        mentioned_at = fact.event_date or datetime.now(UTC)

    return ExtractedFact(
        fact_text=fact.text,
        fact_type=fact.fact_type,
        entities=list(fact.entities),
        occurred_start=fact.occurred_start,
        occurred_end=fact.occurred_end,
        where=None,
        causal_relations=[
            CausalRelation(relation_type=rel.relation_type, target_fact_index=rel.target_fact_index)
            for rel in fact.causal_relations
            if rel.relation_type == CANONICAL_CAUSAL_LINK_TYPE
        ],
        content_index=0,
        chunk_index=fact.chunk_index,
        context=fact.context or "",
        mentioned_at=mentioned_at,
        metadata=dict(fact.metadata),
        tags=list(fact.tags),
        observation_scopes=fact.observation_scopes,
    )


def _legacy_causal_relations(document: TransferDocument) -> list[list[CausalRelation]]:
    """Return legacy archive edges for transfer-only restoration.

    Invalid archive values are excluded. The write helper repeats the explicit
    compatibility allowlist as a persistence boundary.
    """
    return [
        [
            CausalRelation(relation_type=relation.relation_type, target_fact_index=relation.target_fact_index)
            for relation in fact.causal_relations
            if relation.relation_type in LEGACY_CAUSAL_LINK_TYPES
        ]
        for fact in document.facts
    ]
