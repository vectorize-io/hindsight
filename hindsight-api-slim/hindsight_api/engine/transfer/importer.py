"""Import documents from a transfer archive by replaying the deterministic retain pipeline.

For each document the importer rebuilds the extracted facts, re-embeds them with
the *target* bank's embedding model, then runs entity resolution (Phase 1) and
the fact/link insert (Phase 2) — exactly the steps retain runs after LLM
extraction. No LLM is called. Temporal/semantic/causal links and entity merges
are therefore computed relative to the target bank's existing memories.
"""

from __future__ import annotations

import io
import json
import logging
import uuid
import zipfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Literal

from ..causal_links import CANONICAL_CAUSAL_LINK_TYPE, LEGACY_CAUSAL_LINK_TYPES
from ..db.ops_postgresql import pg_search_vector_expr
from ..db_utils import acquire_with_retry
from ..retain import bank_utils, chunk_storage, embedding_processing, fact_storage, link_utils, orchestrator
from ..retain.types import (
    CausalRelation,
    ChunkMetadata,
    ExtractedFact,
    ProcessedFact,
    RetainContent,
    pack_embedding,
)
from ..schema import fq_table
from .schema import (
    CARRIED_HISTORY_TABLES,
    HISTORY_TABLES,
    SCHEMA_VERSION,
    BankRowsJSONEncoding,
    TransferDocument,
    TransferFact,
    TransferKnowledgePage,
    TransferManifest,
    TransferObservation,
)

logger = logging.getLogger(__name__)

OnConflict = Literal["skip", "replace", "new-id"]
_VALID_CONFLICT_MODES: tuple[OnConflict, ...] = ("skip", "replace", "new-id")

#: Texts per embedding call for the import phases that are sized by the *bank* rather
#: than by a document: observations and mental models arrive as one list covering the
#: whole archive. Slicing here, above the provider, bounds peak memory for every
#: provider — including in-process ones with nothing downstream to slice for them
#: (issue #3891). Retain bounds itself the same way (#3763).
_EMBED_BATCH_SIZE = 128


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
    mental_models_imported: int = 0
    knowledge_pages_imported: int = 0
    #: Pages pushed into the STORE's index. Zero when Postgres owns it (the column writes feed it
    #: instead), so a nonzero value is proof the store-owned path ran.
    knowledge_pages_indexed: int = 0
    skipped_document_ids: list[str] = field(default_factory=list)
    # Original id -> freshly generated id, for documents imported under "new-id".
    remapped_document_ids: dict[str, str] = field(default_factory=dict)
    # Per-document outcomes, for the engine's post-retain hook. Not serialized
    # into operation result_metadata (the worker handler writes counts only).
    imported_documents: list[ImportedDocument] = field(default_factory=list)
    # Source memory-unit id -> regenerated target id. Used by whole-bank import
    # to repair mental-model evidence after the replayed facts are inserted.
    remapped_unit_ids: dict[str, str] = field(default_factory=dict)


@dataclass
class _ObservationOutcome:
    """Counts from the observation import pass."""

    imported: int = 0
    skipped: int = 0
    remapped_unit_ids: dict[str, str] = field(default_factory=dict)


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
    mental_models: list[dict] = field(default_factory=list)
    knowledge_pages: list[TransferKnowledgePage] = field(default_factory=list)


def _open_archive(archive_bytes: bytes, *, produced_by: str) -> zipfile.ZipFile:
    """Open a transfer archive, rejecting non-archives as caller errors.

    Both failure modes here are a wrong file, not a server fault, so they must
    surface as ``ValueError`` (the API maps that to a 400 with the message) —
    a bare ``zipfile.BadZipFile`` or a missing manifest would otherwise escape
    as an opaque 500 or an unexplained "manifest.json is missing".

    Args:
        archive_bytes: The uploaded bytes.
        produced_by: How the caller obtains a valid archive, named in the error.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(archive_bytes), "r")
    except zipfile.BadZipFile as e:
        raise ValueError(f"Invalid transfer archive: the uploaded file is not a readable .zip ({e})") from e
    if "manifest.json" not in set(zf.namelist()):
        zf.close()
        raise ValueError(
            f"Invalid transfer archive: manifest.json is missing. This endpoint only accepts a .zip produced "
            f"by {produced_by} — it is not a way to upload a zip of ordinary files (PDF, text, Markdown). "
            f"Use the file upload / retain endpoint for those."
        )
    return zf


def parse_archive(archive_bytes: bytes) -> ParsedArchive:
    """Parse and validate a transfer ZIP archive produced by ``export_documents``."""
    with _open_archive(archive_bytes, produced_by="the document export endpoint") as zf:
        names = set(zf.namelist())
        manifest = TransferManifest.model_validate_json(zf.read("manifest.json"))
        if manifest.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported transfer archive schema version {manifest.schema_version} "
                f"(this build supports {SCHEMA_VERSION})"
            )
        doc_names = sorted(n for n in names if n.startswith("documents/") and n.endswith(".json"))
        documents = [TransferDocument.model_validate_json(zf.read(name)) for name in doc_names]
        observations: list[TransferObservation] = []
        if "observations.json" in names:
            observations = [TransferObservation.model_validate(o) for o in json.loads(zf.read("observations.json"))]
        mental_models = json.loads(zf.read("mental_models.json")) if "mental_models.json" in names else []
        knowledge_pages = (
            [TransferKnowledgePage.model_validate(p) for p in json.loads(zf.read("knowledge_pages.json"))]
            if "knowledge_pages.json" in names
            else []
        )
    return ParsedArchive(
        manifest=manifest,
        documents=documents,
        observations=observations,
        mental_models=mental_models,
        knowledge_pages=knowledge_pages,
    )


async def import_documents(
    *,
    backend: Any,
    embeddings_model: Any,
    entity_resolver: Any,
    config: Any,
    format_date_fn: Any,
    bank_id: str,
    archive_bytes: bytes,
    on_conflict: OnConflict = "skip",
    ops: Any = None,
    outbox_callback_factory: Any = None,
    restore_bank_scoped_rows: bool = True,
) -> ImportResult:
    """Import every document in ``archive_bytes`` into ``bank_id``.

    Args:
        backend: Database backend (provides ``acquire()`` and ``ops``).
        embeddings_model: Target bank's embedding model (used to re-embed facts).
        entity_resolver: Shared entity resolver for the target bank.
        config: Resolved bank config for the target bank.
        format_date_fn: Date formatter used when augmenting fact text for embedding
            (must match retain so embeddings are consistent).
        bank_id: Target bank.
        archive_bytes: A ZIP archive produced by ``export_documents``.
        on_conflict: How to handle a document id that already exists in the target
            bank — ``skip`` (default), ``replace`` (delete old data and re-import),
            or ``new-id`` (import under a freshly generated id).
        ops: Backend ``DataAccessOps``. Defaults to ``backend.ops``.
        restore_bank_scoped_rows: Whether to restore the archive's mental models
            and knowledge pages. True for a documents archive, which merges them
            into an existing bank. False for a whole-bank restore, which owns
            those rows: ``import_bank`` carries the same ``mental_models.json``
            in ``bank_rows``, restores it with the archive's own JSON encoding
            alongside ``mental_model_history``, and — the reason this switch
            exists — rewrites the persisted ``based_on`` evidence ids onto the
            replayed facts first. ``_restore_rows`` is ON CONFLICT DO NOTHING,
            so a copy inserted here would win and strip that repair (#3833).

    Returns:
        An :class:`ImportResult` with per-document counts.
    """
    if on_conflict not in _VALID_CONFLICT_MODES:
        raise ValueError(f"Invalid on_conflict '{on_conflict}'; expected one of {_VALID_CONFLICT_MODES}")
    if ops is None:
        ops = backend.ops

    parsed = parse_archive(archive_bytes)
    if parsed.knowledge_pages and not parsed.mental_models:
        raise ValueError("Knowledge Pages require their backing mental models in the transfer archive")
    # Document transfer merges into an existing target bank. Reuse the logical
    # mental-model ids, but always pin the carried rows to the destination bank;
    # otherwise ON CONFLICT can silently match the source-bank row on same-schema
    # transfers and leave the imported page without a backing model.
    for row in parsed.mental_models:
        row["bank_id"] = bank_id
    result = ImportResult()

    # (original document_id, fact ordinal) -> freshly inserted unit id. Used to
    # resolve observation source references after all facts exist.
    ref_map: dict[tuple[str, int], str] = {}

    for document in parsed.documents:
        target_id = await _resolve_target_id(backend, bank_id, document.id, on_conflict)
        if target_id is None:
            result.documents_skipped += 1
            result.skipped_document_ids.append(document.id)
            continue
        if target_id != document.id:
            result.remapped_document_ids[document.id] = target_id

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
        )
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
        for ordinal, unit_id in zip(imported_facts.original_ordinals, imported_facts.unit_ids, strict=True):
            source_id = document.facts[ordinal].source_id
            if source_id is not None:
                result.remapped_unit_ids[source_id] = unit_id

    if parsed.observations:
        outcome = await _import_observations(
            backend=backend,
            embeddings_model=embeddings_model,
            bank_id=bank_id,
            observations=parsed.observations,
            ref_map=ref_map,
            ops=ops,
        )
        result.observations_imported = outcome.imported
        result.observations_skipped = outcome.skipped
        result.remapped_unit_ids.update(outcome.remapped_unit_ids)

    if restore_bank_scoped_rows and (parsed.mental_models or parsed.knowledge_pages):
        mm_embeddings = await _regenerate_mental_model_embeddings(embeddings_model, parsed.mental_models)
        async with acquire_with_retry(backend) as conn:
            # Document archives serialize bank-row JSON values directly. Keep
            # JSON strings as raw JSON here; treating them as decoded values
            # would encode an object such as trigger_data as a JSON string,
            # which later makes mental-model listing/tree endpoints fail.
            result.mental_models_imported = await _restore_rows(
                conn,
                "mental_models",
                parsed.mental_models,
                bank_rows_json_encoding="serialized",
            )
            await _apply_mental_model_derived_state(conn, bank_id, mm_embeddings, config)
            result.knowledge_pages_imported = await _restore_knowledge_pages(conn, bank_id, parsed.knowledge_pages)
            # After the tree AND its mental models exist: the index is derived from rows that
            # must already be there, and is read back from them rather than from the archive.
            result.knowledge_pages_indexed = await _index_restored_pages(
                conn, bank_id, parsed.knowledge_pages, mm_embeddings
            )

    logger.info(
        "[transfer] Imported %d document(s), %d fact(s), %d observation(s) into bank %s "
        "(%d docs skipped, %d observations skipped, %d mental model(s), %d knowledge page(s))",
        result.documents_imported,
        result.facts_imported,
        result.observations_imported,
        bank_id,
        result.documents_skipped,
        result.observations_skipped,
        result.mental_models_imported,
        result.knowledge_pages_imported,
    )
    return result


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
    knowledge_pages_imported: int = 0
    #: Pages pushed into the STORE's index. Zero when Postgres owns it (the column writes feed it
    #: instead), so a nonzero value is proof the store-owned path ran.
    knowledge_pages_indexed: int = 0
    directives_imported: int = 0
    webhooks_imported: int = 0
    history_rows_imported: int = 0


@dataclass
class ParsedBankArchive:
    """The bank-level sections of a whole-bank archive (documents read separately)."""

    manifest: TransferManifest
    # table name -> list of verbatim row dicts (banks, mental_models, directives, webhooks)
    bank_rows: dict[str, list[dict]] = field(default_factory=dict)
    # Typed knowledge-base tree (folders + pages), restored parent-first.
    knowledge_pages: list[TransferKnowledgePage] = field(default_factory=list)
    # table name -> rows (audit_log, llm_requests), present only with --include-history
    history_rows: dict[str, list[dict]] = field(default_factory=dict)


def parse_bank_archive(archive_bytes: bytes) -> ParsedBankArchive:
    """Parse the bank-level sections of a whole-bank archive (``archive_type='bank'``)."""
    with _open_archive(archive_bytes, produced_by="the bank export endpoint") as zf:
        names = set(zf.namelist())
        manifest = TransferManifest.model_validate_json(zf.read("manifest.json"))
        if manifest.archive_type != "bank":
            raise ValueError(
                f"Not a whole-bank archive (archive_type={manifest.archive_type!r}); use import_documents instead"
            )
        bank_rows: dict[str, list[dict]] = {}
        for table in ("banks", *_BANK_CHILD_TABLES, *CARRIED_HISTORY_TABLES):
            fname = f"{table}.json"
            bank_rows[table] = json.loads(zf.read(fname)) if fname in names else []
        # Typed tree — absent on pre-tree archives, which restore with no pages.
        knowledge_pages: list[TransferKnowledgePage] = []
        if "knowledge_pages.json" in names:
            knowledge_pages = [
                TransferKnowledgePage.model_validate(p) for p in json.loads(zf.read("knowledge_pages.json"))
            ]
        history_rows: dict[str, list[dict]] = {}
        for table in HISTORY_TABLES:
            fname = f"history/{table}.json"
            if fname in names:
                history_rows[table] = json.loads(zf.read(fname))
    return ParsedBankArchive(
        manifest=manifest, bank_rows=bank_rows, knowledge_pages=knowledge_pages, history_rows=history_rows
    )


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


async def _embed_in_batches(embeddings_model: Any, texts: list[str]) -> list[list[float]]:
    """Embed ``texts`` in ``_EMBED_BATCH_SIZE`` slices, preserving input order."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        vectors.extend(
            await embedding_processing.generate_embeddings_batch(
                embeddings_model, texts[start : start + _EMBED_BATCH_SIZE]
            )
        )
    return vectors


async def _regenerate_mental_model_embeddings(embeddings_model: Any, mm_rows: list[dict]) -> dict[str, list[float]]:
    """Re-embed each restored mental model with the *target* model.

    Export strips the source embedding (target-derived). Embeds the same
    ``"{name} {content}"`` text ``create_mental_model`` embeds so a restored model
    ranks identically to a freshly written one. Runs off-connection (no DB conn is
    held across the embedding call — see the retain path); returns id -> vector
    literal for the caller to apply in the restore transaction.
    """
    if not mm_rows:
        return {}
    texts = [f"{(r.get('name') or '')} {(r.get('content') or '')}" for r in mm_rows]
    vectors = await _embed_in_batches(embeddings_model, texts)
    # The vectors themselves, not the ``str(...)`` Postgres wants: a store-owned bank indexes the
    # same vector, and re-parsing a repr to recover it would be lossy for nothing. The one caller
    # that needs the literal stringifies at its own INSERT.
    return {r["id"]: list(v) for r, v in zip(mm_rows, vectors, strict=True)}


async def _apply_mental_model_derived_state(
    conn: Any,
    bank_id: str,
    mm_embeddings: dict[str, list[float]],
    config: Any,
) -> None:
    """Write the regenerated embedding (and vchord lexical state) onto restored models.

    ``search_vector`` is rebuilt only for vchord: native's column is GENERATED and
    already repopulated when the row was inserted, and pg_search / pg_textsearch /
    pgroonga index the base ``name`` / ``content`` columns directly. Same
    per-backend expression the live mental-model writes use (``pg_search_vector_expr``).
    """
    if not mm_embeddings:
        return
    sv_expr = pg_search_vector_expr(
        config, text_col="name", context_col="content", signals_col=None, native_inline=False
    )
    sv_clause = f", search_vector = {sv_expr}" if sv_expr else ""
    for mm_id, vector in mm_embeddings.items():
        await conn.execute(
            f"UPDATE {fq_table('mental_models')} SET embedding = $3::vector{sv_clause} WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
            str(vector),
        )


def _topological_page_order(pages: list[TransferKnowledgePage]) -> list[TransferKnowledgePage]:
    """Order nodes parents-before-children so the self-referential ``parent_id`` FK
    always resolves on insert. A node whose parent is absent from the archive (only
    possible in a corrupt export) or part of a cycle is emitted last so the FK, not
    a silent drop, surfaces it."""
    by_id = {p.id: p for p in pages}
    ordered: list[TransferKnowledgePage] = []
    placed: set[str] = set()
    remaining = list(pages)
    while remaining:
        ready = [p for p in remaining if p.parent_id is None or p.parent_id not in by_id or p.parent_id in placed]
        if not ready:
            # Unresolvable parents (cycle / dangling) — emit the rest as-is.
            ordered.extend(remaining)
            break
        for p in ready:
            ordered.append(p)
            placed.add(p.id)
        ready_ids = {p.id for p in ready}
        remaining = [p for p in remaining if p.id not in ready_ids]
    return ordered


async def _index_restored_pages(
    conn: Any,
    bank_id: str,
    pages: list[TransferKnowledgePage],
    mm_embeddings: dict[str, list[float]],
) -> int:
    """Push restored pages into the store's search index.

    Restoring a page writes its row and rebuilds the Postgres derived columns. For a bank whose
    store owns the knowledge index that is the wrong half: search routes to the store, the store
    was never told these pages exist, and it answers with an empty list rather than an error. The
    bank reads back perfectly -- tree, bodies, everything -- and is findable by nothing, which is
    why it survives a restore that looks entirely successful.

    Rows are read back rather than taken from the archive, matching the live write path: both index
    ``"{name} {content}"``, and a second place assembling that string differently is a second place
    for the index to disagree with the row it is derived from.
    """
    from ..memories import KnowledgePageEntry, get_memories

    store = get_memories()
    if not store.store_owned_for(bank_id):
        return 0
    mm_ids = [p.mental_model_id for p in pages if p.kind == "page" and p.mental_model_id]
    if not mm_ids:
        return 0

    rows = await conn.fetch(
        f"SELECT id, name, content, tags, last_refreshed_at FROM {fq_table('mental_models')} "
        f"WHERE bank_id = $1 AND id = ANY($2::text[])",
        bank_id,
        mm_ids,
    )
    entries = [
        KnowledgePageEntry(
            page_id=r["id"],
            index_text=f"{r['name'] or ''} {r['content'] or ''}",
            # A page whose re-embed produced nothing is still indexed, for text search only:
            # better findable by its words than findable by nothing.
            embedding=mm_embeddings.get(r["id"]),
            tags=list(r["tags"] or []),
            updated_at=r["last_refreshed_at"],
        )
        for r in rows
    ]
    if entries:
        await store.index_knowledge_pages(bank_id, entries)
    return len(entries)


async def _restore_knowledge_pages(conn: Any, bank_id: str, pages: list[TransferKnowledgePage]) -> int:
    """Restore the knowledge-base tree into ``bank_id`` parents-first.

    IDs, ``parent_id``, ``mental_model_id``, ``managed``, ``sort_order``, name and
    timestamps are preserved; ``bank_id`` is applied to the target. Pages are
    restored after their backing mental models (the caller sequences that), and
    folders before their children (topological order here). ``ON CONFLICT DO
    NOTHING`` keeps the import idempotent.
    """
    if not pages:
        return 0
    inserted = 0
    id_map: dict[str, str] = {}
    for page in pages:
        # Knowledge-page ids are globally unique within a schema, unlike the
        # bank-scoped mental-model ids. A document transfer is a merge, so a
        # source page can already exist in the target schema; remap only those
        # collisions and rewrite parent links below.
        page_id = page.id
        if await conn.fetchval(
            f"SELECT 1 FROM {fq_table('knowledge_pages')} WHERE id = $1 AND bank_id <> $2", page_id, bank_id
        ):
            page_id = str(uuid.uuid4())
        id_map[page.id] = page_id
    for page in _topological_page_order(pages):
        await conn.execute(
            f"""
            INSERT INTO {fq_table("knowledge_pages")}
                (id, bank_id, parent_id, kind, name, mental_model_id, sort_order, managed, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, COALESCE($9, now()), COALESCE($10, now()))
            ON CONFLICT DO NOTHING
            """,
            id_map[page.id],
            bank_id,
            id_map.get(page.parent_id) if page.parent_id else None,
            page.kind,
            page.name,
            page.mental_model_id,
            page.sort_order,
            page.managed,
            page.created_at,
            page.updated_at,
        )
        inserted += 1
    return inserted


async def import_bank(
    *,
    backend: Any,
    embeddings_model: Any,
    entity_resolver: Any,
    resolve_config: Callable[[], Awaitable[Any]],
    format_date_fn: Any,
    archive_bytes: bytes,
    target_bank_id: str | None = None,
    include_history: bool = False,
    ops: Any = None,
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

    Takes ``resolve_config`` rather than a resolved config because the only correct
    moment to resolve one is *inside* this function, after the archive's bank row
    lands. Before that the target bank does not exist (import refuses to write into
    an existing one), so any config a caller resolved carries global + tenant values
    and none of the bank's own — which is exactly the bug in #3236.
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

    # `internal_id` is a globally-unique (banks_internal_id_unique) local identifier
    # used only for per-bank index naming — it is NOT part of the bank's logical
    # state and nothing in the archive references it. Drop it so the column DEFAULT
    # (gen_random_uuid) mints a fresh one on insert. Keeping the source value makes
    # the banks INSERT collide with the source bank on a same-instance re-import,
    # where `ON CONFLICT DO NOTHING` then silently skips the parent row and every
    # child (mental_models, …) trips its bank_id foreign key. See #3270.
    for row in parsed.bank_rows.get("banks", []):
        row.pop("internal_id", None)

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
        #
        # A no-op when a size threshold is set: entitlement is by size then, and
        # the restored rows land through the normal import path, where the
        # maintenance operation picks them up (#3485).
        internal_id = await conn.fetchval(f"SELECT internal_id FROM {fq_table('banks')} WHERE bank_id = $1", bank_id)
        if internal_id is not None:
            await bank_utils.create_bank_vector_indexes(conn, bank_id, str(internal_id), ops=ops)

    # Only now does the bank row — and with it the archive's own config — exist, so
    # this is where the config the documents are replayed with has to come from.
    # Until #3236 the import ran on a config resolved before the restore, which
    # could not contain the bank's `entity_labels`: every label entity was
    # classified as a regular one, which both exposed label values to fuzzy merging
    # (#3187) and left them inside the trigram index that the partial index is
    # supposed to keep them out of (#3208), so an imported bank silently lost that
    # fix.
    config = await resolve_config()

    doc_result = await import_documents(
        backend=backend,
        embeddings_model=embeddings_model,
        entity_resolver=entity_resolver,
        config=config,
        format_date_fn=format_date_fn,
        bank_id=bank_id,
        archive_bytes=archive_bytes,
        ops=ops,
        outbox_callback_factory=None,
        # The bank path restores mental models and knowledge pages itself, below,
        # after remapping their evidence ids onto the facts just replayed.
        restore_bank_scoped_rows=False,
    )

    result = BankImportResult(
        bank_id=bank_id,
        documents_imported=doc_result.documents_imported,
        facts_imported=doc_result.facts_imported,
        observations_imported=doc_result.observations_imported,
    )

    # Facts and observations are replayed with fresh ids, but mental-model rows
    # keep their ids and are restored verbatim below. Repair their grounding
    # references before insertion so current and historical based_on data points
    # at the target bank's units rather than the source bank's units.
    unit_id_map = doc_result.remapped_unit_ids
    _remap_mental_model_evidence(parsed.bank_rows.get("mental_models", []), unit_id_map)
    _remap_mental_model_evidence(parsed.bank_rows.get("mental_model_history", []), unit_id_map)

    # Re-embed restored mental models off-connection (the source embedding was
    # stripped on export), so no DB connection is held across the embedding call.
    mm_rows = parsed.bank_rows.get("mental_models", [])
    mm_embeddings = await _regenerate_mental_model_embeddings(embeddings_model, mm_rows)

    async with acquire_with_retry(backend) as conn:
        result.mental_models_imported = await _restore_rows(
            conn,
            "mental_models",
            mm_rows,
            bank_rows_json_encoding=bank_rows_json_encoding,
        )
        # Apply the regenerated embedding + backend-specific lexical state onto the
        # restored rows (native search_vector already repopulated on insert).
        await _apply_mental_model_derived_state(conn, bank_id, mm_embeddings, config)
        # Restored after mental_models so the (mental_model_id, bank_id) FK resolves.
        result.mental_model_history_imported = await _restore_rows(
            conn,
            "mental_model_history",
            parsed.bank_rows.get("mental_model_history", []),
            bank_rows_json_encoding=bank_rows_json_encoding,
        )
        # Knowledge-base tree after its backing mental models exist (page FK) and
        # parents-first (self-referential parent_id FK).
        result.knowledge_pages_imported = await _restore_knowledge_pages(conn, bank_id, parsed.knowledge_pages)
        # After the tree AND its mental models exist: the index is derived from rows that
        # must already be there, and is read back from them rather than from the archive.
        result.knowledge_pages_indexed = await _index_restored_pages(
            conn, bank_id, parsed.knowledge_pages, mm_embeddings
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
        "%d mental model(s), %d mm-history row(s), %d knowledge page(s), %d directive(s), "
        "%d webhook(s), %d history row(s)",
        bank_id,
        result.documents_imported,
        result.facts_imported,
        result.observations_imported,
        result.mental_models_imported,
        result.mental_model_history_imported,
        result.knowledge_pages_imported,
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
    # Ask whichever store actually holds the document. A bank whose document store is external
    # leaves the SQL `documents` table empty, so the query below finds nothing and EVERY conflict
    # mode goes inert: `skip` re-imports the document it was asked to leave alone, `new-id` keeps
    # the original id instead of duplicating under a fresh one, and `replace` degenerates to a
    # plain insert. Silent in all three cases — the import reports success either way.
    from ..memories import get_memories

    _store = get_memories()
    if _store.store_owned_for(bank_id):
        exists = await _store.get_document_record(bank_id=bank_id, document_id=document_id) is not None
    else:
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

    # Put the document itself where the store expects it. Retain does this via
    # `_store_document_bodies`; the importer never did, so for a bank whose document store is
    # external the import wrote the SQL metadata row and the chunk rows and NOTHING to the store —
    # the restored bank listed no documents at all, because that listing reads the store. A no-op
    # for Postgres, which keeps the body in the `documents` row written below.
    #
    # Before the connection is taken, like the retain path: this is the slow object-store write and
    # it has no business inside the write transaction.
    await orchestrator._store_document_bodies(
        bank_id=bank_id,
        document_id=target_id,
        combined_content=document.original_text or "",
        chunk_texts=[c.chunk_text for c in sorted(document.chunks, key=lambda c: c.chunk_index)],
        merged_tags=list(document.tags or []),
        config=config,
        retain_params=document.retain_params,
    )

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

            # Restore the source consolidation lifecycle. A whole-bank transfer
            # preserves exact eligibility: a fact that was consolidated (or that
            # failed consolidation) in the source is never re-consolidated on the
            # target, so the maintenance reconciler sees no phantom backlog and
            # observations are not re-derived. Archives predating these fields
            # carry None for all three -> skipped here, leaving the
            # observation-driven marking in _import_observations as the only
            # (lossy) signal, exactly as before.
            if result_unit_ids:
                await _restore_fact_lifecycle(
                    conn,
                    bank_id,
                    document.facts,
                    retained_index_by_original,
                    result_unit_ids[0],
                )

    # Best-effort, and only after the acquire() block above has exited: this
    # takes its own connection, and on Oracle the write above is not committed
    # until that block exits, so flushing while still holding the connection
    # deadlocks (see the retain orchestrator for the full explanation).
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


async def _restore_fact_lifecycle(
    conn: Any,
    bank_id: str,
    facts: list[TransferFact],
    retained_index_by_original: list[int | None],
    retained_unit_ids: list[str],
) -> None:
    """Apply each imported fact's source consolidation timestamps to its new row.

    ``retained_unit_ids`` follows the retained fact order; ``retained_index_by_original[i]``
    maps original fact ``i`` to its position there (or ``None`` if it was dropped
    on insert, e.g. a duplicate). ``created_at`` restores source provenance only
    when present (mirroring the document-row handling); ``consolidated_at`` /
    ``consolidation_failed_at`` are set verbatim — a source-``NULL`` (unconsolidated)
    fact stays eligible, which is correct.

    No ``updated_at`` stamp (see :data:`~..memories.base.META_UPDATED_AT`): this fixup
    runs in the same transaction as the insert that created the row, so the column
    already carries this transaction's timestamp. The same holds for the observation
    fixups below.
    """
    rows: list[tuple[uuid.UUID, datetime | None, datetime | None, datetime | None]] = []
    for original_index, fact in enumerate(facts):
        retained_index = retained_index_by_original[original_index]
        if retained_index is None:
            continue
        if fact.created_at is None and fact.consolidated_at is None and fact.consolidation_failed_at is None:
            # Legacy archive without lifecycle fields — nothing to restore.
            continue
        rows.append(
            (
                uuid.UUID(retained_unit_ids[retained_index]),
                fact.created_at,
                fact.consolidated_at,
                fact.consolidation_failed_at,
            )
        )
    if not rows:
        return
    await conn.executemany(
        f"UPDATE {fq_table('memory_units')} "
        f"SET created_at = COALESCE($2, created_at), consolidated_at = $3, consolidation_failed_at = $4 "
        f"WHERE id = $1 AND bank_id = $5",
        [
            (unit_id, created_at, consolidated_at, failed_at, bank_id)
            for unit_id, created_at, consolidated_at, failed_at in rows
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
    embeddings = await _embed_in_batches(embeddings_model, [obs.text for obs, _ in resolved])
    processed = [
        ProcessedFact(
            fact_text=obs.text,
            fact_type="observation",
            embedding=pack_embedding(embedding),
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
                await _link_observation_sources(conn, ops, bank_id, observation_uuid, source_uuids, obs.proof_count)
                if obs.source_id is not None:
                    outcome.remapped_unit_ids[obs.source_id] = str(observation_uuid)

            # Mark source facts consolidated so the target consolidator skips
            # them. COALESCE keeps the exact source timestamp already restored by
            # _restore_fact_lifecycle (new archives); now() is the fallback only
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


def _remap_based_on_ids(payload: dict[str, Any] | None, unit_id_map: dict[str, str]) -> None:
    """Rewrite memory-unit ids in a persisted reflect response after transfer.

    Mental-model rows retain their ids during a whole-bank restore, while facts
    and observations are replayed and receive new ids. The response is otherwise
    copied verbatim, so update only evidence ids and preserve all generated text
    and metadata. Older archives without source ids simply have no entries to
    rewrite.
    """
    if not payload or not unit_id_map:
        return
    based_on = payload.get("based_on")
    if not isinstance(based_on, dict):
        return
    for entries in based_on.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                source_id = entry.get("id")
                if isinstance(source_id, str) and source_id in unit_id_map:
                    entry["id"] = unit_id_map[source_id]


def _remap_mental_model_evidence(rows: list[dict], unit_id_map: dict[str, str]) -> None:
    """Repair current and historical mental-model reflect-response evidence.

    Handles both row shapes. A ``mental_models`` row carries the response in its
    own ``reflect_response`` column. A ``mental_model_history`` row carries one
    snapshot as a single JSONB ``content`` blob holding ``previous_content`` and
    ``previous_reflect_response`` — there is no top-level column of that name, so
    the nested payload has to be decoded, rewritten and put back.
    """
    for row in rows:
        reflect_response = _decode_json_object(row.get("reflect_response"))
        if isinstance(reflect_response, dict):
            _remap_based_on_ids(reflect_response, unit_id_map)
            row["reflect_response"] = reflect_response
        content = _decode_json_object(row.get("content"))
        if isinstance(content, dict):
            previous = _decode_json_object(content.get("previous_reflect_response"))
            if isinstance(previous, dict):
                _remap_based_on_ids(previous, unit_id_map)
                content["previous_reflect_response"] = previous
                row["content"] = content


def _decode_json_object(value: Any) -> Any:
    """Accept decoded JSONB values and one or more serialized JSON layers."""
    for _ in range(3):
        if not isinstance(value, str):
            return value
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        if decoded == value:
            return value
        value = decoded
    return value


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
