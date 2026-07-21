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
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Literal

from ..causal_links import CANONICAL_CAUSAL_LINK_TYPE, LEGACY_CAUSAL_LINK_TYPES
from ..db_utils import acquire_with_retry
from ..retain import bank_utils, chunk_storage, embedding_processing, fact_storage, link_utils, orchestrator
from ..retain.types import (
    CausalRelation,
    ChunkMetadata,
    ExtractedFact,
    ProcessedFact,
    RetainContent,
)
from ..schema import fq_table
from .schema import (
    CARRIED_HISTORY_TABLES,
    HISTORY_TABLES,
    SCHEMA_VERSION,
    TransferDocument,
    TransferFact,
    TransferManifest,
    TransferObservation,
)

logger = logging.getLogger(__name__)

OnConflict = Literal["skip", "replace", "new-id"]
_VALID_CONFLICT_MODES: tuple[OnConflict, ...] = ("skip", "replace", "new-id")


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


def parse_archive(archive_bytes: bytes) -> ParsedArchive:
    """Parse and validate a transfer ZIP archive produced by ``export_documents``."""
    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as zf:
        names = set(zf.namelist())
        if "manifest.json" not in names:
            raise ValueError("Invalid transfer archive: manifest.json is missing")
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
    return ParsedArchive(manifest=manifest, documents=documents, observations=observations)


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

    Returns:
        An :class:`ImportResult` with per-document counts.
    """
    if on_conflict not in _VALID_CONFLICT_MODES:
        raise ValueError(f"Invalid on_conflict '{on_conflict}'; expected one of {_VALID_CONFLICT_MODES}")
    if ops is None:
        ops = backend.ops

    parsed = parse_archive(archive_bytes)
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


def parse_bank_archive(archive_bytes: bytes) -> ParsedBankArchive:
    """Parse the bank-level sections of a whole-bank archive (``archive_type='bank'``)."""
    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as zf:
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


async def _restore_rows(conn: Any, table: str, rows: list[dict]) -> int:
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
                # text and cast. Values may already be str (no codec on export) or
                # a Python object (codec on export) — normalize to text either way.
                values.append(value if isinstance(value, str) or value is None else json.dumps(value))
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
    """
    if ops is None:
        ops = backend.ops
    parsed = parse_bank_archive(archive_bytes)
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
        await _restore_rows(conn, "banks", parsed.bank_rows.get("banks", []))
    # Ensure the bank's per-bank vector indexes exist (no-op for global-index
    # extensions); idempotent and keeps the restored banks row (ON CONFLICT DO NOTHING).
    await bank_utils.get_or_create_bank_profile(backend, bank_id)

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
    )

    result = BankImportResult(
        bank_id=bank_id,
        documents_imported=doc_result.documents_imported,
        facts_imported=doc_result.facts_imported,
        observations_imported=doc_result.observations_imported,
    )
    async with acquire_with_retry(backend) as conn:
        result.mental_models_imported = await _restore_rows(
            conn, "mental_models", parsed.bank_rows.get("mental_models", [])
        )
        # Restored after mental_models so the (mental_model_id, bank_id) FK resolves.
        result.mental_model_history_imported = await _restore_rows(
            conn, "mental_model_history", parsed.bank_rows.get("mental_model_history", [])
        )
        result.directives_imported = await _restore_rows(conn, "directives", parsed.bank_rows.get("directives", []))
        result.webhooks_imported = await _restore_rows(conn, "webhooks", parsed.bank_rows.get("webhooks", []))
        if include_history:
            for table in HISTORY_TABLES:
                result.history_rows_imported += await _restore_rows(conn, table, parsed.history_rows.get(table, []))

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
                source_uuids = [uuid.UUID(s) for s in sources]
                all_source_ids.update(source_uuids)
                await _link_observation_sources(
                    conn, ops, bank_id, uuid.UUID(obs_unit_id), source_uuids, obs.proof_count
                )

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
