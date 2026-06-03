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
from datetime import UTC, datetime
from typing import Any, Literal

from ..db_utils import acquire_with_retry
from ..retain import chunk_storage, embedding_processing, fact_storage, orchestrator
from ..retain.types import (
    CausalRelation,
    ChunkMetadata,
    ExtractedFact,
    ProcessedFact,
    RetainContent,
)
from ..schema import fq_table
from .schema import (
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

        unit_ids = await _import_one_document(
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
        result.facts_imported += len(unit_ids)
        result.imported_documents.append(
            ImportedDocument(
                document_id=target_id,
                unit_ids=unit_ids,
                content=document.original_text or "",
                tags=list(document.tags),
            )
        )
        for ordinal, unit_id in enumerate(unit_ids):
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
) -> list[str]:
    """Re-embed and insert a single document; returns the new unit ids in fact order."""
    log_buffer: list[str] = []

    # Fire the same retain.completed webhook retain emits, transactionally inside
    # this document's insert. Factory returns None when no webhook manager exists.
    outbox_callback = (
        outbox_callback_factory([{"document_id": target_id, "tags": list(document.tags)}])
        if outbox_callback_factory
        else None
    )

    extracted_facts = [_to_extracted_fact(fact) for fact in document.facts]

    processed_facts: list[ProcessedFact] = []
    if extracted_facts:
        augmented = embedding_processing.augment_texts_with_dates(extracted_facts, format_date_fn)
        embeddings = await embedding_processing.generate_embeddings_batch(embeddings_model, augmented)
        processed_facts = [ProcessedFact.from_extracted_fact(ef, emb) for ef, emb in zip(extracted_facts, embeddings)]

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
                resolved_entity_ids=phase1.entities.resolved_entity_ids,
                entity_to_unit=phase1.entities.entity_to_unit,
                unit_to_entity_ids=phase1.entities.unit_to_entity_ids,
                semantic_ann_links=phase1.semantic_ann_links,
                skip_semantic_links=False,
                outbox_callback=outbox_callback,
                ops=ops,
            )

        try:
            await entity_resolver.flush_pending_stats()
        except Exception:
            logger.warning("[transfer] Entity stats flush failed for document %s", target_id, exc_info=True)

    logger.debug("[transfer] Imported document %s:\n%s", target_id, "\n".join(log_buffer))
    # Single content item -> result_unit_ids[0] holds the new unit ids in fact order.
    return list(result_unit_ids[0]) if result_unit_ids else []


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
        ],
        content_index=0,
        chunk_index=fact.chunk_index,
        context=fact.context or "",
        mentioned_at=mentioned_at,
        metadata=dict(fact.metadata),
        tags=list(fact.tags),
        observation_scopes=fact.observation_scopes,
    )
