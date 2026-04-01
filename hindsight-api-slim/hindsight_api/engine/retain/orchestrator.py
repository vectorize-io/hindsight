"""
Main orchestrator for the retain pipeline.

Coordinates all retain pipeline modules to store memories efficiently.
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from ..db_utils import acquire_with_retry
from ..memory_engine import fq_table
from . import bank_utils


def utcnow():
    """Get current UTC time."""
    return datetime.now(UTC)


def parse_datetime_flexible(value: Any) -> datetime:
    """
    Parse a datetime value that could be either a datetime object or an ISO string.

    This handles datetime values from both direct Python calls and deserialized JSON
    (where datetime objects are serialized as ISO strings).

    Args:
        value: Either a datetime object or an ISO format string

    Returns:
        datetime object (timezone-aware)

    Raises:
        TypeError: If value is neither datetime nor string
        ValueError: If string is not a valid ISO datetime
    """
    if isinstance(value, datetime):
        # Ensure timezone-aware
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
    elif isinstance(value, str):
        # Parse ISO format string (handles both 'Z' and '+00:00' timezone formats)
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        # Ensure timezone-aware
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt
    else:
        raise TypeError(f"Expected datetime or string, got {type(value).__name__}")


import asyncpg

from ..response_models import TokenUsage
from . import (
    chunk_storage,
    embedding_processing,
    entity_processing,
    fact_extraction,
    fact_storage,
    link_creation,
)
from .types import (
    ChunkMetadata,
    EntityResolutionResult,
    ExtractedFact,
    Phase1Result,
    Phase3Context,
    ProcessedFact,
    RetainContent,
    RetainContentDict,
)

logger = logging.getLogger(__name__)


def _build_retain_params(contents_dicts, document_tags=None, doc_contents=None):
    """Build retain_params and merged_tags from content dicts."""
    if doc_contents is not None:
        # Per-document mode: doc_contents is list of (idx, content_dict)
        items = [item for _, item in doc_contents]
    else:
        items = contents_dicts

    all_tags = set(document_tags or [])
    for item in items:
        item_tags = item.get("tags", []) or []
        all_tags.update(item_tags)
    merged_tags = list(all_tags)

    retain_params = {}
    if items:
        first_item = items[0]
        if first_item.get("context"):
            retain_params["context"] = first_item["context"]
        if first_item.get("event_date"):
            retain_params["event_date"] = (
                first_item["event_date"].isoformat()
                if hasattr(first_item["event_date"], "isoformat")
                else str(first_item["event_date"])
            )
        if first_item.get("metadata"):
            retain_params["metadata"] = first_item["metadata"]

    return retain_params, merged_tags


async def _pre_resolve_phase1(
    pool,
    entity_resolver,
    bank_id: str,
    contents: list[RetainContent],
    processed_facts: list[ProcessedFact],
    config,
    log_buffer: list[str],
    skip_semantic_ann: bool = False,
) -> Phase1Result:
    """
    Phase 1: Run expensive read-heavy operations on a separate connection
    OUTSIDE the write transaction.

    - Entity resolution: trigram GIN scan + co-occurrence fetch + scoring
    - Semantic ANN: HNSW index probes to find similar existing units

    Running these outside the transaction avoids holding row locks during
    slow reads, eliminating TimeoutErrors under concurrent load.
    """
    from .link_utils import compute_semantic_links_ann

    user_entities_per_content = {idx: content.entities for idx, content in enumerate(contents) if content.entities}

    # Use placeholder unit_ids for grouping during resolution.  The actual
    # unit_ids are created later by insert_facts_batch inside the transaction,
    # but entity resolution and ANN search only need them as grouping keys.
    placeholder_unit_ids = [str(i) for i in range(len(processed_facts))]
    embeddings = [fact.embedding for fact in processed_facts]

    async with acquire_with_retry(pool) as resolve_conn:
        resolved_entity_ids, entity_to_unit, unit_to_entity_ids = await entity_processing.resolve_entities(
            entity_resolver,
            resolve_conn,
            bank_id,
            placeholder_unit_ids,
            processed_facts,
            log_buffer,
            user_entities_per_content=user_entities_per_content,
            entity_labels=getattr(config, "entity_labels", None),
        )

        # Semantic ANN search on the same connection (autocommit, no transaction).
        # Skipped in streaming mode — deferred to Phase 3 to avoid O(bank_size)
        # scaling bottleneck that makes later streaming batches progressively slower.
        semantic_ann_links = []
        if not skip_semantic_ann:
            fact_types = [fact.fact_type for fact in processed_facts]
            semantic_ann_links = await compute_semantic_links_ann(
                resolve_conn, bank_id, placeholder_unit_ids, embeddings, fact_types=fact_types, log_buffer=log_buffer
            )

    return Phase1Result(
        entities=EntityResolutionResult(
            resolved_entity_ids=resolved_entity_ids,
            entity_to_unit=entity_to_unit,
            unit_to_entity_ids=unit_to_entity_ids,
        ),
        semantic_ann_links=semantic_ann_links,
    )


def _remap_phase1_results(
    resolved_entity_ids: list[str],
    entity_to_unit: list[tuple],
    unit_to_entity_ids: dict[str, list[str]],
    semantic_ann_links: list[tuple],
    actual_unit_ids: list[str],
) -> tuple[list[tuple], dict[str, list[str]], list[tuple]]:
    """
    Remap Phase 1 results from placeholder unit IDs to actual unit IDs.

    During Phase 1 we use str(fact_index) as placeholder unit IDs.
    After insert_facts_batch creates real UUIDs, this function replaces the
    placeholders so that all rows reference the correct memory_units.
    """
    # Build placeholder -> actual mapping
    placeholder_to_actual = {str(i): actual_id for i, actual_id in enumerate(actual_unit_ids)}

    # Remap entity_to_unit tuples
    remapped_entity_to_unit = [
        (placeholder_to_actual.get(unit_id, unit_id), local_idx, fact_date)
        for unit_id, local_idx, fact_date in entity_to_unit
    ]

    # Remap unit_to_entity_ids keys
    remapped_unit_to_entity_ids: dict[str, list[str]] = {}
    for placeholder_id, entity_ids in unit_to_entity_ids.items():
        actual_id = placeholder_to_actual.get(placeholder_id, placeholder_id)
        remapped_unit_to_entity_ids[actual_id] = entity_ids

    # Remap semantic ANN links (from_id uses placeholder)
    remapped_semantic = [
        (placeholder_to_actual.get(lnk[0], lnk[0]), lnk[1], lnk[2], lnk[3], lnk[4]) for lnk in semantic_ann_links
    ]

    return remapped_entity_to_unit, remapped_unit_to_entity_ids, remapped_semantic


async def _insert_facts_and_links(
    conn,
    entity_resolver,
    bank_id: str,
    contents: list[RetainContent],
    extracted_facts: list,
    processed_facts: list[ProcessedFact],
    config,
    log_buffer: list[str],
    resolved_entity_ids: list[str],
    entity_to_unit: list[tuple],
    unit_to_entity_ids: dict[str, list[str]],
    semantic_ann_links: list[tuple],
    skip_semantic_links: bool = False,
    outbox_callback=None,
) -> tuple[list[list[str]], Phase3Context]:
    """
    Phase 2 of the retain pipeline: insert facts and retrieval-critical links.

    Runs inside a single database transaction to ensure atomicity of the data
    that retrieval depends on (facts, unit_entities, temporal/semantic/causal links).

    Entity link generation and insertion for UI visualization are NOT done here —
    only the unit_entities INSERT (FK to memory_units) stays in the transaction.
    Entity link building is deferred to Phase 3 (post-transaction, best-effort).
    """
    unit_ids = await fact_storage.insert_facts_batch(conn, bank_id, processed_facts)
    step_start = time.time()
    log_buffer.append(f"  Insert facts: {len(unit_ids)} units in {time.time() - step_start:.3f}s")

    # Context for Phase 3 entity link building (after transaction commits)
    phase3_context = Phase3Context()

    if unit_ids:
        # Entity resolution was done in Phase 1 (separate connection).
        # Remap placeholder IDs to actual unit IDs.
        step_start = time.time()
        remapped_entity_to_unit, remapped_unit_to_entity_ids, remapped_semantic = _remap_phase1_results(
            resolved_entity_ids, entity_to_unit, unit_to_entity_ids, semantic_ann_links or [], unit_ids
        )
        # Update semantic_ann_links with remapped IDs for Phase 2
        semantic_ann_links = remapped_semantic
        # INSERT unit_entities (FK to memory_units, must be in transaction)
        unit_entity_pairs = [
            (unit_id, resolved_entity_ids[idx])
            for idx, (unit_id, _local_idx, _fact_date) in enumerate(remapped_entity_to_unit)
        ]
        await entity_resolver.link_units_to_entities_batch(unit_entity_pairs, conn=conn)
        log_buffer.append(f"  Insert unit_entities: {len(unit_entity_pairs)} pairs in {time.time() - step_start:.3f}s")
        # Save context for Phase 3 entity link building (after commit)
        phase3_context = Phase3Context(
            unit_ids=unit_ids,
            resolved_entity_ids=resolved_entity_ids,
            entity_to_unit=remapped_entity_to_unit,
            unit_to_entity_ids=remapped_unit_to_entity_ids,
        )

        # Create temporal links
        step_start = time.time()
        temporal_link_count = await link_creation.create_temporal_links_batch(conn, bank_id, unit_ids)
        log_buffer.append(f"  Temporal links: {temporal_link_count} links in {time.time() - step_start:.3f}s")

        # Create semantic links (within-batch + pre-computed ANN from Phase 1)
        if skip_semantic_links:
            log_buffer.append("  Semantic links: skipped (deferred to final ANN pass)")
            semantic_link_count = 0
        else:
            step_start = time.time()
            embeddings_for_links = [fact.embedding for fact in processed_facts]
            semantic_link_count = await link_creation.create_semantic_links_batch(
                conn,
                bank_id,
                unit_ids,
                embeddings_for_links,
                pre_computed_ann_links=semantic_ann_links,
            )
            log_buffer.append(f"  Semantic links: {semantic_link_count} links in {time.time() - step_start:.3f}s")

        # NOTE: Entity links are NOT inserted here. They are deferred to
        # Phase 3 (post-transaction, best-effort) since retrieval uses the
        # unit_entities self-join instead. Entity links only serve UI visualization.

        # Create causal links
        step_start = time.time()
        causal_link_count = await link_creation.create_causal_links_batch(conn, bank_id, unit_ids, processed_facts)
        log_buffer.append(f"  Causal links: {causal_link_count} links in {time.time() - step_start:.3f}s")

    # Map results back to original content items
    result_unit_ids = _map_results_to_contents(contents, extracted_facts, unit_ids if unit_ids else [])

    if outbox_callback:
        await outbox_callback(conn)

    return result_unit_ids, phase3_context


async def _build_and_insert_entity_links_phase3(
    pool,
    entity_resolver,
    bank_id: str,
    phase3_ctx: Phase3Context,
    log_buffer: list[str],
) -> None:
    """
    Phase 3 helper: build entity links from resolved data and insert them.

    Runs on a fresh connection after the main transaction has committed.
    Entity links are for UI graph visualization only — retrieval uses
    the unit_entities self-join instead.
    """
    p3_unit_ids = phase3_ctx.unit_ids
    p3_resolved = phase3_ctx.resolved_entity_ids
    p3_entity_to_unit = phase3_ctx.entity_to_unit
    p3_unit_to_entity_ids = phase3_ctx.unit_to_entity_ids

    if not p3_unit_ids or not p3_resolved:
        return

    async with acquire_with_retry(pool) as conn:
        step_start = time.time()
        entity_links = await entity_processing.build_entity_links(
            entity_resolver,
            conn,
            bank_id,
            p3_unit_ids,
            p3_resolved,
            p3_entity_to_unit,
            p3_unit_to_entity_ids,
            log_buffer,
            skip_unit_entities_insert=True,  # Already inserted in Phase 2
        )
        if entity_links:
            await entity_processing.insert_entity_links_batch(conn, entity_links, bank_id)
        log_buffer.append(f"  Entity links (viz): {len(entity_links)} links in {time.time() - step_start:.3f}s")


async def _extract_and_embed(
    contents: list[RetainContent],
    llm_config,
    agent_name: str,
    config,
    embeddings_model,
    format_date_fn,
    fact_type_override: str | None,
    log_buffer: list[str],
    pool=None,
    operation_id: str | None = None,
    schema: str | None = None,
) -> tuple[list, list[ProcessedFact], list[ChunkMetadata], TokenUsage]:
    """
    Shared pipeline: extract facts from contents and generate embeddings.

    Returns:
        Tuple of (extracted_facts, processed_facts, chunks_metadata, usage)
    """
    step_start = time.time()
    extracted_facts, chunks, usage = await fact_extraction.extract_facts_from_contents(
        contents, llm_config, agent_name, config, pool, operation_id, schema
    )
    log_buffer.append(
        f"  Extract facts: {len(extracted_facts)} facts, {len(chunks)} chunks "
        f"from {len(contents)} contents in {time.time() - step_start:.3f}s"
    )

    if not extracted_facts:
        return extracted_facts, [], chunks, usage

    if fact_type_override:
        for fact in extracted_facts:
            fact.fact_type = fact_type_override

    step_start = time.time()
    augmented_texts = embedding_processing.augment_texts_with_dates(extracted_facts, format_date_fn)
    embeddings = await embedding_processing.generate_embeddings_batch(embeddings_model, augmented_texts)
    log_buffer.append(f"  Generate embeddings: {len(embeddings)} embeddings in {time.time() - step_start:.3f}s")

    processed_facts = [ProcessedFact.from_extracted_fact(ef, emb) for ef, emb in zip(extracted_facts, embeddings)]

    return extracted_facts, processed_facts, chunks, usage


async def retain_batch(
    pool,
    embeddings_model,
    llm_config,
    entity_resolver,
    format_date_fn,
    bank_id: str,
    contents_dicts: list[RetainContentDict],
    config,
    document_id: str | None = None,
    is_first_batch: bool = True,
    fact_type_override: str | None = None,
    document_tags: list[str] | None = None,
    operation_id: str | None = None,
    schema: str | None = None,
    outbox_callback: Callable[["asyncpg.Connection"], Awaitable[None]] | None = None,
    db_semaphore: "asyncio.Semaphore | None" = None,
) -> tuple[list[list[str]], TokenUsage]:
    """
    Process a batch of content through the retain pipeline.

    Supports delta retain: when upserting a document that already has chunks,
    only re-processes chunks whose content has changed. Unchanged chunks keep
    their existing facts, entities, and links.
    """
    start_time = time.time()
    total_chars = sum(len(item.get("content", "")) for item in contents_dicts)

    log_buffer = []
    log_buffer.append(f"{'=' * 60}")
    log_buffer.append(f"RETAIN_BATCH START: {bank_id}")
    log_buffer.append(f"Batch size: {len(contents_dicts)} content items, {total_chars:,} chars")
    log_buffer.append(f"{'=' * 60}")

    # Get bank profile
    profile = await bank_utils.get_bank_profile(pool, bank_id)
    agent_name = profile["name"]

    # Convert dicts to RetainContent objects
    contents = _build_contents(contents_dicts, document_tags)

    # Resolve effective document_id early so both delta and streaming paths
    # can find existing chunks from a prior attempt. On retry, the generated
    # document_id is recovered from operation result_metadata.
    effective_doc_id = document_id
    if not effective_doc_id:
        doc_ids = {item.get("document_id") for item in contents_dicts if item.get("document_id")}
        if len(doc_ids) == 1:
            effective_doc_id = doc_ids.pop()
    if not effective_doc_id and operation_id:
        try:
            async with acquire_with_retry(pool) as conn:
                row = await conn.fetchrow(
                    f"SELECT result_metadata FROM {fq_table('async_operations')} WHERE operation_id = $1",
                    uuid.UUID(operation_id),
                )
                if row and row["result_metadata"]:
                    meta = (
                        row["result_metadata"]
                        if isinstance(row["result_metadata"], dict)
                        else json.loads(row["result_metadata"])
                    )
                    effective_doc_id = meta.get("generated_document_id")
        except Exception:
            pass
    if not effective_doc_id:
        effective_doc_id = str(uuid.uuid4())
        # Persist so retries reuse the same document_id
        if operation_id:
            try:
                async with acquire_with_retry(pool) as conn:
                    await conn.execute(
                        f"""
                        UPDATE {fq_table("async_operations")}
                        SET result_metadata = result_metadata || $1::jsonb, updated_at = now()
                        WHERE operation_id = $2
                        """,
                        json.dumps({"generated_document_id": effective_doc_id}),
                        uuid.UUID(operation_id),
                    )
            except Exception:
                logger.warning("Failed to persist generated document_id", exc_info=True)

    # --- Delta retain: check if we can skip unchanged chunks ---
    if is_first_batch:
        delta_result = await _try_delta_retain(
            pool,
            embeddings_model,
            llm_config,
            entity_resolver,
            format_date_fn,
            bank_id,
            contents_dicts,
            contents,
            config,
            effective_doc_id,
            fact_type_override,
            document_tags,
            agent_name,
            log_buffer,
            start_time,
            operation_id,
            schema,
            outbox_callback,
            db_semaphore,
        )
        if delta_result is not None:
            return delta_result

    # --- Always use the streaming pipeline (producer-consumer batching) ---
    # Even small documents go through the same path — they just end up as a
    # single batch. This eliminates the maintenance burden of two separate
    # retain code paths.
    chunk_batch_size = getattr(config, "retain_chunk_batch_size", 100)
    chunk_size = getattr(config, "retain_chunk_size", 3000)
    all_pre_chunks = []
    for content in contents:
        content_chunks = fact_extraction.chunk_text(content.content, chunk_size)
        all_pre_chunks.extend(content_chunks)

    total_pre_chunks = len(all_pre_chunks)
    num_batches = (total_pre_chunks + chunk_batch_size - 1) // chunk_batch_size if total_pre_chunks > 0 else 1
    log_buffer.append(
        f"[streaming] {total_pre_chunks} chunks, batch_size {chunk_batch_size} — "
        f"{num_batches} batch{'es' if num_batches != 1 else ''}"
    )

    return await _streaming_retain_batch(
        pool=pool,
        embeddings_model=embeddings_model,
        llm_config=llm_config,
        entity_resolver=entity_resolver,
        format_date_fn=format_date_fn,
        bank_id=bank_id,
        contents_dicts=contents_dicts,
        contents=contents,
        config=config,
        document_id=effective_doc_id,
        is_first_batch=is_first_batch,
        fact_type_override=fact_type_override,
        document_tags=document_tags,
        agent_name=agent_name,
        log_buffer=log_buffer,
        start_time=start_time,
        all_pre_chunks=all_pre_chunks,
        chunk_batch_size=chunk_batch_size,
        operation_id=operation_id,
        schema=schema,
        outbox_callback=outbox_callback,
        db_semaphore=db_semaphore,
    )


# ---------------------------------------------------------------------------
# Final semantic ANN pass (post-commit)
# ---------------------------------------------------------------------------

_ANN_CHUNK_SIZE = 1000  # Max seeds per ANN query — smaller chunks avoid timeouts
_ANN_PARALLELISM = 4  # Max concurrent ANN chunks to avoid pool saturation


async def _run_final_semantic_ann(
    pool,
    bank_id: str,
    unit_ids: list[str],
    log_buffer: list[str],
) -> None:
    """
    Create semantic links for all committed units in a single pass.

    Called after all streaming batches have committed. Loads embeddings and
    fact_types from the database, then runs ANN in chunks of _ANN_CHUNK_SIZE
    seeds. This replaces per-batch within-batch + fire-and-forget ANN with
    one efficient pass that sees the full bank.
    """
    from .link_utils import _bulk_insert_links, compute_semantic_links_ann

    if not unit_ids:
        return

    # Load embeddings and fact_types for all committed units
    load_start = time.time()
    async with acquire_with_retry(pool) as conn:
        rows = await conn.fetch(
            f"""
            SELECT id::text, embedding::text, fact_type
            FROM {fq_table("memory_units")}
            WHERE bank_id = $1 AND id = ANY($2::uuid[])
            ORDER BY id
            """,
            bank_id,
            unit_ids,
        )

    if not rows:
        log_buffer.append("[streaming] Final ANN: no units found in DB (unexpected)")
        return

    # Build lookup: unit_id -> (embedding_text, fact_type)
    unit_map: dict[str, tuple[str, str]] = {}
    for row in rows:
        unit_map[row["id"]] = (row["embedding"], row["fact_type"])

    # Filter to units that have embeddings
    ann_unit_ids = []
    ann_embeddings = []
    ann_fact_types = []
    for uid in unit_ids:
        if uid in unit_map and unit_map[uid][0] is not None:
            ann_unit_ids.append(uid)
            ann_embeddings.append(unit_map[uid][0])  # embedding as text (for temp table)
            ann_fact_types.append(unit_map[uid][1])

    log_buffer.append(
        f"[streaming] Final ANN: loaded {len(ann_unit_ids)} units with embeddings in {time.time() - load_start:.3f}s"
    )

    if not ann_unit_ids:
        return

    # Process in parallel chunks — each chunk runs ANN query + INSERT on its own connection.
    # Parallelism bounded by _ANN_PARALLELISM to avoid saturating the connection pool.
    num_chunks = (len(ann_unit_ids) + _ANN_CHUNK_SIZE - 1) // _ANN_CHUNK_SIZE
    ann_semaphore = asyncio.Semaphore(_ANN_PARALLELISM)
    chunk_link_counts: list[int] = [0] * num_chunks

    async def _process_ann_chunk(chunk_idx: int) -> None:
        chunk_start = chunk_idx * _ANN_CHUNK_SIZE
        chunk_end = min(chunk_start + _ANN_CHUNK_SIZE, len(ann_unit_ids))
        chunk_ids = ann_unit_ids[chunk_start:chunk_end]
        chunk_embs = ann_embeddings[chunk_start:chunk_end]
        chunk_ftypes = ann_fact_types[chunk_start:chunk_end]

        async with ann_semaphore:
            t0 = time.time()
            async with acquire_with_retry(pool) as conn:
                await conn.execute("SET statement_timeout = '300s'")
                ann_links = await compute_semantic_links_ann(
                    conn,
                    bank_id,
                    chunk_ids,
                    chunk_embs,
                    fact_types=chunk_ftypes,
                    top_k=20,  # Recall uses at most 20 neighbors
                    log_buffer=log_buffer,
                )
                if ann_links:
                    await _bulk_insert_links(conn, ann_links, bank_id=bank_id)
                chunk_link_counts[chunk_idx] = len(ann_links)
                await conn.execute("RESET statement_timeout")
            logger.info(
                f"[streaming] Final ANN chunk {chunk_idx + 1}/{num_chunks}: "
                f"{len(ann_links)} links in {time.time() - t0:.3f}s"
            )

    await asyncio.gather(*[_process_ann_chunk(i) for i in range(num_chunks)])
    total_links = sum(chunk_link_counts)
    log_buffer.append(f"[streaming] Final ANN: {total_links} total semantic links")


# ---------------------------------------------------------------------------
# Streaming chunk batching
# ---------------------------------------------------------------------------


async def _streaming_retain_batch(
    pool,
    embeddings_model,
    llm_config,
    entity_resolver,
    format_date_fn,
    bank_id: str,
    contents_dicts: list[RetainContentDict],
    contents: list[RetainContent],
    config,
    document_id: str | None,
    is_first_batch: bool,
    fact_type_override: str | None,
    document_tags: list[str] | None,
    agent_name: str,
    log_buffer: list[str],
    start_time: float,
    all_pre_chunks: list[str],
    chunk_batch_size: int,
    operation_id: str | None = None,
    schema: str | None = None,
    outbox_callback: Callable[["asyncpg.Connection"], Awaitable[None]] | None = None,
    db_semaphore: "asyncio.Semaphore | None" = None,
) -> tuple[list[list[str]], TokenUsage]:
    """
    Process a large document in streaming mini-batches to bound memory usage.

    Instead of extracting facts from ALL chunks at once (which can OOM for 17k+
    chunk documents), this splits the pre-chunked content into batches of
    ``chunk_batch_size`` chunks.  Each mini-batch goes through the full
    extract -> embed -> Phase 1/2/3 pipeline and commits to the DB before the
    next batch starts, so memory is released between batches.

    All mini-batches share the same ``document_id`` so that:
    - Delta retain can detect already-committed chunks on retry
    - The document row tracks the full content
    - Chunks are associated with the correct document
    """
    total_chunks = len(all_pre_chunks)
    total_usage = TokenUsage()
    all_unit_ids: list[str] = []

    # document_id is already resolved by retain_batch (includes recovery from
    # operation result_metadata on retry).
    effective_doc_id = document_id

    # Use the first content item as the template for metadata (context, event_date, etc.)
    template_content = contents[0] if contents else RetainContent(content="")

    # Load existing chunk hashes BEFORE document tracking to detect recovery.
    # If chunks exist AND the document content hash matches, this is a retry of
    # the same content — preserve existing data. If content differs, this is an
    # update — cascade-delete old data and start fresh.
    existing_chunk_hashes: set[str] = set()
    combined_content = "\n".join([c.get("content", "") for c in contents_dicts])
    new_content_hash = hashlib.sha256(combined_content.encode()).hexdigest()
    is_recovery = False

    try:
        async with acquire_with_retry(pool) as conn:
            # Check if document exists with matching content hash
            doc_row = await conn.fetchrow(
                f"SELECT content_hash FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
                effective_doc_id,
                bank_id,
            )
            if doc_row and doc_row["content_hash"] == new_content_hash:
                # Same content — load chunk hashes for recovery skip
                existing_rows = await chunk_storage.load_existing_chunks(conn, bank_id, effective_doc_id)
                existing_chunk_hashes = {c.content_hash for c in existing_rows if c.content_hash}
                if existing_chunk_hashes:
                    is_recovery = True
                    log_buffer.append(
                        f"[streaming] RECOVERY: found {len(existing_chunk_hashes)} already-committed chunks — "
                        f"will skip matching and preserve existing data"
                    )
    except Exception:
        pass  # If we can't load, just process all chunks

    # Create/update the document row.
    retain_params, merged_tags = _build_retain_params(contents_dicts, document_tags)
    async with acquire_with_retry(pool) as conn:
        async with conn.transaction():
            if is_recovery:
                # Recovery: same content, partially committed — preserve existing data
                await fact_storage.upsert_document_metadata(
                    conn, bank_id, effective_doc_id, combined_content, retain_params, merged_tags
                )
                log_buffer.append(
                    f"[streaming] Document {effective_doc_id} updated (recovery, preserving existing chunks)"
                )
            else:
                # Fresh or update: cascade-delete old data if document exists
                await fact_storage.handle_document_tracking(
                    conn, bank_id, effective_doc_id, combined_content, is_first_batch, retain_params, merged_tags
                )
                log_buffer.append(f"[streaming] Document {effective_doc_id} tracked (full content)")

    # ---------------------------------------------------------------------------
    # Producer-consumer pipeline: LLM extraction runs concurrently with DB writes
    # ---------------------------------------------------------------------------
    num_batches = (total_chunks + chunk_batch_size - 1) // chunk_batch_size

    # Queue for enriched chunks (extracted facts + embeddings).
    # Buffer up to 2x batch_size items so the producer can stay ahead of the consumer.
    chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=chunk_batch_size * 2)

    # Shared mutable state for the producer to report skipped chunks and usage
    producer_error: list[BaseException] = []

    # ---- LLM Producer ----
    # Fires all chunk extractions as concurrent tasks (bounded by the LLM
    # semaphore inside fact_extraction to 32 concurrent).  As each completes
    # it pushes the enriched result into the queue for the DB consumer.
    async def _llm_producer() -> None:
        async def _extract_one(global_idx: int, chunk_text: str) -> None:
            content = RetainContent(
                content=chunk_text,
                context=template_content.context,
                event_date=template_content.event_date,
                metadata=template_content.metadata,
                entities=template_content.entities,
                tags=template_content.tags,
                observation_scopes=template_content.observation_scopes,
            )
            extracted, processed, chunk_meta, usage = await _extract_and_embed(
                [content],
                llm_config,
                agent_name,
                config,
                embeddings_model,
                format_date_fn,
                fact_type_override,
                log_buffer,
                pool,
                operation_id,
                schema,
            )
            await chunk_queue.put((global_idx, content, extracted, processed, chunk_meta, usage))

        tasks: list[asyncio.Task] = []
        skipped_total = 0
        for i, chunk_text in enumerate(all_pre_chunks):
            chunk_hash = chunk_storage.compute_chunk_hash(chunk_text)
            if chunk_hash in existing_chunk_hashes:
                skipped_total += 1
                continue
            tasks.append(asyncio.create_task(_extract_one(i, chunk_text)))

        if skipped_total > 0:
            log_buffer.append(f"[streaming] Producer: skipped {skipped_total}/{total_chunks} already-committed chunks")

        # Wait for all extractions; collect exceptions
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException):
                producer_error.append(r)

        # Signal the consumer that production is done
        await chunk_queue.put(None)

    # ---- DB Consumer ----
    # Drains enriched chunks from the queue in batches and runs
    # Phase 1 (entity resolution) -> Phase 2 (write txn) -> Phase 3 (ANN fire-and-forget).
    async def _db_consumer() -> None:
        batch: list[tuple] = []
        global_chunk_offset = 0
        consumer_batch_idx = 0

        while True:
            item = await chunk_queue.get()
            if item is None:
                # Process any remaining items
                if batch:
                    await _process_db_batch(
                        batch,
                        global_chunk_offset,
                        consumer_batch_idx,
                        is_last=True,
                    )
                break

            batch.append(item)

            if len(batch) >= chunk_batch_size:
                await _process_db_batch(
                    batch,
                    global_chunk_offset,
                    consumer_batch_idx,
                    is_last=False,
                )
                global_chunk_offset += len(batch)
                consumer_batch_idx += 1
                batch = []

    async def _process_db_batch(
        batch: list[tuple],
        global_chunk_offset: int,
        consumer_batch_idx: int,
        is_last: bool,
    ) -> None:
        """Run Phase 1 + Phase 2 + Phase 3 for a batch of pre-extracted chunks."""
        # Combine results from individual chunk extractions
        batch_contents: list[RetainContent] = []
        batch_extracted: list = []
        batch_processed: list[ProcessedFact] = []
        batch_chunk_meta: list[ChunkMetadata] = []
        batch_usage = TokenUsage()

        for global_idx, content, extracted, processed, chunk_meta, usage in batch:
            content_idx_in_batch = len(batch_contents)
            # Adjust chunk indices to global offsets and remap content_index
            for fact in extracted:
                fact.content_index = content_idx_in_batch
                if fact.chunk_index is not None:
                    fact.chunk_index = global_chunk_offset + content_idx_in_batch
            for pf in processed:
                pf.content_index = content_idx_in_batch
            for cm in chunk_meta:
                cm.chunk_index = global_chunk_offset + content_idx_in_batch

            batch_contents.append(content)
            batch_extracted.extend(extracted)
            batch_processed.extend(processed)
            batch_chunk_meta.extend(chunk_meta)
            batch_usage = batch_usage + usage

        nonlocal total_usage
        total_usage = total_usage + batch_usage

        if not batch_extracted:
            log_buffer.append(
                f"[streaming] Consumer batch {consumer_batch_idx + 1}: "
                f"0 facts extracted from {len(batch)} chunks, skipping"
            )
            return

        log_buffer.append(
            f"[streaming] Consumer batch {consumer_batch_idx + 1}: "
            f"processing {len(batch_extracted)} facts from {len(batch)} chunks"
        )

        async def _run_mini_batch_db_work() -> None:
            entity_resolver.discard_pending_stats()
            mb_start = time.time()

            # Phase 1 — Entity Resolution only (no ANN — deferred to Phase 3)
            p1_start = time.time()
            phase1 = await _pre_resolve_phase1(
                pool,
                entity_resolver,
                bank_id,
                batch_contents,
                batch_processed,
                config,
                log_buffer,
                skip_semantic_ann=True,
            )

            logger.info(f"[streaming] Phase 1 (entity resolution): {time.time() - p1_start:.3f}s")

            # Phase 2 — Write transaction (within-batch semantic links only)
            p2_start = time.time()
            async with acquire_with_retry(pool) as conn:
                async with conn.transaction():
                    # Store chunks with correct global indices
                    step_start = time.time()
                    chunk_id_map = {}
                    if batch_chunk_meta:
                        chunk_id_map = await chunk_storage.store_chunks_batch(
                            conn, bank_id, effective_doc_id, batch_chunk_meta
                        )
                        log_buffer.append(
                            f"  Store chunks: {len(batch_chunk_meta)} chunks in {time.time() - step_start:.3f}s"
                        )

                    # Map document_id and chunk_id to processed facts
                    for fact, processed_fact in zip(batch_extracted, batch_processed):
                        processed_fact.document_id = effective_doc_id
                        if batch_chunk_meta and fact.chunk_index is not None:
                            chunk_id = chunk_id_map.get(fact.chunk_index)
                            if chunk_id:
                                processed_fact.chunk_id = chunk_id

                    # Insert facts and links — skip semantic links entirely in streaming
                    # mode; they are created in a single final ANN pass after all batches.
                    batch_result_ids, phase3_ctx = await _insert_facts_and_links(
                        conn,
                        entity_resolver,
                        bank_id,
                        batch_contents,
                        batch_extracted,
                        batch_processed,
                        config,
                        log_buffer,
                        resolved_entity_ids=phase1.entities.resolved_entity_ids,
                        entity_to_unit=phase1.entities.entity_to_unit,
                        unit_to_entity_ids=phase1.entities.unit_to_entity_ids,
                        semantic_ann_links=[],
                        skip_semantic_links=True,
                        outbox_callback=outbox_callback if is_last else None,
                    )

                logger.info(f"[streaming] Phase 2 (write txn): {time.time() - p2_start:.3f}s")

                # Best-effort: entity viz + stats (fast, not semantic ANN)
                try:
                    await entity_resolver.flush_pending_stats()
                    await _build_and_insert_entity_links_phase3(pool, entity_resolver, bank_id, phase3_ctx, log_buffer)
                except Exception:
                    logger.warning(f"Phase 3 stats (consumer batch {consumer_batch_idx + 1}) failed", exc_info=True)

            logger.info(
                f"[streaming] Consumer batch {consumer_batch_idx + 1} total "
                f"(excluding fire-and-forget): {time.time() - mb_start:.3f}s"
            )

            # Collect unit_ids from this batch
            for content_ids in batch_result_ids:
                all_unit_ids.extend(content_ids)

        if db_semaphore is not None:
            async with db_semaphore:
                await _run_mini_batch_db_work()
        else:
            await _run_mini_batch_db_work()

    # ---------------------------------------------------------------------------
    # Check if facts are already committed (recovery from previous crash).
    # If so, skip extraction+writes and jump straight to final ANN pass.
    # ---------------------------------------------------------------------------
    facts_already_committed = False
    if operation_id:
        try:
            async with acquire_with_retry(pool) as conn:
                row = await conn.fetchrow(
                    f"SELECT result_metadata FROM {fq_table('async_operations')} WHERE operation_id = $1",
                    uuid.UUID(operation_id),
                )
                if row and row["result_metadata"]:
                    meta = (
                        row["result_metadata"]
                        if isinstance(row["result_metadata"], dict)
                        else json.loads(row["result_metadata"])
                    )
                    if meta.get("facts_committed"):
                        facts_already_committed = True
                        log_buffer.append(
                            f"[streaming] Recovery: facts already committed ({meta.get('unit_ids_count', '?')} units), "
                            f"skipping to final ANN pass"
                        )
        except Exception:
            logger.warning("Failed to check operation recovery state", exc_info=True)

    if not facts_already_committed:
        # Run producer and consumer concurrently
        await asyncio.gather(_llm_producer(), _db_consumer())

        # Propagate producer errors (e.g. LLM failures)
        if producer_error:
            raise producer_error[0]

        # Mark facts as committed in operation metadata (crash recovery checkpoint)
        if operation_id and all_unit_ids:
            try:
                async with acquire_with_retry(pool) as conn:
                    await conn.execute(
                        f"""
                        UPDATE {fq_table("async_operations")}
                        SET result_metadata = result_metadata || $1::jsonb, updated_at = now()
                        WHERE operation_id = $2
                        """,
                        json.dumps({"facts_committed": True, "unit_ids_count": len(all_unit_ids)}),
                        uuid.UUID(operation_id),
                    )
                log_buffer.append(f"[streaming] Checkpoint: {len(all_unit_ids)} facts committed, ANN pass next")
            except Exception:
                logger.warning("Failed to save facts_committed checkpoint", exc_info=True)
    else:
        # Recovery path: load committed unit IDs from DB
        async with acquire_with_retry(pool) as conn:
            rows = await conn.fetch(
                f"""
                SELECT id::text FROM {fq_table("memory_units")}
                WHERE bank_id = $1 AND document_id = $2
                ORDER BY created_at
                """,
                bank_id,
                effective_doc_id,
            )
            all_unit_ids = [row["id"] for row in rows]
            log_buffer.append(f"[streaming] Recovery: loaded {len(all_unit_ids)} unit IDs from DB")

    # ---------------------------------------------------------------------------
    # Final ANN pass: create semantic links for ALL committed units at once.
    # This replaces per-batch within-batch + fire-and-forget ANN with a single
    # efficient pass after all facts are in the database.
    # ---------------------------------------------------------------------------
    if all_unit_ids:
        ann_start = time.time()
        await _run_final_semantic_ann(pool, bank_id, all_unit_ids, log_buffer)
        log_buffer.append(f"[streaming] Final ANN pass: {time.time() - ann_start:.3f}s for {len(all_unit_ids)} units")

    total_time = time.time() - start_time
    log_buffer.append(f"{'=' * 60}")
    log_buffer.append(
        f"STREAMING RETAIN COMPLETE: {len(all_unit_ids)} units across {num_batches} batches in {total_time:.3f}s"
    )
    log_buffer.append(f"Document: {effective_doc_id}")
    log_buffer.append(f"{'=' * 60}")
    logger.info("\n" + "\n".join(log_buffer) + "\n")

    # Map all unit_ids back to the original content items.
    # For streaming mode with a single document, all units belong to content 0.
    result_unit_ids = [all_unit_ids] + [[] for _ in contents[1:]]
    return result_unit_ids, total_usage


# ---------------------------------------------------------------------------
# Delta retain
# ---------------------------------------------------------------------------


async def _try_delta_retain(
    pool,
    embeddings_model,
    llm_config,
    entity_resolver,
    format_date_fn,
    bank_id,
    contents_dicts,
    contents,
    config,
    document_id,
    fact_type_override,
    document_tags,
    agent_name,
    log_buffer,
    start_time,
    operation_id,
    schema,
    outbox_callback,
    db_semaphore: "asyncio.Semaphore | None" = None,
):
    """
    Attempt delta retain for a document upsert. Returns result tuple if delta
    was performed, or None to fall back to full retain.
    """
    # Need a single document_id
    effective_doc_id = document_id
    if not effective_doc_id:
        doc_ids = {item.get("document_id") for item in contents_dicts if item.get("document_id")}
        if len(doc_ids) != 1:
            return None
        effective_doc_id = doc_ids.pop()

    # Load existing chunks
    async with acquire_with_retry(pool) as conn:
        existing_chunks = await chunk_storage.load_existing_chunks(conn, bank_id, effective_doc_id)

    if not existing_chunks:
        return None

    if any(c.content_hash is None for c in existing_chunks):
        logger.info(f"Delta retain skipped for {effective_doc_id}: existing chunks lack content_hash (pre-migration)")
        return None

    # Chunk new content and classify changes
    step_start = time.time()
    new_chunks_with_contents = _chunk_contents_for_delta(contents, config)
    log_buffer.append(
        f"[delta] Chunked new content: {len(new_chunks_with_contents)} chunks in {time.time() - step_start:.3f}s"
    )

    existing_by_index = {c.chunk_index: c for c in existing_chunks}
    new_hashes = {idx: chunk_storage.compute_chunk_hash(text) for idx, text in new_chunks_with_contents.items()}

    unchanged_indices, changed_indices, new_indices, removed_indices = [], [], [], []
    for idx, new_hash in new_hashes.items():
        existing = existing_by_index.get(idx)
        if existing and existing.content_hash == new_hash:
            unchanged_indices.append(idx)
        elif existing:
            changed_indices.append(idx)
        else:
            new_indices.append(idx)
    for idx in existing_by_index:
        if idx not in new_hashes:
            removed_indices.append(idx)

    log_buffer.append(
        f"[delta] Chunk diff: {len(unchanged_indices)} unchanged, "
        f"{len(changed_indices)} changed, {len(new_indices)} new, "
        f"{len(removed_indices)} removed"
    )

    if not unchanged_indices:
        logger.info(f"Delta retain: no unchanged chunks for {effective_doc_id}, falling back to full retain")
        return None

    chunks_to_process = changed_indices + new_indices

    if not chunks_to_process and not removed_indices:
        # Nothing changed — just update document metadata/tags
        log_buffer.append("[delta] No chunk changes detected — updating document metadata only")
        return await _delta_metadata_only(
            pool,
            bank_id,
            contents_dicts,
            contents,
            effective_doc_id,
            document_tags,
            log_buffer,
            start_time,
            outbox_callback,
        )

    # Build content items for only the changed/new chunks
    delta_contents, delta_chunk_map = _build_delta_contents(contents, new_chunks_with_contents, chunks_to_process)

    if not delta_contents:
        return await _delta_metadata_only(
            pool,
            bank_id,
            contents_dicts,
            contents,
            effective_doc_id,
            document_tags,
            log_buffer,
            start_time,
            outbox_callback,
        )

    # Extract facts and generate embeddings (shared pipeline)
    extracted_facts, processed_facts, new_chunk_metadata, usage = await _extract_and_embed(
        delta_contents,
        llm_config,
        agent_name,
        config,
        embeddings_model,
        format_date_fn,
        fact_type_override,
        log_buffer,
        pool,
        operation_id,
        schema,
    )

    # Database transaction
    result_unit_ids: list[list[str]] = []
    log_buffer_pre_db = len(log_buffer)

    async def _run_delta_db_work() -> None:
        nonlocal result_unit_ids
        del log_buffer[log_buffer_pre_db:]
        for pf in processed_facts:
            pf.document_id = None
            pf.chunk_id = None
        entity_resolver.discard_pending_stats()

        # PHASE 1 — Entity Resolution + Semantic ANN (separate connection, read-heavy)
        phase1 = await _pre_resolve_phase1(
            pool, entity_resolver, bank_id, delta_contents, processed_facts, config, log_buffer
        )

        # PHASE 2 — Core Write Transaction (atomic)
        async with acquire_with_retry(pool) as conn:
            async with conn.transaction():
                # Update document metadata (no delete)
                step_start = time.time()
                combined_content = "\n".join([c.get("content", "") for c in contents_dicts])
                retain_params, merged_tags = _build_retain_params(contents_dicts, document_tags)
                await fact_storage.upsert_document_metadata(
                    conn,
                    bank_id,
                    effective_doc_id,
                    combined_content,
                    retain_params,
                    merged_tags,
                )
                log_buffer.append(f"  Document metadata update in {time.time() - step_start:.3f}s")

                # Delete changed and removed chunks (cascades to memory_units and links)
                step_start = time.time()
                chunks_to_delete = [
                    existing_by_index[idx].chunk_id
                    for idx in changed_indices + removed_indices
                    if idx in existing_by_index
                ]
                await chunk_storage.delete_chunks_by_ids(conn, chunks_to_delete)
                log_buffer.append(
                    f"  Deleted {len(chunks_to_delete)} chunks "
                    f"({len(changed_indices)} changed + {len(removed_indices)} removed) "
                    f"in {time.time() - step_start:.3f}s"
                )

                # Update tags on unchanged chunks' memory units
                step_start = time.time()
                updated_count = await fact_storage.update_memory_units_tags(
                    conn, bank_id, effective_doc_id, merged_tags
                )
                log_buffer.append(
                    f"  Updated tags on {updated_count} existing memory units in {time.time() - step_start:.3f}s"
                )

                # Store new/changed chunks
                step_start = time.time()
                chunk_id_map_by_doc = {}
                if new_chunk_metadata:
                    remapped_chunks = [
                        ChunkMetadata(
                            chunk_text=cm.chunk_text,
                            fact_count=cm.fact_count,
                            content_index=cm.content_index,
                            chunk_index=delta_chunk_map.get(cm.chunk_index, cm.chunk_index),
                        )
                        for cm in new_chunk_metadata
                    ]
                    chunk_id_map = await chunk_storage.store_chunks_batch(
                        conn, bank_id, effective_doc_id, remapped_chunks
                    )
                    for chunk_idx, chunk_id in chunk_id_map.items():
                        chunk_id_map_by_doc[(effective_doc_id, chunk_idx)] = chunk_id
                    log_buffer.append(
                        f"  Stored {len(remapped_chunks)} new/changed chunks in {time.time() - step_start:.3f}s"
                    )

                # Map chunk_ids and document_ids to processed facts
                for ef, pf in zip(extracted_facts, processed_facts):
                    pf.document_id = effective_doc_id
                    if ef.chunk_index is not None:
                        original_idx = delta_chunk_map.get(ef.chunk_index, ef.chunk_index)
                        chunk_id = chunk_id_map_by_doc.get((effective_doc_id, original_idx))
                        if chunk_id:
                            pf.chunk_id = chunk_id

                # Insert facts and retrieval-critical links.
                # Use delta_contents (the changed/new chunks) as the content list,
                # since extracted_facts have content_index relative to delta_contents.
                result_unit_ids, phase3_ctx = await _insert_facts_and_links(
                    conn,
                    entity_resolver,
                    bank_id,
                    delta_contents,
                    extracted_facts,
                    processed_facts,
                    config,
                    log_buffer,
                    resolved_entity_ids=phase1.entities.resolved_entity_ids,
                    entity_to_unit=phase1.entities.entity_to_unit,
                    unit_to_entity_ids=phase1.entities.unit_to_entity_ids,
                    semantic_ann_links=phase1.semantic_ann_links,
                    outbox_callback=outbox_callback,
                )

            # PHASE 3 — Best-Effort Display Data (post-transaction)
            try:
                await entity_resolver.flush_pending_stats()
                await _build_and_insert_entity_links_phase3(pool, entity_resolver, bank_id, phase3_ctx, log_buffer)
            except Exception:
                logger.warning("Phase 3 (best-effort display data) failed — retrieval unaffected", exc_info=True)

            total_time = time.time() - start_time
            log_buffer.append(f"{'=' * 60}")
            log_buffer.append(
                f"DELTA RETAIN COMPLETE: {len(processed_facts)} new units, "
                f"{len(unchanged_indices)} chunks unchanged in {total_time:.3f}s"
            )
            log_buffer.append(f"Document: {effective_doc_id}")
            log_buffer.append(f"{'=' * 60}")
            logger.info("\n" + "\n".join(log_buffer) + "\n")

    if db_semaphore is not None:
        async with db_semaphore:
            await _run_delta_db_work()
    else:
        await _run_delta_db_work()
    return result_unit_ids, usage


async def _delta_metadata_only(
    pool,
    bank_id,
    contents_dicts,
    contents,
    document_id,
    document_tags,
    log_buffer,
    start_time,
    outbox_callback,
):
    """Handle the case where no chunks changed — just update document metadata and tags."""
    async with acquire_with_retry(pool) as conn:
        async with conn.transaction():
            combined_content = "\n".join([c.get("content", "") for c in contents_dicts])
            retain_params, merged_tags = _build_retain_params(contents_dicts, document_tags)
            await fact_storage.upsert_document_metadata(
                conn,
                bank_id,
                document_id,
                combined_content,
                retain_params,
                merged_tags,
            )
            await fact_storage.update_memory_units_tags(conn, bank_id, document_id, merged_tags)
            if outbox_callback:
                await outbox_callback(conn)

    total_time = time.time() - start_time
    log_buffer.append(f"DELTA RETAIN (no changes): metadata updated in {total_time:.3f}s")
    logger.info("\n" + "\n".join(log_buffer) + "\n")
    return [[] for _ in contents], TokenUsage()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_contents(contents_dicts: list[RetainContentDict], document_tags: list[str] | None) -> list[RetainContent]:
    """Convert content dicts to RetainContent objects."""
    contents = []
    for item in contents_dicts:
        item_tags = item.get("tags", []) or []
        merged_tags = list(set(item_tags + (document_tags or [])))

        if "event_date" in item and item["event_date"] is None:
            event_date_value = None
        elif item.get("event_date"):
            event_date_value = parse_datetime_flexible(item["event_date"])
        else:
            event_date_value = utcnow()

        content = RetainContent(
            content=item["content"],
            context=item.get("context", ""),
            event_date=event_date_value,
            metadata=item.get("metadata", {}),
            entities=item.get("entities", []),
            tags=merged_tags,
            observation_scopes=item.get("observation_scopes"),
        )
        contents.append(content)
    return contents


def _chunk_contents_for_delta(contents: list[RetainContent], config) -> dict[int, str]:
    """
    Chunk contents the same way the streaming path does, returning a map of
    global_chunk_index -> chunk_text.

    Must use the same chunk_size as the streaming path (default 3000) so that
    chunk boundaries match and delta can detect unchanged chunks.
    Previously defaulted to 120000, causing all chunks to appear changed on retry.
    """
    result = {}
    global_chunk_idx = 0
    for content in contents:
        chunk_size = getattr(config, "retain_chunk_size", 3000)
        chunks = fact_extraction.chunk_text(content.content, chunk_size)
        for chunk_text in chunks:
            result[global_chunk_idx] = chunk_text
            global_chunk_idx += 1
    return result


def _build_delta_contents(
    original_contents: list[RetainContent],
    new_chunks_with_contents: dict[int, str],
    chunks_to_process: list[int],
) -> tuple[list[RetainContent], dict[int, int]]:
    """
    Build RetainContent items containing only the chunks that need processing.

    Returns:
        - List of RetainContent items (one per chunk to process)
        - Map of delta_chunk_index -> original_chunk_index
    """
    if not chunks_to_process or not original_contents:
        return [], {}

    template_content = original_contents[0]
    delta_contents = []
    delta_chunk_map = {}

    for original_chunk_idx in sorted(chunks_to_process):
        chunk_text = new_chunks_with_contents.get(original_chunk_idx)
        if not chunk_text:
            continue
        delta_content = RetainContent(
            content=chunk_text,
            context=template_content.context,
            event_date=template_content.event_date,
            metadata=template_content.metadata,
            entities=template_content.entities,
            tags=template_content.tags,
            observation_scopes=template_content.observation_scopes,
        )
        delta_contents.append(delta_content)
        delta_chunk_map[len(delta_contents) - 1] = original_chunk_idx

    return delta_contents, delta_chunk_map


def _map_results_to_contents(
    contents: list[RetainContent],
    extracted_facts: list[ExtractedFact],
    unit_ids: list[str],
) -> list[list[str]]:
    """Map created unit IDs back to original content items."""
    facts_by_content: dict[int, list[int]] = {i: [] for i in range(len(contents))}
    for i, fact in enumerate(extracted_facts):
        facts_by_content[fact.content_index].append(i)

    result_unit_ids = []
    unit_idx = 0
    for content_index in range(len(contents)):
        content_unit_ids = []
        for _ in facts_by_content[content_index]:
            content_unit_ids.append(unit_ids[unit_idx])
            unit_idx += 1
        result_unit_ids.append(content_unit_ids)

    return result_unit_ids
