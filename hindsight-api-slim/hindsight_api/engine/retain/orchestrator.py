"""
Main orchestrator for the retain pipeline.

Coordinates all retain pipeline modules to store memories efficiently.
"""

import asyncio
import dataclasses
import hashlib
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ...extensions.memory_defense import (
    DefenseAction,
    DefenseDecision,
    MemoryDefenseExtension,
    apply_redaction,
    parse_policy,
)
from ...metrics import get_metrics_collector
from ...worker.stage import set_stage
from ..db_utils import acquire_with_retry
from ..memory_engine import count_tokens, fq_table


@dataclass
class BlockedViolation:
    """One item blocked by the Memory Defense policy (surfaced in the 422 body)."""

    index: int
    detector: str | None
    message: str


class MemoryDefenseAllBlockedError(Exception):
    """Raised when every item in a retain batch is blocked by the Memory Defense policy."""

    def __init__(self, violations: list[BlockedViolation]) -> None:
        self.violations = violations
        super().__init__(f"all {len(violations)} items blocked by Memory Defense policy")


def utcnow():
    """Get current UTC time."""
    return datetime.now(UTC)


def redact_document_body(body: str, config: Any) -> str:
    """Apply Memory Defense redaction to a document body.

    Per-item screening only scrubs the chunked content that goes through
    `screen()`. A `document_body_override` (the full original text of an
    oversized item — see `_split_contents_into_sub_batches`) never goes through
    `screen()` and would otherwise persist verbatim into
    `documents.original_text`, so the splitting caller runs it through this
    redactor once, before handing the same body to every slice.

    **Callers of this module must pass an override that is already screened.**
    The retain path here deliberately does not re-screen it: every slice of an
    oversized item carries the identical body, so re-screening would rescan the
    whole document once per sub-batch (issue #3282).
    """
    # No try/except around the parse. A malformed policy must fail the retain, not
    # quietly skip screening: this is a security control, so fail-open is the wrong
    # default even for an unreachable state (the HTTP layer 422s a bad policy on
    # write, so one can only reach the store by a direct DB edit).
    #
    # The blanket ``except Exception: return body`` this replaces could not buy a
    # successful retain anyway — ``retain_batch`` parses the very same
    # ``config.memory_defense`` unguarded before extracting, so anything raising here
    # raises there moments later and the retain fails with nothing persisted. All the
    # catch did was swallow the traceback that says *why* the policy did not parse,
    # and hide that screening had been skipped first. It also made the large-batch
    # path (the only caller) diverge from the small-batch path, which has always let
    # the same error propagate.
    policy = parse_policy(config.memory_defense)
    if not policy.enabled:
        return body
    if not any(r.on == "sensitive_data" for r in policy.rules):
        return body
    return apply_redaction(body).content


def _is_strict_append_of_stored_document(
    stored_original_text: str | None,
    document_body_override: str | None,
) -> bool:
    """Return whether an oversized document body strictly appends stored text.

    ``documents.original_text`` is sanitized before persistence (the override
    arrives Memory Defense redacted — see ``redact_document_body``), so apply
    the same sanitization before comparing it with the stored prefix.
    """
    if stored_original_text is None or document_body_override is None:
        return False

    sanitized_body = fact_extraction._sanitize_text(document_body_override) or ""
    return len(sanitized_body) > len(stored_original_text) and sanitized_body.startswith(stored_original_text)


async def _fire_memory_defense_webhook(
    webhook_manager: Any,
    *,
    conn: Any,
    schema: str | None,
    bank_id: str,
    operation_id: str | None,
    document_id: str | None,
    decision: DefenseDecision,
) -> None:
    """Fire a memory_defense.triggered webhook for a non-allow decision.

    No-op when no webhook manager is wired or none is subscribed. Delivery
    failures are swallowed so screening never blocks a retain.
    """
    if webhook_manager is None:
        return
    try:
        from ...webhooks import (
            MemoryDefenseEventData,
            MemoryDefenseHit,
            WebhookEvent,
            WebhookEventType,
        )

        # Translate per-match raw dicts on the decision into MemoryDefenseHit
        # entries on the wire. The decision's hits list is already fingerprinted
        # by apply_redaction (the raw value never lands in hits, by contract),
        # so this is purely a shape conversion. None when no per-hit data is
        # available so receivers can distinguish "no preview info" from
        # "scanned, nothing matched" (the latter wouldn't be a webhook delivery
        # in the first place).
        decision_hits = getattr(decision, "hits", None) or []
        hits: list[MemoryDefenseHit] | None = [
            MemoryDefenseHit(
                detector=str(h.get("detector") or ""),
                preview=str(h.get("preview") or ""),
            )
            for h in decision_hits
            if h.get("detector") and h.get("preview")
        ] or None

        event = WebhookEvent(
            event=WebhookEventType.MEMORY_DEFENSE_TRIGGERED,
            bank_id=bank_id,
            operation_id=operation_id or "",
            status=decision.action.value,
            timestamp=utcnow(),
            data=MemoryDefenseEventData(
                action=decision.action.value,
                detector=decision.detector,
                document_id=document_id,
                matched_types=decision.matched_types or None,
                message=decision.message or None,
                hits=hits,
                # Optional SIEM-enrichment fields populated by downstream
                # extensions (e.g. hindsight-cloud's _CloudDefenseDecision
                # subclass). Read via getattr so OSS doesn't need to know
                # about extension subclasses. Combined with the manager's
                # exclude_none serialization, missing values stay absent
                # from the wire entirely rather than appearing as null.
                severity=getattr(decision, "severity", None),
                api_key_name=getattr(decision, "api_key_name", None),
                memory_unit_id=getattr(decision, "memory_unit_id", None),
                receipt_uri=getattr(decision, "receipt_uri", None),
            ),
        )
        await webhook_manager.fire_event_with_conn(event, conn, schema=schema)
    except Exception:
        logger.warning("memory_defense webhook delivery failed", exc_info=True)


async def _audit_memory_defense(
    audit_logger: Any,
    *,
    bank_id: str,
    document_id: str | None,
    decision: DefenseDecision,
) -> None:
    """Write a fire-and-forget ``memory_defense`` audit entry for a non-allow decision.

    No-op when auditing is off for this bank. ``audit_log_enabled`` is per-bank
    overridable, so the decision must be awaited here rather than relying on the
    logger's synchronous allowlist check alone.
    The action taken (redact/block) and what matched live in the entry metadata.
    """
    if audit_logger is None:
        return
    if not await audit_logger.should_log("memory_defense", bank_id):
        return
    from ..audit import AuditEntry

    entry = AuditEntry(
        action="memory_defense",
        transport="system",
        bank_id=bank_id,
        metadata={
            "action": decision.action.value,
            "detector": decision.detector,
            "document_id": document_id,
            "matched_types": decision.matched_types,
            "message": decision.message,
        },
    )
    entry.ended_at = entry.started_at  # point-in-time policy decision (duration 0)
    audit_logger.log_fire_and_forget(entry)


def _merge_processed_content_tokens(a: int | None, b: int | None) -> int | None:
    """Combine the processed-content-tokens signal across sub-results.

    Semantics (see RetainResult.processed_content_tokens):
      * None means "this part of the retain did not go through chunk-level
        dedup" — i.e. the entire submitted payload was processed. If any
        sub-result is None, the aggregate is None so callers conservatively
        bill the full content.
      * Otherwise, accumulate the int values.
    """
    if a is None or b is None:
        return None
    return a + b


def _count_delta_content_tokens(delta_contents: list["RetainContent"]) -> int:
    """Sum content + context tokens across the chunk items that were
    actually fed into the extraction pipeline on a partial-delta retain.
    """
    total = 0
    for c in delta_contents:
        total += count_tokens(c.content or "")
        total += count_tokens(c.context or "")
    return total


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

from ... import config as config_module
from ..response_models import TokenUsage
from . import (
    chunk_storage,
    embedding_processing,
    entity_processing,
    fact_extraction,
    fact_storage,
    link_creation,
)
from . import timing as _timing
from .embedding_coalescer import CoalescingEmbedder
from .memory_budget import RetainMemoryBudget, estimate_chunk_bytes
from .types import (
    CausalRelation,
    ChunkMetadata,
    ConcurrentAppendConflict,
    EntityResolutionResult,
    ExtractedFact,
    Phase1Result,
    ProcessedFact,
    ResolvedEntity,
    RetainContent,
    RetainContentDict,
    UserEntities,
)

logger = logging.getLogger(__name__)

# Sentinel append base: the append read found no document row at all. Distinct
# from None (not an append) and from any real content_hash, so the write gate
# can tell "nobody had written this document yet" apart from "we didn't look".
_APPEND_BASE_ABSENT = "__absent__"

RetainOutboxCallback = Callable[[asyncpg.Connection], Awaitable[None]]
RetainOutboxCallbackFactory = Callable[[list[RetainContentDict]], RetainOutboxCallback | None]


@dataclass
class _ProcessedFactBatch:
    """Aligned survivors from converting extracted facts for storage."""

    extracted_facts: list[ExtractedFact]
    processed_facts: list[ProcessedFact]
    retained_index_by_original: list[int | None]


async def _record_retain_document_outcome(pool: Any, bank_id: str, document_id: str, units_created: int) -> None:
    """Emit the per-document retain outcome metric.

    The metric reports the document's unit count *after* this retain, not what
    this call created: a delta retain that touches one chunk can legitimately
    create zero units while the document keeps the units of its unchanged chunks,
    and an oversized document is retained as several sequential sub-batches. Only
    a document left with zero units is unreachable through recall/reflect and
    worth alerting on (#3040).

    ``units_created > 0`` already settles the outcome, so the count query only
    runs on the zero case — which is exactly the cheap path (nothing was written).
    Best-effort: telemetry must never fail a retain.
    """
    try:
        total = units_created
        if total == 0:
            async with acquire_with_retry(pool) as conn:
                total = await fact_storage.count_document_memory_units(conn, bank_id, document_id)
        get_metrics_collector().record_retain_document(bank_id=bank_id, memory_unit_count=total)
    except Exception:
        logger.debug("Failed to record retain document outcome metric", exc_info=True)


# What a reprocess must NOT replay, because it supplies its own: `content` is the
# document's stored original_text, `document_id` and `update_mode` are set by the
# reprocess itself, and `tags` live on the document row and are read from there.
#
# An EXCLUSION list rather than an inclusion one, deliberately. The inclusion list
# this replaces silently dropped three fields in turn — `strategy`, `entities`, and
# `resolve_entities` — each written by api_retain and each never captured, so a
# reprocess re-extracted under the bank's default strategy with entity resolution
# it was told not to do. Every one of those was invisible: the reprocess succeeds
# and only the resulting facts are wrong. Inverting it makes the safe case the
# default — a new retain field round-trips unless someone deliberately excludes it,
# and the single source of truth becomes what api_retain puts on the content dict.
_RETAIN_PARAMS_NOT_REPLAYED = frozenset({"content", "document_id", "update_mode", "tags", "force_reextract"})


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
        for key, value in first_item.items():
            if key in _RETAIN_PARAMS_NOT_REPLAYED or value is None:
                continue
            # event_date arrives as a datetime from some callers and a string from
            # others; retain_params is JSON, so normalise here rather than at every
            # reader.
            if key == "event_date":
                value = value.isoformat() if hasattr(value, "isoformat") else str(value)
            retain_params[key] = value

    return retain_params, merged_tags


async def _pre_resolve_phase1(
    pool: Any,
    entity_resolver,
    bank_id: str,
    contents: list[RetainContent],
    processed_facts: list[ProcessedFact],
    config,
    log_buffer: list[str],
    skip_semantic_ann: bool = False,
    skip_entity_resolution: bool = False,
) -> Phase1Result:
    """
    Phase 1: Run expensive read-heavy operations on a separate connection
    OUTSIDE the write transaction.

    - Entity resolution: trigram GIN scan + co-occurrence fetch + scoring
    - Semantic ANN: HNSW index probes to find similar existing units

    Running these outside the transaction avoids holding row locks during
    slow reads, eliminating TimeoutErrors under concurrent load.

    ``skip_entity_resolution`` is set for a store that resolves/mints entities itself
    (``store_owned``): the store owns its own entity registry and resolves raw names
    server-side inside its atomic retain, so the Postgres trigram scan + entity INSERTs here
    are pure waste (and the whole point of the PG-free path is to not touch Postgres). We return an
    empty entity result; the store-owned write path reconstructs raw names straight from the facts.
    """
    set_stage("retain.phase1.resolve")
    from .link_utils import compute_semantic_links_ann

    if skip_entity_resolution:
        # No Postgres. Semantic ANN is already deferred in streaming, so there is nothing read-heavy
        # left to do — the store-owned write path carries entity NAMES to the server itself.
        empty = EntityResolutionResult(resolved_entities=[], entity_to_unit=[], unit_to_entity_ids={})
        return Phase1Result(entities=empty, semantic_ann_links=[])

    user_entities_per_content = {
        idx: UserEntities(entities=content.entities, resolve=content.resolve_entities)
        for idx, content in enumerate(contents)
        if content.entities
    }

    # Use placeholder unit_ids for grouping during resolution.  The actual
    # unit_ids are created later by insert_facts_batch inside the transaction,
    # but entity resolution and ANN search only need them as grouping keys.
    placeholder_unit_ids = [str(i) for i in range(len(processed_facts))]
    embeddings = [fact.embedding for fact in processed_facts]

    async with acquire_with_retry(pool) as resolve_conn:
        entity_resolution = await entity_processing.resolve_entities(
            entity_resolver,
            resolve_conn,
            bank_id,
            placeholder_unit_ids,
            processed_facts,
            log_buffer,
            user_entities_per_content=user_entities_per_content,
            entity_labels=config.entity_labels,
        )

        # Semantic ANN search on the same connection (autocommit, no transaction).
        # Skipped in streaming mode — deferred to Phase 3 to avoid O(bank_size)
        # scaling bottleneck that makes later streaming batches progressively slower.
        semantic_ann_links = []
        if not skip_semantic_ann:
            fact_types = [fact.fact_type for fact in processed_facts]
            semantic_ann_links = await compute_semantic_links_ann(
                resolve_conn,
                bank_id,
                placeholder_unit_ids,
                embeddings,
                fact_types=fact_types,
                threshold=config.semantic_link_min_similarity,
                log_buffer=log_buffer,
            )

    return Phase1Result(
        entities=entity_resolution,
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
    resolved_entities: list[ResolvedEntity],
    entity_to_unit: list[tuple],
    unit_to_entity_ids: dict[str, list[str]],
    semantic_ann_links: list[tuple],
    skip_semantic_links: bool = False,
    outbox_callback=None,
    ops=None,
) -> list[list[str]]:
    """
    Phase 2 of the retain pipeline: insert facts and retrieval-critical links.

    Runs inside a single database transaction to ensure atomicity of the data
    that retrieval depends on (facts, unit_entities, temporal/semantic/causal links).

    Entity edges for UI graph visualization are derived on demand from
    unit_entities by the /graph endpoint, so no entity rows are written to
    memory_links here.
    """
    set_stage("retain.phase2.insert_facts")
    unit_ids = await fact_storage.insert_facts_batch(conn, bank_id, processed_facts, ops=ops)
    step_start = time.time()
    log_buffer.append(f"  Insert facts: {len(unit_ids)} units in {time.time() - step_start:.3f}s")

    if unit_ids:
        # Entity resolution was done in Phase 1 (separate connection).
        # Remap placeholder IDs to actual unit IDs.
        step_start = time.time()
        resolved_entity_ids = [entity.entity_id for entity in resolved_entities]
        remapped_entity_to_unit, _remapped_unit_to_entity_ids, remapped_semantic = _remap_phase1_results(
            resolved_entity_ids, entity_to_unit, unit_to_entity_ids, semantic_ann_links or [], unit_ids
        )
        # Update semantic_ann_links with remapped IDs for Phase 2
        semantic_ann_links = remapped_semantic
        # INSERT unit_entities (FK to memory_units, must be in transaction).
        # Pass fact_date alongside so entity_cooccurrences.last_cooccurred
        # tracks the event timeline, not the ingest moment.
        unit_entity_pairs = [
            (unit_id, resolved_entity_ids[idx], fact_date)
            for idx, (unit_id, _local_idx, fact_date) in enumerate(remapped_entity_to_unit)
        ]
        # Lock/re-create the resolved parents on THIS transaction before linking,
        # closing the window where prune_orphan_entities could have deleted one
        # between Phase-1 resolution and this insert (#2662).
        await entity_resolver.reassert_entities_batch(bank_id, resolved_entities, conn=conn)
        await entity_resolver.link_units_to_entities_batch(unit_entity_pairs, conn=conn, bank_id=bank_id)
        log_buffer.append(f"  Insert unit_entities: {len(unit_entity_pairs)} pairs in {time.time() - step_start:.3f}s")

        # Create temporal links
        step_start = time.time()
        temporal_link_count = await link_creation.create_temporal_links_batch(conn, bank_id, unit_ids, ops=ops)
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
                threshold=config.semantic_link_min_similarity,
                pre_computed_ann_links=semantic_ann_links,
                ops=ops,
            )
            log_buffer.append(f"  Semantic links: {semantic_link_count} links in {time.time() - step_start:.3f}s")

        # NOTE: Entity links are NOT inserted here. They are deferred to
        # Phase 3 (post-transaction, best-effort) since retrieval uses the
        # unit_entities self-join instead. Entity links only serve UI visualization.

        # Create causal links
        step_start = time.time()
        causal_link_count = await link_creation.create_causal_links_batch(
            conn, bank_id, unit_ids, processed_facts, ops=ops
        )
        log_buffer.append(f"  Causal links: {causal_link_count} links in {time.time() - step_start:.3f}s")

    # Map results back to original content items. Use processed_facts (not
    # extracted_facts) because unit_ids has 1:1 alignment with processed_facts —
    # any upstream drop between extraction and processing would otherwise cause
    # an IndexError (see issue #1037).
    result_unit_ids = _map_results_to_contents(contents, processed_facts, unit_ids if unit_ids else [])

    if outbox_callback is not None:
        await outbox_callback(conn)

    return result_unit_ids


def attempts_delta_retain(provider, bank_id: str, is_first_batch: bool) -> bool:
    """Whether this bank may take the delta path (rewrite only the chunks that changed).

    Delta runs only on the FIRST sub-batch; see the call site for why widening that breaks the
    caller's per-slice bookkeeping. Otherwise every bank deltas, store-owned or not — a store-owned
    delta is one `retain` whose replace names the chunks that moved.
    """
    return is_first_batch


async def _streaming_session_retain(
    *,
    session,
    bank_id: str,
    batch_contents: list,
    batch_extracted: list,
    batch_processed: list,
    batch_chunk_meta: list,
    chunk_index_offset: int,
    effective_doc_id: str,
    combined_content: str,
    content_hash: str | None,
    merged_tags: list[str] | None,
    retain_params: dict | None,
    is_first_batch: bool,
    doc_tracking_done: list[bool],
    doc_replace_done: list[bool],
    entity_resolver,
    log_buffer: list[str],
) -> list[list[str]]:
    """Hand one consumer batch to the store's retain session.

    This is the whole of the engine's persistence work for a store that owns it: mint the unit ids,
    label the facts, and pass the part along. What used to follow — a document write, a retain
    write, and the round trips each costs — is the store's business now.

    The unit ids are minted WITHOUT writing (`defer_index`), exactly as the previous path did, so
    the caller still gets ids to map back to its inputs whether or not the session has committed
    yet. That is deliberate: the ids are the engine's, and a retain that is still buffering has to
    be able to answer with them.
    """
    from ..memories.base import RetainDocumentPart, build_fact_records
    from . import entity_processing, fact_storage
    from .entity_processing import UserEntities

    chunk_id_by_index = {}
    if batch_chunk_meta:
        chunk_id_by_index = {
            cm.chunk_index: f"{bank_id}_{effective_doc_id}_{cm.chunk_index}" for cm in batch_chunk_meta
        }
    for fact, processed_fact in zip(batch_extracted, batch_processed, strict=True):
        processed_fact.document_id = effective_doc_id
        if batch_chunk_meta and fact.chunk_index is not None:
            cid = chunk_id_by_index.get(fact.chunk_index)
            if cid:
                processed_fact.chunk_id = cid

    unit_ids = await fact_storage.insert_facts_batch(None, bank_id, batch_processed, ops=None, defer_index=True)
    batch_result_ids = _map_results_to_contents(batch_contents, batch_processed, unit_ids or [])

    records = list(build_fact_records(unit_ids or [], batch_processed, effective_doc_id))

    # Entity NAMES, unresolved — the store resolves and mints. Built exactly as the path this
    # replaces builds them, and it must be: `UserEntities` (not a bare list), because the merge
    # reads `.entities` and `.resolve` off it, and `resolve` is what distinguishes a caller
    # correcting a name from the extractor guessing one. The user-supplied entities on the content
    # item are merged with whatever extraction found.
    user_entities_per_content = {
        idx: UserEntities(entities=content.entities, resolve=getattr(content, "resolve_entities", True))
        for idx, content in enumerate(batch_contents)
        if getattr(content, "entities", None)
    }
    _texts, _dates, entities_per_fact = entity_processing._prepare_facts_for_entity_processing(
        batch_processed, user_entities_per_content
    )
    names = {
        (unit_ids or [])[i]: [e["text"] for e in entities_per_fact[i]]
        for i in range(min(len(unit_ids or []), len(entities_per_fact)))
    }

    # Only the FIRST batch of a document may replace: a later one would tombstone the siblings this
    # same retain just wrote. Latched here, where the replace is actually handed over.
    replace_chunk_ids = None
    if is_first_batch and not doc_replace_done[0]:
        replace_chunk_ids = []  # empty list == replace the whole document
        doc_replace_done[0] = True

    await session.add(
        RetainDocumentPart(
            document_id=effective_doc_id,
            document_body=combined_content,
            content_hash=content_hash or "",
            chunk_offset=chunk_index_offset,
            # Facts only: the texts were handed over where they were still resident.
            chunk_texts=[],
            facts=records,
            tags=list(merged_tags or []),
            metadata=({"retain_params": json.dumps(retain_params)} if retain_params else {}),
            entity_names=names,
            replace_chunk_ids=replace_chunk_ids,
        )
    )
    # Tracking is complete once the part is handed over, and NOT latching it is data loss rather
    # than waste: the post-loop finalizer runs `store.delete_document`, which tombstones by
    # document_id at a LATER seq than the session's entry — the same-entry sparing protects only a
    # replace INSIDE the atomic write, not a separate tombstone after it. So the finalizer would
    # delete the very memories this retain just wrote. Set even when a batch produced 0 units: the
    # document is tracked regardless of what extraction found.
    doc_tracking_done[0] = True

    log_buffer.append(
        f"[streaming] session facts doc={effective_doc_id} offset={chunk_index_offset} facts={len(records)}"
    )
    return batch_result_ids


async def _streaming_store_owned_retain(
    *,
    provider,
    pool,
    bank_id: str,
    batch_contents: list,
    batch_extracted: list,
    batch_processed: list,
    batch_chunk_meta: list,
    effective_doc_id: str,
    config,
    log_buffer: list[str],
    is_first_batch: bool,
    append_base_hash,
    doc_tracking_done: list[bool],
    doc_replace_done: list[bool],
    p2_start: float,
) -> list[list[str]]:
    """The PG-free retain write for a store that owns entity resolution + atomicity.

    ONE server-side retain does everything the old two-phase path split across an object-store
    write and a Postgres connection phase:

    * resolves each fact's raw entity NAMES (reconstructed here with no Postgres, exactly the merge
      the PG resolver used to do) against the store's own entity registry, minting deterministic ids
      for new names;
    * writes the memories with their entity ids attached;
    * on the document's first batch, tombstones the prior version of the document in the SAME atomic
      entry (a re-retain replaces; the same-entry upserts are spared).

    No connection is acquired and no ``documents``/``chunks``/``entities`` rows are written — the
    store's single write is already atomic, so there is nothing for a second store to be atomic
    *with*. Document/chunk BODIES were already sent to the store's document store by
    ``_store_document_bodies`` (``store_owned``).

    Known gaps, tracked for the follow-on phases:
    * Concurrent same-document ownership/takeover is no longer serialized by a Postgres row lock;
      the store's atomic replace gives last-writer-wins. A store-side document content-hash CAS is a
      follow-up. ``append_base_hash`` (strict-append base check) is likewise deferred to that CAS,
      so an append here does not verify its base — it just does not replace.
    * The transactional-outbox (webhook delivery) is not emitted here; the intended design is
      at-least-once emission from the store, which replaces the Postgres outbox row.
    """
    # Tag each fact with its document + deterministic chunk id (mirrors
    # chunk_storage.store_chunks_batch, so a fact's chunk_id matches its chunk metadata).
    chunk_id_by_index = {}
    if batch_chunk_meta:
        chunk_id_by_index = {
            cm.chunk_index: f"{bank_id}_{effective_doc_id}_{cm.chunk_index}" for cm in batch_chunk_meta
        }
    for fact, processed_fact in zip(batch_extracted, batch_processed):
        processed_fact.document_id = effective_doc_id
        if batch_chunk_meta and fact.chunk_index is not None:
            cid = chunk_id_by_index.get(fact.chunk_index)
            if cid:
                processed_fact.chunk_id = cid

    # Mint the unit ids WITHOUT writing (defer_index) — connection-free and Postgres-free; the
    # single server-side retain below is the only write.
    unit_ids = await fact_storage.insert_facts_batch(None, bank_id, batch_processed, ops=pool.ops, defer_index=True)
    batch_result_ids = _map_results_to_contents(batch_contents, batch_processed, unit_ids if unit_ids else [])

    if unit_ids:
        # Reconstruct each fact's raw entity NAMES (LLM-extracted ∪ user-supplied), the exact merge
        # the Postgres resolver did — but without touching Postgres. The server resolves/mints.
        # Same shape the Postgres resolver builds (`UserEntities`, not a bare list): the entity
        # merge reads `.entities` and `.resolve` off it, and `resolve` is what distinguishes a
        # caller correcting a name from the extractor guessing one (#3479). Passing the raw list
        # here made every store-owned retain that carried user entities fail on `.entities`.
        user_entities_per_content = {
            idx: UserEntities(entities=content.entities, resolve=getattr(content, "resolve_entities", True))
            for idx, content in enumerate(batch_contents)
            if getattr(content, "entities", None)
        }
        _texts, _dates, entities_per_fact = entity_processing._prepare_facts_for_entity_processing(
            batch_processed, user_entities_per_content
        )
        unit_entity_names = {
            unit_ids[i]: [e["text"] for e in entities_per_fact[i]]
            for i in range(min(len(unit_ids), len(entities_per_fact)))
        }
        # Replace the document's prior version only on its FIRST batch — later batches append to
        # what batch 1 just wrote, and replacing again would tombstone those siblings.
        #
        # An append replaces too, and must. `retain_batch` prepends the stored body to this same
        # first batch, so what is being written already covers the old content: leaving the prior
        # version in place keeps a SECOND copy of every earlier fact, carrying the metadata of the
        # retain that created it. That is what `update_mode="append"` then looked like — the old
        # units surviving with stale metadata beside the new ones, where Postgres reprocesses the
        # whole document and ends with one set. The exception here was load-bearing only while the
        # prepend was broken (the base text was read from a SQL column a store-owned bank leaves
        # NULL, so an append's first batch carried the NEW content alone and replacing would have
        # dropped every earlier turn). With the base read from the store that holds it, the first
        # batch is the whole document again and replace is the correct — and the only safe — move.
        # `is_first_batch` is a parameter of the whole `retain_batch` call and stays True for every
        # consumer batch a streaming retain produces, so it cannot express "the first batch of this
        # document" on its own. `doc_tracking_done` is the latch that can — it is set below, after
        # the first batch writes. Without it every batch replaced, tombstoning the siblings the
        # comment above says must survive: measured on a ten-batch document, the retain returned 100
        # unit ids and the bank held 10, the last batch's.
        replace_id = effective_doc_id if (is_first_batch and not doc_replace_done[0]) else ""
        # 0.0 → the server's default trigram-Jaccard threshold; a configured value overrides it.
        threshold = float(getattr(config, "entity_similarity_threshold", 0.0) or 0.0)
        resp = await provider.retain(
            bank_id,
            unit_ids,
            batch_processed,
            document_id=effective_doc_id,
            unit_entity_names=unit_entity_names,
            replace_document_id=replace_id,
            resolve_threshold=threshold,
            # The bank's recall toggles, on the WRITE path: a store that owns its index has no
            # reason to build one for an arm this bank has switched off. Read from the resolved
            # config on every retain, so a bank that changes either one is followed by the store
            # rather than needing an out-of-band call.
            enable_text_search=bool(getattr(config, "enable_text_search", True)),
            enable_graph_retrieval=bool(getattr(config, "enable_graph_retrieval", True)),
        )
        # Latched HERE, where the replace was actually issued — not after the write, which also runs
        # when this batch produced no units and therefore replaced nothing.
        if replace_id:
            doc_replace_done[0] = True
        log_buffer.append(
            f"[streaming] pg-free retain doc={effective_doc_id} units={len(unit_ids)} "
            f"seq={resp.seq} new_entities={resp.new_entities}"
        )
    # Mark the document tracked so the post-loop "no facts / not-yet-tracked" finalizer does NOT
    # fire. That finalizer (a) writes a Postgres documents row and (b) runs handle_document_tracking,
    # whose store.delete_document tombstones by document_id — which, landing at a LATER seq than this
    # retain, would delete the very memories we just wrote (the same-entry sparing only protects the
    # replace inside the store's atomic retain, not a later separate tombstone). The atomic retain
    # above is the whole write, so tracking is complete here. Set it even when there were 0 units:
    # the document is tracked regardless of what extraction produced.
    doc_tracking_done[0] = True
    logger.info(f"[streaming] Phase 2 (pg-free retain): {time.time() - p2_start:.3f}s")
    return batch_result_ids


async def _delta_store_owned_write(
    *,
    provider,
    pool,
    bank_id: str,
    effective_doc_id: str,
    config,
    log_buffer: list,
    entity_resolver,
    contents_dicts: list,
    delta_contents: list,
    document_tags,
    document_body_override,
    extracted_facts: list,
    processed_facts: list,
    new_chunk_metadata,
    delta_chunk_map: dict,
    new_chunks_with_contents: dict,
    existing_by_index: dict,
    changed_indices: list,
    removed_indices: list,
    doc_watermark_at_load,
) -> "tuple[bool, list]":
    """A store-owned bank's delta write: ONE `retain`, scoped to the chunks that moved.

    Separate from `_try_delta_retain` so the contract can be tested directly: what matters here is
    not only the result but that NO Postgres connection is held across it. The store write and the retain are both slow and both
    connection-free; a delta that quietly took a connection would serialise every concurrent retain
    on the pool.

    Returns `(committed, result_unit_ids)`. `False` means the caller falls back to the streaming
    retain — the document moved under this write, and the diff it planned is stale.
    """
    if document_body_override is not None:
        combined_content = document_body_override
    else:
        combined_content = "\n".join([c.get("content", "") for c in contents_dicts])
    retain_params, merged_tags = _build_retain_params(contents_dicts, document_tags)

    # The fence, and it must be this batch's FIRST store write. `expect_watermark` guards on
    # the namespace's WAL head, and the fact write below MOVES that head — fencing after it
    # would fence the batch against itself, so a plain sequential append fails with
    # "required WAL head < 10, but head was 12". Postgres does not need this here because it
    # locks the `documents` row in PHASE 2; a store-owned bank has no such row, which is why
    # parallel appends would otherwise plan against the same base and overwrite each other
    # with every call returning success.
    try:
        await _store_document_bodies(
            bank_id=bank_id,
            document_id=effective_doc_id,
            combined_content=combined_content,
            chunk_texts=[new_chunks_with_contents[i] for i in sorted(new_chunks_with_contents)],
            merged_tags=merged_tags,
            config=config,
            retain_params=retain_params,
            expect_watermark=doc_watermark_at_load,
        )
    except ConcurrentAppendConflict:
        # `_store_document_bodies` already translates the store's StoreWriteConflict into
        # this; catching the store exception instead lets it escape to the caller and turns
        # a losable race into a failed request.
        log_buffer.append(
            f"[delta] Document {effective_doc_id} moved under this delta write — "
            f"falling back to the full streaming retain"
        )
        logger.info("\n" + "\n".join(log_buffer) + "\n")
        return False, []

    # Deterministic chunk ids for the new/changed chunks, after the delta remap, so a fact's
    # chunk_id matches the chunk that carries it.
    chunk_id_by_index = {
        cm.chunk_index: f"{bank_id}_{effective_doc_id}_{cm.chunk_index}" for cm in (new_chunk_metadata or [])
    }
    for ef, pf in zip(extracted_facts, processed_facts):
        pf.document_id = effective_doc_id
        if ef.chunk_index is not None:
            original_idx = delta_chunk_map.get(ef.chunk_index, ef.chunk_index)
            cid = chunk_id_by_index.get(original_idx)
            if cid:
                pf.chunk_id = cid

    # The chunks whose prior facts must go: the ones that CHANGED and the ones REMOVED. A
    # removed chunk has no replacement upsert to supersede it, so naming it is the only
    # thing that takes it out — the case a "replace only what I re-sent" scope would miss.
    replace_chunk_ids = [
        existing_by_index[idx].chunk_id
        for idx in list(changed_indices) + list(removed_indices)
        if idx in existing_by_index
    ]

    # The observations standing on the facts we are about to retire have to go with them, and
    # BEFORE the replace: consolidation batches are built from facts, so once the sources are gone
    # an observation derived from them is never selected again and stays recallable as stale
    # knowledge from the previous version of the document (issue #3294). The SQL delta gets this
    # from its own cascade in `chunk_storage`; a store-owned delta retires its facts through the
    # replace below instead, so this is the only place that can catch them.
    if replace_chunk_ids:
        outgoing = await chunk_storage.memory_ids_for_chunks(None, bank_id, replace_chunk_ids, store=provider)
        if outgoing:
            swept = await fact_storage.delete_stale_observations_for_memories(None, bank_id, outgoing, ops=pool.ops)
            if swept:
                log_buffer.append(f"[delta] swept {swept} observation(s) whose sources are being replaced")

    # Mint the ids without writing; the retain below is the only fact write.
    unit_ids = await fact_storage.insert_facts_batch(None, bank_id, processed_facts, ops=pool.ops, defer_index=True)
    result_unit_ids = _map_results_to_contents(delta_contents, processed_facts, unit_ids if unit_ids else [])

    if unit_ids or replace_chunk_ids:
        unit_entity_names: dict[str, list[str]] = {}
        if unit_ids:
            # Raw entity NAMES, the same merge the Postgres resolver performs — the server
            # resolves and mints them, which is what owning the retain means.
            user_entities_per_content = {
                idx: UserEntities(
                    entities=content.entities,
                    resolve=getattr(content, "resolve_entities", True),
                )
                for idx, content in enumerate(delta_contents)
                if getattr(content, "entities", None)
            }
            _t, _d, entities_per_fact = entity_processing._prepare_facts_for_entity_processing(
                processed_facts, user_entities_per_content
            )
            unit_entity_names = {
                unit_ids[i]: [e["text"] for e in entities_per_fact[i]]
                for i in range(min(len(unit_ids), len(entities_per_fact)))
            }

        threshold = float(getattr(config, "entity_similarity_threshold", 0.0) or 0.0)
        resp = await provider.retain(
            bank_id,
            unit_ids or [],
            processed_facts if unit_ids else [],
            document_id=effective_doc_id,
            unit_entity_names=unit_entity_names,
            # Scoped: only the chunks named above are superseded. Empty `replace_chunk_ids`
            # would be a scope of NOTHING rather than of everything, so when nothing changed
            # this is a plain append and names no document to replace.
            replace_document_id=effective_doc_id if replace_chunk_ids else "",
            replace_chunk_ids=replace_chunk_ids or None,
            resolve_threshold=threshold,
            # The bank's current recall toggles, same as the streaming path above: a delta retain
            # is still a write, and a store that owns its index should not build one for an arm
            # this bank has switched off.
            enable_text_search=bool(getattr(config, "enable_text_search", True)),
            enable_graph_retrieval=bool(getattr(config, "enable_graph_retrieval", True)),
        )
        log_buffer.append(
            f"[delta] store-owned retain doc={effective_doc_id} units={len(unit_ids or [])} "
            f"replaced_chunks={len(replace_chunk_ids)} seq={resp.seq} "
            f"new_entities={resp.new_entities}"
        )

    log_buffer.append(f"DELTA RETAIN COMPLETE (store-owned): {len(processed_facts)} new units")
    logger.info("\n" + "\n".join(log_buffer) + "\n")
    try:
        async with _timing.timed("entity.stats"):
            await entity_resolver.flush_pending_stats()
    except Exception:
        logger.warning("Entity stats flush failed — retrieval unaffected", exc_info=True)
    return True, result_unit_ids


async def _extract_and_embed(
    contents: list[RetainContent],
    llm_config,
    config,
    embeddings_model,
    format_date_fn,
    fact_type_override: str | None,
    log_buffer: list[str],
    pool: Any = None,
    operation_id: str | None = None,
    schema: str | None = None,
) -> tuple[list, list[ProcessedFact], list[ChunkMetadata], TokenUsage]:
    """
    Shared pipeline: extract facts from contents and generate embeddings.

    Returns:
        Tuple of (extracted_facts, processed_facts, chunks_metadata, usage)
    """
    set_stage("retain.extract_and_embed")
    step_start = time.time()
    # No narrator: extraction takes none from this path at all. A "Narrator: {name}" line is
    # stamped into the who-dimension of every first-person fact, so whatever primes it ends up
    # verbatim in stored fact text. Retain used to prime it with the bank's `name` — a display
    # label (#1680 already had to suppress it when it defaulted to the bank_id, itself typically
    # a routing key), which leaked project/tenant names like "AuditProject_0825" into memories
    # that never mentioned them (#3962). A caller that genuinely wants to name the speaker says
    # so in the item's `context`, which extraction already reads and which the dry-run
    # `agent_name` override is deprecated in favour of.
    extracted_facts, chunks, usage = await fact_extraction.extract_facts_from_contents(
        contents, llm_config, config, pool, operation_id, schema
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
    async with _timing.timed("embed"):
        embeddings = await embedding_processing.generate_embeddings_batch(embeddings_model, augmented_texts)
    log_buffer.append(f"  Generate embeddings: {len(embeddings)} embeddings in {time.time() - step_start:.3f}s")

    fact_batch = _process_extracted_facts(extracted_facts, embeddings)

    return fact_batch.extracted_facts, fact_batch.processed_facts, chunks, usage


def _remap_causal_relations(
    relations_per_fact: list[list[CausalRelation]],
    retained_index_by_original: list[int | None],
) -> list[list[CausalRelation]]:
    """Remap a causal relation matrix after facts have been filtered.

    Both the source row and each ``target_fact_index`` use fact ordinals. A
    rejected source disappears with its row; a relation to a rejected target
    must disappear rather than silently pointing at the next surviving fact.
    """
    remapped = [[] for retained_index in retained_index_by_original if retained_index is not None]
    for original_source, retained_source in enumerate(retained_index_by_original):
        if retained_source is None:
            continue
        for relation in relations_per_fact[original_source]:
            original_target = relation.target_fact_index
            retained_target = (
                retained_index_by_original[original_target]
                if 0 <= original_target < len(retained_index_by_original)
                else None
            )
            if retained_target is None:
                continue
            remapped[retained_source].append(
                CausalRelation(
                    relation_type=relation.relation_type,
                    target_fact_index=retained_target,
                )
            )
    return remapped


def _process_extracted_facts(
    extracted_facts: list[ExtractedFact],
    embeddings: list[list[float]],
) -> _ProcessedFactBatch:
    """Process facts while preserving their positional relationships.

    ``ProcessedFact.from_extracted_fact`` may reject a degenerate fact. Keep
    the surviving extracted and processed facts in lockstep, and translate
    causal ordinals from the original extraction into that retained sequence.
    The returned index table is also used by transfer import for archive-only
    links and observation source references.
    """
    if len(extracted_facts) != len(embeddings):
        raise ValueError(
            f"Extracted facts/embeddings length mismatch: {len(extracted_facts)} facts, {len(embeddings)} embeddings"
        )

    retained_extracted: list[ExtractedFact] = []
    processed_facts: list[ProcessedFact] = []
    retained_index_by_original: list[int | None] = [None] * len(extracted_facts)

    for original_index, (extracted_fact, embedding) in enumerate(zip(extracted_facts, embeddings, strict=True)):
        processed_fact = ProcessedFact.from_extracted_fact(extracted_fact, embedding)
        if processed_fact is None:
            continue
        retained_index_by_original[original_index] = len(processed_facts)
        retained_extracted.append(extracted_fact)
        processed_facts.append(processed_fact)

    remapped_relations = _remap_causal_relations(
        [fact.causal_relations for fact in extracted_facts],
        retained_index_by_original,
    )
    for extracted_fact, processed_fact, causal_relations in zip(
        retained_extracted,
        processed_facts,
        remapped_relations,
        strict=True,
    ):
        extracted_fact.causal_relations = causal_relations
        processed_fact.causal_relations = causal_relations

    return _ProcessedFactBatch(
        extracted_facts=retained_extracted,
        processed_facts=processed_facts,
        retained_index_by_original=retained_index_by_original,
    )


async def retain_batch(
    pool: Any,
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
    outbox_callback: RetainOutboxCallback | None = None,
    outbox_callback_factory: RetainOutboxCallbackFactory | None = None,
    db_semaphore: "asyncio.Semaphore | None" = None,
    document_body_override: str | None = None,
    document_body_hash: str | None = None,
    chunk_index_offset: int = 0,
    body_accum: "dict[str, DocumentBodyAccumulator] | None" = None,
    retain_session=None,
    document_prefetch: "dict[str, dict] | asyncio.Task | None" = None,
    progress_callback: "Callable[..., Awaitable[None]] | None" = None,
    webhook_manager: Any = None,
    memory_defense_extension: "MemoryDefenseExtension | None" = None,
    audit_logger: Any = None,
) -> tuple[list[list[str]], TokenUsage, int | None]:
    """
    Process a batch of content through the retain pipeline.

    Supports delta retain: when upserting a document that already has chunks,
    only re-processes chunks whose content has changed. Unchanged chunks keep
    their existing facts, entities, and links.

    ``chunk_index_offset`` shifts the chunk_index (and therefore the derived
    ``chunk_id = {bank}_{doc}_{index}``) of every chunk this call stores. The
    in-process splitter slices an oversized single item into several
    sub-batches that all share one document_id and run sequentially; without
    a per-document offset each sub-batch would restart chunk_index at 0, so
    their chunk_ids collide and later sub-batches overwrite earlier chunks —
    leaving only one sub-batch's worth of chunks/memories behind (issue #1888).

    Returns a three-tuple of:
      * per-content-item unit ID lists
      * aggregate LLM token usage
      * processed_content_tokens — content+context tokens that actually went
        through extraction after chunk-level dedup, or ``None`` if this path
        didn't dedup (caller should treat as "bill full submitted content").
        See ``RetainResult.processed_content_tokens`` for details.
    """
    # Before anything is written. A retain is the one operation that writes to BOTH stores —
    # documents, chunks and entities go to SQL through paths that never touch the memories
    # interface — so a store that needs a bank closed to writes (a backend cutover) cannot enforce
    # it from its own methods alone. Checked here, at the single entry every retain passes through,
    # rather than at each of the writes it fans out into.
    from ..memories import get_memories

    await get_memories().assert_writable(bank_id)

    start_time = time.time()
    total_chars = sum(len(item.get("content", "")) for item in contents_dicts)

    log_buffer = []
    log_buffer.append(f"{'=' * 60}")
    log_buffer.append(f"RETAIN_BATCH START: {bank_id}")
    log_buffer.append(f"Batch size: {len(contents_dicts)} content items, {total_chars:,} chars")
    log_buffer.append(f"{'=' * 60}")

    # Convert dicts to RetainContent objects
    contents = _build_contents(contents_dicts, document_tags)

    # When contents have multiple distinct per-content document_ids and no
    # batch-level document_id, group by doc_id and process each group
    # independently so each document is tracked separately.
    if not document_id:
        per_content_doc_ids = [item.get("document_id") for item in contents_dicts]
        unique_doc_ids = {d for d in per_content_doc_ids if d}
        if len(unique_doc_ids) > 1:
            # Group contents by document_id, preserving original order
            groups: dict[str, tuple[list[RetainContentDict], list[RetainContent]]] = {}
            original_indices: dict[str, list[int]] = {}
            for idx, (cd, c) in enumerate(zip(contents_dicts, contents)):
                doc_key = cd.get("document_id") or str(uuid.uuid4())
                if doc_key not in groups:
                    groups[doc_key] = ([], [])
                    original_indices[doc_key] = []
                groups[doc_key][0].append(cd)
                groups[doc_key][1].append(c)
                original_indices[doc_key].append(idx)

            # Process each group and merge results back in original order
            result_unit_ids: list[list[str]] = [[] for _ in contents_dicts]
            total_usage = TokenUsage()
            total_processed_tokens: int | None = 0
            # One read for every document this retain touches, before the groups fan out. The
            # delta check needs each document's stored hashes, and it needs them BEFORE extraction
            # — so it cannot ride the write. What it can do is happen once: asking per document
            # was a round trip per document on a path whose cost is round trips.
            #
            # A miss is as meaningful as a hit here (the document is new), so the prefetch records
            # BOTH: the dict is what was found, and `prefetched` is the fact that we asked.
            if retain_session is not None and document_prefetch is None:
                from ..memories import get_memories as _gm_prefetch

                # Started, not awaited. Delta needs the answer before it decides which chunks to
                # extract, but chunking the new content does not -- and on a store whose cost is
                # round trips, this read is a whole one on the critical path of every retain.
                # Handing the TASK down lets the groups chunk while it is in flight; each awaits it
                # where the answer is actually consumed. Awaiting a task more than once is fine,
                # and the concurrent groups do exactly that.
                async def _prefetch():
                    async with _timing.timed("delta.read"):
                        return await _gm_prefetch().get_document_records(
                            bank_id=bank_id, document_ids=sorted(groups.keys())
                        )

                document_prefetch = asyncio.create_task(_prefetch())

            # Without sub-batching there is nothing else running these in parallel, so the groups
            # do it themselves. Hardcoded rather than a knob: it is not a capacity dial, it is how
            # many documents of ONE retain may be in flight, and the retain is already bounded by
            # the caller's own concurrency and by the session's flush threshold.
            _GROUP_CONCURRENCY = 8
            _group_sem = asyncio.Semaphore(_GROUP_CONCURRENCY) if retain_session is not None else None

            async def _run_group(doc_key, group_dicts, group_contents):
                if _group_sem is None:
                    return await _one_group(doc_key, group_dicts, group_contents)
                async with _group_sem:
                    return await _one_group(doc_key, group_dicts, group_contents)

            async def _one_group(doc_key, group_dicts, group_contents):
                group_outbox_callback = (
                    outbox_callback_factory(group_dicts) if outbox_callback_factory is not None else outbox_callback
                )

                group_ids, group_usage, group_processed = await retain_batch(
                    pool=pool,
                    embeddings_model=embeddings_model,
                    llm_config=llm_config,
                    entity_resolver=entity_resolver,
                    format_date_fn=format_date_fn,
                    bank_id=bank_id,
                    contents_dicts=group_dicts,
                    config=config,
                    document_id=doc_key,
                    is_first_batch=is_first_batch,
                    fact_type_override=fact_type_override,
                    document_tags=document_tags,
                    operation_id=operation_id,
                    schema=schema,
                    outbox_callback=group_outbox_callback,
                    outbox_callback_factory=outbox_callback_factory,
                    db_semaphore=db_semaphore,
                    document_body_override=document_body_override,
                    chunk_index_offset=chunk_index_offset,
                    progress_callback=progress_callback,
                    webhook_manager=webhook_manager,
                    memory_defense_extension=memory_defense_extension,
                    audit_logger=audit_logger,
                    # Forward the accumulator. Without it every document in a multi-document
                    # retain took the non-accumulating branch and wrote its body immediately and
                    # individually -- so the accumulator only ever applied to the single-oversized-
                    # document case, and a bulk ingest (the case it matters most for) issued one
                    # record write per document. Each of those is an append to the namespace's one
                    # WAL head, which concurrent appends contend for.
                    body_accum=body_accum,
                    retain_session=retain_session,
                    document_prefetch=document_prefetch,
                )
                # Returned rather than merged in place: the groups may run concurrently, and the
                # usage totals are not safe to accumulate from several tasks at once. The driver
                # below merges them in one place, in group order, so the result does not depend on
                # which group happened to finish first.
                return doc_key, group_ids, group_usage, group_processed

            group_results = await asyncio.gather(*(_run_group(k, gd, gc) for k, (gd, gc) in groups.items()))
            for doc_key, group_ids, group_usage, group_processed in group_results:
                for group_idx, orig_idx in enumerate(original_indices[doc_key]):
                    if group_idx < len(group_ids):
                        result_unit_ids[orig_idx] = group_ids[group_idx]
                total_usage = total_usage + group_usage
                total_processed_tokens = _merge_processed_content_tokens(total_processed_tokens, group_processed)
            return result_unit_ids, total_usage, total_processed_tokens

    # --- Memory Defense pre-extraction screening ---
    # Delegate to the loaded extension. `config` is a resolved HindsightConfig
    # object at this point (see _retain_batch_async_internal). On a non-allow
    # decision we redact in place or drop the item, and fire a
    # memory_defense.triggered webhook when one is configured.
    _policy = parse_policy(config.memory_defense)
    _blocked_violations: list[BlockedViolation] = []

    if memory_defense_extension is not None and _policy.enabled:
        async with acquire_with_retry(pool) as _defense_conn:
            for _idx, _content in enumerate(contents):
                # Prefer the per-item document_id over the batch-level value so
                # the decision and webhook carry the document the caller
                # submitted, not whichever doc_id the batch happens to share.
                _item_doc_id = contents_dicts[_idx].get("document_id") or document_id

                _decision = await memory_defense_extension.screen(
                    policy=_policy,
                    bank_id=bank_id,
                    document_id=_item_doc_id,
                    content=_content.content,
                    tags=_content.tags,
                )

                if _decision.action is DefenseAction.ALLOW:
                    continue

                if _decision.action is DefenseAction.REDACT:
                    _redacted = _decision.redacted_content or _content.content
                    _content.content = _redacted
                    # Mirror the redaction into the raw dict so the document
                    # body persisted further down the pipeline also stores the
                    # redacted text, not the verbatim secret.
                    contents_dicts[_idx]["content"] = _redacted
                elif _decision.action is DefenseAction.BLOCK:
                    _blocked_violations.append(
                        BlockedViolation(
                            index=_idx,
                            detector=_decision.detector,
                            message=_decision.message,
                        )
                    )

                await _fire_memory_defense_webhook(
                    webhook_manager,
                    conn=_defense_conn,
                    schema=schema,
                    bank_id=bank_id,
                    operation_id=operation_id,
                    document_id=_item_doc_id,
                    decision=_decision,
                )
                await _audit_memory_defense(
                    audit_logger,
                    bank_id=bank_id,
                    document_id=_item_doc_id,
                    decision=_decision,
                )

    if _blocked_violations:
        # All items blocked → raise so the HTTP layer can return 422.
        if len(_blocked_violations) == len(contents):
            raise MemoryDefenseAllBlockedError(_blocked_violations)

        # Remove blocked items from the pipeline.
        _skip_indices = {v.index for v in _blocked_violations}
        if _skip_indices:
            _surviving = [i for i in range(len(contents)) if i not in _skip_indices]
            contents = [contents[i] for i in _surviving]
            contents_dicts = [contents_dicts[i] for i in _surviving]
            # If nothing survives, return empty results immediately.
            if not contents:
                return [[] for _ in contents_dicts], TokenUsage(), 0

    # Resolve effective document_id early so both delta and streaming paths
    # can find existing chunks from a prior attempt. On retry, a generated
    # document_id is recovered from operation result_metadata.document_ids[0].
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
                    recovered = meta.get("document_ids") or []
                    if recovered:
                        effective_doc_id = recovered[0]
        except Exception:
            pass
    if not effective_doc_id:
        effective_doc_id = str(uuid.uuid4())

    # Record effective_doc_id on the operation (idempotent set-append). Captures
    # both user-provided and generated ids so the operation shows every document
    # it touched, and lets retries reuse the same generated id.
    if operation_id:
        try:
            async with acquire_with_retry(pool) as conn:
                await conn.execute(
                    f"""
                    UPDATE {fq_table("async_operations")}
                    SET result_metadata = jsonb_set(
                        COALESCE(result_metadata, '{{}}'::jsonb),
                        '{{document_ids}}',
                        CASE
                            WHEN COALESCE(result_metadata->'document_ids', '[]'::jsonb) @> $1::jsonb
                                THEN result_metadata->'document_ids'
                            ELSE COALESCE(result_metadata->'document_ids', '[]'::jsonb) || $1::jsonb
                        END,
                        true
                    ),
                    updated_at = now()
                    WHERE operation_id = $2
                    """,
                    json.dumps([effective_doc_id]),
                    uuid.UUID(operation_id),
                )
        except Exception:
            logger.warning("Failed to persist document_id", exc_info=True)

    # --- Append mode: prepend existing document content to new content ---
    # When update_mode="append", fetch the existing document text and prepend it
    # so the full document is reprocessed (delta retain will skip unchanged chunks).
    update_mode = None
    for item in contents_dicts:
        item_mode = item.get("update_mode")
        if item_mode:
            update_mode = item_mode
            break

    # --- Forced re-extraction ---
    # Two independent skips make a re-retain of byte-identical content a no-op: the delta
    # path finds no changed chunk and updates document metadata only, and the recovery gate
    # in `_streaming_retain_batch` treats a matching content_hash plus surviving chunk hashes
    # as a crashed retain being resumed and preserves every existing unit. Both are right for
    # a re-push of unchanged content; both are wrong for `reprocess_document`, whose whole
    # purpose is "extract this again under the CURRENT config", where the content is unchanged
    # by definition (#3899). The flag rides on the content item, so it survives the async
    # operation payload and the oversized-item splitter (which copies every field onto each
    # slice) without a parameter on every frame in between.
    force_reextract = any(bool(item.get("force_reextract")) for item in contents_dicts)

    # The document version this append was built on. Captured with the text it
    # reads so the write path can prove nothing else appended in between — see
    # ``ConcurrentAppendConflict`` and the gate in ``_streaming_retain_batch``.
    # ``_APPEND_BASE_ABSENT`` distinguishes "read a document that wasn't there"
    # from "not an append", which None alone cannot express.
    append_base_hash: str | None = None
    # The store-side watermark for that same base — the token a conditional write uses to prove
    # nothing landed in between. The content_hash gate above only covers stores that keep the
    # document row in Postgres and can lock it; a store that owns the document store has no such
    # row, so this is what serializes concurrent appends there.
    append_base_watermark: int | None = None
    is_append = update_mode == "append" and bool(effective_doc_id) and is_first_batch

    if is_append:
        async with acquire_with_retry(pool) as conn:
            base_row = await conn.fetchrow(
                f"SELECT original_text, content_hash FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
                effective_doc_id,
                bank_id,
            )
        existing_text = base_row["original_text"] if base_row else None
        append_base_hash = base_row["content_hash"] if base_row else _APPEND_BASE_ABSENT
        # The base text comes from whichever store HOLDS it. For a store that owns the document
        # store the SQL row keeps only metadata and `original_text` is NULL by construction
        # (`fact_storage.upsert_document_metadata`), so this read returned nothing to prepend and
        # the append silently became a replace — every earlier turn dropped. The content_hash
        # above stays authoritative for the race gate either way: it is in SQL for both stores.
        from ..memories import get_memories

        _store = get_memories()
        if not existing_text and _store.store_owned_for(bank_id):
            _record = await _store.get_document_record(bank_id=bank_id, document_id=effective_doc_id, include_text=True)
            if _record:
                existing_text = _record.get("original_text")
                # Read WITH the base, not later: a watermark taken after the read would already
                # include a writer that beat us, and the guard would pass while the base was stale.
                append_base_watermark = _record.get("watermark")
                # A store-owned bank may have no SQL documents row at all, in which case the hash
                # the gate compares against has to come from the record too — otherwise a document
                # that plainly exists reads as absent and the append base check is comparing
                # against nothing.
                if base_row is None:
                    append_base_hash = _record.get("content_hash") or _APPEND_BASE_ABSENT
        if existing_text:
            # Prepend existing text as a new content item at the beginning
            existing_content: RetainContentDict = {"content": existing_text}
            # Copy context/tags from first item for consistency
            first = contents_dicts[0]
            if first.get("context"):
                existing_content["context"] = first["context"]
            if first.get("event_date"):
                existing_content["event_date"] = first["event_date"]
            if first.get("metadata"):
                existing_content["metadata"] = first["metadata"]
            if first.get("observation_scopes") is not None:
                existing_content["observation_scopes"] = first["observation_scopes"]
            if first.get("tags"):
                existing_content["tags"] = first["tags"]
            contents_dicts = [existing_content, *contents_dicts]
            # Merge JSON arrays to keep original_text valid (#2409).
            # Without this, combined_content joins items with "\n", producing
            # "[...]\n[...]" which is not valid JSON. On the next append cycle
            # chunk_text() fails to parse it and falls through to sentence-
            # boundary text splitting, breaking speaker attribution.
            try:
                _merged = []
                for _item in contents_dicts:
                    _parsed = json.loads(_item.get("content", ""))
                    if isinstance(_parsed, list) and all(isinstance(_e, dict) for _e in _parsed):
                        _merged.extend(_parsed)
                    else:
                        _merged = None
                        break
                if _merged is not None:
                    contents_dicts = [{"content": json.dumps(_merged, ensure_ascii=False)}]
                    if first.get("context"):
                        contents_dicts[0]["context"] = first["context"]
                    if first.get("event_date"):
                        contents_dicts[0]["event_date"] = first["event_date"]
                    if first.get("metadata"):
                        contents_dicts[0]["metadata"] = first["metadata"]
                    if first.get("observation_scopes") is not None:
                        contents_dicts[0]["observation_scopes"] = first["observation_scopes"]
                    if first.get("tags"):
                        contents_dicts[0]["tags"] = first["tags"]
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
            # Rebuild contents list to match
            contents = _build_contents(contents_dicts, document_tags)
            log_buffer.append(
                f"[append] Prepended {len(existing_text):,} chars from existing document {effective_doc_id}"
            )

    # --- Stale-request check (best-effort, before LLM extraction) ---
    # If the document was already updated by a more recent retain (updated_at > our
    # start_time), skip this request entirely to avoid overwriting newer content
    # (e.g. a longer conversation) with older data. This is an optimization — the
    # real correctness guarantee comes from the FOR UPDATE + content_hash check
    # inside each batch TXN (see _run_mini_batch_db_work).
    # Skipped for a store that owns its documents: there is no SQL `documents` row to read, so
    # this always came back None and the check below never fired -- one pool acquire and one query
    # per document to learn nothing. What actually serializes writers for such a bank is the
    # store's own compare-and-set (`put_document(expect_watermark=...)`), which is the same thing
    # the comment above defers to when it calls this best-effort.
    from ..memories import get_memories as _get_memories_stale

    doc_row = None
    if not _get_memories_stale().store_owned_for(bank_id):
        async with acquire_with_retry(pool) as conn:
            doc_row = await conn.fetchrow(
                f"SELECT updated_at FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
                effective_doc_id,
                bank_id,
            )
    if doc_row and doc_row["updated_at"]:
        doc_updated = doc_row["updated_at"].timestamp()
        if doc_updated > start_time:
            # Under replace semantics dropping this request is right: a newer
            # submission of the same document already superseded it. Under
            # append semantics it is data loss — our content is a turn the
            # winner never saw — so raise and let the caller retry on top of
            # the newer document instead.
            if is_append:
                log_buffer.append(
                    f"[append] Document {effective_doc_id} advanced before extraction — "
                    f"retrying this append on the newer document"
                )
                logger.info("\n" + "\n".join(log_buffer) + "\n")
                raise ConcurrentAppendConflict(
                    f"Document {effective_doc_id} was updated by a concurrent retain after this append read its content"
                )
            log_buffer.append(
                f"[stale] Skipping retain: document {effective_doc_id} was updated at "
                f"{doc_row['updated_at'].isoformat()} (after this request started at "
                f"{datetime.fromtimestamp(start_time, tz=UTC).isoformat()})"
            )
            logger.info("\n" + "\n".join(log_buffer) + "\n")
            # No new content was processed — report 0 so callers can skip
            # billing cleanly instead of falling back to full-content billing.
            return [[] for _ in contents], TokenUsage(), 0

    # --- Delta retain: check if we can skip unchanged chunks ---
    #
    # An APPEND is the one shape where `document_body_override` is not the body being written: the
    # splitter fills it with the incoming item's own text, and the append above then PREPENDS the
    # stored body onto slice 1 — so the body actually being written is `existing + override`, and
    # the override alone is only the new tail. Diffing the whole stored document against that tail
    # classifies every pre-existing chunk as REMOVED and drops it; measured on an oversized append,
    # chunks ended up covering 4,348 of 18,538 chars. `contents` already carries the prepend, so an
    # append keeps diffing against that and only slice 1 (the slice holding the prepend) may delta,
    # exactly as before. `update_mode` is readable on every slice because the splitter copies it
    # onto each one, so slices 2..N opt out here too rather than re-deleting the body slice 1 wrote.
    _delta_full_body = document_body_override if update_mode != "append" else None

    # Every slice of an OVERSIZED replacement gets to try, not just the first. Each one diffs the
    # same complete body against what is stored, so the first slice does the real work and the rest
    # find nothing left to change and fall through to the metadata-only path. Gating on the first
    # slice alone left slices 2..N doing a full extraction of their own content regardless, which is
    # what made an oversized replacement re-extract a document it had just diffed correctly.
    # Delta runs ONLY on the first sub-batch. Widening this to every slice changes the Postgres path
    # too, and three things downstream assume the narrow gate: the caller keeps one result list per
    # sub-batch item (`sub_origins` is length 1 for an oversized slice, so a multi-chunk delta's
    # extra ids are dropped), `chunk_index_offset` advances by the splitter's per-slice count rather
    # than by what a delta wrote, and a brand-new oversized document would extract its whole tail in
    # one step — the bound the sub-batch splitting exists to keep.
    # A store that owns its whole retain has exactly ONE write path: `provider.retain()`, via
    # `_streaming_store_owned_retain`. It deliberately does NOT delta.
    #
    # Delta cannot be expressed as a Retain today. Retain replaces a document wholesale
    # (`replace_document_id` tombstones the document's prior-seq facts) whereas delta rewrites only
    # the chunks that changed, so routing delta through it would need a chunk-scoped replace the RPC
    # has no way to say. The alternative — writing the new facts with `Write` and tombstoning the
    # superseded ones separately — is what this path used to do, and it is precisely the second
    # retain operation being removed here: two RPCs, non-atomic, and bypassing the server-side
    # entity resolution that owning the retain is for.
    #
    # It had also stopped working. The store-owned delta write was reached only when the store
    # minted a cross-store write-group handle, and a store whose single write is already atomic
    # mints none. So a store-owned bank fell through to the Postgres branch below, which resolves
    # entities against Postgres and takes its document lock on a `documents` row that store-owned
    # banks no longer have: `current_hash` comes back None, the stale-chunk guard never fires, and
    # delta ran with no concurrency control at all.
    from ..memories import get_memories as _get_memories_delta

    _delta_provider = _get_memories_delta()
    if not force_reextract and attempts_delta_retain(_delta_provider, bank_id, is_first_batch):
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
            log_buffer,
            start_time,
            operation_id,
            schema,
            outbox_callback,
            db_semaphore,
            document_prefetch=document_prefetch,
            document_body_override=document_body_override,
            delta_full_body=_delta_full_body,
            append_base_hash=append_base_hash,
        )
        if delta_result is not None:
            return delta_result

    # --- Always use the streaming pipeline (producer-consumer batching) ---
    # Even small documents go through the same path — they just end up as a
    # single batch. This eliminates the maintenance burden of two separate
    # retain code paths.
    chunk_batch_size = config.retain_chunk_batch_size
    # Direct attribute access, never getattr-with-default: these two decide chunk
    # boundaries, and the delta path must derive them from the very same resolved
    # config object. A getattr default silently substitutes the global value when
    # handed the wrong config (StaticConfigProxy raises ConfigFieldAccessError,
    # an AttributeError subclass, for bank-configurable fields) — which re-chunks
    # at different boundaries and makes every stored chunk look changed. Fail loud.
    chunk_size = config.retain_chunk_size
    structured_chunk_size = config.retain_structured_chunk_size
    all_pre_chunks: list[str] = []
    chunk_to_content: list[int] = []  # maps chunk index -> index into contents
    for content_idx, content in enumerate(contents):
        # Streamed, not materialised per content: `iter_chunks` yields each chunk as it is
        # cut, so the peak here is one chunk rather than the intermediate splits an eager
        # chunker builds for the whole body (a 45 MB one cost ~130 MB live before #3756).
        for chunk in fact_extraction.iter_chunks(
            content.content,
            chunk_size,
            structured_chunk_size=structured_chunk_size,
        ):
            all_pre_chunks.append(chunk)
            chunk_to_content.append(content_idx)

    # Memory: after chunking, the original content bodies in RetainContent are
    # no longer needed (all_pre_chunks holds the working set). Clear them so
    # Python can reclaim the (potentially multi-MB) strings.
    # Note: contents_dicts["content"] is still needed briefly for hash computation
    # inside _streaming_retain_batch, but gets cleared there after use.
    for content in contents:
        content.content = ""

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
        log_buffer=log_buffer,
        start_time=start_time,
        all_pre_chunks=all_pre_chunks,
        chunk_to_content=chunk_to_content,
        chunk_batch_size=chunk_batch_size,
        operation_id=operation_id,
        schema=schema,
        outbox_callback=outbox_callback,
        db_semaphore=db_semaphore,
        document_body_override=document_body_override,
        document_body_hash=document_body_hash,
        chunk_index_offset=chunk_index_offset,
        body_accum=body_accum,
        retain_session=retain_session,
        document_prefetch=document_prefetch,
        progress_callback=progress_callback,
        append_base_hash=append_base_hash,
        append_base_watermark=append_base_watermark,
        force_reextract=force_reextract,
    )


# ---------------------------------------------------------------------------
# Final semantic ANN pass (post-commit)
# ---------------------------------------------------------------------------

_ANN_CHUNK_SIZE = 1000  # Max seeds per ANN query — smaller chunks avoid timeouts
_ANN_PARALLELISM = 4  # Max concurrent ANN chunks to avoid pool saturation


async def _run_final_semantic_ann(
    pool: Any,
    bank_id: str,
    unit_ids: list[str],
    *,
    threshold: float,
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
                ann_links = await compute_semantic_links_ann(
                    conn,
                    bank_id,
                    chunk_ids,
                    chunk_embs,
                    fact_types=chunk_ftypes,
                    top_k=20,  # Recall uses at most 20 neighbors
                    threshold=threshold,
                    log_buffer=log_buffer,
                )
                if ann_links:
                    await _bulk_insert_links(conn, ann_links, bank_id=bank_id, ops=pool.ops)
                chunk_link_counts[chunk_idx] = len(ann_links)
            logger.info(
                f"[streaming] Final ANN chunk {chunk_idx + 1}/{num_chunks}: "
                f"{len(ann_links)} links in {time.time() - t0:.3f}s"
            )

    await asyncio.gather(*[_process_ann_chunk(i) for i in range(num_chunks)])
    total_links = sum(chunk_link_counts)
    log_buffer.append(f"[streaming] Final ANN: {total_links} total semantic links")


# ---------------------------------------------------------------------------
# Document bodies → the store's dedicated document store (when it owns one)
# ---------------------------------------------------------------------------


async def _store_document_bodies(
    *,
    bank_id: str,
    document_id: str,
    combined_content: str,
    chunk_texts: list[str],
    merged_tags: list[str] | None,
    config: Any,
    content_hash: str | None = None,
    retain_params: dict | None = None,
    chunk_index_offset: int = 0,
    expect_watermark: int | None = None,
) -> None:
    """Route a document's bulky bodies — its extracted text and ordered chunk texts — to the
    store's dedicated document store, when the store owns one. No-op for Postgres.

    Content-addressed and idempotent, so this is safe to call up front, before the facts commit:
    a re-ingest re-uploads only the bodies whose hash changed, and a retain that later rolls back
    leaves only orphan bodies the store's sweep reclaims (they are referenced by no committed
    record). Cold, never-searched, key-based — see docs/documents-chunks.md.

    ``retain_params`` rides along in the record's metadata. It used to be true that "the SQL
    ``documents`` row still carries the small metadata" — it is not, for a store that owns the
    whole retain: there is no SQL row at all, so anything not carried here is simply lost, and
    ``get_document`` returned null ``retain_params`` / ``document_metadata`` /
    ``observation_scopes`` for such a bank. The store's metadata map is ``string -> string``, so
    the params are carried as one JSON value rather than flattened.
    """
    from ..memories import get_memories
    from ..memories.base import StoreWriteConflict

    store = get_memories()
    if not store.store_owned_for(bank_id):
        return
    # `put_document` REPLACES a document's chunk list — it takes the ordered texts whole, because
    # the store packs them into one object. A sub-batched retain calls this once per sub-batch
    # with only that slice's chunks, so every call after the first replaced the document's chunks
    # with its own few and the body ended up covered by one slice (issue #1888's store-side twin:
    # the SQL rows were already offset by `chunk_index_offset`, the store's packed object was not).
    # Restore the prefix the earlier sub-batches stored, so what goes in is the whole document.
    # Sub-batches run sequentially, so [0, offset) is complete by the time this runs.
    chunk_texts = list(chunk_texts)
    if chunk_index_offset > 0:
        prior = await store.get_chunk_texts(bank_id=bank_id, refs=[(document_id, i) for i in range(chunk_index_offset)])
        # A missing prior chunk becomes "" rather than being dropped: the list is positional, and
        # shortening it would shift every following chunk's index — silent corruption in place of
        # one absent chunk.
        chunk_texts = [t or "" for t in prior] + chunk_texts
    # The record's content_hash must equal what the SQL documents row stores, so a read is
    # consistent whichever it comes from: sanitize + sha256 the same combined_content. The
    # streaming path already has it (passes it in); delta computes it here.
    if content_hash is None:
        _sanitized = fact_extraction._sanitize_text(combined_content) or ""
        content_hash = hashlib.sha256(_sanitized.encode()).hexdigest()
    # `expect_watermark` makes this a compare-and-set (the append case: the body being written was
    # derived from the stored one). The store reports a lost race as `StoreWriteConflict`, which is
    # exactly the retain-level `ConcurrentAppendConflict` the caller already knows how to redo —
    # the Postgres path raises it from its own document-row gate. Translating here keeps the two
    # stores' append semantics the same instead of one of them being last-writer-wins.
    try:
        await store.put_document(
            bank_id=bank_id,
            document_id=document_id,
            content_hash=content_hash,
            # Honour store_document_text: when a deployment opts out of keeping the full text,
            # only the chunk texts (needed for citation) go to the store, not the whole body.
            original_text=combined_content if config.store_document_text else None,
            chunk_texts=list(chunk_texts),
            tags=list(merged_tags or []),
            metadata=({"retain_params": json.dumps(retain_params)} if retain_params else {}),
            expect_watermark=expect_watermark,
        )
    except StoreWriteConflict as e:
        raise ConcurrentAppendConflict(str(e)) from e


# A document body is flushed once its unwritten chunk text has at least DOUBLED since the last
# flush. Doubling from the FIRST slice makes the number of writes O(log chunks) and the total bytes
# written ~2x the document, where flushing per sub-batch is O(chunks^2).
#
# There is deliberately no minimum size below which nothing is written. A floor would mean any
# document under it is written only at the very end, so an interrupted retain would leave its
# memories with no body at all — worse than the per-sub-batch writes this replaces, which at least
# left a partial body. Doubling from the first slice keeps the guarantee the size claim rests on:
# an interruption never loses more than half of what had accumulated.


@dataclasses.dataclass
class DocumentBodyMeta:
    """What a document-body write needs, beyond the chunk texts themselves.

    Every sub-batch of a document carries the same values here — `combined_content` is the WHOLE
    document on each of them, so the content hash matches too — so whichever sub-batch arrives
    first fills this in and the rest reuse it.
    """

    bank_id: str
    content_hash: str | None
    combined_content: str
    merged_tags: list[str] | None
    config: Any
    retain_params: dict | None
    expect_watermark: int | None


@dataclasses.dataclass
class DocumentBodyAccumulator:
    """One document's chunk texts as its sub-batches produce them, plus what has been written.

    A dataclass rather than a dict so the shape is checkable: `slices` is positional (offset ->
    that sub-batch's chunks), `meta` carries what the write needs and is filled by whichever
    sub-batch gets there first, and `flushed_bytes` is how much of the prefix is already durable.
    """

    slices: dict[int, list[str]] = dataclasses.field(default_factory=dict)
    meta: DocumentBodyMeta | None = None
    flushed_bytes: int = 0
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)


def _contiguous_prefix(slices: dict[int, list[str]]) -> list[str]:
    """The document's chunk texts from index 0, stopping at the first gap.

    Sub-batches may complete out of order, so the accumulator can hold slice 0 and slice 2 while
    slice 1 is still in flight. The chunk list is POSITIONAL — `put_document` takes it whole and
    index N is chunk N — so writing across a gap would shift every following chunk. Writing only
    the gap-free prefix is exactly what a sequential retain would have written by that point.
    """
    out: list[str] = []
    for offset in sorted(slices):
        if offset != len(out):
            break
        out.extend(slices[offset])
    return out


async def _document_body_write_args(acc: DocumentBodyAccumulator, document_id: str, *, force: bool) -> dict | None:
    """The arguments for this document's next body write, or None if it does not need one.

    Split out from the write itself so the end-of-retain flush can collect several documents'
    writes and issue them as ONE store call, while the incremental (doubling) flush during the
    retain still writes as it goes. Both compute the write here, so they cannot diverge.

    Advances `flushed_bytes` and collapses the slice map exactly as writing would: the caller is
    expected to perform the returned write.
    """
    meta = acc.meta
    if not meta:
        return None
    async with acc.lock:
        chunks = _contiguous_prefix(acc.slices)
        if not chunks:
            return None
        pending = sum(len(c) for c in chunks)
        # Batching mode: hold every write for the end-of-retain flush, which issues them as ONE
        # store call. Without this the first slice of each document writes immediately and the
        # final flush has nothing left to batch. See `retain_batch_document_writes` for when this
        # is the right trade and what it gives up.
        if not force and config_module.get_config().retain_batch_document_writes:
            return None
        # "Nothing new" holds regardless of `force`. force means "write the remainder even though
        # it has not doubled", not "write again what is already durable" -- and the end-of-retain
        # flush passes force=True for every document, so without this it re-issued a record write
        # per document whose body was already fully written. The bodies dedup on hash, but the
        # RECORD write is still an append to the namespace's WAL head.
        if pending <= acc.flushed_bytes:
            return None
        if not force and pending < max(1, 2 * acc.flushed_bytes):
            return None
        args = dict(
            bank_id=meta.bank_id,
            document_id=document_id,
            content_hash=meta.content_hash,
            combined_content=meta.combined_content,
            chunk_texts=chunks,
            merged_tags=meta.merged_tags,
            config=meta.config,
            retain_params=meta.retain_params,
            # The append CAS belongs to the write derived from the stored base, which is the first
            # one this retain issues; later flushes build on what it wrote.
            expect_watermark=meta.expect_watermark if acc.flushed_bytes == 0 else None,
            # The accumulator holds the document from index 0, so the write needs no offset — it
            # IS the prefix, which is what `put_document` wants.
            chunk_index_offset=0,
        )
        acc.flushed_bytes = pending
        # Collapse what was just written into one entry. `put_document` REPLACES the chunk list, so
        # the next flush needs these strings again and they cannot be dropped — but the per-slice
        # entries can, which keeps the prefix walk O(1) instead of O(sub-batches) and stops the dict
        # growing for the rest of the retain. Slices past the write stay keyed where they are.
        rest = {off: sl for off, sl in acc.slices.items() if off >= len(chunks)}
        acc.slices = {0: chunks, **rest}
        return args


async def _flush_document_body(acc: DocumentBodyAccumulator, document_id: str, *, force: bool) -> None:
    """Write the accumulated body if enough has accumulated (or the retain is finishing)."""
    args = await _document_body_write_args(acc, document_id, force=force)
    if args is not None:
        async with _timing.timed("store.document"):
            await _store_document_bodies(**args)


async def flush_document_bodies(body_accum: dict[str, DocumentBodyAccumulator]) -> None:
    """Write out every accumulated document body. Call once a retain's sub-batches have all run.

    Documents whose body is already fully written are skipped, so for a multi-document retain --
    where each document's chunks arrive in one slice and are written as they arrive -- this is
    usually a no-op. It matters for a document split across sub-batches whose last slice did not
    trigger the doubling rule.

    Deliberately one write per document rather than one batched write for all of them: batching
    them into a single store call was measured SLOWER (892 KB/s against 1,348 KB/s on a 64-document
    retain). One large call serialises what independent writes overlap, and that overlap is worth
    more than the WAL-head contention the batch would avoid.
    """
    from ..memories import get_memories

    pending = list(body_accum.items())
    body_accum.clear()
    if not pending:
        return

    store = get_memories()
    batch: list[dict] = []
    for document_id, acc in pending:
        args = await _document_body_write_args(acc, document_id, force=True)
        if args is None:
            continue
        # A guarded write cannot share a batch's single precondition (the guard is on the
        # namespace's WAL head, and a batch carries one), and a bank whose documents live in SQL
        # has no batch call to make. Both go the single-document route.
        if args["expect_watermark"] is not None or not store.store_owned_for(args["bank_id"]):
            await _store_document_bodies(**args)
            continue
        batch.append(args)

    if not batch:
        return
    if len(batch) == 1:
        await _store_document_bodies(**batch[0])
        return

    # Assert rather than assume: a batch is one namespace's write, and silently splitting
    # documents across banks here would write them to whichever bank sorted first.
    bank_id = batch[0]["bank_id"]
    assert all(a["bank_id"] == bank_id for a in batch), "flush spans banks"
    async with _timing.timed("store.document"):
        await store.put_documents(
            bank_id=bank_id,
            documents=[
                {
                    "document_id": a["document_id"],
                    "content_hash": a["content_hash"] or "",
                    "original_text": (a["combined_content"] if a["config"].store_document_text else None),
                    "chunk_texts": a["chunk_texts"],
                    "tags": list(a["merged_tags"] or []),
                    "metadata": ({"retain_params": json.dumps(a["retain_params"])} if a["retain_params"] else {}),
                }
                for a in batch
            ],
        )


# ---------------------------------------------------------------------------
# Streaming chunk batching
# ---------------------------------------------------------------------------


async def _streaming_retain_batch(
    pool: Any,
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
    log_buffer: list[str],
    start_time: float,
    all_pre_chunks: list[str],
    chunk_to_content: list[int],
    chunk_batch_size: int,
    operation_id: str | None = None,
    schema: str | None = None,
    outbox_callback: Callable[["asyncpg.Connection"], Awaitable[None]] | None = None,
    db_semaphore: "asyncio.Semaphore | None" = None,
    document_body_override: str | None = None,
    document_body_hash: str | None = None,
    chunk_index_offset: int = 0,
    body_accum: "dict[str, DocumentBodyAccumulator] | None" = None,
    retain_session=None,
    document_prefetch: "dict[str, dict] | asyncio.Task | None" = None,
    progress_callback: "Callable[..., Awaitable[None]] | None" = None,
    append_base_hash: str | None = None,
    append_base_watermark: int | None = None,
    force_reextract: bool = False,
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

    # Default template for metadata (context, event_date, etc.) when content list is empty.
    _default_content = RetainContent(content="")

    # ---------------------------------------------------------------------------
    # Recovery detection (read-only, before LLM extraction)
    # ---------------------------------------------------------------------------
    # Check if this is a retry of the same content (crash recovery). If the
    # document exists with a matching content_hash and has committed chunks,
    # the producer can skip already-extracted chunks to avoid duplicate work.
    existing_chunk_hashes: set[str] = set()
    # When the caller is processing a sub-batch sliced out of an oversized
    # item (see _split_contents_into_sub_batches), document_body_override
    # carries the full original document body. Use it for the doc-row write
    # so documents.original_text stores the complete payload, not just this
    # slice (issue #1838).
    if document_body_override is not None:
        # Already Memory Defense screened by the caller that produced it
        # (see redact_document_body) — do not rescan it per slice.
        combined_content = document_body_override
    else:
        combined_content = "\n".join([c.get("content", "") for c in contents_dicts])
    # Memory: contents_dicts content strings are now captured in combined_content.
    # Clear them from the dicts to release the per-item copies (can be multi-MB each).
    for d in contents_dicts:
        d.pop("content", None)
    # Sanitize before hashing to match what handle_document_tracking stores.
    #
    # `document_body_hash` is that same hash, already computed by the caller that screened
    # the body. Taking it skips the one remaining piece of retain work that scaled with
    # (sub-batches x document size): every slice of an oversized item carries the identical
    # body, so each one re-sanitized and re-hashed the whole document to derive a value the
    # slice before it had already derived — ~0.9s per slice on a 45 MB body, and such a body
    # splits into ~1,200 slices (#3756). Recomputed here only when no caller supplied it
    # (the un-sliced path, where the body is this submission's own content).
    if document_body_hash is not None:
        new_content_hash = document_body_hash
    else:
        sanitized_content = fact_extraction._sanitize_text(combined_content) or ""
        new_content_hash = hashlib.sha256(sanitized_content.encode()).hexdigest()
        # Memory: sanitized_content is only needed for the hash; free it immediately.
        sanitized_content = ""
    is_recovery = False

    # Same reason as the stale-request check above: the recovery probe reads the SQL `documents`
    # row and the SQL chunk rows, and a store that owns its documents has neither -- so this found
    # nothing, `is_recovery` stayed False, and it cost a pool acquire and a query per document.
    from ..memories import get_memories as _get_memories_recov

    # A forced re-extraction is an operator saying "extract this again under the current
    # config", so it must never be classified as a crashed retain being resumed: recovery
    # preserves every existing unit and skips every matching chunk, which is exactly the
    # silent no-op #3899 reports. Skipping the probe also skips its two queries.
    _sql_recovery_possible = not force_reextract and not _get_memories_recov().store_owned_for(bank_id)
    try:
        if _sql_recovery_possible:
            async with acquire_with_retry(pool) as conn:
                doc_row = await conn.fetchrow(
                    f"SELECT content_hash FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
                    effective_doc_id,
                    bank_id,
                )
                if doc_row and doc_row["content_hash"] == new_content_hash:
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

    # ---------------------------------------------------------------------------
    # Document tracking is DEFERRED to the first consumer batch TXN.
    # ---------------------------------------------------------------------------
    # Previously, document tracking (cascade-delete old data + insert doc row)
    # ran in a separate transaction BEFORE LLM extraction. This left a gap
    # between the cascade-delete and the first chunk write, allowing concurrent
    # requests to interleave and produce duplicates.
    #
    # Now, document tracking runs atomically inside the first batch's write TXN,
    # using SELECT ... FOR UPDATE on the document row for serialization across
    # workers. Each batch TXN also verifies document ownership via content_hash
    # to detect when a concurrent request has taken over the document.
    # See _run_mini_batch_db_work() for the implementation.
    retain_params, merged_tags = _build_retain_params(contents_dicts, document_tags)

    # Route the document's bulky bodies (extracted text + ordered chunk texts) to the store's
    # dedicated document store when it owns one — up front, before the streaming batches
    # write facts. Idempotent and content-addressed, so this is safe here and dedups a re-ingest;
    # a no-op for a Postgres store (which keeps the text in its own columns below). ``all_pre_chunks``
    # is the full ordered chunk-text list; ``combined_content`` is the full document text (both are
    # released as the batches stream, so the write happens now while they are still resident).
    # Accumulate only when the store actually owns a document store. `_store_document_bodies`
    # early-returns on the SAME predicate, so on a SQL deployment accumulating would hold the whole
    # document's chunk texts for the retain and then flush them into a no-op — and worse, it would
    # pin exactly the strings the streaming producer frees as it goes (`all_pre_chunks[i] = ""`).
    from ..memories import get_memories
    from ..memories.base import RetainDocumentPart

    # A session owns the document body: it carries the chunk texts in the same entry as the facts,
    # so accumulating them here as well would write them twice.
    if retain_session is not None and effective_doc_id:
        # The chunk texts go to the session HERE, not at the consumer batch: the streaming producer
        # frees each one as it is extracted (`all_pre_chunks[i] = ""`), so this is the last point
        # they are live. The facts follow from the consumer batch, and the session merges the two.
        async with _timing.timed("store.bodies"):
            await retain_session.add(
                RetainDocumentPart(
                    document_id=effective_doc_id,
                    document_body=combined_content,
                    content_hash=new_content_hash or "",
                    chunk_offset=chunk_index_offset,
                    chunk_texts=list(all_pre_chunks),
                    facts=[],
                    tags=list(merged_tags or []),
                    metadata=({"retain_params": json.dumps(retain_params)} if retain_params else {}),
                )
            )
    elif body_accum is not None and effective_doc_id and get_memories().store_owned_for(bank_id):
        # Accumulating path — see below. Written as the positive branch so `body_accum` and
        # `effective_doc_id` are both narrowed inside it.
        acc = body_accum.get(effective_doc_id)
        if acc is None:
            acc = DocumentBodyAccumulator()
            body_accum[effective_doc_id] = acc
        acc.slices[chunk_index_offset] = list(all_pre_chunks)
        # Every sub-batch carries the WHOLE document as `combined_content` (and so the same content
        # hash), so any one of them can supply the metadata for the writes.
        if acc.meta is None:
            acc.meta = DocumentBodyMeta(
                bank_id=bank_id,
                content_hash=new_content_hash,
                combined_content=combined_content,
                merged_tags=merged_tags,
                config=config,
                retain_params=retain_params,
                expect_watermark=append_base_watermark,
            )
        await _flush_document_body(acc, effective_doc_id, force=False)
    else:
        await _store_document_bodies(
            bank_id=bank_id,
            document_id=effective_doc_id,
            content_hash=new_content_hash,
            combined_content=combined_content,
            chunk_texts=all_pre_chunks,
            merged_tags=merged_tags,
            config=config,
            retain_params=retain_params,
            # An append derives the new body from the stored one, so its write is conditional on
            # that base still being current. Only the first sub-batch carries it: it is the one
            # that read the base, and the later sub-batches build on what it just wrote.
            expect_watermark=append_base_watermark if is_first_batch else None,
            # `all_pre_chunks` is THIS sub-batch's chunks; the offset is where they sit in the
            # document, and is what lets the store keep the earlier sub-batches' chunks instead of
            # being handed one slice as if it were the whole document. The delta path's two calls
            # need no offset: delta retain only runs on the first sub-batch, where the offset is 0.
            chunk_index_offset=chunk_index_offset,
        )

    # Track whether document tracking has been done (by the first batch)
    doc_tracking_done = [False]
    # Whether the document's prior version has actually been REPLACED. Distinct from
    # `doc_tracking_done`, which records that tracking completed and is set even when a batch wrote
    # zero units — there `provider.retain` was never called and no replace was issued, so reusing
    # that latch would let batch 1 replace nothing, latch, and leave the prior version standing
    # beside the new memories for every batch after it.
    doc_replace_done = [False]
    # Track whether the transactional-outbox callback has already fired inside a
    # batch write TXN. The in-TXN fire only runs on a final facts-bearing batch
    # (is_last=True); two success paths never reach it — a committed-chunk count
    # that lands exactly on a chunk_batch_size boundary (the sentinel drains an
    # empty batch), and a final batch that extracts zero facts (it returns before
    # the insert). A post-loop fallback fires the callback in those cases, so this
    # flag exists to guarantee the callback fires exactly once.
    outbox_fired = [False]

    # ---------------------------------------------------------------------------
    # Producer-consumer pipeline: LLM extraction runs concurrently with DB writes
    # ---------------------------------------------------------------------------
    # How many batches the consumer actually wrote. Counted rather than derived from
    # `total_chunks / chunk_batch_size`: since #3756 the consumer also flushes when the open
    # batch grows past its memory budget, so the chunk count only ever gives a lower bound.
    batches_written = [0]

    # Queue for enriched chunks (extracted facts + embeddings).
    # Buffer up to 2x batch_size items so the producer can stay ahead of the consumer.
    chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=chunk_batch_size * 2)

    # ...and a bound on what those items WEIGH, which the queue's item count cannot express:
    # a chunk carries as many facts as the extractor found in it, so "2x batch_size chunks"
    # is anywhere between a few hundred KB and a few hundred MB. The producer reserves a
    # chunk's estimated cost before queueing it and the consumer releases it once written,
    # which is what makes the pipeline's peak a number a worker can be sized against
    # regardless of the document (#3756).
    memory_budget = RetainMemoryBudget(limit_bytes=config.retain_memory_budget_mb * 1024 * 1024)
    # What each queued chunk reserved, so the consumer gives back exactly that. Keyed by the
    # chunk's global index because completion order is not queue order.
    reserved_by_chunk: dict[int, int] = {}

    # Shared mutable state for the producer to report skipped chunks and usage
    producer_error: list[BaseException] = []
    # Set to True by _run_mini_batch_db_work when a concurrent request takes
    # over the document (content_hash mismatch). The consumer checks this and
    # stops processing further batches.
    pipeline_aborted: list[bool] = [False]

    def _assert_append_base_unchanged(existing_hash: str | None) -> None:
        """Fail the append if the document moved since it read its base text.

        Called under the document row lock, on the write that establishes
        ownership. ``append_base_hash`` is the ``content_hash`` the append read
        alongside the text it concatenated onto; the row can only still carry
        that hash if no one else committed in between. A freshly created row
        reads back ``'__pending__'``, which is the expected value exactly when
        the append found no document at all.

        No-op for replace-mode retains (``append_base_hash is None``), whose
        last-writer-wins semantics make a moved document the correct outcome
        rather than a conflict.
        """
        if append_base_hash is None or existing_hash is None:
            return
        expected = "__pending__" if append_base_hash == _APPEND_BASE_ABSENT else append_base_hash
        if existing_hash == expected:
            return
        log_buffer.append(
            f"[append] Document {effective_doc_id} moved between the append read and this "
            f"write (expected {expected[:12]}, found {existing_hash[:12]}) — retrying on the newer document"
        )
        logger.info("\n" + "\n".join(log_buffer) + "\n")
        raise ConcurrentAppendConflict(
            f"Document {effective_doc_id} was updated by a concurrent retain while this append was extracting"
        )

    # Every chunk task embeds only its own chunk's facts, which makes each embedding
    # call one text wide — and in `chunks` extraction mode, where there is no LLM call
    # to overlap, that single round trip is the whole per-chunk cost (issue #3784).
    # The coalescer keeps the fan-out and batches the concurrent embedding calls
    # underneath it. One per retain: the backends read the ambient bank id for cost
    # attribution, so texts from different banks must not share a request.
    coalescing_embedder = CoalescingEmbedder(embeddings_model)

    # ---- LLM Producer ----
    # Fires all chunk extractions as concurrent tasks (bounded by the LLM
    # semaphore inside fact_extraction to 32 concurrent).  As each completes
    # it pushes the enriched result into the queue for the DB consumer.
    async def _llm_producer() -> None:
        async def _extract_one(global_idx: int, chunk_text: str) -> None:
            source = contents[chunk_to_content[global_idx]] if contents else _default_content
            content = RetainContent(
                content=chunk_text,
                context=source.context,
                event_date=source.event_date,
                metadata=source.metadata,
                entities=source.entities,
                resolve_entities=source.resolve_entities,
                tags=source.tags,
                observation_scopes=source.observation_scopes,
            )
            # Attribute this chunk's extraction LLM call to its document, so the
            # trace row carries document_id (a document accrues one such trace
            # per retain/re-retain). Per-call: the operation-level trace context
            # is shared across a batch's documents.
            from ..llm_trace import reset_call_metadata, set_call_metadata

            meta_token = set_call_metadata({"document_id": effective_doc_id})
            try:
                extracted, processed, chunk_meta, usage = await _extract_and_embed(
                    [content],
                    llm_config,
                    config,
                    coalescing_embedder,
                    format_date_fn,
                    fact_type_override,
                    log_buffer,
                    pool,
                    operation_id,
                    schema,
                )
            finally:
                reset_call_metadata(meta_token)
            # Reserve before queueing, so a producer running ahead of a slow write path
            # waits here instead of piling extracted facts up behind the queue. Extraction
            # for chunks already in flight continues; only the handover is throttled.
            chunk_bytes = estimate_chunk_bytes(processed, extracted, chunk_meta)
            await memory_budget.reserve(chunk_bytes)
            reserved_by_chunk[global_idx] = chunk_bytes
            await chunk_queue.put((global_idx, content, extracted, processed, chunk_meta, usage))
            # Memory: release the chunk text from the shared list now that it's
            # been extracted and queued. The queued RetainContent holds its own copy.
            all_pre_chunks[global_idx] = ""

        tasks: list[asyncio.Task] = []
        skipped_total = 0
        try:
            for i, chunk_text in enumerate(all_pre_chunks):
                chunk_hash = chunk_storage.compute_chunk_hash(chunk_text)
                if chunk_hash in existing_chunk_hashes:
                    # Memory: skipped chunks aren't needed either.
                    all_pre_chunks[i] = ""
                    skipped_total += 1
                    continue
                tasks.append(asyncio.create_task(_extract_one(i, chunk_text)))

            if skipped_total > 0:
                log_buffer.append(
                    f"[streaming] Producer: skipped {skipped_total}/{total_chunks} already-committed chunks"
                )

            # Wait for all extractions; collect exceptions
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, BaseException):
                    producer_error.append(r)

            # Signal the consumer that production is done
            await chunk_queue.put(None)
        finally:
            # Cancellation arriving mid-fan-out (the consumer failed, or the worker's
            # wall-clock ceiling fired) must not strand extraction tasks. Cancelling
            # the gather above already propagates to them, but tasks created before
            # we reach it would otherwise survive and park on `chunk_queue.put()`
            # for the life of the process.
            for extraction in tasks:
                if not extraction.done():
                    extraction.cancel()
            # Same reasoning for the coalescer: its dispatcher is a task of its own and
            # a cancelled fan-out would otherwise leave it — and anything parked on
            # it — alive for the life of the process.
            coalescing_embedder.close()
            log_buffer.append(f"[streaming] {coalescing_embedder.stats.describe()}")

    # ---- DB Consumer ----
    # Drains enriched chunks from the queue in batches and runs
    # Phase 1 (entity resolution) -> Phase 2 (write transaction) -> Phase 3 (ANN fire-and-forget).
    async def _db_consumer() -> None:
        batch: list[tuple] = []
        batch_bytes = 0
        consumer_batch_idx = 0
        chunks_committed = 0

        def _release_batch(written: list[tuple]) -> None:
            """Hand the budget back what ``written`` reserved, now that it is committed.

            Released after the batch is written and dropped rather than as each item leaves
            the queue: until then the facts are still resident, and giving the producer room
            to extract more against memory that is still in use is exactly the accounting
            error the budget exists to prevent.
            """
            for global_idx, *_rest in written:
                memory_budget.release(reserved_by_chunk.pop(global_idx, 0))

        # Best-effort durable progress: how many chunks of this document have been
        # extracted+committed so far. Written per consumer batch so an operator polling
        # the retain operation sees "storing 200/1200 chunks" advancing instead of a
        # single opaque sub-batch tick. Never lets a heartbeat failure break retain.
        async def _emit_chunk_progress() -> None:
            if not (progress_callback and operation_id):
                return
            try:
                await progress_callback(
                    operation_id,
                    stage="storing",
                    processed=chunks_committed,
                    total=total_chunks,
                    detail={"facts_committed": len(all_unit_ids)},
                )
            except Exception:
                logger.debug("retain chunk-progress write failed", exc_info=True)

        while True:
            item = await chunk_queue.get()
            if item is None:
                # Process any remaining items
                if batch and not pipeline_aborted[0]:
                    await _process_db_batch(
                        batch,
                        consumer_batch_idx,
                        is_last=True,
                    )
                    chunks_committed += len(batch)
                    await _emit_chunk_progress()
                _release_batch(batch)
                batch = []
                batch_bytes = 0
                break

            batch.append(item)
            batch_bytes += reserved_by_chunk.get(item[0], 0)

            # Write on whichever comes first: the configured chunk count, or an open batch
            # heavy enough that holding more would crowd out the producer (#3756). The
            # count alone let a batch of fact-dense chunks grow far past any memory a
            # worker was sized for.
            if len(batch) >= chunk_batch_size or memory_budget.should_flush(batch_bytes):
                if pipeline_aborted[0]:
                    # Another request took over the document — discard this batch
                    log_buffer.append(
                        f"[streaming] Consumer: discarding batch of {len(batch)} chunks "
                        f"(pipeline aborted due to concurrent takeover)"
                    )
                    _release_batch(batch)
                    batch = []
                    batch_bytes = 0
                    continue
                await _process_db_batch(
                    batch,
                    consumer_batch_idx,
                    is_last=False,
                )
                consumer_batch_idx += 1
                chunks_committed += len(batch)
                await _emit_chunk_progress()
                _release_batch(batch)
                batch = []
                batch_bytes = 0

    async def _process_db_batch(
        batch: list[tuple],
        consumer_batch_idx: int,
        is_last: bool,
    ) -> None:
        """Run Phase 1 + Phase 2 + Phase 3 for a batch of pre-extracted chunks."""
        # Allow clearing combined_content after the no-facts skip path runs
        # doc tracking — see the assignment further below.
        nonlocal combined_content
        batches_written[0] += 1
        # Combine results from individual chunk extractions
        batch_contents: list[RetainContent] = []
        batch_extracted: list = []
        batch_processed: list[ProcessedFact] = []
        batch_chunk_meta: list[ChunkMetadata] = []
        batch_usage = TokenUsage()

        for global_idx, content, extracted, processed, chunk_meta, usage in batch:
            content_idx_in_batch = len(batch_contents)
            # Adjust chunk indices to use the original global position (global_idx)
            # so that chunk_id = {bank}_{doc}_{chunk_index} is deterministic regardless
            # of task completion order. content_index is batch-relative for result grouping.
            #
            # chunk_index_offset continues the document's chunk_index sequence
            # when this call is one of several sequential sub-batches sliced
            # from a single oversized item sharing one document_id — without it
            # each sub-batch restarts at 0 and their chunk_ids collide (#1888).
            doc_chunk_index = global_idx + chunk_index_offset
            fact_index_offset = len(batch_processed)
            for fact, processed_fact in zip(extracted, processed, strict=True):
                fact.content_index = content_idx_in_batch
                if fact.chunk_index is not None:
                    fact.chunk_index = doc_chunk_index
                processed_fact.content_index = content_idx_in_batch

                # Each producer call extracts one chunk, so its causal ordinals
                # start at zero. Translate them into the combined consumer-batch
                # sequence before link creation; otherwise later chunks can point
                # at equally numbered facts from the first completed chunk.
                causal_relations = [
                    CausalRelation(
                        relation_type=relation.relation_type,
                        target_fact_index=relation.target_fact_index + fact_index_offset,
                    )
                    for relation in processed_fact.causal_relations
                ]
                fact.causal_relations = causal_relations
                processed_fact.causal_relations = causal_relations
            for cm in chunk_meta:
                cm.chunk_index = doc_chunk_index

            batch_contents.append(content)
            batch_extracted.extend(extracted)
            batch_processed.extend(processed)
            batch_chunk_meta.extend(chunk_meta)
            batch_usage = batch_usage + usage

        nonlocal total_usage
        total_usage = total_usage + batch_usage

        # ``batch_extracted`` contains only survivors after the degenerate-text
        # guard. Chunk metadata still records whether extraction originally
        # produced facts, so an all-rejected batch follows the normal write path
        # and preserves chunk/outbox behavior from before filtering was added.
        had_extracted_facts = bool(batch_extracted) or any(chunk.fact_count for chunk in batch_chunk_meta)
        if not had_extracted_facts:
            # Even with 0 facts, the first batch must still run document tracking
            # (cascade-delete + insert doc row) to establish ownership and prevent
            # concurrent requests from interleaving. Later batches can safely skip.
            if not doc_tracking_done[0]:
                from ..memories import get_memories

                _edge_provider = get_memories()
                if _edge_provider.store_owned_for(bank_id):
                    # Store-owned 0-fact (re-)ingest: the document's bodies are already in the store
                    # (via _store_document_bodies) and there are no new memories. A re-ingest that now
                    # yields 0 facts must drop the document's PRIOR memories — ONE plain store-side
                    # delete-by-document. No Postgres documents row and no lock — this is the
                    # 0-fact analogue of the fact-bearing PG-free path's replace-tombstone.
                    await _edge_provider.delete_document(
                        conn=None, fq_table=fq_table, bank_id=bank_id, document_id=effective_doc_id
                    )
                    doc_tracking_done[0] = True
                    combined_content = ""
                    log_buffer.append(
                        f"[streaming] Document {effective_doc_id} tracked (0 facts, store-owned, PG-free)"
                    )
                    log_buffer.append(
                        f"[streaming] Consumer batch {consumer_batch_idx + 1}: "
                        f"0 facts extracted from {len(batch)} chunks, skipping"
                    )
                    return
                async with acquire_with_retry(pool) as conn:
                    async with conn.transaction():
                        # Same create-and-lock the fact-bearing path uses. Routed
                        # through the ops layer so this branch takes the row lock
                        # on Oracle too, and so the append gate below sees the
                        # pre-existing hash rather than discarding it.
                        existing_hash = await pool.ops.lock_document_for_write(
                            conn,
                            fq_table("documents"),
                            effective_doc_id,
                            bank_id,
                        )
                        _assert_append_base_unchanged(existing_hash)
                        if is_recovery:
                            await fact_storage.upsert_document_metadata(
                                conn,
                                bank_id,
                                effective_doc_id,
                                combined_content,
                                retain_params,
                                merged_tags,
                                store_document_text=config.store_document_text,
                            )
                        else:
                            # A 0-fact re-ingest still deletes the outgoing memories, in the same
                            # transaction as the document row.
                            await fact_storage.handle_document_tracking(
                                conn,
                                bank_id,
                                effective_doc_id,
                                combined_content,
                                is_first_batch,
                                retain_params,
                                merged_tags,
                                ops=pool.ops,
                                store_document_text=config.store_document_text,
                            )
                        doc_tracking_done[0] = True
                        # Memory: combined_content has been persisted; release
                        # it now so the rest of the consumer loop doesn't pin
                        # a multi-MB string. Nothing reads it after tracking.
                        combined_content = ""
                        log_buffer.append(f"[streaming] Document {effective_doc_id} tracked (0 facts in first batch)")
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
            # Allow clearing combined_content after the doc-tracking call so
            # subsequent batches don't carry the per-document text in memory.
            nonlocal combined_content
            entity_resolver.discard_pending_stats()
            mb_start = time.time()

            # Phase 1 — Entity Resolution only (no ANN — deferred to Phase 3). A store that resolves
            # and mints entities itself (server-side, ``store_owned``) skips the Postgres
            # trigram scan + entity INSERTs entirely — the PG-free path touches no Postgres in retain.
            from ..memories import get_memories as _get_memories_p1

            _store_owned = _get_memories_p1().store_owned_for(bank_id)
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
                skip_entity_resolution=_store_owned,
            )

            logger.info(f"[streaming] Phase 1 (entity resolution): {time.time() - p1_start:.3f}s")

            # Phase 2 — Write transaction
            # -----------------------------------------------------------------
            # Concurrent-safety via row-level locking:
            #
            # The streaming pipeline splits work across multiple batch TXNs.
            # Without protection, two concurrent retains for the same document
            # can interleave: Request A writes batch1, Request B cascade-deletes
            # A's doc and writes its own batch1, then A's batch2 adds stale data
            # on top of B's → duplicates.
            #
            # To prevent this, every batch TXN:
            #   1. SELECT ... FOR UPDATE on the document row — serializes all
            #      writers for this document at the DB level (works across workers).
            #   2. Check content_hash — if it doesn't match ours, another request
            #      took over the document → abort remaining batches.
            #   3. First batch only: run handle_document_tracking (cascade-delete
            #      old data + insert doc row) atomically with the first chunk write.
            #      This eliminates the gap between "delete old" and "insert new"
            #      that previously allowed interleaving.
            # -----------------------------------------------------------------

            p2_start = time.time()
            batch_result_ids = None

            # A store that owns its whole retain writes the batch in ONE server-side call and
            # holds no data-plane connection at all, so it takes its own path. Postgres falls
            # through to the single-transaction path below, unchanged.
            #
            # The gate is the store's CAPABILITY. It was once "did the store mint a cross-store
            # write-group handle", which asked the same question only while every non-SQL store
            # ran that commit protocol; a store whose single write is already atomic mints
            # nothing, so the handle came back None, this branch stopped being taken, and
            # store-owned banks silently fell through to the Postgres path with their `Retain`
            # RPC never called while `store_owned_for()` went on reporting True. Asking what the
            # store CAN do cannot drift out of sync with what it does.
            from ..memories import get_memories

            _ext_provider = get_memories()
            if retain_session is not None:
                # The store owns persistence: hand it this batch and let it decide when to write.
                # The session was opened for the WHOLE retain, above the per-document grouping, so
                # parts from every document accumulate into one commit — the grouping stays and
                # chunk identity stays document-local, which is what makes this safe.
                async with _timing.timed("store.retain"):
                    ext_result_ids = await _streaming_session_retain(
                        session=retain_session,
                        bank_id=bank_id,
                        batch_contents=batch_contents,
                        batch_extracted=batch_extracted,
                        batch_processed=batch_processed,
                        batch_chunk_meta=batch_chunk_meta,
                        chunk_index_offset=chunk_index_offset,
                        effective_doc_id=effective_doc_id,
                        combined_content=combined_content,
                        content_hash=new_content_hash,
                        merged_tags=merged_tags,
                        retain_params=retain_params,
                        is_first_batch=is_first_batch,
                        doc_tracking_done=doc_tracking_done,
                        doc_replace_done=doc_replace_done,
                        entity_resolver=entity_resolver,
                        log_buffer=log_buffer,
                    )
                combined_content = ""
                try:
                    await entity_resolver.flush_pending_stats()
                except Exception:
                    logger.warning(
                        f"Entity stats flush (consumer batch {consumer_batch_idx + 1}) failed", exc_info=True
                    )
                for content_ids in ext_result_ids:
                    all_unit_ids.extend(content_ids)
                return

            if _ext_provider.store_owned_for(bank_id):
                async with _timing.timed("store.retain"):
                    ext_result_ids = await _streaming_store_owned_retain(
                        provider=_ext_provider,
                        pool=pool,
                        bank_id=bank_id,
                        batch_contents=batch_contents,
                        batch_extracted=batch_extracted,
                        batch_processed=batch_processed,
                        batch_chunk_meta=batch_chunk_meta,
                        effective_doc_id=effective_doc_id,
                        config=config,
                        log_buffer=log_buffer,
                        is_first_batch=is_first_batch,
                        append_base_hash=append_base_hash,
                        doc_tracking_done=doc_tracking_done,
                        doc_replace_done=doc_replace_done,
                        p2_start=p2_start,
                    )
                # Doc-tracking consumed combined_content on the first batch; release it (mirrors
                # the Postgres path's first-batch reset).
                combined_content = ""
                # `outbox_fired` is deliberately NOT set: this path writes no transactional-outbox
                # row (there is no Postgres transaction to put one in), so the post-loop fallback
                # is what delivers the webhook. Marking it fired here would drop the delivery.
                #
                # Deferred-stats flush + unit collection — mirrors the shared tail the Postgres
                # path reaches after its connection block exits.
                try:
                    await entity_resolver.flush_pending_stats()
                except Exception:
                    logger.warning(
                        f"Entity stats flush (consumer batch {consumer_batch_idx + 1}) failed", exc_info=True
                    )
                for content_ids in ext_result_ids:
                    all_unit_ids.extend(content_ids)
                return

            async with acquire_with_retry(pool) as conn:
                async with conn.transaction():
                    # --- Document ownership gate ---
                    # Ensure the document row exists, lock it to serialize all
                    # concurrent same-document writers, and read its pre-existing
                    # hash. The lock prevents interleaved retains from corrupting
                    # each other in handle_document_tracking; the returned hash
                    # ('__pending__' for a freshly inserted row) drives the
                    # takeover check for later batches below. The PG/Oracle split
                    # lives in the ops layer because Oracle can't do this upsert +
                    # RETURNING in a single statement.
                    existing_hash = await pool.ops.lock_document_for_write(
                        conn,
                        fq_table("documents"),
                        effective_doc_id,
                        bank_id,
                    )

                    if not doc_tracking_done[0]:
                        # Append compare-and-swap, under the row lock and before any write: an
                        # append that lost its read-modify-write race must abort here rather than
                        # commit over the winner.
                        _assert_append_base_unchanged(existing_hash)

                        # --- First batch: document tracking (atomic with chunk write) ---
                        if is_recovery:
                            await fact_storage.upsert_document_metadata(
                                conn,
                                bank_id,
                                effective_doc_id,
                                combined_content,
                                retain_params,
                                merged_tags,
                                store_document_text=config.store_document_text,
                            )
                            log_buffer.append(
                                f"[streaming] Document {effective_doc_id} updated "
                                f"(recovery, preserving existing chunks)"
                            )
                        else:
                            await fact_storage.handle_document_tracking(
                                conn,
                                bank_id,
                                effective_doc_id,
                                combined_content,
                                is_first_batch,
                                retain_params,
                                merged_tags,
                                ops=pool.ops,
                                store_document_text=config.store_document_text,
                            )
                            log_buffer.append(f"[streaming] Document {effective_doc_id} tracked (full content)")
                        doc_tracking_done[0] = True
                        # Memory: combined_content is no longer needed after
                        # this first-batch tracking call. Release it so the
                        # remaining consumer batches don't pin the string.
                        combined_content = ""
                    else:
                        # --- Later batches: verify we still own the document ---
                        # If another request took over (cascade-deleted our doc and
                        # inserted its own), the content_hash won't match ours.
                        if existing_hash is not None and existing_hash != new_content_hash:
                            log_buffer.append(
                                f"[streaming] Document {effective_doc_id} taken over by "
                                f"concurrent request (hash mismatch) — aborting remaining batches"
                            )
                            logger.info("\n" + "\n".join(log_buffer) + "\n")
                            # Discarding the rest is only acceptable under replace
                            # semantics, where the winner's content supersedes ours.
                            # An append's remaining batches carry content nobody
                            # else has, so raise and redo the whole append instead.
                            if append_base_hash is not None:
                                raise ConcurrentAppendConflict(
                                    f"Document {effective_doc_id} was taken over by a concurrent "
                                    f"retain while this append was storing its batches"
                                )
                            # Signal the consumer to stop processing further batches
                            pipeline_aborted[0] = True
                            return

                    # Store chunks with correct global indices
                    step_start = time.time()
                    chunk_id_map = {}
                    if batch_chunk_meta:
                        chunk_id_map = await chunk_storage.store_chunks_batch(
                            conn,
                            bank_id,
                            effective_doc_id,
                            batch_chunk_meta,
                            ops=pool.ops,
                            store_document_text=config.store_document_text,
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
                    batch_result_ids = await _insert_facts_and_links(
                        conn,
                        entity_resolver,
                        bank_id,
                        batch_contents,
                        batch_extracted,
                        batch_processed,
                        config,
                        log_buffer,
                        resolved_entities=phase1.entities.resolved_entities,
                        entity_to_unit=phase1.entities.entity_to_unit,
                        unit_to_entity_ids=phase1.entities.unit_to_entity_ids,
                        semantic_ann_links=[],
                        skip_semantic_links=True,
                        outbox_callback=outbox_callback if is_last else None,
                        ops=pool.ops,
                    )

                logger.info(f"[streaming] Phase 2 (write transaction): {time.time() - p2_start:.3f}s")

                # The write TXN above committed the transactional-outbox row in the
                # same transaction as this batch's facts. Record it so the post-loop
                # fallback doesn't queue a duplicate delivery.
                if is_last and outbox_callback is not None:
                    outbox_fired[0] = True

            # Best-effort: flush entity_cooccurrences and other deferred stats.
            #
            # This MUST run after the `acquire_with_retry` block above has exited,
            # not inside it: flush_pending_stats() acquires its own connection, and
            # the write above is only committed when the enclosing acquire() block
            # exits. On Oracle (oracledb does not autocommit — the backend commits
            # on clean exit of acquire()) doing this inside the block deadlocks
            # permanently: connection #2 waits on the row locks the still-open
            # connection #1 holds on `entities`, while connection #1 cannot commit
            # until this call returns. Oracle never reports ORA-00060 for it,
            # because session #1 is blocked in Python rather than on the database.
            try:
                await entity_resolver.flush_pending_stats()
            except Exception:
                logger.warning(f"Entity stats flush (consumer batch {consumer_batch_idx + 1}) failed", exc_info=True)

            logger.info(
                f"[streaming] Consumer batch {consumer_batch_idx + 1} total "
                f"(excluding fire-and-forget): {time.time() - mb_start:.3f}s"
            )

            # Collect unit_ids from this batch
            if batch_result_ids:
                for content_ids in batch_result_ids:
                    all_unit_ids.extend(content_ids)

        if db_semaphore is not None:
            async with db_semaphore:
                await _run_mini_batch_db_work()
        else:
            await _run_mini_batch_db_work()

        # Memory: after DB write, clear the batch-local lists that hold extracted
        # facts and embedding vectors. These can be large (384 floats per fact ×
        # thousands of facts) and are no longer needed after commit.
        batch_contents.clear()
        batch_extracted.clear()
        batch_processed.clear()
        batch_chunk_meta.clear()

    # ---------------------------------------------------------------------------
    # Check if facts are already committed (recovery from previous crash).
    # If so, skip extraction+writes and jump straight to final ANN pass.
    # ---------------------------------------------------------------------------
    # Only the call that starts a document at chunk 0 may take the whole-document
    # skip. When an oversized single item is split into several sequential
    # sub-batches that SHARE one document_id AND one operation_id (see
    # _split_contents_into_sub_batches), the first sub-batch commits its chunks
    # and stamps effective_doc_id into result_metadata.facts_committed_document_ids.
    # Without the offset gate, every later sub-batch (chunk_index_offset > 0) would
    # then see its own document already "committed" and skip extraction, dropping
    # all chunks past the first slice. A non-zero offset inherently means this call
    # continues a document another sub-batch already started, so it must always do
    # its work — crash-safety for those chunks still comes from the per-chunk hash
    # recovery (existing_chunk_hashes) below.
    facts_already_committed = False
    if operation_id and chunk_index_offset == 0:
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
                    committed_doc_ids = meta.get("facts_committed_document_ids") or []
                    document_ids = meta.get("document_ids") or []
                    # Legacy path: operations created before per-document checkpoint
                    # tracking only wrote facts_committed=true without document IDs.
                    # Treat those as committed only for single-doc operations.
                    legacy_single_doc_checkpoint = (
                        meta.get("facts_committed")
                        and not committed_doc_ids
                        and (len(document_ids) <= 1 or document_ids == [effective_doc_id])
                    )
                    if effective_doc_id in committed_doc_ids or legacy_single_doc_checkpoint:
                        facts_already_committed = True
                        log_buffer.append(
                            f"[streaming] Recovery: facts already committed ({meta.get('unit_ids_count', '?')} units), "
                            f"skipping to final ANN pass"
                        )
        except Exception:
            logger.warning("Failed to check operation recovery state", exc_info=True)

    if not facts_already_committed:
        # Run producer and consumer concurrently.
        #
        # Cancellation is explicit because plain gather() leaks: when the consumer
        # raises (a deadlock victim, a lock timeout) gather propagates that error
        # immediately but leaves the producer — and every extraction task under it
        # — running. Those tasks then block forever on `chunk_queue.put()` into a
        # queue nobody drains, pinning their chunk payloads and still spending LLM
        # permits and tokens on an operation that already failed (#3002). The same
        # applies when the worker's wall-clock ceiling cancels us from above.
        producer_task = asyncio.create_task(_llm_producer())
        consumer_task = asyncio.create_task(_db_consumer())
        try:
            await asyncio.gather(producer_task, consumer_task)
        finally:
            for pipeline_task in (producer_task, consumer_task):
                if not pipeline_task.done():
                    pipeline_task.cancel()
            # Await the cancellations so neither half outlives this call; the
            # results are already accounted for by the gather above (or by the
            # exception that is propagating).
            await asyncio.gather(producer_task, consumer_task, return_exceptions=True)

        # Propagate producer errors (e.g. LLM failures)
        if producer_error:
            raise producer_error[0]

        # If no batch was processed (e.g. zero facts extracted from gibberish
        # content, or all chunks skipped in recovery), the document row was
        # never created by the first batch TXN. Create it now so the document
        # is tracked regardless of extraction results.
        if not doc_tracking_done[0] and not pipeline_aborted[0]:
            from ..memories import get_memories

            _edge_provider = get_memories()
            if _edge_provider.store_owned_for(bank_id):
                # Store-owned zero-batch retain — the post-loop analogue of the per-batch 0-fact
                # PG-free branch above. Reached when NO batch ran at all (empty/gibberish content,
                # or a recovery where every chunk was already committed as a prior attempt). The
                # document's bodies are already in the store (via _store_document_bodies); a
                # re-ingest that yields no facts must still drop the document's PRIOR memories —
                # ONE plain store-side delete-by-document, with no Postgres documents row.
                await _edge_provider.delete_document(
                    conn=None, fq_table=fq_table, bank_id=bank_id, document_id=effective_doc_id
                )
                doc_tracking_done[0] = True
                combined_content = ""
                log_buffer.append(f"[streaming] Document {effective_doc_id} tracked (no facts, store-owned, PG-free)")
            else:
                async with acquire_with_retry(pool) as conn:
                    async with conn.transaction():
                        await conn.execute(
                            f"INSERT INTO {fq_table('documents')} (id, bank_id, original_text, content_hash) "
                            f"VALUES ($1, $2, '', '__pending__') "
                            f"ON CONFLICT (id, bank_id) DO NOTHING",
                            effective_doc_id,
                            bank_id,
                        )
                        await conn.fetchval(
                            f"SELECT content_hash FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2 FOR UPDATE",
                            effective_doc_id,
                            bank_id,
                        )
                        if is_recovery:
                            await fact_storage.upsert_document_metadata(
                                conn,
                                bank_id,
                                effective_doc_id,
                                combined_content,
                                retain_params,
                                merged_tags,
                                store_document_text=config.store_document_text,
                            )
                        else:
                            # A no-facts re-ingest still deletes the outgoing memories, in the same
                            # transaction as the document row.
                            await fact_storage.handle_document_tracking(
                                conn,
                                bank_id,
                                effective_doc_id,
                                combined_content,
                                is_first_batch,
                                retain_params,
                                merged_tags,
                                ops=pool.ops,
                                store_document_text=config.store_document_text,
                            )
                        doc_tracking_done[0] = True
                        # Memory: combined_content has been persisted and won't be
                        # read again — release the per-document text now.
                        combined_content = ""
                        log_buffer.append(f"[streaming] Document {effective_doc_id} tracked (no facts extracted)")

        # Transactional-outbox fallback. The in-TXN fire only runs on a final
        # facts-bearing batch (is_last=True). When the committed-chunk count lands
        # exactly on a chunk_batch_size boundary the sentinel drains an empty batch
        # and never marks one last; when the final batch extracts zero facts it
        # returns before the insert; and when every chunk is skipped as already
        # committed no batch runs at all. In each of those the retain still
        # succeeded, so the retain.completed delivery must be queued — exactly once,
        # in its own transaction (there is no batch TXN left to attach it to). Skip
        # it on a concurrent takeover: an aborted request must not emit completion.
        if outbox_callback is not None and not outbox_fired[0] and not pipeline_aborted[0]:
            async with acquire_with_retry(pool) as conn:
                async with conn.transaction():
                    await outbox_callback(conn)
            outbox_fired[0] = True

        # Mark facts as committed in operation metadata (crash recovery checkpoint)
        if operation_id and all_unit_ids:
            try:
                async with acquire_with_retry(pool) as conn:
                    # Append effective_doc_id to the committed document set if not
                    # already present, so multi-doc batches track each document
                    # independently for crash recovery.
                    await conn.execute(
                        f"""
                        UPDATE {fq_table("async_operations")}
                        SET result_metadata = jsonb_set(
                            result_metadata || $1::jsonb,
                            '{{facts_committed_document_ids}}',
                            CASE
                                WHEN COALESCE(result_metadata->'facts_committed_document_ids', '[]'::jsonb) @> $2::jsonb
                                    THEN result_metadata->'facts_committed_document_ids'
                                ELSE COALESCE(result_metadata->'facts_committed_document_ids', '[]'::jsonb) || $2::jsonb
                            END,
                            true
                        ),
                        updated_at = now()
                        WHERE operation_id = $3
                        """,
                        json.dumps({"facts_committed": True, "unit_ids_count": len(all_unit_ids)}),
                        json.dumps([effective_doc_id]),
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
    # A store that derives its own semantic links makes this pass pure waste: the units it would
    # read are not in `memory_units` at all, so it acquires a connection and queries an empty table
    # once per retain and derives nothing.
    from ..memories import get_memories as _get_memories_ann

    _derives_links_itself = _get_memories_ann().derives_semantic_links_internally_for(bank_id)
    if all_unit_ids and not pipeline_aborted[0] and not _derives_links_itself:
        ann_start = time.time()
        try:
            await _run_final_semantic_ann(
                pool,
                bank_id,
                all_unit_ids,
                threshold=config.semantic_link_min_similarity,
                log_buffer=log_buffer,
            )
        except Exception:
            # ANN pass is best-effort. FK violations can occur if a concurrent
            # retain cascade-deleted our units between the batch commit and here.
            logger.warning(
                f"[streaming] Final ANN pass failed for document {effective_doc_id} "
                f"(units may have been superseded by concurrent retain)",
                exc_info=True,
            )
        log_buffer.append(f"[streaming] Final ANN pass: {time.time() - ann_start:.3f}s for {len(all_unit_ids)} units")

    total_time = time.time() - start_time
    log_buffer.append(f"{'=' * 60}")
    if pipeline_aborted[0]:
        log_buffer.append(
            f"STREAMING RETAIN ABORTED: document {effective_doc_id} was taken over by "
            f"a concurrent request after {total_time:.3f}s — data from this request was discarded"
        )
    else:
        log_buffer.append(
            f"STREAMING RETAIN COMPLETE: {len(all_unit_ids)} units across {batches_written[0]} batches in {total_time:.3f}s"
        )
    log_buffer.append(f"Document: {effective_doc_id}")
    log_buffer.append(f"{'=' * 60}")
    logger.info("\n" + "\n".join(log_buffer) + "\n")

    if not pipeline_aborted[0]:
        await _record_retain_document_outcome(pool, bank_id, effective_doc_id, len(all_unit_ids))

    # Map all unit_ids back to the original content items.
    # For streaming mode with a single document, all units belong to content 0.
    result_unit_ids = [all_unit_ids] + [[] for _ in contents[1:]]
    # The streaming path doesn't compute per-chunk content-hash dedup in
    # a way that lets us report a partial-processed tokens count — signal
    # ``None`` so callers bill against the full submitted payload.
    return result_unit_ids, total_usage, None


# ---------------------------------------------------------------------------
# Delta retain
# ---------------------------------------------------------------------------


@dataclass
class _ChunkDiff:
    """Classification of chunk indices when diffing new content vs stored chunks."""

    unchanged: list[int]
    changed: list[int]
    new: list[int]
    removed: list[int]


def _classify_chunk_diff(existing_by_index: dict[int, Any], new_hashes: dict[int, str]) -> _ChunkDiff:
    """Classify chunk indices by comparing freshly computed ``new_hashes``
    (index -> content hash) against the currently stored chunks
    (``existing_by_index``: index -> chunk row)."""
    diff = _ChunkDiff(unchanged=[], changed=[], new=[], removed=[])
    for idx, new_hash in new_hashes.items():
        existing = existing_by_index.get(idx)
        if existing and existing.content_hash == new_hash:
            diff.unchanged.append(idx)
        elif existing:
            diff.changed.append(idx)
        else:
            diff.new.append(idx)
    for idx in existing_by_index:
        if idx not in new_hashes:
            diff.removed.append(idx)
    return diff


async def _try_delta_retain(
    pool: Any,
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
    log_buffer,
    start_time,
    operation_id,
    schema,
    outbox_callback,
    db_semaphore: "asyncio.Semaphore | None" = None,
    document_prefetch: "dict[str, dict] | asyncio.Task | None" = None,
    *,
    document_body_override: str | None = None,
    # The complete body to diff against, when the caller could establish one. Distinct from
    # `document_body_override`, which an append fills with only the new tail.
    delta_full_body: str | None = None,
    append_base_hash: str | None = None,
) -> tuple[list[list[str]], TokenUsage, int | None] | None:
    """
    Attempt delta retain for a document upsert. Returns result tuple if delta
    was performed, or None to fall back to full retain.

    When a result tuple is returned, the third element is the content+context
    token count for the chunks that actually went through extraction
    (``0`` if the submission matched prior content exactly and nothing was
    re-extracted).
    """
    # Delta RUNS for a store-owned (PG-free) bank. It did not always: the two things this path
    # relies on are absent for such a bank, and each had to be replaced rather than assumed.
    #
    #   * the concurrency control. The Postgres delta write serializes concurrent writers on
    #     `SELECT content_hash FROM documents ... FOR UPDATE`. A store-owned bank has no such row,
    #     so parallel appends each planned against the same base and overwrote each other — turns
    #     lost silently, every call returning success. Its place is taken by the store's own
    #     compare-and-set: `put_document(expect_watermark=...)`, which must be the batch's FIRST
    #     store write, because the guard is on the namespace's WAL head and this batch's own fact
    #     writes move it.
    #   * the document write itself. The delta path updates the document through
    #     `upsert_document_metadata`, which is SQL and writes nothing for such a bank, so the record
    #     would keep its pre-delta body. `_store_document_bodies` carries it instead.
    #
    # Leaving delta off was not free, which is why it was worth replacing both. A full replace
    # deletes the document's facts and rebuilds them, so re-submitting a document that changed
    # NOTHING orphans or destroys every observation standing on those facts and requeues them for
    # consolidation — test_delta_retain_orphan_observations.py covers that, and enabling delta
    # without the store-side CAS would have traded it for silent append loss, which is worse. Both
    # directions are covered by tests, so the trade was measurable rather than a matter of opinion.
    from ..memories import get_memories as _get_memories_delta

    _delta_store = _get_memories_delta()
    _store_owned_delta = _delta_store.store_owned_for(bank_id)

    # Need a single document_id
    effective_doc_id = document_id
    if not effective_doc_id:
        doc_ids = {item.get("document_id") for item in contents_dicts if item.get("document_id")}
        if len(doc_ids) != 1:
            return None
        effective_doc_id = doc_ids.pop()

    # Load existing chunks and snapshot the document's content_hash. This is
    # outside the write TXN, so a concurrent retain could modify the document
    # between this read and the write. The write TXN verifies the hash hasn't
    # changed; if it has, we fall back to streaming (which has full protection).
    if _store_owned_delta:
        # The same two reads, asked of the store that actually holds them. A chunk_id is
        # `{bank_id}_{document_id}_{index}` by construction, so the records carry no separate id,
        # and the hash is recomputed with the same function that wrote it — the comparison below
        # is against like. ONE record read, not two: it carries the content hash, the text when
        # asked for it, and the watermark the write below compare-and-sets against. All three come
        # from THIS record: un-pairing the watermark from the hash is what the CAS exists to
        # prevent.
        # The prefetch answered this for every document of the retain in one read. It is used ONLY
        # when no text is wanted: the prefetch deliberately carries no bodies, and an append needs
        # the stored text, so that case still reads its own record. Absent from the prefetch means
        # the document does not exist — which is an answer, not a miss to retry.
        if document_prefetch is not None and document_body_override is None:
            # May be the in-flight read rather than its result -- see where it is started. Resolved
            # here, which is the first point that actually needs it.
            if isinstance(document_prefetch, asyncio.Task):
                document_prefetch = await document_prefetch
            record = document_prefetch.get(effective_doc_id)
        else:
            record = await _delta_store.get_document_record(
                bank_id=bank_id,
                document_id=effective_doc_id,
                include_text=document_body_override is not None,
            )
        doc_hash_at_load = (record or {}).get("content_hash")
        doc_watermark_at_load = (record or {}).get("watermark")
        original_text_at_load = (record or {}).get("original_text") if document_body_override is not None else None
        # The record's own chunk hashes, not a download of every chunk's text. Delta compares
        # hashes; the record already stores them, computed with the same
        # `sha256(chunk.encode()).hexdigest()` that `compute_chunk_hash` uses. Reading the texts
        # back to recompute them cost a round trip AND the document's whole body, per document, to
        # arrive at a value the first read already had.
        existing_chunks = [
            chunk_storage.ExistingChunk(
                chunk_id=f"{bank_id}_{effective_doc_id}_{index}",
                chunk_index=index,
                content_hash=chunk_hash,
            )
            for index, chunk_hash in enumerate((record or {}).get("chunk_hashes") or [])
        ]
    else:
        doc_watermark_at_load = None  # SQL serializes on the documents row instead
        async with acquire_with_retry(pool) as conn:
            if document_body_override is not None:
                doc_row_at_load = await conn.fetchrow(
                    f"SELECT content_hash, original_text FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
                    effective_doc_id,
                    bank_id,
                )
                doc_hash_at_load = doc_row_at_load["content_hash"] if doc_row_at_load else None
                original_text_at_load = doc_row_at_load["original_text"] if doc_row_at_load else None
            else:
                doc_hash_at_load = await conn.fetchval(
                    f"SELECT content_hash FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
                    effective_doc_id,
                    bank_id,
                )
                original_text_at_load = None

            # Load chunks after the document version. If a concurrent writer commits
            # between these reads, the hash precondition on metadata-only writes (or
            # the extraction freshness recheck below) forces a streaming fallback.
            existing_chunks = await chunk_storage.load_existing_chunks(conn, bank_id, effective_doc_id)

    # For an append, the document this delta plans against must still be the one
    # whose text the append concatenated onto. Every write below is gated on
    # ``doc_hash_at_load``, so a document that already moved would let the delta
    # commit content assembled from a stale base — losing the turn that moved it.
    # Cheapest possible place to notice: before any chunking or extraction.
    if append_base_hash is not None:
        expected_base = "__pending__" if append_base_hash == _APPEND_BASE_ABSENT else append_base_hash
        if doc_hash_at_load is not None and doc_hash_at_load != expected_base:
            raise ConcurrentAppendConflict(
                f"Document {effective_doc_id} was updated by a concurrent retain "
                f"between this append's read and its delta plan"
            )

    if not existing_chunks:
        return None

    if any(c.content_hash is None for c in existing_chunks):
        logger.info(f"Delta retain skipped for {effective_doc_id}: existing chunks lack content_hash (pre-migration)")
        return None

    # Chunk new content and classify changes.
    #
    # For an OVERSIZED item the retain is split into slices, and `contents` is only this slice while
    # `existing_chunks` covers the whole stored document. Diffing those two classifies every chunk
    # merely absent from this slice as REMOVED — measured on a 20-chunk document:
    # `unchanged=1 changed=0 new=0 removed=19` — so their memories were tombstoned and the later
    # slices re-added and re-extracted them. `delta_full_body` is the complete body for exactly the
    # shapes where the caller could establish one, so the diff is taken against that and compares
    # like with like. It is None for an append, whose complete body is the already-prepended
    # `contents` — see the gate in `retain_batch`.
    step_start = time.time()
    _diff_contents = [RetainContent(content=delta_full_body)] if delta_full_body is not None else contents
    new_chunks_with_contents = _chunk_contents_for_delta(_diff_contents, config)
    log_buffer.append(
        f"[delta] Chunked new content: {len(new_chunks_with_contents)} chunks in {time.time() - step_start:.3f}s"
    )

    existing_by_index = {c.chunk_index: c for c in existing_chunks}
    new_hashes = {idx: chunk_storage.compute_chunk_hash(text) for idx, text in new_chunks_with_contents.items()}

    diff = _classify_chunk_diff(existing_by_index, new_hashes)
    unchanged_indices = diff.unchanged
    changed_indices = diff.changed
    new_indices = diff.new
    removed_indices = diff.removed

    log_buffer.append(
        f"[delta] Chunk diff: {len(unchanged_indices)} unchanged, "
        f"{len(changed_indices)} changed, {len(new_indices)} new, "
        f"{len(removed_indices)} removed"
    )

    if not unchanged_indices:
        if _is_strict_append_of_stored_document(
            original_text_at_load,
            document_body_override,
        ):
            log_buffer.append(
                "[delta] First oversized slice has no stored chunk match, but "
                "the complete document strictly appends the stored source — "
                "preserving historical chunks and advancing document metadata"
            )
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
                document_body_override=document_body_override,
                config=config,
                expected_content_hash=doc_hash_at_load,
            )
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
            document_body_override=document_body_override,
            config=config,
            expected_content_hash=doc_hash_at_load,
        )

    # Build content items for only the changed/new chunks
    delta_contents, delta_chunk_map = _build_delta_contents(contents, new_chunks_with_contents, chunks_to_process)

    # Only bail out to the metadata-only path when there is genuinely nothing to WRITE — not
    # merely nothing to EXTRACT. A re-retain that only DELETES content (drop the last section of
    # a document, leave the rest byte-identical) classifies as `changed=0 new=0 removed=N`, so
    # `delta_contents` is empty while `removed_indices` is not. Returning here updated the
    # document row to the shrunken body and deleted nothing: the removed sections' chunks and
    # facts stayed recallable as part of a document that no longer contained them, and the state
    # was stable — the document's stored hash now matched the new body, so re-submitting it was a
    # no-op that never revisited the leftovers. The write path below deletes `removed_indices`
    # whether or not it also has facts to insert, so let it run.
    if not delta_contents and not removed_indices:
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
            document_body_override=document_body_override,
            config=config,
            expected_content_hash=doc_hash_at_load,
        )

    # Freshness recheck BEFORE the (expensive) LLM extraction.
    #
    # We snapshotted the document hash and chunks outside any lock. A concurrent
    # retain for the same document may have committed a new version while we were
    # chunking and diffing. Re-read the current hash; if it changed, recompute the
    # diff against the now-committed chunk state. If the concurrent writer already
    # produced content identical to ours, there is nothing left to extract — skip
    # the LLM call entirely (metadata-only). If it still differs, fall back to the
    # streaming path (which dedups per-chunk and re-locks the document).
    #
    # This narrows — but cannot fully close — the race window: a writer can still
    # commit during our extraction. The post-extraction hash gate inside the write
    # transaction remains the correctness backstop; this check exists purely to
    # avoid burning LLM tokens on work a concurrent request already did.
    async with acquire_with_retry(pool) as conn:
        recheck_hash = await conn.fetchval(
            f"SELECT content_hash FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
            effective_doc_id,
            bank_id,
        )
    if recheck_hash is not None and doc_hash_at_load is not None and recheck_hash != doc_hash_at_load:
        log_buffer.append(
            f"[delta] Document {effective_doc_id} changed before extraction "
            f"(concurrent retain) — rechecking diff against current state"
        )
        async with acquire_with_retry(pool) as conn:
            current_chunks = await chunk_storage.load_existing_chunks(conn, bank_id, effective_doc_id)
        if not current_chunks or any(c.content_hash is None for c in current_chunks):
            log_buffer.append("[delta] Recheck: current chunks unavailable — falling back to full retain")
            logger.info("\n" + "\n".join(log_buffer) + "\n")
            return None
        current_by_index = {c.chunk_index: c for c in current_chunks}
        recheck = _classify_chunk_diff(current_by_index, new_hashes)
        if not (recheck.changed or recheck.new or recheck.removed):
            log_buffer.append(
                "[delta] Recheck: concurrent retain already stored identical content — "
                "skipping extraction, updating metadata only"
            )
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
                document_body_override=document_body_override,
                config=config,
                expected_content_hash=recheck_hash,
            )
        log_buffer.append(
            f"[delta] Recheck: {len(recheck.changed) + len(recheck.new) + len(recheck.removed)} chunks still differ — "
            f"falling back to full retain"
        )
        logger.info("\n" + "\n".join(log_buffer) + "\n")
        return None

    # Extract facts and generate embeddings (shared pipeline). Attribute these
    # extraction calls to the document so the delta re-retain's trace also binds
    # to it (a document accrues one trace per full/delta retain).
    from ..llm_trace import reset_call_metadata, set_call_metadata

    meta_token = set_call_metadata({"document_id": effective_doc_id})
    try:
        extracted_facts, processed_facts, new_chunk_metadata, usage = await _extract_and_embed(
            delta_contents,
            llm_config,
            config,
            embeddings_model,
            format_date_fn,
            fact_type_override,
            log_buffer,
            pool,
            operation_id,
            schema,
        )
    finally:
        reset_call_metadata(meta_token)

    # Database transaction
    result_unit_ids: list[list[str]] = []
    log_buffer_pre_db = len(log_buffer)

    async def _run_delta_db_work() -> bool:
        """Write this delta. Returns False when the document moved underneath it.

        The caller must translate False into "fall back to the streaming path" —
        this used to be declared ``-> None`` with a bare ``return None`` on the
        abort branch, so the guard logged that it was falling back while the
        delta actually committed on top of the concurrent writer.
        """
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

        # A store-owned bank's delta is ONE `retain`, scoped to the chunks that moved. It does not
        # take PHASE 2 below: that locks a `documents` row such a bank does not have, so
        # `current_hash` would come back None, the stale-chunk guard would never fire, and the delta
        # would run with no concurrency control at all.
        if _store_owned_delta:
            ok, result_unit_ids = await _delta_store_owned_write(
                provider=_delta_store,
                pool=pool,
                bank_id=bank_id,
                effective_doc_id=effective_doc_id,
                config=config,
                log_buffer=log_buffer,
                entity_resolver=entity_resolver,
                contents_dicts=contents_dicts,
                delta_contents=delta_contents,
                document_tags=document_tags,
                document_body_override=document_body_override,
                extracted_facts=extracted_facts,
                processed_facts=processed_facts,
                new_chunk_metadata=new_chunk_metadata,
                delta_chunk_map=delta_chunk_map,
                new_chunks_with_contents=new_chunks_with_contents,
                existing_by_index=existing_by_index,
                changed_indices=changed_indices,
                removed_indices=removed_indices,
                doc_watermark_at_load=doc_watermark_at_load,
            )
            return ok

        # PHASE 2 — Core Write Transaction (atomic)
        # Lock the document row and verify ownership. Delta loaded existing
        # chunks OUTSIDE this TXN, so a concurrent retain may have cascade-deleted
        # and replaced the document since then. If the content_hash changed,
        # the chunk state we based our delta diff on is stale — abort.
        async with acquire_with_retry(pool) as conn:
            async with conn.transaction():
                current_hash = await conn.fetchval(
                    f"SELECT content_hash FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2 FOR UPDATE",
                    effective_doc_id,
                    bank_id,
                )
                # Verify the document hasn't been replaced since we loaded chunks.
                # Compare the current hash against what we snapshotted at load time.
                if current_hash is not None and doc_hash_at_load is not None and current_hash != doc_hash_at_load:
                    log_buffer.append(
                        f"[delta] Document {effective_doc_id} was modified by concurrent request "
                        f"since chunks were loaded — aborting delta, falling back to full retain"
                    )
                    logger.info("\n" + "\n".join(log_buffer) + "\n")
                    # Fall back to streaming, which re-locks the document and (for
                    # an append) verifies the base this content was built on.
                    return False

                # Update document metadata (no delete)
                step_start = time.time()
                # When this sub-batch is one slice of an oversized item
                # split across multiple sub-batches, store the full body
                # (issue #1838) instead of just the slice. The override
                # arrives already screened (see redact_document_body).
                if document_body_override is not None:
                    combined_content = document_body_override
                else:
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
                # Re-store the document's bodies in the store's document store with the
                # FULL new chunk set — put_document dedups by content hash, so unchanged chunks and
                # text re-upload nothing; only what the delta changed moves. A no-op for Postgres.
                await _store_document_bodies(
                    bank_id=bank_id,
                    document_id=effective_doc_id,
                    combined_content=combined_content,
                    chunk_texts=[new_chunks_with_contents[i] for i in sorted(new_chunks_with_contents)],
                    merged_tags=merged_tags,
                    config=config,
                    retain_params=retain_params,
                )
                log_buffer.append(f"  Document metadata update in {time.time() - step_start:.3f}s")

                # Delete changed and removed chunks (cascades to memory_units and links)
                step_start = time.time()
                chunks_to_delete = [
                    existing_by_index[idx].chunk_id
                    for idx in changed_indices + removed_indices
                    if idx in existing_by_index
                ]
                invalidated_obs = await chunk_storage.delete_chunks_by_ids(
                    conn, chunks_to_delete, bank_id, ops=pool.ops
                )
                log_buffer.append(
                    f"  Deleted {len(chunks_to_delete)} chunks "
                    f"({len(changed_indices)} changed + {len(removed_indices)} removed), "
                    f"invalidated {invalidated_obs} observation(s) "
                    f"in {time.time() - step_start:.3f}s"
                )

                # Changed/removed units were deleted above. Sync unchanged
                # survivors; new/changed units inserted below already carry
                # the current metadata and tags.
                step_start = time.time()
                updated_count = await fact_storage.update_memory_units_metadata_and_tags(
                    conn,
                    bank_id,
                    effective_doc_id,
                    merged_tags,
                    retain_params.get("metadata", {}),
                )
                log_buffer.append(
                    f"  Updated tags and metadata on {updated_count} existing memory units "
                    f"in {time.time() - step_start:.3f}s"
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
                        conn,
                        bank_id,
                        effective_doc_id,
                        remapped_chunks,
                        ops=pool.ops,
                        store_document_text=config.store_document_text,
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
                result_unit_ids = await _insert_facts_and_links(
                    conn,
                    entity_resolver,
                    bank_id,
                    delta_contents,
                    extracted_facts,
                    processed_facts,
                    config,
                    log_buffer,
                    resolved_entities=phase1.entities.resolved_entities,
                    entity_to_unit=phase1.entities.entity_to_unit,
                    unit_to_entity_ids=phase1.entities.unit_to_entity_ids,
                    semantic_ann_links=phase1.semantic_ann_links,
                    outbox_callback=outbox_callback,
                    ops=pool.ops,
                )

            total_time = time.time() - start_time
            log_buffer.append(f"{'=' * 60}")
            log_buffer.append(
                f"DELTA RETAIN COMPLETE: {len(processed_facts)} new units, "
                f"{len(unchanged_indices)} chunks unchanged in {total_time:.3f}s"
            )
            log_buffer.append(f"Document: {effective_doc_id}")
            log_buffer.append(f"{'=' * 60}")
            logger.info("\n" + "\n".join(log_buffer) + "\n")

        # Flush deferred entity_cooccurrences stats (best-effort). Must run after
        # the acquire() block above has exited — see the streaming path for why
        # doing this while still holding the connection deadlocks on Oracle.
        try:
            await entity_resolver.flush_pending_stats()
        except Exception:
            logger.warning("Entity stats flush failed — retrieval unaffected", exc_info=True)

        return True

    if db_semaphore is not None:
        async with db_semaphore:
            delta_committed = await _run_delta_db_work()
    else:
        delta_committed = await _run_delta_db_work()
    if not delta_committed:
        # The document moved while this delta was extracting. Nothing was
        # written; the streaming path redoes the work under its own lock, and
        # for an append its base check turns the loss into a retry.
        return None
    await _record_retain_document_outcome(pool, bank_id, effective_doc_id, sum(len(ids) for ids in result_unit_ids))
    # Count content + context tokens that actually went through extraction.
    # ``delta_contents`` holds the per-chunk RetainContent items for the
    # changed/new chunks (see ``_build_delta_contents``) — i.e. exactly what
    # the LLM pipeline saw this call. Unchanged chunks contribute zero.
    processed_tokens = _count_delta_content_tokens(delta_contents)
    return result_unit_ids, usage, processed_tokens


async def _delta_metadata_only(
    pool: Any,
    bank_id,
    contents_dicts,
    contents,
    document_id,
    document_tags,
    log_buffer,
    start_time,
    outbox_callback,
    *,
    document_body_override: str | None = None,
    config: Any = None,
    expected_content_hash: str | None = None,
) -> tuple[list[list[str]], TokenUsage, int] | None:
    """Handle the case where no chunks changed — just update document metadata and tags."""
    from ..memories import get_memories as _get_memories_meta

    _meta_store = _get_memories_meta()
    if _meta_store.store_owned_for(bank_id):
        # The document and its chunks are not in SQL, so the row lock and the hash read above have
        # nothing to read: the whole point of this path — "the document has not moved, so leave its
        # facts alone" — would otherwise decide it HAD moved and fall back to a full retain, which
        # rebuilds the facts and orphans every observation standing on them.
        current_content_hash = await _meta_store.document_content_hash(bank_id=bank_id, document_id=document_id)
        if expected_content_hash is not None and current_content_hash != expected_content_hash:
            log_buffer.append(
                f"[delta] Document {document_id} changed before metadata update — falling back to full retain"
            )
            return None
        combined_content = (
            document_body_override
            if document_body_override is not None
            else "\n".join([c.get("content", "") for c in contents_dicts])
        )
        retain_params, merged_tags = _build_retain_params(contents_dicts, document_tags)
        # Re-put the record with the SAME body hashes: PutDocuments mints upload URLs only for
        # bodies it is missing, and none are missing, so this changes labels without moving bytes.
        chunk_texts = await _meta_store.list_chunk_texts(bank_id=bank_id, document_id=document_id) or []
        await _meta_store.put_document(
            bank_id=bank_id,
            document_id=document_id,
            content_hash=current_content_hash or "",
            original_text=combined_content,
            chunk_texts=chunk_texts,
            tags=merged_tags,
            metadata=retain_params,
        )
        # The document record now carries the new labels, but the memories do not: the SQL branch
        # below propagates them onto the units with the same call, and without it a tags-only
        # re-retain relabelled the document and left every unit on the OLD tags and metadata —
        # measured, v2 units still read ['team-a'] after a retain carrying ['team-b', 'important'].
        # This is the whole work of a metadata-only retain for such a bank, not a detail of it.
        async with acquire_with_retry(pool) as conn:
            await fact_storage.update_memory_units_metadata_and_tags(
                conn, bank_id, document_id, merged_tags, retain_params.get("metadata", {})
            )
        if outbox_callback is not None:
            # The outbox is still SQL for every deployment, so it keeps its own connection.
            async with acquire_with_retry(pool) as conn:
                await outbox_callback(conn)
        total_time = time.time() - start_time
        log_buffer.append(f"DELTA RETAIN (no changes): metadata updated in {total_time:.3f}s")
        logger.info("\n" + "\n".join(log_buffer) + "\n")
        return [[] for _ in contents], TokenUsage(), 0

    async with acquire_with_retry(pool) as conn:
        async with conn.transaction():
            # Lock the document row to serialize with concurrent retains
            current_content_hash = await conn.fetchval(
                f"SELECT content_hash FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2 FOR UPDATE",
                document_id,
                bank_id,
            )
            if expected_content_hash is not None and current_content_hash != expected_content_hash:
                log_buffer.append(
                    f"[delta] Document {document_id} changed before metadata update — falling back to full retain"
                )
                return None
            # When this sub-batch is a slice of an oversized item, write the
            # full original body (issue #1838) instead of just the slice. The
            # override arrives already screened (see redact_document_body).
            if document_body_override is not None:
                combined_content = document_body_override
            else:
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
            await fact_storage.update_memory_units_metadata_and_tags(
                conn,
                bank_id,
                document_id,
                merged_tags,
                retain_params.get("metadata", {}),
            )
            if outbox_callback is not None:
                await outbox_callback(conn)

    total_time = time.time() - start_time
    log_buffer.append(f"DELTA RETAIN (no changes): metadata updated in {total_time:.3f}s")
    logger.info("\n" + "\n".join(log_buffer) + "\n")
    # Nothing went through the extraction pipeline — report 0 processed
    # content tokens so callers can bill accordingly (a caller that's been
    # told ``0`` knows the retain was a pure metadata update and should
    # charge nothing for content).
    return [[] for _ in contents], TokenUsage(), 0


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
            resolve_entities=item.get("resolve_entities", True),
            tags=merged_tags,
            observation_scopes=item.get("observation_scopes"),
        )
        contents.append(content)
    return contents


def _chunk_contents_for_delta(contents: list[RetainContent], config) -> dict[int, str]:
    """
    Chunk contents the same way the streaming path does, returning a map of
    global_chunk_index -> chunk_text.

    Must read chunk_size/structured_chunk_size off the same resolved ``config``
    the streaming path uses, so chunk boundaries match and delta can detect
    unchanged chunks. Two earlier incidents came from this drifting: a 120000
    default made all chunks appear changed on retry, and a getattr default
    silently swapped a bank's resolved size for the global one.
    """
    result = {}
    global_chunk_idx = 0
    # Same resolved-config invariant as the streaming path — see the note there.
    chunk_size = config.retain_chunk_size
    structured_chunk_size = config.retain_structured_chunk_size
    for content in contents:
        chunks = fact_extraction.chunk_text(
            content.content,
            chunk_size,
            structured_chunk_size=structured_chunk_size,
        )
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
            resolve_entities=template_content.resolve_entities,
            tags=template_content.tags,
            observation_scopes=template_content.observation_scopes,
        )
        delta_contents.append(delta_content)
        delta_chunk_map[len(delta_contents) - 1] = original_chunk_idx

    return delta_contents, delta_chunk_map


def _map_results_to_contents(
    contents: list[RetainContent],
    processed_facts: list[ProcessedFact],
    unit_ids: list[str],
) -> list[list[str]]:
    """Map created unit IDs back to original content items.

    `processed_facts` and `unit_ids` must have the same length: each unit_id
    corresponds to the processed_fact at the same index.
    """
    if len(processed_facts) != len(unit_ids):
        raise ValueError(f"processed_facts ({len(processed_facts)}) and unit_ids ({len(unit_ids)}) length mismatch")

    facts_by_content: dict[int, list[int]] = {i: [] for i in range(len(contents))}
    for i, fact in enumerate(processed_facts):
        # Normalize content_index: some LLM providers return 1-indexed values.
        # Clamp to valid range to prevent KeyError.
        idx = fact.content_index
        if idx < 0 or idx >= len(contents):
            idx = min(max(idx, 0), len(contents) - 1) if len(contents) > 0 else 0
        facts_by_content[idx].append(i)

    result_unit_ids = []
    for content_index in range(len(contents)):
        content_unit_ids = [unit_ids[i] for i in facts_by_content[content_index]]
        result_unit_ids.append(content_unit_ids)

    return result_unit_ids
