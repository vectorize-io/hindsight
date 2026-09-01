"""Consolidation engine for automatic observation creation from memories.

The consolidation engine runs as a background job after retain operations complete.
It processes new memories and either:
- Creates new observations from novel facts
- Updates existing observations when new evidence supports/contradicts/refines them

Observations are stored in memory_units with fact_type='observation' and include:
- proof_count: Number of supporting memories
- source_memory_ids: Array of memory UUIDs that contribute to this observation
- history: JSONB tracking changes over time

NOTE: Observations are distinct from mental models (pinned reflections).
- Observations: auto-generated bottom-up by this engine from raw facts (memory_units table, fact_type='observation')
- Mental models: user-defined queries stored in the mental_models table, refreshed on demand via reflect
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from fnmatch import fnmatchcase
from itertools import combinations
from typing import TYPE_CHECKING, Any, Literal

import asyncpg
from pydantic import BaseModel, ValidationError, field_validator

from ...config import get_config
from ...worker.stage import set_stage
from ..db import DatabaseBackend
from ..db_utils import acquire_with_retry
from ..llm_interface import OutputTooLongError, ProviderRateLimitResetError
from ..llm_trace import (
    record_created_memory_ids,
    record_source_memory_ids,
    reset_trace_context,
    set_trace_context,
    trace_context_of,
)
from ..llm_wrapper import sanitize_llm_output
from ..memories import FactRecord, get_memories
from ..memory_engine import Budget, fq_table
from ..retain import embedding_utils
from .prompts import (
    build_consolidation_input,
    build_consolidation_system_prompt,
)

if TYPE_CHECKING:
    from asyncpg import Connection

    from ...api.http import RequestContext
    from ..memories.base import StoredMemory
    from ..memory_engine import MemoryEngine
    from ..response_models import MemoryFact, RecallResult

logger = logging.getLogger(__name__)


async def _gather_or_cancel(coros: list[Any]) -> list[Any]:
    """``asyncio.gather`` that leaves no task running behind it.

    Plain ``asyncio.gather`` re-raises the first exception immediately but does
    NOT cancel its siblings — they keep running detached. In consolidation that
    is actively harmful: the failure propagates out of ``run_consolidation_job``
    to the worker, which marks the operation failed and re-queues it with a 5s
    base backoff, while the orphaned tag groups are still calling the LLM,
    stamping ``mark_consolidated`` and committing write-groups. The per-scope
    ``scope_locks`` are local to one dispatch, so nothing serialises an orphan
    against the retry, and the "batches within a group run serially" invariant
    that keeps two consolidators out of the same observation scope is broken
    exactly when it matters.

    So: cancel the outstanding tasks and await them before propagating. A
    cancelled batch's writes stay invisible (its witness row is never
    committed) and are resolved by the recovery sweep, which is the same state
    a crash would leave.

    Deliberately not ``asyncio.TaskGroup``: it wraps failures in an
    ``ExceptionGroup``, and the worker's ``_is_non_retryable_task_error`` does
    ``isinstance`` checks on the raised exception — a wrapped
    ``IntegrityConstraintViolationError`` would be misclassified as retryable
    and retried forever. This helper re-raises the original exception unchanged.
    """
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            if not t.done():
                t.cancel()
        # Await the cancellations before propagating: returning while they are
        # still unwinding would reintroduce the very overlap this prevents.
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _native_search_vector_update(config, param: str) -> str:
    """UPDATE-clause fragment that repopulates ``search_vector`` inline, or ''
    when the backend does not maintain a native tsvector column that way.

    ``to_tsvector(...)::regconfig`` is PostgreSQL-only. On Oracle ``search_vector``
    is a CLOB maintained by Oracle's own text index rather than an inline
    tsvector, so emit nothing there (mirrors the insert path, which gates
    ``search_vector`` on the PG-only ``pg_search_vector_expr``). Without this
    guard the PG expression reaches Oracle and fails with DPY-4010 (the
    ``::regconfig`` cast becomes an unbound ``:REGCONFIG`` placeholder).
    """
    from ..schema import _is_oracle  # noqa: PLC0415

    if config.text_search_extension != "native" or _is_oracle():
        return ""
    lang = config.text_search_extension_native_language
    return f",\n            search_vector = to_tsvector('{lang}'::regconfig, COALESCE({param}, ''))"


def _norm_obs_text(text: str) -> str:
    """Whitespace-normalised observation text for exact-duplicate matching.

    Collapses runs of whitespace only; case is preserved. The reconciliation guard
    drops a CREATE on the premise that an exact-text match loses no information — but
    case-folding would also drop a create differing only in case (e.g. "TLS" vs "tls"),
    which *does* lose information, so we match case-sensitively.
    """
    return " ".join((text or "").split()).strip()


def _duplicate_create_target(
    create_text: str,
    shown_obs_by_text: "dict[str, MemoryFact]",
    update_texts: set[str],
) -> str | None:
    """Return a human label for what ``create_text`` duplicates, or None if novel.

    A CREATE is a duplicate when its normalised text matches an observation that was
    already shown to the LLM, or the text of an UPDATE issued in the same response
    (the model occasionally UPDATEs the twin to text X and also CREATEs X). Exact-text
    match means no information is lost by dropping the CREATE.
    """
    norm = _norm_obs_text(create_text)
    matched = shown_obs_by_text.get(norm)
    if matched is not None:
        return f"shown observation {str(matched.id)[:8]}"
    if norm in update_texts:
        return "an UPDATE in this response"
    return None


# Top-K existing observations probed (by the new observation's own embedding) when
# semantic dedup is enabled. Small: we only need the nearest few candidates.
_DEDUP_TOP_K = 5


class _DedupDecision(BaseModel):
    """Focused 1-by-1 verdict for whether a new observation duplicates an existing one."""

    action: Literal["merge", "keep"] = "keep"
    text: str = ""  # the synthesized merged observation (when action == "merge")
    reason: str = ""

    @field_validator("action", mode="before")
    @classmethod
    def _normalize_action(cls, value: object) -> str:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"merge", "keep"}:
                return normalized

        logger.warning("Invalid consolidation dedup action %r; defaulting to keep", value)
        return "keep"


def _dedup_decision_from_response(raw: Any) -> _DedupDecision:
    try:
        if isinstance(raw, _DedupDecision):
            return raw
        if isinstance(raw, str):
            return _DedupDecision.model_validate_json(raw)
        return _DedupDecision.model_validate(raw)
    except ValueError as exc:
        logger.warning("Invalid consolidation dedup response %r; defaulting to keep: %s", raw, exc)
        return _DedupDecision(action="keep", reason="invalid structured response")


_DEDUP_PROMPT = """You reconcile long-term memory observations. A NEW observation is about to be \
stored, and it is highly similar to an EXISTING one:

[NEW] {new}
[EXISTING] {existing}

Respond with ONLY one valid JSON object matching one of these shapes:

For duplicate facts:
{{"action": "merge", "text": "...", "reason": "..."}}

For distinct facts:
{{"action": "keep", "text": "", "reason": "..."}}

Do NOT use key=value lines, markdown fences, or any text outside the JSON object.

If they assert the SAME fact (wording aside), set "action" to "merge" and provide "text": a \
single observation that preserves EVERY detail from both. If they differ in ANY important detail \
— a number/quantity, a named entity or language, a negation, or a condition — set "action" to \
"keep" and "text" to an empty string."""


def _dedup_active(config: Any) -> bool:
    """Whether create/update semantic dedup runs for this consolidation.

    Enabled when the resolved threshold is < 1.0, EXCEPT on Oracle: the merge path uses
    Postgres-only SQL (``unnest``/``array_agg``, ``UPDATE ... FROM``), so on Oracle dedup is
    skipped — it behaves exactly as it did before this feature, regardless of the configured
    threshold. This is why the feature can ship enabled-by-default without breaking Oracle.
    """
    if config is None or config.consolidation_dedup_threshold >= 1.0:
        return False
    return get_config().database_backend != "oracle"


@dataclass(frozen=True)
class _TemporalBounds:
    """The temporal columns an observation inherits from the facts behind it.

    Merging two observations (or an observation and a fresh set of source facts) must widen
    these, never replace them: ``event_date``/``occurred_start`` keep the earliest known value
    and ``occurred_end``/``mentioned_at`` the latest, with a missing value on either side
    ignored. That is exactly the ``_aggregate_source_fields`` rule, and the Python mirror of the
    ``LEAST``/``GREATEST`` the SQL paths apply.

    The SQL spelling differs by reach, deliberately. The dedup folds only ever run on PostgreSQL
    (``_dedup_active`` disables dedup on Oracle) and use the plain
    ``LEAST(col, COALESCE(x, col))``, which is enough there because PostgreSQL ignores NULL
    arguments. ``_apply_update_action`` also runs on Oracle, where LEAST/GREATEST return NULL
    if any argument is NULL, so it wraps the whole expression in one more COALESCE — see the
    comment there.
    """

    event_date: "datetime | None" = None
    occurred_start: "datetime | None" = None
    occurred_end: "datetime | None" = None
    mentioned_at: "datetime | None" = None

    @classmethod
    def of(cls, row: "StoredMemory | _SourceAggregation") -> "_TemporalBounds":
        """The bounds carried by a stored memory or by an aggregation over source facts.

        Deliberately not a recall ``MemoryFact``: that model has no ``event_date`` at all and
        keeps the rest as ISO strings, so it has to be read field by field where it is used.
        """
        return cls(
            event_date=row.event_date,
            occurred_start=row.occurred_start,
            occurred_end=row.occurred_end,
            mentioned_at=row.mentioned_at,
        )

    def merged_with(self, other: "_TemporalBounds") -> "_TemporalBounds":
        return _TemporalBounds(
            event_date=_merge_min(self.event_date, other.event_date),
            occurred_start=_merge_min(self.occurred_start, other.occurred_start),
            occurred_end=_merge_max(self.occurred_end, other.occurred_end),
            mentioned_at=_merge_max(self.mentioned_at, other.mentioned_at),
        )


@dataclass
class _DedupOutcome:
    """Result of probing one observation against its in-scope neighbours.

    ``best_id`` is the nearest observation at/above the threshold (None if none),
    ``merged_text`` is the LLM-synthesized union text (set only when ``should_merge``).
    """

    best_id: str | None
    merged_text: str
    should_merge: bool
    # The twin's text at probe time. Guards the fold against a concurrent survivor
    # rewrite during the connection-free LLM window (set on the two non-None returns).
    best_text: str = ""


async def _dedup_adjudicate(
    pool: DatabaseBackend,
    memory_engine: "MemoryEngine",
    bank_id: str,
    config: Any,
    dedup_llm_config: Any,
    anchor_text: str,
    anchor_emb_str: str | None,
    tags: list[str] | None,
    exclude_id: str | None,
) -> _DedupOutcome:
    """Probe one observation's embedding against in-scope observations and adjudicate a merge.

    Anchored on the observation text — the correct obs<->obs comparison, unlike consolidation
    recall which is anchored on the raw fact. Returns the nearest observation at/above
    ``consolidation_dedup_threshold`` and, when found, the LLM's focused 1-by-1 merge-or-keep
    verdict (scope ``consolidation_dedup``): the LLM reads both texts, so a word-level difference
    (number / negation / entity) is respected. ``exclude_id`` skips the anchor observation itself
    (used by the UPDATE path, where the anchor row already exists and would self-match at 1.0).
    ``anchor_emb_str`` reuses an already-computed embedding (the UPDATE path just embedded it);
    pass None to embed ``anchor_text`` here (the CREATE path).

    The embedder and the LLM both run with NO connection held; only the semantic+BM25 probe
    briefly borrows a short-lived connection.
    """
    from ..memories import get_memories

    threshold = config.consolidation_dedup_threshold
    if anchor_emb_str is None:
        embs = await embedding_utils.generate_embeddings_batch(memory_engine.embeddings, [anchor_text])
        if not embs:
            return _DedupOutcome(best_id=None, merged_text="", should_merge=False)
        anchor_emb_str = str(embs[0])
    tags_match = "all_strict" if tags else "any"
    # Dedup only needs the dense/keyword arms over observations — no graph, no temporal window.
    grouped = await get_memories().recall_unified(
        conn=pool,
        bank_id=bank_id,
        fact_types=["observation"],
        query_embedding=anchor_emb_str,
        query_text=anchor_text,
        limit=_DEDUP_TOP_K,
        tags=tags,
        tags_match=tags_match,
        enable_graph=False,
        temporal_window=None,
    )
    results = grouped["observation"].semantic
    best_id: str | None = None
    best_text = ""
    best_sim = threshold  # only candidates at/above the threshold are considered
    for r in results:
        rid = str(r.id)
        if exclude_id is not None and rid == exclude_id:
            continue  # never match the anchor observation against itself
        sim = r.similarity or 0.0
        if sim >= best_sim:
            best_id, best_text, best_sim = rid, r.text, sim

    if best_id is None:
        return _DedupOutcome(best_id=None, merged_text="", should_merge=False)

    decision = _dedup_decision_from_response(
        await dedup_llm_config.call(
            messages=[{"role": "user", "content": _DEDUP_PROMPT.format(new=anchor_text, existing=best_text)}],
            response_format=_DedupDecision,
            temperature=config.llm_temperature_consolidation,
            scope="consolidation_dedup",
            strict_schema=get_config().llm_strict_schema_consolidation,
        )
    )
    if decision.action != "merge":
        return _DedupOutcome(best_id=best_id, merged_text="", should_merge=False, best_text=best_text)
    merged_text = (sanitize_llm_output(decision.text) or "").strip() or best_text
    return _DedupOutcome(best_id=best_id, merged_text=merged_text, should_merge=True, best_text=best_text)


async def _apply_dedup_create_fold(
    conn,
    memory_engine: "MemoryEngine",
    bank_id: str,
    config: Any,
    outcome: _DedupOutcome,
    create_source_ids: list[uuid.UUID],
    source_bounds: _TemporalBounds,
) -> str | None:
    """Fold a CREATE the adjudicator called a duplicate into its existing twin.

    Runs on the caller's connection, inside the caller's transaction: this is one of the
    writes derived from a single consolidation LLM response, and all of them commit or roll
    back together (#3876). The slow half — the embed, the semantic probe and the
    merge-or-keep adjudication that produced ``outcome`` — already ran connection-free in
    the prepare phase (:func:`_dedup_adjudicate`).

    Returns the twin's id (the caller then skips the CREATE), or None when the fold did not
    happen and the observation must be inserted after all.

    ``source_bounds`` are the dates the skipped CREATE would have been stamped with. They are
    folded into the twin too: this path bypasses the CREATE writer, so without them the twin
    would cite dated source facts while reporting the dates of its original sources only (#3477).
    """
    if not outcome.should_merge or outcome.best_id is None:
        return None

    # Fold the new source facts into the twin and persist the merged text. The SQL path keeps the
    # twin's existing embedding (the merged text is >= threshold similar, so it stays
    # representative and avoids a re-embed + a dialect-specific vector UPDATE).
    store = get_memories()
    # Re-check liveness inside the write transaction; CREATE performed the slow embed/LLM
    # work off-connection, so sources may have been deleted since the decision was made.
    live_source_ids = await _filter_live_source_memories(conn, bank_id, create_source_ids)
    if not live_source_ids:
        return None
    if not store.store_owned_for(bank_id):
        # Oracle-safe: _native_search_vector_update emits the to_tsvector clause only for a
        # native PG tsvector column, "" otherwise (see #3021 — the raw ::regconfig cast
        # breaks Oracle). RETURNING-gate on the twin's probe-time text so a concurrent
        # survivor rewrite during the connection-free LLM window can't be clobbered.
        search_vector_clause = _native_search_vector_update(config, "$1")
        folded = await conn.fetchval(
            f"""
            UPDATE {fq_table("memory_units")}
            SET text = $1,
                source_memory_ids = (SELECT array_agg(DISTINCT e) FROM unnest(source_memory_ids || $2::uuid[]) e),
                proof_count = (SELECT count(DISTINCT e) FROM unnest(source_memory_ids || $2::uuid[]) e),
                event_date = LEAST(event_date, COALESCE($5, event_date)),
                occurred_start = LEAST(occurred_start, COALESCE($6, occurred_start)),
                occurred_end = GREATEST(occurred_end, COALESCE($7, occurred_end)),
                mentioned_at = GREATEST(mentioned_at, COALESCE($8, mentioned_at)),
                updated_at = now(){search_vector_clause}
            WHERE id = $3::uuid AND text = $4
            RETURNING id
            """,
            outcome.merged_text,
            live_source_ids,
            uuid.UUID(outcome.best_id),
            outcome.best_text,
            source_bounds.event_date,
            source_bounds.occurred_start,
            source_bounds.occurred_end,
            source_bounds.mentioned_at,
        )
        if folded is None:
            # The twin vanished (or was rewritten) during the connection-free LLM window.
            # Don't skip the CREATE: returning None lets the caller insert the observation
            # so nothing is lost.
            logger.debug(
                "[CONSOLIDATION] dedup-merge target %s vanished before fold; proceeding with CREATE",
                outcome.best_id[:8],
            )
            return None
    else:
        await _reconcile_merge_via_store(
            store,
            conn,
            memory_engine,
            bank_id,
            outcome.best_id,
            outcome.merged_text,
            live_source_ids,
            source_bounds,
        )
    return outcome.best_id


async def _apply_dedup_update_fold(
    conn,
    memory_engine: "MemoryEngine",
    bank_id: str,
    config: Any,
    outcome: _DedupOutcome,
    updated_id: str,
    updated_text: str,
) -> bool:
    """Fold a just-rewritten observation into the twin the adjudicator matched it to.

    An UPDATE rewrites an observation's text and re-embeds it, so its vector can drift to
    within threshold of a DIFFERENT existing observation. The create-time guard never sees
    this (it only runs on CREATE), so without this the two persist as a near-duplicate pair —
    the measured residual-duplicate source. On "merge", fold the just-updated observation's
    sources into the twin, persist the merged text, and DELETE the updated row. Unlike the
    CREATE path the row already exists, so reconciliation is a fold-and-delete, not a skip.

    Like :func:`_apply_dedup_create_fold` this runs on the caller's connection inside the
    caller's transaction (#3876); ``outcome`` comes from the connection-free prepare phase.
    Returns True when the updated row was folded away.
    """
    if not outcome.should_merge or outcome.best_id is None:
        return False

    store = get_memories()
    if not store.store_owned_for(bank_id):
        # Snapshot the updated row's sources with a PLAIN read (no FOR UPDATE). Lock order
        # must be sources-before-observation: _filter_live_source_memories below takes
        # FOR SHARE on the SOURCE rows first, then the fold UPDATE locks the observation
        # rows -- the same order as _apply_dedup_create_fold and the normal write paths
        # (_apply_create_observation / _apply_update_action). Locking the observation
        # here (FOR UPDATE) would invert that against the invalidation path and deadlock.
        updated_row = await conn.fetchrow(
            f"""
            SELECT source_memory_ids
            FROM {fq_table("memory_units")}
            WHERE id = $1::uuid AND text = $2
            """,
            uuid.UUID(updated_id),
            updated_text,
        )
        if updated_row is None:
            return False
        live_u_sources = await _filter_live_source_memories(conn, bank_id, list(updated_row["source_memory_ids"] or []))
        if not live_u_sources:
            return False
        # Oracle-safe search_vector clause (#3021): "" unless a native PG tsvector column.
        # RETURNING-gate on both rows' probe-time text so a survivor/updated rewrite during
        # the connection-free LLM window can't be clobbered or fold a stale row.
        search_vector_clause = _native_search_vector_update(config, "$1")
        folded = await conn.fetchval(
            f"""
            UPDATE {fq_table("memory_units")} t
            SET text = $1,
                source_memory_ids = (
                    SELECT array_agg(DISTINCT e) FROM unnest(t.source_memory_ids || $6::uuid[]) e
                ),
                proof_count = (
                    SELECT count(DISTINCT e) FROM unnest(t.source_memory_ids || $6::uuid[]) e
                ),
                event_date = LEAST(t.event_date, COALESCE(u.event_date, t.event_date)),
                occurred_start = LEAST(t.occurred_start, COALESCE(u.occurred_start, t.occurred_start)),
                occurred_end = GREATEST(t.occurred_end, COALESCE(u.occurred_end, t.occurred_end)),
                mentioned_at = GREATEST(t.mentioned_at, COALESCE(u.mentioned_at, t.mentioned_at)),
                updated_at = now(){search_vector_clause}
            FROM {fq_table("memory_units")} u
            WHERE t.id = $2::uuid AND u.id = $3::uuid AND t.text = $4 AND u.text = $5
            RETURNING t.id
            """,
            outcome.merged_text,
            uuid.UUID(outcome.best_id),
            uuid.UUID(updated_id),
            outcome.best_text,
            updated_text,
            live_u_sources,
        )
        if folded is None:
            # Twin or updated row vanished during the LLM window — keep the updated row
            # as a distinct observation instead of deleting it unfolded.
            return False
    else:
        updated_obs = await store.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[updated_id])
        updated_sources = list(updated_obs[0].source_memory_ids or []) if updated_obs else []
        live_u_sources = await _filter_live_source_memories(conn, bank_id, updated_sources)
        if not live_u_sources:
            return False
        await _reconcile_merge_via_store(
            store,
            conn,
            memory_engine,
            bank_id,
            outcome.best_id,
            outcome.merged_text,
            live_u_sources,
            _TemporalBounds.of(updated_obs[0]),
        )
    await _execute_delete_action(conn, bank_id, updated_id)
    logger.info(
        "[CONSOLIDATION] dedup-merged updated observation %s into %s (cosine>=%.2f)",
        updated_id[:8],
        outcome.best_id[:8],
        config.consolidation_dedup_threshold,
    )
    return True


@dataclass
class _BatchDeltas:
    """Per-LLM-batch deltas, merged into the job's running stats after dispatch.

    Returned by value rather than mutated into the outer ``stats`` /
    ``consolidated_tags`` so parallel batches cannot race on those shared
    structures (the merge happens once, serially, after dispatch completes).
    """

    stats: dict[str, int]
    tags: set[str]
    cancelled: bool


def _parse_observation_scopes(memory: dict[str, Any]) -> Any:
    """Parse the per-memory ``observation_scopes`` value.

    The value arrives already decoded when read through the memories store (its
    reader coerces the JSONB column) or as raw JSON text from a driver without a
    JSONB codec. A scalar mode such as ``"per_tag"`` decodes to a bare string that
    is not itself valid JSON, so a blind ``json.loads`` would raise on it — try to
    parse, but treat an unparseable string as an already-decoded scalar.
    """
    raw = memory.get("observation_scopes")
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _resolve_obs_tags_list(memory: dict[str, Any]) -> list[list[str]] | None:
    """Resolve a memory's ``observation_scopes`` spec into concrete scope tags.

    Returns ``None`` for the default ``combined``-mode single pass (caller uses
    the memory's own tags). Returns a list[list[str]] when the memory requested
    multi-pass scoping (``per_tag``, ``all_combinations``, ``shared``, or an
    explicit list).

    ``shared`` resolves to ``[[]]`` — a single pass over the empty (untagged)
    scope. The created observation carries no tags and recall/dedup match it with
    ``tags_match="any"``, so every memory consolidates into one shared observation
    regardless of its own tags. Use it to deduplicate across volatile per-call
    provenance tags (e.g. per-session ids) without dropping those tags from the
    source facts.
    """
    parsed = _parse_observation_scopes(memory)
    tags = list(memory.get("tags") or [])

    if parsed == "per_tag":
        return [[t] for t in tags] if tags else None
    if parsed == "all_combinations":
        if not tags:
            return None
        return [list(c) for r in range(1, len(tags) + 1) for c in combinations(tags, r)]
    if parsed == "shared":
        return [[]]
    if parsed == "combined" or parsed is None:
        return None
    return parsed  # explicit list[list[str]]


def _resolve_write_scopes(memory: dict[str, Any]) -> list[frozenset[str]]:
    """Return the observation scopes a memory will write to, as frozensets.

    Used by the parallel dispatcher to acquire one lock per scope before
    processing a tag group, so that two groups whose write-scope sets overlap
    serialise on the overlapping scopes rather than racing on the same
    observation row. The mapping mirrors ``_resolve_obs_tags_list`` exactly:

    - ``combined`` / ``None``    -> ``[frozenset(memory.tags)]``
    - ``per_tag``                -> ``[frozenset({t}) for t in memory.tags]``
    - ``all_combinations``       -> one frozenset per nonempty subset of tags
    - ``shared``                 -> ``[frozenset()]`` (the single untagged scope)
    - explicit ``list[list[str]]`` -> one frozenset per declared scope

    Empty-tag memories collapse to a single ``frozenset()`` in all modes so they
    still take exactly one lock and serialise against other untagged work.
    """
    parsed = _parse_observation_scopes(memory)
    tags = list(memory.get("tags") or [])

    if parsed == "per_tag":
        return [frozenset([t]) for t in tags] if tags else [frozenset()]
    if parsed == "all_combinations":
        if not tags:
            return [frozenset()]
        return [frozenset(c) for r in range(1, len(tags) + 1) for c in combinations(tags, r)]
    if parsed == "shared":
        return [frozenset()]
    if parsed == "combined" or parsed is None:
        return [frozenset(tags)]
    # Explicit list[list[str]]. An *empty* list resolves to no passes at all, and
    # the pass loop (``if obs_tags_list:``) then falls back to the combined
    # single pass over the memory's own tags — so report that scope here too,
    # or the group takes no lock for the scope it actually writes.
    return [frozenset(s) for s in parsed] or [frozenset(tags)]


def _batch_scope_signature(memory: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    """The exact set of observation scopes consolidating this memory will write.

    Derived from the pass loop rather than from the grouping key, so it can be
    used to *check* the key: a truthy ``_resolve_obs_tags_list`` yields one pass
    per resolved scope (each written with that ``obs_tags_override``), and a
    falsy one — ``None`` from ``combined``, or a degenerate empty explicit list —
    yields the single combined pass whose tags come from the memory's own tag set.

    Two memories may share an LLM call only if their signatures are equal, because
    the batch resolves its scope once from ``sub_batch[0]`` and applies it to all.
    """
    obs_tags_list = _resolve_obs_tags_list(memory)
    if not obs_tags_list:
        return (tuple(sorted(memory.get("tags") or [])),)
    return tuple(sorted(tuple(sorted(scope)) for scope in obs_tags_list))


def _consolidation_batch_key(memory: dict[str, Any]) -> tuple[str, ...]:
    """Return the key that decides which memories may share an LLM batch.

    The real security requirement is "memories targeting different observation
    scopes must never share an LLM call" — every branch below keys on the
    memory's *resolved* scope(s), never on raw tags, so two memories with the
    same native tags but different ``observation_scopes`` modes can never
    collide into the same group:

    - default ``combined`` (``resolved is None``), and the degenerate empty
      resolution (an explicit ``[]``, which the pass loop below treats as
      combined because ``if obs_tags_list:`` is falsy): the memory's target
      scope *is* its own tag set, so it keys on those tags.
    - a single alternate scope (``shared``, an explicit one-scope list, or
      ``per_tag`` with exactly one tag): keys on that resolved scope instead,
      so it batches with any other memory naming the identical scope
      regardless of native tags — that is the whole point of requesting it.
    - fan-out to *multiple* scopes (``per_tag`` with more than one tag,
      ``all_combinations``, or a multi-scope explicit list): writes several
      observations from one LLM call, so it keys on the full resolved
      scope-list, not native tags — two fan-out memories only share a batch
      when every one of their target scopes matches exactly. A distinct
      leading marker per branch keeps e.g. a ``combined`` memory tagged
      ``["a","b"]`` out of the same key as a ``per_tag`` memory tagged
      ``["a","b"]``, even though both would otherwise sort to ``("a","b")``.
    """
    resolved = _resolve_obs_tags_list(memory)
    if not resolved:
        # ``None`` (combined) and ``[]`` (an explicit empty scope list) both fall
        # through to the combined pass downstream — ``if obs_tags_list:`` is
        # falsy for each — so they must key the same way. Testing ``is None``
        # alone sent ``[]`` down the fan-out branch, where every such memory
        # keyed to the tag-free ``("fanout",)`` and pooled with memories of
        # unrelated tags; ``obs_tags_override`` then being ``None``, the whole
        # group took ``memories[0]``'s tags and leaked across scopes.
        return ("combined", *sorted(memory.get("tags") or []))
    if len(resolved) == 1:
        return ("scope", *sorted(resolved[0]))
    return ("fanout", *sorted("\x1f".join(sorted(scope)) for scope in resolved))


def _scope_sort_key(scope: frozenset[str]) -> tuple[str, ...]:
    """Total ordering on scope frozensets for deadlock-free lock acquisition.

    Every parallel group acquires its scope locks in this same order, so two
    groups that share any subset of scopes cannot acquire them in opposite
    orders and deadlock.
    """
    return tuple(sorted(scope))


async def _filter_live_source_memories(
    conn: "Connection",
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
) -> list[uuid.UUID]:
    """Return only the source memory ids that still exist in the bank.

    The SQL store takes a ``FOR SHARE`` lock on the surviving rows so a concurrent
    delete can't remove one between this check and the observation write that
    follows — without it, a source deleted in that window would leave an orphan
    observation until the next sweep, because the delete path's stale-observation
    sweep only catches observations that already exist when it runs. (Oracle has no
    ``FOR SHARE``; the SQL rewriter promotes it to ``FOR UPDATE`` — more
    conservative, still correct.) A store that keeps memories outside SQL has its
    own concurrency model, so it answers with an unlocked existence check.
    """
    if not source_memory_ids:
        return []
    store = get_memories()
    if not store.store_owned_for(bank_id):
        rows = await conn.fetch(
            f"SELECT id FROM {fq_table('memory_units')} WHERE id = ANY($1::uuid[]) AND bank_id = $2 FOR SHARE",
            source_memory_ids,
            bank_id,
        )
        live = {str(r["id"]) for r in rows}
    else:
        present = await store.get_memories(
            conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[str(mid) for mid in source_memory_ids]
        )
        live = {str(m.unit_id) for m in present}
    return [mid for mid in source_memory_ids if str(mid) in live]


async def _any_live_source_memory(
    conn: "Connection",
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
) -> bool:
    """Cheap, non-locking existence check used as a preflight before embedding.

    Lets the create/update executors skip the (slow) embedder when every source
    memory is already gone, restoring the pre-refactor short-circuit. The
    authoritative, FOR SHARE liveness check still runs inside the write transaction.
    """
    if not source_memory_ids:
        return False
    store = get_memories()
    if not store.store_owned_for(bank_id):
        found = await conn.fetchval(
            f"SELECT 1 FROM {fq_table('memory_units')} WHERE id = ANY($1::uuid[]) AND bank_id = $2 LIMIT 1",
            source_memory_ids,
            bank_id,
        )
        return found is not None
    present = await store.get_memories(
        conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[str(mid) for mid in source_memory_ids]
    )
    return bool(present)


class _CreateAction(BaseModel):
    text: str
    source_fact_ids: list[str]  # memory UUIDs from the NEW FACTS list
    # One-sentence justification from the LLM (why CREATE vs UPDATE). Diagnostic
    # only — surfaced in the consolidation trace to explain duplicate creates.
    reason: str = ""

    @field_validator("text", mode="before")
    @classmethod
    def sanitize_text(cls, v: str) -> str:
        return sanitize_llm_output(v) or ""

    @field_validator("source_fact_ids", mode="before")
    @classmethod
    def ensure_list(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [v]
        return v


class _UpdateAction(BaseModel):
    text: str
    observation_id: str  # UUID of the existing observation to update
    source_fact_ids: list[str]  # memory UUIDs from the NEW FACTS list
    reason: str = ""  # LLM's one-sentence justification (diagnostic only)

    @field_validator("text", mode="before")
    @classmethod
    def sanitize_text(cls, v: str) -> str:
        return sanitize_llm_output(v) or ""

    @field_validator("source_fact_ids", mode="before")
    @classmethod
    def ensure_list(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [v]
        return v


class _DeleteAction(BaseModel):
    observation_id: str  # UUID of the observation to remove
    reason: str = ""  # LLM's one-sentence justification (diagnostic only)


class _ConsolidationBatchResponse(BaseModel):
    creates: list[_CreateAction] = []
    updates: list[_UpdateAction] = []
    deletes: list[_DeleteAction] = []


@dataclass
class _PreparedUpdate:
    """One UPDATE from an LLM response, with every slow step already done.

    Consolidation applies all of a response's writes in a single transaction (#3876), and a
    transaction must never be held open across an embedder or LLM call. So each action is
    first *prepared* connection-free — the source facts resolved and security-checked, the
    new text embedded, the semantic-dedup verdict adjudicated — and the resulting value
    object carries everything the write needs.
    """

    update: _UpdateAction
    #: Pre-update snapshot of the observation, from the batch's recall.
    model: "MemoryFact"
    source_mems: list[dict[str, Any]]
    source_memory_ids: list[uuid.UUID]
    source_fact_tags: list[str]
    source_bounds: _TemporalBounds
    embedding_str: str | None
    #: Fold target for the rewritten observation, when create/update dedup is enabled and
    #: the re-embed drifted it into a near-twin.
    dedup: _DedupOutcome | None = None


@dataclass
class _PreparedCreate:
    """One CREATE from an LLM response, with every slow step already done.

    See :class:`_PreparedUpdate`. ``dedup`` here is a fold into an existing near-twin,
    which replaces the insert rather than following it.
    """

    text: str
    source_mems: list[dict[str, Any]]
    source_memory_ids: list[uuid.UUID]
    source_fact_tags: list[str]
    agg: "_SourceAggregation"
    embedding_str: str | None
    dedup: _DedupOutcome | None = None


@dataclass
class _BatchLLMResult:
    creates: list[_CreateAction] = field(default_factory=list)
    updates: list[_UpdateAction] = field(default_factory=list)
    deletes: list[_DeleteAction] = field(default_factory=list)
    obs_count: int = 0
    prompt_chars: int = 0
    failed: bool = False


@dataclass
class _SourceAggregation:
    """Fields inherited by an observation from its source memories."""

    event_date: datetime | None
    occurred_start: datetime | None
    occurred_end: datetime | None
    mentioned_at: datetime | None
    tags: list[str]


def _aggregate_source_fields(source_mems: list[dict[str, Any]], tags: list[str] | None = None) -> _SourceAggregation:
    """Compute the observation fields inherited from a set of source memories.

    Temporal aggregation rules:
    - ``event_date``    — earliest across sources (min)
    - ``occurred_start`` — earliest across sources (min)
    - ``occurred_end``   — latest across sources (max)
    - ``mentioned_at``   — latest across sources (max)

    Fields remain ``None`` when no source memory carries that information, so
    observations are never stamped with an artificial timestamp.

    ``tags`` defaults to those of the first source memory when not explicitly
    provided (all memories in a consolidation batch share the same tag set).
    """
    effective_tags = tags if tags is not None else (source_mems[0].get("tags") or [] if source_mems else [])
    return _SourceAggregation(
        event_date=_min_date(m.get("event_date") for m in source_mems),
        occurred_start=_min_date(m.get("occurred_start") for m in source_mems),
        occurred_end=_max_date(m.get("occurred_end") for m in source_mems),
        mentioned_at=_max_date(m.get("mentioned_at") for m in source_mems),
        tags=effective_tags,
    )


async def _count_observations_for_scope(
    conn: "Connection",
    bank_id: str,
    tags: list[str],
) -> int:
    """Count existing observations matching the given tag scope.

    Returns the count of observations whose tags contain all specified tags.
    Observations with no tags are not counted (the limit does not apply to them).
    """
    store = get_memories()
    if not store.store_owned_for(bank_id):
        return await conn.fetchval(
            f"SELECT COUNT(*) FROM {fq_table('memory_units')} "
            f"WHERE bank_id = $1 AND fact_type = 'observation' AND tags @> $2::varchar[]",
            bank_id,
            tags,
        )
    # A store that keeps observations outside Postgres: count them through it (tag containment).
    total = 0
    page_token = ""
    for _ in range(100):
        page = await store.scan_memories(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            fact_types=["observation"],
            tags=tags or None,
            tags_match="all",
            limit=500,
            page_token=page_token,
        )
        total += len(page.memories)
        page_token = page.next_page_token
        if not page_token:
            break
    return total


@dataclass(frozen=True)
class _ScopeLimitRule:
    """One ``observation_scope_limits`` rule: a scope pattern -> an observation cap.

    ``globs`` is a tuple of fnmatch tag-globs describing one consolidation scope.
    A concrete scope (the set of ``fact_tags`` for a consolidation pass) matches
    under *exact cover*: every tag is matched by some glob AND every glob matches
    some tag. So ``["shared"]`` matches the scope ``{shared}`` but not
    ``{run_1, shared}``, and ``["run_*", "shared"]`` matches ``{run_1, shared}``
    but not ``{shared}``.

    ``limit`` is the cap applied to matching scopes (-1 = unlimited, 0 = no new
    observations, >0 = hard cap), mirroring ``max_observations_per_scope``.
    """

    globs: tuple[str, ...]
    limit: int


def _parse_scope_limit_rules(raw: Any) -> list[_ScopeLimitRule]:
    """Parse the raw ``observation_scope_limits`` config into ordered rules.

    The config round-trips as JSON through env and the bank-config API, so this
    is defensive: malformed entries are skipped rather than raising, and list
    order is preserved (first match wins in :func:`_effective_scope_limit`).
    """
    if not isinstance(raw, list):
        return []
    rules: list[_ScopeLimitRule] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        scope = entry.get("scope")
        limit = entry.get("limit")
        if not isinstance(scope, list) or not scope:
            continue
        if not all(isinstance(g, str) and g for g in scope):
            continue
        # bool is an int subclass — reject True/False masquerading as a limit.
        if not isinstance(limit, int) or isinstance(limit, bool):
            continue
        rules.append(_ScopeLimitRule(globs=tuple(scope), limit=limit))
    return rules


def _scope_matches_globs(globs: tuple[str, ...], tags: list[str]) -> bool:
    """Exact-cover match between a scope pattern and a concrete tag set.

    True iff every tag is covered by at least one glob AND every glob covers at
    least one tag (no uncovered tags, no vacuous globs). Untagged scopes never
    match, so a scope limit never applies to untagged observations (consistent
    with the ``and fact_tags`` guard at the call site). Matching is
    case-sensitive (``fnmatchcase``) for deterministic cross-platform behaviour.
    """
    tagset = set(tags)
    if not tagset:
        return False
    if not all(any(fnmatchcase(t, g) for g in globs) for t in tagset):
        return False
    if not all(any(fnmatchcase(t, g) for t in tagset) for g in globs):
        return False
    return True


def _effective_scope_limit(config: Any, fact_tags: list[str]) -> int:
    """Resolve the observation cap for one concrete consolidation scope.

    The first rule in ``observation_scope_limits`` whose pattern exact-covers
    ``fact_tags`` wins; otherwise falls back to the bank-wide
    ``max_observations_per_scope``. Wildcards live only here, matched against the
    already-resolved concrete tags — the SQL count stays exact and indexed.
    """
    if config is None:
        return -1
    for rule in _parse_scope_limit_rules(config.observation_scope_limits):
        if _scope_matches_globs(rule.globs, fact_tags):
            return rule.limit
    return config.max_observations_per_scope


def _build_response_model(
    max_creates: int | None = None,
    *,
    supports_max_items: bool = True,
) -> type[_ConsolidationBatchResponse]:
    """Build a response model, optionally constraining creates via JSON schema.

    Some structured-output backends (notably Bedrock Converse) reject the JSON
    Schema ``maxItems`` keyword emitted by Pydantic's list ``max_length``. Operators
    can disable the schema hint for those backends; the prompt capacity note and
    post-response truncation still enforce the observation cap.
    """
    if not supports_max_items or max_creates is None or max_creates < 0:
        return _ConsolidationBatchResponse

    from pydantic import Field as PydanticField

    clamped = max(max_creates, 0)

    class _ConstrainedConsolidationBatchResponse(_ConsolidationBatchResponse):
        creates: list[_CreateAction] = PydanticField(default=[], max_length=clamped)

    return _ConstrainedConsolidationBatchResponse


class ConsolidationPerfLog:
    """Performance logging for consolidation operations."""

    def __init__(self, bank_id: str):
        self.bank_id = bank_id
        self.start_time = time.time()
        self.lines: list[str] = []
        self.timings: dict[str, float] = {}
        self.timing_counts: dict[str, int] = {}
        self.llm_calls: int = 0
        self.total_obs_in_context: int = 0
        self.total_prompt_chars: int = 0

    def log(self, message: str) -> None:
        """Add a log line."""
        self.lines.append(message)

    def record_timing(self, key: str, duration: float) -> None:
        """Record a timing measurement.

        Tracks both total seconds and call count so the summary can
        distinguish one slow call from many fast calls in aggregate.
        """
        self.timings[key] = self.timings.get(key, 0.0) + duration
        self.timing_counts[key] = self.timing_counts.get(key, 0) + 1

    def record_llm_call(self, obs_count: int, prompt_chars: int) -> None:
        """Record stats for a single LLM call."""
        self.llm_calls += 1
        self.total_obs_in_context += obs_count
        self.total_prompt_chars += prompt_chars

    def merge_from(self, other: "ConsolidationPerfLog") -> None:
        """Merge a per-batch perf log into this (job-level) one.

        Used by the parallel dispatcher: each in-flight batch records into its
        own ``ConsolidationPerfLog`` so the per-batch log line shows only that
        batch's timings (no cross-batch interleaving). After the batch finishes
        we fold the local counters into the job-level perf, which then drives
        the final ``flush()`` summary.

        ``lines`` is intentionally NOT merged — log lines are emitted directly
        in ``logger.info`` calls by the dispatcher; the perf object's ``lines``
        buffer is only used by the top-level job summary.
        """
        for key, value in other.timings.items():
            self.timings[key] = self.timings.get(key, 0.0) + value
        for key, count in other.timing_counts.items():
            self.timing_counts[key] = self.timing_counts.get(key, 0) + count
        self.llm_calls += other.llm_calls
        self.total_obs_in_context += other.total_obs_in_context
        self.total_prompt_chars += other.total_prompt_chars

    def flush(self) -> None:
        """Flush all log lines to the logger."""
        total_time = time.time() - self.start_time
        header = f"\n{'=' * 60}\nCONSOLIDATION for bank {self.bank_id}"
        footer = f"{'=' * 60}\nCONSOLIDATION COMPLETE: {total_time:.3f}s total\n{'=' * 60}"

        log_output = header + "\n" + "\n".join(self.lines) + "\n" + footer
        logger.info(log_output)


def _as_dt(v: "datetime | str | None") -> "datetime | None":
    """Coerce an ISO string to a datetime. Recall results can carry timestamps as strings while
    the store's addressed reads hand back datetimes, so normalise before comparing."""
    return datetime.fromisoformat(v) if isinstance(v, str) else v


def _merge_min(a: "datetime | str | None", b: "datetime | str | None") -> "datetime | None":
    """SQL ``LEAST(a, COALESCE(b, a))`` in Python: the earlier of two times, ignoring None."""
    a, b = _as_dt(a), _as_dt(b)
    return a if b is None else b if a is None else min(a, b)


def _merge_max(a: "datetime | str | None", b: "datetime | str | None") -> "datetime | None":
    """SQL ``GREATEST(a, COALESCE(b, a))`` in Python: the later of two times, ignoring None."""
    a, b = _as_dt(a), _as_dt(b)
    return a if b is None else b if a is None else max(a, b)


async def _reconcile_merge_via_store(
    store,
    conn,
    memory_engine: "MemoryEngine",
    bank_id: str,
    observation_id: str,
    merged_text: str,
    add_source_ids: list,
    add_bounds: _TemporalBounds,
) -> None:
    """Dedup merge for a store that owns its rows: fold the extra source facts and the merged text
    into the twin observation and re-upsert it, preserving its other fields. Re-embeds the merged
    text because ``get_memories`` does not return the stored vector (the SQL path reuses it in
    place instead).

    ``add_bounds`` are the folded-in side's dates, widened onto the twin exactly as the SQL
    path's LEAST/GREATEST does."""
    current = await store.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[observation_id])
    cur = current[0] if current else None
    if cur is None:
        return
    merged_sources = list(dict.fromkeys([*(cur.source_memory_ids or []), *(str(s) for s in add_source_ids)]))
    merged_bounds = _TemporalBounds.of(cur).merged_with(add_bounds)
    embeddings = await embedding_utils.generate_embeddings_batch(memory_engine.embeddings, [merged_text])
    await store.upsert_observation(
        conn=conn,
        bank_id=bank_id,
        record=FactRecord(
            unit_id=observation_id,
            text=merged_text,
            embedding=str(embeddings[0]) if embeddings else None,
            fact_type="observation",
            tags=list(cur.tags or []),
            proof_count=len(merged_sources),
            source_memory_ids=merged_sources,
            event_date=merged_bounds.event_date,
            occurred_start=merged_bounds.occurred_start,
            occurred_end=merged_bounds.occurred_end,
            mentioned_at=merged_bounds.mentioned_at,
            created_at=cur.created_at,
        ),
    )


async def _fetch_unconsolidated_rows(
    conn,
    bank_id: str,
    fact_types: list[str],
    limit: int,
    observation_scopes: list[list[str]] | None,
) -> list[dict[str, Any]]:
    """Unconsolidated candidate facts, read through the memories store.

    The store owns the memories, so this must ask it rather than query ``memory_units``
    directly — otherwise a store that keeps its rows elsewhere yields nothing and
    consolidation silently produces no observations. Returns the same row-dict shape the
    consolidation loop consumes. Mirrors the job's scope filter: with scopes, OR each
    "tags ⊇ scope" and merge oldest-first; without, one unscoped read.
    """
    store = get_memories()
    scopes: list[list[str] | None] = list(observation_scopes) if observation_scopes else [None]
    by_id: dict[str, Any] = {}
    for scope in scopes:
        for m in await store.find_unconsolidated(
            conn=conn, fq_table=fq_table, bank_id=bank_id, fact_types=fact_types, limit=limit, scope_tags=scope
        ):
            by_id.setdefault(m.unit_id, m)
    ordered = sorted(by_id.values(), key=lambda m: (m.created_at is None, m.created_at))[:limit]
    return [
        {
            "id": uuid.UUID(m.unit_id),
            "text": m.text,
            "fact_type": m.fact_type,
            "occurred_start": m.occurred_start,
            "occurred_end": m.occurred_end,
            "event_date": m.event_date,
            "tags": list(m.tags or []),
            "mentioned_at": m.mentioned_at,
            "observation_scopes": m.observation_scopes,
        }
        for m in ordered
    ]


#: Cap on the store-side count of unconsolidated facts. Used only for the "is there work?"
#: gate and progress reporting, so a floor at this size is harmless on a huge backlog.
_COUNT_LIMIT = 100_000


async def _count_unconsolidated_rows(
    conn,
    bank_id: str,
    fact_types: list[str],
    observation_scopes: list[list[str]] | None,
) -> int:
    """Count of unconsolidated candidate facts, from the store (bounded by ``_COUNT_LIMIT``).

    Asks the store for a *count* rather than fetching the rows and taking ``len`` — on the SQL
    store that is one bounded ``COUNT(*)`` instead of shipping up to ``_COUNT_LIMIT`` full memory
    rows across the wire on every job start / progress tick.
    """
    scopes: list[list[str] | None] = list(observation_scopes) if observation_scopes else [None]
    return await get_memories().count_unconsolidated(
        conn=conn, fq_table=fq_table, bank_id=bank_id, fact_types=fact_types, scopes=scopes, limit=_COUNT_LIMIT
    )


def _as_op_uuid(operation_id: str | uuid.UUID) -> uuid.UUID:
    return uuid.UUID(operation_id) if isinstance(operation_id, str) else operation_id


async def _persist_pending_refresh_tags(conn, operation_id: str, new_tags: list[str]) -> None:
    """Union ``new_tags`` into the consolidation op's durable ``pending_refresh_tags``.

    Called inside each batch's witness transaction, so the tags of an
    already-consolidated batch are durable the instant that batch is — a mid-round
    worker crash no longer loses them. On retry the op re-reads ``task_payload`` and the
    final round still refreshes those models (#3411); without this, a crash after batch 1
    committed but before the round finished would drop batch 1's tags, because the retry
    skips its now-consolidated rows and never re-collects them. ``SELECT ... FOR UPDATE``
    serialises the concurrent batches of one op so their unions don't clobber each other.
    """
    op_uuid = _as_op_uuid(operation_id)
    row = await conn.fetchrow(
        f"SELECT task_payload FROM {fq_table('async_operations')} WHERE operation_id = $1 FOR UPDATE",
        op_uuid,
    )
    if row is None:
        return
    payload = row["task_payload"]
    payload = json.loads(payload) if isinstance(payload, str) else (payload or {})
    existing = set(payload.get("pending_refresh_tags") or [])
    merged = existing | set(new_tags)
    if merged == existing:
        return
    payload["pending_refresh_tags"] = sorted(merged)
    await conn.execute(
        f"UPDATE {fq_table('async_operations')} SET task_payload = $1::jsonb, updated_at = now() "
        f"WHERE operation_id = $2",
        json.dumps(payload),
        op_uuid,
    )


async def _read_pending_refresh_tags(pool, operation_id: str) -> set[str]:
    """Read the op's durably-accumulated ``pending_refresh_tags`` (crash-safe source of
    truth for the final-round flush)."""
    async with acquire_with_retry(pool) as conn:
        row = await conn.fetchrow(
            f"SELECT task_payload FROM {fq_table('async_operations')} WHERE operation_id = $1",
            _as_op_uuid(operation_id),
        )
    if row is None:
        return set()
    payload = row["task_payload"]
    payload = json.loads(payload) if isinstance(payload, str) else (payload or {})
    return set(payload.get("pending_refresh_tags") or [])


async def run_consolidation_job(
    memory_engine: "MemoryEngine",
    bank_id: str,
    request_context: "RequestContext",
    operation_id: str | None = None,
    observation_scopes: list[list[str]] | None = None,
    pending_refresh_tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run consolidation job for a bank.

    This is called after retain operations to consolidate new memories into mental models.

    Args:
        memory_engine: MemoryEngine instance
        bank_id: Bank identifier
        request_context: Request context for authentication
        operation_id: Optional operation ID for tracking
        observation_scopes: Optional list of tag scopes. When provided, only
            unconsolidated memories whose tags contain all tags in at least one
            scope are processed.
        pending_refresh_tags: Tags of memories consolidated by earlier rounds of this
            round-limited chain, carried through the re-queue so the final round can
            refresh every affected mental model exactly once (#3411).

    Returns:
        Dict with consolidation results
    """
    # Resolve bank-specific config with hierarchical overrides
    config = await memory_engine._config_resolver.resolve_full_config(bank_id, request_context)

    # Build a configured LLM wrapper that applies per-bank settings (e.g. safety settings)
    # to every call without leaking across operations.
    llm_config = memory_engine._consolidation_llm_config.with_config(config, bank_id=bank_id, operation="consolidation")

    # Bind the operation trace context for the whole run so the create/update DB
    # sites (deep inside _process_memory_batch) can accumulate the observations
    # this consolidation produced and the source memories it consumed onto the
    # trace — flushed onto every trace row on exit by attach_memory_ids.
    trace_ctx = trace_context_of(llm_config)
    trace_token = set_trace_context(trace_ctx) if trace_ctx is not None else None
    try:
        return await _run_consolidation_job(
            memory_engine,
            bank_id,
            request_context,
            config,
            llm_config,
            operation_id,
            observation_scopes,
            pending_refresh_tags,
        )
    finally:
        if trace_token is not None:
            reset_trace_context(trace_token)
            # Fire-and-forget: patched on a background task, off the consolidation
            # critical path.
            memory_engine._llm_recorder.attach_memory_ids(trace_ctx)


async def _run_consolidation_job(
    memory_engine: "MemoryEngine",
    bank_id: str,
    request_context: "RequestContext",
    config: Any,
    llm_config: Any,
    operation_id: str | None = None,
    observation_scopes: list[list[str]] | None = None,
    pending_refresh_tags: list[str] | None = None,
) -> dict[str, Any]:
    """Core consolidation flow. See ``run_consolidation_job`` for the public entrypoint."""
    perf = ConsolidationPerfLog(bank_id)
    max_memories_per_batch = config.consolidation_batch_size
    max_memories_per_round = config.consolidation_max_memories_per_round
    llm_batch_size = max(1, config.consolidation_llm_batch_size)

    # Check if consolidation is enabled
    if not config.enable_observations:
        logger.debug(f"Consolidation disabled for bank {bank_id}")
        return {"status": "disabled", "bank_id": bank_id}

    pool = memory_engine._backend

    # Get bank profile
    async with acquire_with_retry(pool) as conn:
        t0 = time.time()
        bank_row = await conn.fetchrow(
            f"""
            SELECT bank_id, name
            FROM {fq_table("banks")}
            WHERE bank_id = $1
            """,
            bank_id,
        )

        if not bank_row:
            logger.warning(f"Bank {bank_id} not found for consolidation")
            return {"status": "bank_not_found", "bank_id": bank_id}

        perf.record_timing("fetch_bank", time.time() - t0)

        # Count total unconsolidated memories for progress logging — through the store.
        total_count = await _count_unconsolidated_rows(conn, bank_id, ["experience", "world"], observation_scopes)

    if total_count == 0:
        logger.debug(f"No new memories to consolidate for bank {bank_id}")
        return {"status": "no_new_memories", "bank_id": bank_id, "memories_processed": 0}

    logger.info(f"[CONSOLIDATION] bank={bank_id} total_unconsolidated={total_count}")
    perf.log(f"[1] Found {total_count} pending memories to consolidate")

    # Initial durable progress snapshot so an operator polling the operation status
    # API sees the job has started and how much work it found, before the first batch
    # of LLM work completes (which can take minutes on a dense bank). Uses the same
    # "consolidating" stage as the per-batch heartbeat so the operator sees a single
    # phase advancing 0/N -> N/N rather than an opaque "scanning" -> "processing" hop.
    set_stage("consolidation.consolidating")
    await memory_engine._write_operation_progress(operation_id, stage="consolidating", processed=0, total=total_count)

    async def _count_unconsolidated() -> int:
        """Re-count memories still pending consolidation in this job's scope.

        ``total_count`` is a point-in-time estimate from job start; memories retained
        while consolidation runs get picked up by later fetches, so processed can pass
        it. When that happens we re-count to report a real total (processed + remaining)
        instead of pinning the bar at 100%."""
        async with acquire_with_retry(pool) as count_conn:
            return await _count_unconsolidated_rows(count_conn, bank_id, ["experience", "world"], observation_scopes)

    async def _progress_total(processed: int) -> int:
        # Cheap path: while we're still within the start-of-job estimate it's exact, so
        # no extra query. Only re-count once the estimate is exhausted (≈the final batch
        # normally, or repeatedly only if memories keep arriving mid-run).
        if processed < total_count:
            return total_count
        return processed + await _count_unconsolidated()

    # Process each memory with individual commits for crash recovery
    stats: dict[str, int] = {
        "memories_processed": 0,
        "observations_created": 0,
        "observations_updated": 0,
        "observations_merged": 0,
        "observations_deleted": 0,
        "actions_executed": 0,
        "skipped": 0,
        "memories_failed": 0,
    }

    # Track all unique tags from consolidated memories for mental model refresh filtering
    consolidated_tags: set[str] = set()

    round_limit_enabled = max_memories_per_round > 0
    round_remaining = max_memories_per_round if round_limit_enabled else float("inf")
    hit_round_limit = False

    llm_batch_num = 0
    # Cumulative counters across the whole job, shared by the per-batch log and the
    # durable progress snapshot so both report processed/total (and observation
    # tallies) under parallelism. Mutable container so the inner closure can update
    # without a `nonlocal`.
    cumulative_progress = {
        "processed": 0,
        "observations_created": 0,
        "observations_updated": 0,
        "observations_merged": 0,
        "observations_deleted": 0,
        "memories_failed": 0,
    }
    while True:
        # Cap fetch size by remaining round budget
        fetch_limit = (
            min(max_memories_per_batch, int(round_remaining)) if round_limit_enabled else max_memories_per_batch
        )

        # Fetch next batch of unconsolidated memories — through the store, so a store that
        # keeps its rows outside Postgres is read too.
        async with acquire_with_retry(pool) as conn:
            t0 = time.time()
            memories = await _fetch_unconsolidated_rows(
                conn, bank_id, ["experience", "world"], fetch_limit, observation_scopes
            )
            perf.record_timing("fetch_memories", time.time() - t0)

        if not memories:
            break  # No more unconsolidated memories

        # Group memories by target observation scope before batching — security
        # requirement: memories targeting different scopes must never share an
        # LLM call. See _consolidation_batch_key for why that's scope, not raw
        # tags: a memory requesting observation_scopes="shared" (or another
        # single-scope override) targets a scope that can differ from its own
        # tags, and must only batch with peers naming that same scope (#3924).
        tag_groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for m in memories:
            tag_key = _consolidation_batch_key(m)
            tag_groups.setdefault(tag_key, []).append(dict(m))

        # Split each tag group into LLM batches respecting llm_batch_size, keeping
        # the group boundary intact so the dispatcher can parallelise across
        # distinct groups while running each group's batches serially.
        grouped_batches: list[list[list[dict[str, Any]]]] = []
        for group in tag_groups.values():
            grouped_batches.append([group[i : i + llm_batch_size] for i in range(0, len(group), llm_batch_size)])

        # Compute each group's union write-scope set. Used below to acquire
        # per-scope locks: any two groups whose write-scope sets share a scope S
        # will serialise on the lock for S, leaving truly disjoint groups to run
        # concurrently. We union over every memory because per-memory
        # observation_scopes can differ within a group.
        group_scopes: list[list[frozenset[str]]] = []
        for batches in grouped_batches:
            scopes: set[frozenset[str]] = set()
            for batch in batches:
                for memory in batch:
                    scopes.update(_resolve_write_scopes(memory))
            group_scopes.append(sorted(scopes, key=_scope_sort_key))

        async def _process_one_llm_batch(llm_batch_local: list[dict[str, Any]], batch_num_local: int) -> _BatchDeltas:
            """Process one LLM batch independently. Returns local deltas + cancelled flag.

            Each batch records timings/llm-call counters into its OWN
            ``ConsolidationPerfLog`` so the per-batch log line reflects only
            this batch's work — not interleaved timings from concurrent batches
            sharing the global ``perf``. The local perf is merged into the
            job-level ``perf`` once at the end so the final summary still totals
            everything.
            """
            llm_batch_start = time.time()
            batch_perf = ConsolidationPerfLog(bank_id)

            local_tags: set[str] = set()
            for memory in llm_batch_local:
                memory_tags = memory.get("tags") or []
                if memory_tags:
                    local_tags.update(memory_tags)

            # Adaptive splitting: on LLM failure, halve the sub-batch and retry,
            # down to batch_size=1. Only if a single-memory batch still fails is
            # the memory marked with consolidation_failed_at.
            all_results: list[dict[str, Any]] = []
            all_deleted = 0
            succeeded_ids: list[Any] = []
            failed_ids: list[Any] = []

            pending: list[list[dict[str, Any]]] = [llm_batch_local]
            while pending:
                sub_batch = pending.pop(0)

                # Defence in depth. Everything below writes the sub-batch at ONE
                # resolved scope, read off ``sub_batch[0]``. If a memory with a
                # different scope ever reaches this batch — a regression in
                # ``_consolidation_batch_key``, or in how groups are split into
                # batches — that single read stamps one memory's scope onto
                # another's observation: an untagged, globally recallable
                # observation built from a tagged fact, or a dropped override
                # (#3953). So verify the invariant against the scopes the pass
                # loop will actually write, independently of the grouping key,
                # and split instead of trusting it. The cost in the case that
                # must never happen is one extra LLM call.
                if len(sub_batch) > 1:
                    by_scope: dict[tuple[tuple[str, ...], ...], list[dict[str, Any]]] = {}
                    for m in sub_batch:
                        by_scope.setdefault(_batch_scope_signature(m), []).append(m)
                    if len(by_scope) > 1:
                        logger.error(
                            f"[CONSOLIDATION] bank={bank_id} sub-batch of {len(sub_batch)} memories mixes"
                            f" {len(by_scope)} observation scopes {sorted(by_scope)} — splitting it."
                            " LLM batches must be scope-homogeneous; this is a grouping bug."
                        )
                        pending[0:0] = list(by_scope.values())
                        continue

                # No connection is held across the batch: recall, the main LLM call, the
                # per-action embeds, and dedup adjudication all run connection-free. Only then
                # does the sub-batch take one connection and write everything that LLM response
                # decided — observations and consolidated_at stamps alike — in one transaction.
                obs_tags_list = _resolve_obs_tags_list(sub_batch[0]) if sub_batch else None

                sub_deleted: int = 0
                sub_llm_failed = False
                sub_ids = [m["id"] for m in sub_batch]
                if obs_tags_list:
                    sub_results: list[dict[str, Any]] = []
                    for pass_index, obs_tags in enumerate(obs_tags_list):
                        # A memory consolidated at several tag scopes gets one LLM call per
                        # scope; the ``consolidated_at`` stamp belongs to the last of them, so
                        # an earlier scope's write and the stamp never commit apart (#3876).
                        is_final_pass = pass_index == len(obs_tags_list) - 1
                        pass_results, pass_deleted, pass_failed = await _process_memory_batch(
                            pool=pool,
                            memory_engine=memory_engine,
                            llm_config=llm_config,
                            bank_id=bank_id,
                            memories=sub_batch,
                            request_context=request_context,
                            perf=batch_perf,
                            config=config,
                            obs_tags_override=obs_tags,
                            mark_consolidated_ids=sub_ids if is_final_pass else None,
                        )
                        sub_deleted += pass_deleted
                        if pass_failed:
                            # Stop the remaining scopes: the sub-batch is going to be bisected
                            # and re-run in full, and every write this pass made is already
                            # committed. Running the rest would only add writes to discard.
                            sub_llm_failed = True
                            break
                        if not sub_results:
                            sub_results = pass_results
                        else:
                            for i, (existing, new) in enumerate(zip(sub_results, pass_results)):
                                if existing.get("action") == "skipped" and new.get("action") != "skipped":
                                    sub_results[i] = new
                                elif existing.get("action") != "skipped" and new.get("action") != "skipped":
                                    existing_created = existing.get(
                                        "created", 1 if existing.get("action") == "created" else 0
                                    )
                                    existing_updated = existing.get(
                                        "updated", 1 if existing.get("action") == "updated" else 0
                                    )
                                    new_created = new.get("created", 1 if new.get("action") == "created" else 0)
                                    new_updated = new.get("updated", 1 if new.get("action") == "updated" else 0)
                                    total = existing_created + existing_updated + new_created + new_updated
                                    sub_results[i] = {
                                        "action": "multiple",
                                        "created": existing_created + new_created,
                                        "updated": existing_updated + new_updated,
                                        "merged": 0,
                                        "total_actions": total,
                                    }
                else:
                    sub_results, sub_deleted, sub_llm_failed = await _process_memory_batch(
                        pool=pool,
                        memory_engine=memory_engine,
                        llm_config=llm_config,
                        bank_id=bank_id,
                        memories=sub_batch,
                        request_context=request_context,
                        perf=batch_perf,
                        config=config,
                        mark_consolidated_ids=sub_ids,
                    )

                all_deleted += sub_deleted

                if sub_llm_failed and len(sub_batch) > 1:
                    mid = len(sub_batch) // 2
                    logger.warning(
                        f"[CONSOLIDATION] bank={bank_id} LLM failed for sub-batch of {len(sub_batch)},"
                        f" splitting into {mid}/{len(sub_batch) - mid}"
                    )
                    pending[0:0] = [sub_batch[:mid], sub_batch[mid:]]
                elif sub_llm_failed:
                    failed_ids.append(sub_batch[0]["id"])
                    all_results.append({"action": "failed"})
                    logger.warning(
                        f"[CONSOLIDATION] bank={bank_id} LLM failed for single memory"
                        f" {sub_batch[0]['id']}, marking consolidation_failed_at"
                    )
                else:
                    succeeded_ids.extend(m["id"] for m in sub_batch)
                    all_results.extend(sub_results)

            # The successful sub-batches stamped their own ``consolidated_at`` inside the
            # transaction that wrote their observations (#3876) — a stamp and the writes it
            # accounts for must never commit apart. What is left here is the failed ids, which
            # have no writes to share a transaction with, and the refresh tags. Marking goes
            # through the store so the flag lands wherever the source facts live. ONE short
            # transaction, with no LLM work inside it: the sub-batch loop above must not hold a
            # Postgres connection across its LLM calls.
            async with acquire_with_retry(pool) as conn:
                store = get_memories()
                now = datetime.now(timezone.utc)
                if failed_ids:
                    await store.mark_consolidated(
                        conn=conn,
                        fq_table=fq_table,
                        bank_id=bank_id,
                        unit_ids=[str(mem_id) for mem_id in failed_ids],
                        when=now,
                        failed=True,
                    )
                async with conn.transaction():
                    # Persist this batch's mental-model refresh tags atomically with the
                    # witness, so they share the batch's fate: durable iff the batch is
                    # (#3411). Only the succeeded source facts — the ones just marked
                    # consolidated — contribute a tag. (Store-owned: no witness, so this is a
                    # plain best-effort write; a missed tag only defers a mental-model refresh.)
                    if operation_id and succeeded_ids:
                        succeeded_set = {str(mem_id) for mem_id in succeeded_ids}
                        batch_tags = sorted(
                            {t for m in llm_batch_local if str(m["id"]) in succeeded_set for t in (m.get("tags") or [])}
                        )
                        if batch_tags:
                            await _persist_pending_refresh_tags(conn, operation_id, batch_tags)

            cancelled_local = False
            if operation_id and not await memory_engine._check_op_alive(operation_id):
                logger.info(
                    f"[CONSOLIDATION] bank={bank_id} operation {operation_id} cancelled (bank deleted), stopping early"
                )
                cancelled_local = True

            # Per-batch local stats; merged into outer state once, serially,
            # after dispatch completes.
            local_stats: dict[str, int] = {
                "memories_processed": 0,
                "observations_created": 0,
                "observations_updated": 0,
                "observations_merged": 0,
                "observations_deleted": all_deleted,
                "actions_executed": 0,
                "skipped": 0,
                "memories_failed": 0,
            }
            for result in all_results:
                local_stats["memories_processed"] += 1
                action = result.get("action")
                if action == "created":
                    local_stats["observations_created"] += 1
                    local_stats["actions_executed"] += 1
                elif action == "updated":
                    local_stats["observations_updated"] += 1
                    local_stats["actions_executed"] += 1
                elif action == "merged":
                    local_stats["observations_merged"] += 1
                    local_stats["actions_executed"] += 1
                elif action == "multiple":
                    local_stats["observations_created"] += result.get("created", 0)
                    local_stats["observations_updated"] += result.get("updated", 0)
                    local_stats["observations_merged"] += result.get("merged", 0)
                    local_stats["actions_executed"] += result.get("total_actions", 0)
                elif action == "skipped":
                    local_stats["skipped"] += 1
                elif action == "failed":
                    local_stats["memories_failed"] += 1

            # Maintain the cumulative-progress indicator under parallelism:
            # increment shared counters and snapshot under the same statements so
            # the snapshot includes this batch. No await between the reads and
            # writes, so single-threaded asyncio gives us atomicity for free —
            # no lock needed.
            cumulative_progress["processed"] += local_stats["memories_processed"]
            cumulative_progress["observations_created"] += local_stats["observations_created"]
            cumulative_progress["observations_updated"] += local_stats["observations_updated"]
            cumulative_progress["observations_merged"] += local_stats["observations_merged"]
            cumulative_progress["observations_deleted"] += local_stats["observations_deleted"]
            cumulative_progress["memories_failed"] += local_stats["memories_failed"]
            cum_processed = cumulative_progress["processed"]
            cum_snapshot = dict(cumulative_progress)

            # Per-batch log uses batch_perf so timings/llm-calls/tokens reflect
            # only this batch's own work, even when other batches are running
            # concurrently under parallelism > 1. ``processed=`` is the
            # cumulative count across all batches that have finished so far in
            # this job (monotonic, may be reported out of strict batch-number
            # order under parallelism).
            llm_batch_time = time.time() - llm_batch_start
            timing_parts = [
                f"{key}={batch_perf.timings[key]:.3f}s"
                for key in ("recall", "llm", "embedding", "db_write")
                if key in batch_perf.timings
            ]
            input_tokens = int(batch_perf.total_prompt_chars / 4)
            logger.info(
                f"[CONSOLIDATION] bank={bank_id} llm_batch #{batch_num_local}"
                f" ({len(llm_batch_local)} memories, {batch_perf.llm_calls} llm calls)"
                f" | processed={cum_processed}/{total_count}"
                f" | {', '.join(timing_parts)}"
                f" | created={local_stats['observations_created']}"
                f" updated={local_stats['observations_updated']}"
                f" skipped={local_stats['skipped']}"
                + (f" failed={local_stats['memories_failed']}" if local_stats["memories_failed"] else "")
                + f" | input_tokens=~{input_tokens}"
                f" | avg={llm_batch_time / max(1, len(llm_batch_local)):.3f}s/memory"
            )

            # Durable progress snapshot per LLM batch — this is the heartbeat an
            # operator polls. The whole fetched batch is processed inside one outer
            # round, so a round-boundary write would sit at the pre-round count for
            # the entire (often minutes-long) LLM phase; writing here advances
            # processed/total as each batch commits. set_stage mirrors it for the
            # live worker log.
            set_stage(f"consolidation.llm_batch.{batch_num_local}")
            await memory_engine._write_operation_progress(
                operation_id,
                stage="consolidating",
                processed=cum_processed,
                total=await _progress_total(cum_processed),
                detail={
                    "observations_created": cum_snapshot["observations_created"],
                    "observations_updated": cum_snapshot["observations_updated"],
                    "observations_merged": cum_snapshot["observations_merged"],
                    "observations_deleted": cum_snapshot["observations_deleted"],
                    "memories_failed": cum_snapshot["memories_failed"],
                },
            )

            # Fold batch counters into the job-level perf so the final summary
            # (perf.flush) totals every batch correctly. Safe without a lock —
            # ConsolidationPerfLog.merge_from is a series of += on Python ints
            # and floats with no intervening awaits, so single-threaded asyncio
            # gives us atomicity.
            perf.merge_from(batch_perf)

            return _BatchDeltas(stats=local_stats, tags=local_tags, cancelled=cancelled_local)

        # Number every batch up front so log line numbering is deterministic
        # regardless of dispatch order under parallelism. Each group keeps its own
        # (batch, number) list so it can be processed as one serial unit.
        numbered_groups: list[list[tuple[list[dict[str, Any]], int]]] = []
        for batches in grouped_batches:
            numbered: list[tuple[list[dict[str, Any]], int]] = []
            for b in batches:
                llm_batch_num += 1
                numbered.append((b, llm_batch_num))
            numbered_groups.append(numbered)

        async def _process_tag_group(
            group_batches: list[tuple[list[dict[str, Any]], int]],
        ) -> list[_BatchDeltas]:
            # Batches within a group share a tag set and observation scope, so
            # they MUST run serially. Stop early if the op was cancelled mid-group.
            deltas: list[_BatchDeltas] = []
            for b, n in group_batches:
                d = await _process_one_llm_batch(b, n)
                deltas.append(d)
                if d.cancelled:
                    break
            return deltas

        llm_parallelism = max(1, config.consolidation_llm_parallelism)

        if llm_parallelism > 1 and len(numbered_groups) > 1:
            sem = asyncio.Semaphore(llm_parallelism)
            # Per-scope async locks shared across all parallel groups in this
            # fetch iteration. Each group acquires locks for every scope it will
            # write to, in _scope_sort_key order (deadlock-free). Groups with
            # disjoint scope sets never contend; any overlap serialises on the
            # overlapping scopes — covering combined / per_tag / all_combinations
            # / explicit-list modes uniformly without operator opt-in.
            scope_locks: defaultdict[frozenset[str], asyncio.Lock] = defaultdict(asyncio.Lock)

            async def _run_group(
                group_batches: list[tuple[list[dict[str, Any]], int]],
                scopes: list[frozenset[str]],
            ) -> list[_BatchDeltas]:
                async with sem:
                    async with AsyncExitStack() as stack:
                        for s in scopes:
                            await stack.enter_async_context(scope_locks[s])
                        return await _process_tag_group(group_batches)

            group_results = await _gather_or_cancel([_run_group(g, s) for g, s in zip(numbered_groups, group_scopes)])
            batch_results: list[_BatchDeltas] = [d for gd in group_results for d in gd]
            any_cancelled = any(d.cancelled for d in batch_results)
        else:
            batch_results = []
            any_cancelled = False
            for g in numbered_groups:
                group_deltas = await _process_tag_group(g)
                batch_results.extend(group_deltas)
                if any(d.cancelled for d in group_deltas):
                    any_cancelled = True
                    break

        # Merge per-batch deltas into outer state — serial, post-dispatch, so
        # concurrent batches cannot race on the shared counters / tag set.
        for d in batch_results:
            for k, v in d.stats.items():
                stats[k] = stats.get(k, 0) + v
            consolidated_tags.update(d.tags)

        if any_cancelled:
            return {"status": "cancelled", "bank_id": bank_id, **stats}

        # Update round budget after processing this DB fetch batch
        if round_limit_enabled:
            round_remaining -= len(memories)
            if round_remaining <= 0:
                hit_round_limit = True
                break

    # Re-submit consolidation if we hit the round limit and there's likely more work.
    # Any failure here must propagate: swallowing it (the prior behavior) leaves the
    # bank with backlog and no queued work — silently stuck — because the outer op
    # gets marked completed in the success path. Letting the exception bubble up to
    # execute_task's retry handler means the op is retried with backoff; on retry the
    # consolidator skips already-consolidated rows via the consolidated_at filter and
    # picks up the remainder. Issue #1842.
    # The affected-tag union for the whole round-limited chain. Refresh fires once, when
    # the backlog has fully drained (the final round), not once per round — a model's
    # memories can straddle rounds, and gating on the final round alone (the prior
    # behaviour) dropped every model consolidated earlier because the final round's tags
    # no longer named them (#3411). The union is durable: each batch writes its tags into
    # the op's ``task_payload`` in its own transaction (crash-safe), and the re-queue threads
    # the accumulated set forward to the next round. Prefer that durable
    # value; fall back to the in-memory union when there is no backing op (a direct
    # ``run_consolidation_job`` call, e.g. in tests).
    all_refresh_tags = set(pending_refresh_tags or []) | consolidated_tags
    if operation_id:
        all_refresh_tags |= await _read_pending_refresh_tags(pool, operation_id)

    if hit_round_limit:
        remaining = total_count - stats["memories_processed"]
        logger.info(
            f"[CONSOLIDATION] bank={bank_id} hit round limit of {max_memories_per_round} memories,"
            f" ~{remaining} remaining. Re-queuing consolidation."
        )
        await memory_engine.submit_async_consolidation(
            bank_id=bank_id,
            request_context=request_context,
            observation_scopes=observation_scopes,
            pending_refresh_tags=sorted(all_refresh_tags) or None,
        )

    # Build summary
    perf.log(
        f"[3] Results: {stats['memories_processed']} memories -> "
        f"{stats['actions_executed']} actions "
        f"({stats['observations_created']} created, "
        f"{stats['observations_updated']} updated, "
        f"{stats['observations_merged']} merged, "
        f"{stats['skipped']} skipped)"
    )

    # Add timing breakdown. Each phase is recorded once per call, so the count
    # disambiguates a single slow call from many fast calls — important for
    # operators triaging "the recall phase took 15s" log lines, where the
    # total is the sum of many serial sub-calls rather than one slow query.
    def _fmt(key: str) -> str:
        total = perf.timings[key]
        count = perf.timing_counts.get(key, 0)
        if count > 1:
            avg_ms = total * 1000.0 / count
            return f"{key}={total:.3f}s ({count} calls, avg={avg_ms:.0f}ms)"
        return f"{key}={total:.3f}s"

    timing_parts = []
    for key in ("recall", "llm", "embedding", "db_write"):
        if key in perf.timings:
            timing_parts.append(_fmt(key))

    if perf.llm_calls > 0:
        timing_parts.append(f"avg_obs={perf.total_obs_in_context / perf.llm_calls:.1f}")
        timing_parts.append(f"avg_prompt_tokens=~{perf.total_prompt_chars / perf.llm_calls / 4:.0f}")

    if timing_parts:
        perf.log(f"[4] Timing breakdown: {', '.join(timing_parts)}")

    # Trigger mental-model refreshes once, when the chain has fully drained. On a
    # round-limited round we skip and carry the affected tags forward (above); the
    # final round flushes the accumulated union, so a model whose memories were
    # consolidated in ANY round is refreshed exactly once — deduplicated, not dropped
    # (#3411). Each model is still refreshed at most once per drain: a strict tagged
    # model appears once in the trigger's candidate query regardless of how many rounds
    # its tag spanned.
    if hit_round_limit:
        stats["mental_models_refreshed"] = 0
        logger.info(
            f"[CONSOLIDATION] bank={bank_id} deferring mental model refresh to the final round "
            f"(round limit hit; carrying {len(all_refresh_tags)} tags forward)"
        )
    else:
        set_stage("consolidation.refreshing_mental_models")
        await memory_engine._write_operation_progress(
            operation_id,
            stage="refreshing_mental_models",
            processed=stats["memories_processed"],
            total=await _progress_total(stats["memories_processed"]),
        )
        # SECURITY: Only refresh mental models whose scope covers what was consolidated
        mental_models_refreshed = await _trigger_mental_model_refreshes(
            memory_engine=memory_engine,
            bank_id=bank_id,
            request_context=request_context,
            consolidated_tags=sorted(all_refresh_tags) or None,
            perf=perf,
        )
        stats["mental_models_refreshed"] = mental_models_refreshed

    perf.flush()

    return {"status": "completed", "bank_id": bank_id, **stats}


# SQL predicate: "this mental model's refresh scope can contain untagged memories".
#
# A model's scope is NOT its ``tags`` column — it is whatever
# ``_resolve_refresh_tag_filtering`` resolves, and both the refresh and the staleness
# check use that. Three cases reach untagged memories:
#   - no tags at all             -> no tag constraint, every bank memory is in scope
#   - tags_match "any" / "all"   -> non-strict, the clause ORs in untagged rows
#   - trigger.tag_groups         -> overrides the tags column entirely, so the column
#                                   says nothing about what the model can see
# A tagged model left on the default (``all_strict``) is correctly excluded: strict
# matching drops untagged rows, so an untagged-only consolidation cannot make it stale.
# Gating on the tags column alone starved the first two cases (#3053).
_MM_SCOPE_REACHES_UNTAGGED = (
    "((tags IS NULL OR tags = '{}') OR (trigger->>'tags_match') IN ('any', 'all') OR trigger ? 'tag_groups')"
)


async def _trigger_mental_model_refreshes(
    memory_engine: "MemoryEngine",
    bank_id: str,
    request_context: "RequestContext",
    consolidated_tags: list[str] | None = None,
    perf: ConsolidationPerfLog | None = None,
) -> int:
    """
    Trigger refreshes for mental models with refresh_after_consolidation=true.

    SECURITY: Only triggers refresh for mental models whose refresh scope can contain
    what this consolidation touched, preventing unnecessary refreshes across security
    boundaries.

    Args:
        memory_engine: MemoryEngine instance
        bank_id: Bank identifier
        request_context: Request context for authentication
        consolidated_tags: Tags of the memories that were consolidated. None means only
            untagged memories were consolidated (or nothing was), so only models whose
            scope reaches untagged memories are candidates.
        perf: Performance logging

    Returns:
        Number of mental models scheduled for refresh
    """
    pool = memory_engine._backend

    # Find mental models with refresh_after_consolidation=true that are actually stale.
    # The tag predicate on the SELECT is a cheap prefilter that skips models this
    # consolidation cannot have affected; compute_mental_model_is_stale then verifies
    # against the model's *resolved* scope that new memories really were ingested since
    # its last refresh.
    async with acquire_with_retry(pool) as conn:
        if consolidated_tags:
            candidates = await conn.fetch(
                f"""
                SELECT id, name, tags, last_refreshed_at, last_memory_seen_at, trigger
                FROM {fq_table("mental_models")}
                WHERE bank_id = $1
                  AND (trigger->>'refresh_after_consolidation')::boolean = true
                  AND (
                    (tags IS NOT NULL AND tags != '{{}}' AND tags && $2::varchar[])
                    OR {_MM_SCOPE_REACHES_UNTAGGED}
                  )
                """,
                bank_id,
                consolidated_tags,
            )
        else:
            candidates = await conn.fetch(
                f"""
                SELECT id, name, tags, last_refreshed_at, last_memory_seen_at, trigger
                FROM {fq_table("mental_models")}
                WHERE bank_id = $1
                  AND (trigger->>'refresh_after_consolidation')::boolean = true
                  AND {_MM_SCOPE_REACHES_UNTAGGED}
                """,
                bank_id,
            )

        rows = []
        for candidate in candidates:
            if await memory_engine.compute_mental_model_is_stale(conn, bank_id, candidate):
                rows.append(candidate)

    if not rows:
        return 0

    if perf:
        if consolidated_tags:
            perf.log(
                f"[5] Triggering refresh for {len(rows)} mental models with refresh_after_consolidation=true "
                f"(filtered by tags: {consolidated_tags})"
            )
        else:
            perf.log(f"[5] Triggering refresh for {len(rows)} mental models with refresh_after_consolidation=true")

    # Submit refresh tasks for each mental model
    refreshed_count = 0
    for row in rows:
        mental_model_id = row["id"]
        try:
            # skip_if_in_flight: a consolidation chain fires this every round and
            # overlapping consolidations can run on the same bank, so a model still
            # pending/processing a refresh must not be enqueued a second time (#3411).
            await memory_engine.submit_async_refresh_mental_model(
                bank_id=bank_id,
                mental_model_id=mental_model_id,
                request_context=request_context,
                skip_if_in_flight=True,
                automatic=True,
            )
            refreshed_count += 1
            logger.info(
                f"[CONSOLIDATION] Triggered refresh for mental model {mental_model_id} "
                f"(name: {row['name']}) in bank {bank_id}"
            )
        except Exception as e:
            logger.warning(f"[CONSOLIDATION] Failed to trigger refresh for mental model {mental_model_id}: {e}")

    return refreshed_count


async def _process_memory_batch(
    pool: DatabaseBackend,
    memory_engine: "MemoryEngine",
    llm_config: Any,
    bank_id: str,
    memories: list[dict[str, Any]],
    request_context: "RequestContext",
    perf: ConsolidationPerfLog | None = None,
    config: Any = None,
    obs_tags_override: list[str] | None = None,
    mark_consolidated_ids: list[Any] | None = None,
) -> tuple[list[dict[str, Any]], int, bool]:
    """
    Process a batch of memories in a single LLM call.

    Steps:
    1. Parallel recalls — one per fact (read-only; safe to parallelise)
    2. Union of retrieved observations across the batch (deduped by id)
    3. Single LLM call with all N facts + unioned observations
    4. Prepare every action connection-free, then apply them in ONE transaction
    5. Returns one result dict per memory, in the same order as `memories`

    Per-fact security: action execution validates each learning_id against the
    observations that were recalled specifically for that fact, so cross-tag
    updates cannot occur.

    Args:
        obs_tags_override: When set, use these tags for observation recall and
            create/update instead of the memory's own tags. This enables multi-pass
            consolidation where a single memory can contribute to observations
            scoped at different tag levels (e.g., user-level vs session-level).
        mark_consolidated_ids: Source facts to stamp ``consolidated_at`` inside this
            batch's write transaction, so the stamps share the fate of the observations
            derived from the same LLM response (#3876). None on a pass that is not the
            final one for these memories — the caller stamps once, on the last pass.
    """
    # Map the source memories this batch consumes onto the consolidation trace.
    record_source_memory_ids([str(m["id"]) for m in memories])

    # 1. Parallel recalls — one per fact
    # When obs_tags_override is set, use it as the observation scope for all facts.
    t0 = time.time()
    observation_scope_tags = obs_tags_override if obs_tags_override is not None else None
    recall_tasks = [
        _find_related_observations(
            memory_engine=memory_engine,
            bank_id=bank_id,
            query=m["text"],
            request_context=request_context,
            tags=observation_scope_tags if observation_scope_tags is not None else (m.get("tags") or []),
        )
        for m in memories
    ]
    # A failed recall must fail the batch rather than degrade to "no related
    # observations": proceeding with an empty candidate set would hide an
    # existing twin from the LLM and turn an UPDATE into a duplicate CREATE.
    # The batch's memories stay unconsolidated and are picked up on retry.
    per_fact_recalls = await _gather_or_cancel(recall_tasks)
    if perf:
        perf.record_timing("recall", time.time() - t0)

    # 2. Build per-fact observation sets (keyed by memory ID string) for secure action validation
    per_fact_obs_ids: dict[str, set[str]] = {
        str(memories[i]["id"]): {str(obs.id) for obs in r.results} for i, r in enumerate(per_fact_recalls)
    }

    # Union all observations (deduped by id)
    seen_ids: set[str] = set()
    union_observations: list["MemoryFact"] = []
    union_source_facts: dict[str, "MemoryFact"] = {}
    for recall_result in per_fact_recalls:
        for obs in recall_result.results:
            obs_id = str(obs.id)
            if obs_id not in seen_ids:
                seen_ids.add(obs_id)
                union_observations.append(obs)
        if recall_result.source_facts:
            union_source_facts.update(recall_result.source_facts)

    # Determine effective tag scope for observations.
    # When obs_tags_override is set, use it; otherwise use the memory's own tags.
    if obs_tags_override is not None:
        fact_tags = obs_tags_override
    else:
        # All memories in the batch share the same tag set (enforced by batching)
        fact_tags = memories[0].get("tags") or [] if memories else []

    # 2b. Compute remaining observation slots for this scope (if limit configured).
    # The cap is resolved per-scope: an observation_scope_limits rule may override
    # the bank-wide max_observations_per_scope for scopes matching its tag pattern.
    max_obs = _effective_scope_limit(config, fact_tags)
    remaining_observation_slots: int | None = None
    if max_obs >= 0 and fact_tags:
        # max_obs == 0 means "no new observations": there are no slots regardless
        # of the current count, so skip the count query for that case.
        current_count = 0
        if max_obs > 0:
            async with acquire_with_retry(pool) as count_conn:
                current_count = await _count_observations_for_scope(count_conn, bank_id, fact_tags)
        remaining_observation_slots = max(max_obs - current_count, 0)
        if remaining_observation_slots == 0:
            logger.info(
                f"[CONSOLIDATION] bank={bank_id} scope={fact_tags} at observation limit "
                f"({current_count}/{max_obs}), only updates/deletes allowed"
            )

    # 3. Single LLM call
    t0 = time.time()
    llm_result = await _consolidate_batch_with_llm(
        llm_config=llm_config,
        memories=memories,
        union_observations=union_observations,
        union_source_facts=union_source_facts,
        config=config,
        remaining_observation_slots=remaining_observation_slots,
        max_observations_per_scope=max_obs,
    )
    if perf:
        perf.record_timing("llm", time.time() - t0)
        perf.record_llm_call(llm_result.obs_count, llm_result.prompt_chars)

    # 4. Prepare every action connection-free, then apply them all in ONE transaction.
    #
    # Before #3876 each action wrote on its own connection the moment it was decided:
    # deletes first (to free observation slots), then updates, then creates. A batch whose
    # DELETE succeeded and whose replacement CREATE then failed — a raising embedder, a
    # cancelled sibling, a create silently dropped because the model cited a fact id that
    # was not in this batch — left the observation deleted and nothing in its place, while
    # the source facts were still stamped ``consolidated_at`` and so were never
    # re-consolidated. The knowledge was gone with the operation still reporting
    # ``completed``. The writes derived from one LLM response are now all-or-nothing, and
    # the ``consolidated_at`` stamps for the facts that response consumed commit with them.
    #
    # The transaction must therefore contain no LLM or embedder call: each action is first
    # prepared (source facts resolved and security-checked, text embedded, semantic-dedup
    # verdict adjudicated) with no connection held, and only then written.
    per_memory_created: set[str] = set()
    per_memory_updated: set[str] = set()

    mem_by_id = {str(m["id"]): m for m in memories}

    # Semantic dedup: when enabled, an observation that is >= the threshold cosine to a DIFFERENT
    # existing observation is reconciled by a focused 1-by-1 LLM merge (anchored on the observation
    # text, not the source fact). It runs on both CREATE (a near-dup emitted despite the twin being
    # in context — weak-model failure mode) and UPDATE (a rewrite+re-embed that drifts an existing
    # observation into a twin — the create-time guard can't see this). The trace operation/scope is
    # "consolidation_dedup" (routes through the consolidation concurrency bucket via llm_wrapper's
    # "consolidation" prefix; recorded distinctly in llm_requests).
    dedup_enabled = _dedup_active(config)
    dedup_llm_config = (
        memory_engine._consolidation_llm_config.with_config(config, bank_id=bank_id, operation="consolidation_dedup")
        if dedup_enabled
        else None
    )

    # --- Prepare: deletes -----------------------------------------------------
    # Deletes carry no slow work; validating them here keeps the whole decision set in
    # one place. They are still applied first inside the transaction, so an observation
    # slot freed by a delete is available to a create in the same response.
    prepared_deletes: list[str] = []
    for delete in llm_result.deletes:
        # Security: the observation must be present in the unioned recall
        if not any(str(obs.id) == delete.observation_id for obs in union_observations):
            logger.debug(
                f"Batch consolidation: rejected delete — observation {delete.observation_id} not in unioned recall"
            )
            continue
        prepared_deletes.append(delete.observation_id)

    # --- Prepare: updates -----------------------------------------------------
    prepared_updates: list[_PreparedUpdate] = []
    for update in llm_result.updates:
        source_mems = [mem_by_id[fid] for fid in update.source_fact_ids if fid in mem_by_id]
        if not source_mems:
            continue
        # Security: the observation must have been recalled for at least one of the source facts
        if not any(update.observation_id in per_fact_obs_ids.get(str(m["id"]), set()) for m in source_mems):
            logger.debug(
                f"Batch consolidation: rejected update — observation {update.observation_id} "
                f"not in any source fact's recall"
            )
            continue
        model = next((m for m in union_observations if str(m.id) == update.observation_id), None)
        if model is None:
            logger.debug(f"Update skipped: observation {update.observation_id} not found in recall results")
            continue
        agg = _aggregate_source_fields(source_mems, tags=fact_tags)
        source_memory_ids = [m["id"] for m in source_mems]
        # Preflight (non-locking, short-lived conn): if every source memory is already gone,
        # skip BEFORE the slow embed. The write itself re-checks liveness under FOR SHARE.
        async with acquire_with_retry(pool) as conn:
            if not await _any_live_source_memory(conn, bank_id, source_memory_ids):
                logger.debug(
                    f"Update skipped: all {len(source_memory_ids)} source memories for observation "
                    f"{update.observation_id} were deleted before embedding"
                )
                continue
        embedding_str = await _embed_observation_text(memory_engine, update.text, perf)
        prepared = _PreparedUpdate(
            update=update,
            model=model,
            source_mems=source_mems,
            source_memory_ids=source_memory_ids,
            source_fact_tags=agg.tags,
            source_bounds=_TemporalBounds.of(agg),
            embedding_str=embedding_str,
        )
        # Reconcile the rewritten observation against its neighbours: the re-embed may have
        # drifted it into a near-twin of another existing observation (the residual-duplicate
        # source). Adjudicated here, folded inside the transaction below.
        if dedup_enabled and embedding_str is not None:
            prepared.dedup = await _dedup_adjudicate(
                pool,
                memory_engine,
                bank_id,
                config,
                dedup_llm_config,
                update.text,
                embedding_str,
                agg.tags,
                exclude_id=update.observation_id,
            )
        prepared_updates.append(prepared)

    # --- Prepare: creates -----------------------------------------------------
    # Deterministic dedup guard: map the observations the LLM was SHOWN by their
    # normalised text. The model intermittently emits a CREATE whose text is identical
    # to an observation already in its context (over-aggregation / incoherence — it even
    # UPDATEs the twin and creates a sibling). When that happens we drop the duplicate
    # CREATE instead of inserting a redundant row. No extra LLM/embedding cost — the
    # match is exact text against the in-memory set.
    shown_obs_by_text = {_norm_obs_text(o.text): o for o in union_observations}
    # Also collapse a CREATE that reproduces the text of an UPDATE issued in the SAME
    # response (the model occasionally UPDATEs the twin to text X and also CREATEs X).
    update_texts = {_norm_obs_text(u.text) for u in llm_result.updates if u.text}

    prepared_creates: list[_PreparedCreate] = []
    for create in llm_result.creates:
        source_mems = [mem_by_id[fid] for fid in create.source_fact_ids if fid in mem_by_id]
        if not source_mems:
            continue
        agg = _aggregate_source_fields(source_mems, tags=fact_tags)
        create_source_ids = [m["id"] for m in source_mems]

        # Reconcile against observations shown to the LLM: an exact-text match means
        # this CREATE reproduces verbatim an observation the model already had in context.
        # Since that observation already carries this exact text, drop the duplicate CREATE
        # — no row is inserted, nothing is lost. We deliberately do NOT also UPDATE the twin
        # here: the LLM frequently UPDATEd it earlier in this same batch, and a second update
        # would run off the pre-LLM snapshot and clobber that change (see _dedupe_updates).
        duplicate_of = _duplicate_create_target(create.text, shown_obs_by_text, update_texts)
        if duplicate_of is not None:
            logger.warning(
                "[CONSOLIDATION] dropped duplicate observation CREATE — verbatim match of %s; llm_reason=%r",
                duplicate_of,
                create.reason or "(none given)",
            )
            continue

        async with acquire_with_retry(pool) as conn:
            if not await _any_live_source_memory(conn, bank_id, create_source_ids):
                logger.debug(
                    f"Create skipped: all {len(create_source_ids)} source memories were deleted before embedding"
                )
                continue
        embedding_str = await _embed_observation_text(memory_engine, create.text, perf)
        prepared_create = _PreparedCreate(
            text=create.text,
            source_mems=source_mems,
            source_memory_ids=create_source_ids,
            source_fact_tags=agg.tags,
            agg=agg,
            embedding_str=embedding_str,
        )
        # Semantic near-duplicate reconciliation: merge this CREATE into an existing
        # near-identical observation (LLM-adjudicated, 1-by-1) instead of inserting a dup.
        # The probe reads the state the batch started from, so two near-twin CREATEs in ONE
        # response can both land; the next round's probe sees them and folds them, and the
        # exact-text guard above already covers the common case.
        if dedup_enabled:
            prepared_create.dedup = await _dedup_adjudicate(
                pool,
                memory_engine,
                bank_id,
                config,
                dedup_llm_config,
                create.text,
                embedding_str,
                agg.tags,
                exclude_id=None,
            )
        prepared_creates.append(prepared_create)

    # --- Apply: one transaction for everything this LLM response decided ------
    deleted_count = 0
    # A failed LLM call yields no actions, and its memories must NOT be stamped: the caller
    # bisects and retries them, and a stamp would exclude them from pending consolidation
    # for good. With neither writes nor stamps there is nothing to open a transaction for.
    stamp_ids = list(mark_consolidated_ids or []) if not llm_result.failed else []
    if prepared_deletes or prepared_updates or prepared_creates or stamp_ids:
        async with acquire_with_retry(pool) as conn:
            async with conn.transaction():
                for observation_id in prepared_deletes:
                    await _execute_delete_action(conn=conn, bank_id=bank_id, observation_id=observation_id)
                    deleted_count += 1

                for prepared in prepared_updates:
                    updated_emb_str = await _apply_update_action(
                        conn=conn,
                        memory_engine=memory_engine,
                        bank_id=bank_id,
                        prepared=prepared,
                        perf=perf,
                    )
                    if updated_emb_str is None:
                        # Skipped inside the write (sources or the observation vanished).
                        continue
                    for m in prepared.source_mems:
                        per_memory_updated.add(str(m["id"]))
                    if prepared.dedup is not None:
                        await _apply_dedup_update_fold(
                            conn,
                            memory_engine,
                            bank_id,
                            config,
                            prepared.dedup,
                            prepared.update.observation_id,
                            prepared.update.text,
                        )

                for prepared_create in prepared_creates:
                    if prepared_create.dedup is not None:
                        merged_into = await _apply_dedup_create_fold(
                            conn,
                            memory_engine,
                            bank_id,
                            config,
                            prepared_create.dedup,
                            prepared_create.source_memory_ids,
                            _TemporalBounds.of(prepared_create.agg),
                        )
                        if merged_into is not None:
                            logger.info(
                                "[CONSOLIDATION] dedup-merged observation CREATE into %s (cosine>=%.2f)",
                                merged_into[:8],
                                config.consolidation_dedup_threshold,
                            )
                            for m in prepared_create.source_mems:
                                per_memory_created.add(str(m["id"]))
                            continue

                    action = await _apply_create_action(
                        conn=conn,
                        memory_engine=memory_engine,
                        bank_id=bank_id,
                        prepared=prepared_create,
                        perf=perf,
                    )
                    # Count a memory as created only when an observation was actually written (the
                    # source-liveness recheck inside the write can skip it).
                    if action == "created":
                        for m in prepared_create.source_mems:
                            per_memory_created.add(str(m["id"]))

                # The facts this response consumed are marked consolidated in the SAME transaction
                # as the observations that now carry them. Stamping them separately is what let a
                # half-applied batch orphan its sources forever: the pending-consolidation predicate
                # excludes a stamped fact, so nothing would ever rebuild what the batch failed to
                # write (#3876).
                if stamp_ids:
                    await get_memories().mark_consolidated(
                        conn=conn,
                        fq_table=fq_table,
                        bank_id=bank_id,
                        unit_ids=[str(mem_id) for mem_id in stamp_ids],
                        when=datetime.now(timezone.utc),
                        failed=False,
                    )

    # Build per-memory result dicts for the stats tracker in the outer loop
    results: list[dict[str, Any]] = []
    for m in memories:
        mid = str(m["id"])
        created = mid in per_memory_created
        updated = mid in per_memory_updated
        if created and updated:
            results.append({"action": "multiple", "created": 1, "updated": 1, "merged": 0, "total_actions": 2})
        elif created:
            results.append({"action": "created"})
        elif updated:
            results.append({"action": "updated"})
        else:
            results.append({"action": "skipped", "reason": "no_durable_knowledge"})

    return results, deleted_count, llm_result.failed


def _min_date(dates: "Any") -> "datetime | None":
    """Return the minimum non-None datetime from an iterable."""
    return min((d for d in dates if d is not None), default=None)


def _max_date(dates: "Any") -> "datetime | None":
    """Return the maximum non-None datetime from an iterable."""
    return max((d for d in dates if d is not None), default=None)


@dataclass(frozen=True)
class _ObservationHistorySnapshot:
    """Pre-update state of an observation, persisted as the ``content`` JSON blob
    of one observation_history row.

    Temporal fields are the ISO strings carried on MemoryFact; new_source_memory_ids
    are the ids added by the update.
    """

    previous_text: str | None
    previous_tags: list[str]
    previous_occurred_start: str | None
    previous_occurred_end: str | None
    previous_mentioned_at: str | None
    new_source_memory_ids: list[str]


async def _append_observation_history(
    conn: "Connection",
    bank_id: str,
    observation_id: str,
    snapshot: _ObservationHistorySnapshot,
    max_entries: int,
) -> None:
    """Insert one pre-update snapshot into ``observation_history``, then delete the
    oldest rows beyond ``max_entries`` for this observation.

    The snapshot is stored as a single JSONB ``content`` blob (per-row, so it stays
    small). Bounding by row count keeps a frequently-reinforced observation's
    history from growing without bound.
    """
    obs_uuid = uuid.UUID(observation_id)
    try:
        await conn.execute(
            f"""
        INSERT INTO {fq_table("observation_history")} (observation_id, bank_id, content, changed_at)
        VALUES ($1, $2, $3::jsonb, now())
        """,
            obs_uuid,
            bank_id,
            json.dumps(asdict(snapshot)),
        )
    except asyncpg.exceptions.ForeignKeyViolationError:
        logger.warning(
            f"FK violation writing observation_history for {observation_id}: "
            "observation was removed before history could be written (race with parallel consolidation). Skipping."
        )
        return
    if max_entries and max_entries > 0:
        await conn.execute(
            f"""
            DELETE FROM {fq_table("observation_history")}
            WHERE observation_id = $1
              AND id NOT IN (
                  SELECT id FROM {fq_table("observation_history")}
                  WHERE observation_id = $1
                  ORDER BY changed_at DESC, id DESC
                  LIMIT $2
              )
            """,
            obs_uuid,
            max_entries,
        )


async def _apply_update_action(
    conn,
    memory_engine: "MemoryEngine",
    bank_id: str,
    prepared: _PreparedUpdate,
    perf: ConsolidationPerfLog | None = None,
) -> str | None:
    """Write one prepared UPDATE: rewrite the observation and re-attach its sources.

    Extends source_memory_ids with all contributing memories, widens the observation's temporal
    bounds by ``prepared.source_bounds`` (see :class:`_TemporalBounds`), and merges tags.

    Runs entirely on the caller's connection inside the caller's transaction. Everything slow —
    the liveness preflight and the embedding — already happened in the prepare phase, so this
    write commits or rolls back together with every other write derived from the same LLM
    response, including the ``consolidated_at`` stamps for its source facts (#3876).

    Returns the observation's embedding (pgvector literal) so the caller can run UPDATE-path
    dedup without re-embedding, or None when the update was skipped.
    """
    from ...config import get_config

    model = prepared.model
    observation_id = prepared.update.observation_id
    new_text = prepared.update.text
    source_memory_ids = prepared.source_memory_ids
    source_fact_tags = prepared.source_fact_tags
    source_bounds = prepared.source_bounds
    embedding_str = prepared.embedding_str

    config = get_config()
    search_vector_clause = _native_search_vector_update(config, "$1")
    store = get_memories()

    # FOR SHARE liveness + the write share the batch's transaction, so a concurrent
    # delete cannot remove a source row between the check and the UPDATE.
    live_source_memory_ids = await _filter_live_source_memories(conn, bank_id, source_memory_ids)
    if not live_source_memory_ids:
        logger.debug(
            f"Update skipped: all {len(source_memory_ids)} source memories for observation "
            f"{observation_id} were deleted concurrently"
        )
        return None
    live_ids = live_source_memory_ids

    history_entry = _ObservationHistorySnapshot(
        previous_text=model.text,
        previous_tags=list(model.tags or []),
        previous_occurred_start=model.occurred_start,
        previous_occurred_end=model.occurred_end,
        previous_mentioned_at=model.mentioned_at,
        new_source_memory_ids=[str(mid) for mid in live_ids],
    )

    source_ids = list(model.source_fact_ids or []) + live_ids

    # SECURITY: Merge source fact's tags into existing observation tags so all contributors can see it
    existing_tags = set(model.tags or [])
    source_tags = set(source_fact_tags or [])
    merged_tags = list(existing_tags | source_tags)

    t0 = time.time()
    if not store.store_owned_for(bank_id):
        # Unlike the dedup folds this statement also runs on Oracle, where LEAST/GREATEST
        # return NULL as soon as ANY argument is NULL (PostgreSQL ignores NULL arguments).
        # The inner COALESCE covers a NULL *parameter*; the outer one covers a NULL
        # *column* — an observation with no occurred interval yet, which is precisely the
        # #3477 case. Without it Oracle would compute LEAST(NULL, <source date>) = NULL and
        # silently drop the date it was told to inherit. Keep the inner
        # ``COALESCE($n, col)`` spelled exactly like this: the Oracle driver shim keys its
        # TIMESTAMP-TZ input-size hint off that pattern (db/oracle.py::_apply_clob_input_sizes),
        # and a NULL parameter binds as VARCHAR2 (ORA-00932) without it.
        updated_rows = await conn.execute_rows_affected(
            f"""
            UPDATE {fq_table("memory_units")}
            SET text = $1,
                embedding = $2::vector,
                source_memory_ids = $3,
                proof_count = $4,
                tags = $10,
                updated_at = now(),
                event_date = COALESCE(LEAST(event_date, COALESCE($6, event_date)), $6),
                occurred_start = COALESCE(LEAST(occurred_start, COALESCE($7, occurred_start)), $7),
                occurred_end = COALESCE(GREATEST(occurred_end, COALESCE($8, occurred_end)), $8),
                mentioned_at = COALESCE(GREATEST(mentioned_at, COALESCE($9, mentioned_at)), $9){search_vector_clause}
            WHERE id = $5
            """,
            new_text,
            embedding_str,
            source_ids,
            len(source_ids),
            uuid.UUID(observation_id),
            source_bounds.event_date,
            source_bounds.occurred_start,
            source_bounds.occurred_end,
            source_bounds.mentioned_at,
            merged_tags,
        )
        # The source-liveness checks above guard the *source* memories; the
        # observation row itself (WHERE id = $5) can still be invalidated/deleted
        # concurrently, matching 0 rows. Bail out BEFORE the observation_history
        # INSERT below — that INSERT carries an observation_id FK onto memory_units,
        # so appending history for a now-missing row raises ForeignKeyViolationError,
        # a non-retryable integrity failure that would fail the whole consolidation
        # op for a row that simply no longer exists.
        if updated_rows == 0:
            logger.debug(
                f"Update skipped: observation {observation_id} no longer exists "
                "(deleted/invalidated concurrently); not appending history"
            )
            return None
    else:
        # Upsert overwrites the whole observation, so start from its current state (fetched
        # from the store) and apply the same merge the SQL does — LEAST/GREATEST on the
        # times — while preserving fields the update never touches (created_at).
        current = await store.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[observation_id])
        cur = current[0] if current else None
        # Widen the row the store still holds. If it has vanished, fall back to the
        # pre-update recall snapshot — ISO strings, and no event_date on that model.
        current_bounds = (
            _TemporalBounds.of(cur)
            if cur
            else _TemporalBounds(
                occurred_start=_as_dt(model.occurred_start),
                occurred_end=_as_dt(model.occurred_end),
                mentioned_at=_as_dt(model.mentioned_at),
            )
        )
        merged_bounds = current_bounds.merged_with(source_bounds)
        await store.upsert_observation(
            conn=conn,
            bank_id=bank_id,
            record=FactRecord(
                unit_id=observation_id,
                text=new_text,
                embedding=embedding_str,
                fact_type="observation",
                tags=merged_tags,
                proof_count=len(source_ids),
                source_memory_ids=[str(s) for s in source_ids],
                event_date=merged_bounds.event_date,
                occurred_start=merged_bounds.occurred_start,
                occurred_end=merged_bounds.occurred_end,
                mentioned_at=merged_bounds.mentioned_at,
                created_at=cur.created_at if cur else None,
            ),
        )

    # Record the pre-update snapshot in the dedicated observation_history table
    # (one row per change), then trim to the configured cap. History lived in a
    # single unbounded JSONB column before; an often-reinforced observation grew
    # it until it crossed Postgres's 256MB jsonb limit and got stuck.
    if config.enable_observation_history:
        await _append_observation_history(
            conn, bank_id, observation_id, history_entry, config.observation_history_max_entries
        )

    # Sync observation_sources junction table (Oracle only — PG uses native array ops).
    if memory_engine._backend.ops.uses_observation_sources_table:
        obs_uuid = uuid.UUID(observation_id)
        await conn.execute(
            f"DELETE FROM {fq_table('observation_sources')} WHERE observation_id = $1",
            obs_uuid,
        )
        if source_ids:
            await conn.executemany(
                f"""
                INSERT INTO {fq_table("observation_sources")} (observation_id, source_id)
                VALUES ($1, $2)
                ON CONFLICT (observation_id, source_id) DO NOTHING
                """,
                [(obs_uuid, sid) for sid in dict.fromkeys(source_ids)],
            )

    if perf:
        perf.record_timing("db_write", time.time() - t0)

    # Map the updated observation onto the consolidation trace as a produced memory.
    record_created_memory_ids([observation_id])
    logger.debug(f"Updated observation {observation_id} from {len(source_memory_ids)} source memories")
    return embedding_str


async def _apply_create_action(
    conn,
    memory_engine: "MemoryEngine",
    bank_id: str,
    prepared: "_PreparedCreate",
    perf: ConsolidationPerfLog | None = None,
) -> str:
    """
    Create a new observation from one or more source memories.

    Tags are inherited from the source facts (determined algorithmically, not by LLM)
    to maintain visibility scope. Returns the write action ("created" or "skipped").

    Runs on the caller's connection inside the caller's transaction (#3876).
    """
    created = await _apply_create_observation(
        conn=conn,
        memory_engine=memory_engine,
        bank_id=bank_id,
        source_memory_ids=prepared.source_memory_ids,
        observation_text=prepared.text,
        embedding_str=prepared.embedding_str,
        tags=prepared.source_fact_tags or [],
        event_date=prepared.agg.event_date,
        occurred_start=prepared.agg.occurred_start,
        occurred_end=prepared.agg.occurred_end,
        mentioned_at=prepared.agg.mentioned_at,
        perf=perf,
    )
    # Map the new observation onto the consolidation trace as a produced memory.
    new_id = created.get("observation_id")
    if new_id:
        record_created_memory_ids([new_id])
    logger.debug(f"Created observation from {len(prepared.source_memory_ids)} source memories")
    return created["action"]


async def _execute_delete_action(
    conn: "Connection",
    bank_id: str,
    observation_id: str,
) -> None:
    """Delete a superseded or contradicted observation."""
    store = get_memories()
    if not store.store_owned_for(bank_id):
        await conn.execute(
            f"DELETE FROM {fq_table('memory_units')} WHERE id = $1 AND bank_id = $2 AND fact_type = 'observation'",
            uuid.UUID(observation_id),
            bank_id,
        )
    else:
        await store.delete_facts(bank_id, [observation_id])
    # History lives in Postgres regardless of where the observation itself does, and no
    # longer cascades from memory_units (that FK was dropped so it could be recorded for
    # observations kept outside SQL). Drop it explicitly so a deleted observation's
    # snapshots don't accumulate forever.
    await conn.execute(
        f"DELETE FROM {fq_table('observation_history')} WHERE bank_id = $1 AND observation_id = $2",
        bank_id,
        uuid.UUID(observation_id),
    )
    logger.debug(f"Deleted observation {observation_id}")


async def _embed_observation_text(
    memory_engine: "MemoryEngine",
    text: str,
    perf: ConsolidationPerfLog | None = None,
) -> str | None:
    """Embed one observation text into a pgvector literal, with no connection held.

    Every caller is in the prepare phase of a batch (#3876): the embedder is the slowest
    thing consolidation does per action, and it must finish before the batch's single write
    transaction opens.
    """
    t0 = time.time()
    embeddings = await embedding_utils.generate_embeddings_batch(memory_engine.embeddings, [text])
    if perf:
        perf.record_timing("embedding", time.time() - t0)
    return str(embeddings[0]) if embeddings else None


async def _find_related_observations(
    memory_engine: "MemoryEngine",
    bank_id: str,
    query: str,
    request_context: "RequestContext",
    tags: list[str] | None = None,
) -> "RecallResult":
    """
    Find observations related to the given query using optimized recall.

    SECURITY: Filters by tags using all_strict matching to prevent cross-tenant/cross-user
    information leakage. Observations are only consolidated within the same tag scope.

    Uses max_tokens to naturally limit observations (no artificial count limit).
    Includes source memories with dates for LLM context.

    Args:
        tags: Optional tags to filter observations (uses all_strict matching for security)

    Returns:
        List of related observations with their tags, source memories, and dates
    """
    # Use recall to find related observations with token budget
    # max_tokens naturally limits how many observations are returned
    from ...tracing import get_tracer, is_tracing_enabled

    config = await memory_engine._config_resolver.resolve_full_config(bank_id, request_context)

    # SECURITY: Use all_strict matching if tags provided to prevent cross-scope consolidation
    tags_match = "all_strict" if tags else "any"

    # Create span for recall operation within consolidation
    tracer = get_tracer()
    if is_tracing_enabled():
        recall_span = tracer.start_span("hindsight.consolidation_recall")
        recall_span.set_attribute("hindsight.bank_id", bank_id)
        recall_span.set_attribute("hindsight.query", query[:100])  # Truncate for brevity
        recall_span.set_attribute("hindsight.fact_type", "observation")
    else:
        recall_span = None

    # Resolve budget: consolidation doesn't need deep recall, default to LOW to reduce memory fan-out
    recall_budget = Budget(config.consolidation_recall_budget)

    try:
        recall_result = await memory_engine.recall_async(
            bank_id=bank_id,
            query=query,
            budget=recall_budget,
            max_tokens=config.consolidation_max_tokens,  # Token budget for observations (configurable)
            fact_type=["observation"],  # Only retrieve observations
            request_context=request_context,
            tags=tags,  # Filter by source memory's tags
            tags_match=tags_match,  # Use strict matching for security
            include_source_facts=True,  # Embed source facts so we avoid a separate DB fetch
            max_source_facts_tokens=config.consolidation_source_facts_max_tokens,
            max_source_facts_tokens_per_observation=config.consolidation_source_facts_max_tokens_per_observation,
            # Round-robin interleave fusion (no cross-encoder): consolidation is looking
            # for an existing near-identical observation to merge into. Both the
            # cross-encoder (semantic #1 -> reranked #37) and RRF (semantic #1 -> outside
            # the 512-token budget) were measured to bury that twin; interleave guarantees
            # each retrieval arm's top hits a slot, so the semantic-#1 twin is always shown
            # to the LLM, which then UPDATEs instead of creating a duplicate.
            reranking="interleave",
            _quiet=True,  # Suppress logging
        )
    finally:
        if recall_span:
            recall_span.end()

    return recall_result


def _build_observations_for_llm(
    observations: "list[MemoryFact]",
    source_facts: "dict[str, MemoryFact]",
) -> list[dict[str, Any]]:
    """Serialize MemoryFact observations into dicts for the consolidation LLM prompt."""
    obs_list = []
    for obs in observations:
        obs_data: dict[str, Any] = {
            "id": obs.id,
            "text": obs.text,
            "proof_count": len(obs.source_fact_ids or []) or 1,
        }
        if obs.occurred_start:
            obs_data["occurred_start"] = obs.occurred_start
        if obs.occurred_end:
            obs_data["occurred_end"] = obs.occurred_end
        if obs.mentioned_at:
            obs_data["mentioned_at"] = obs.mentioned_at
        source_memories = []
        for sid in obs.source_fact_ids or []:
            sf = source_facts.get(sid)
            if sf is None:
                continue
            sf_data: dict[str, Any] = {"text": sf.text}
            if sf.context:
                sf_data["context"] = sf.context
            if sf.occurred_start:
                sf_data["occurred_start"] = sf.occurred_start
            if sf.occurred_end:
                sf_data["occurred_end"] = sf.occurred_end
            if sf.mentioned_at:
                sf_data["mentioned_at"] = sf.mentioned_at
            source_memories.append(sf_data)
        if source_memories:
            obs_data["source_memories"] = source_memories
        obs_list.append(obs_data)
    return obs_list


def _dedupe_updates(updates: list[_UpdateAction], *, batch_label: str) -> list[_UpdateAction]:
    """Collapse `updates` that target the same `observation_id`.

    LLMs occasionally emit several update entries for one observation in a
    single response (one per facet drawn from the same fact). Without
    deduplication the downstream loop would issue separate DB writes for each
    and the last write would silently overwrite the earlier ones. We keep the
    last text (the LLM's most recent attempt) and union all contributing
    `source_fact_ids`, then warn so the misbehavior is visible in logs.
    """
    if len(updates) < 2:
        return list(updates)

    by_id: dict[str, _UpdateAction] = {}
    collisions = 0
    for upd in updates:
        existing = by_id.get(upd.observation_id)
        if existing is None:
            by_id[upd.observation_id] = upd
            continue
        collisions += 1
        merged_ids = list(dict.fromkeys([*existing.source_fact_ids, *upd.source_fact_ids]))
        by_id[upd.observation_id] = _UpdateAction(
            text=upd.text,
            observation_id=upd.observation_id,
            source_fact_ids=merged_ids,
        )

    if collisions:
        logger.warning(
            f"[CONSOLIDATION] {batch_label}: LLM emitted {collisions} duplicate update(s) targeting "
            f"the same observation_id ({len(updates)} updates -> {len(by_id)} after dedup). "
            "Kept the last text and unioned source_fact_ids."
        )

    return list(by_id.values())


# Backoff for the OUTER batch retry ladder (the provider runs its own, independently).
# Deliberately short: it only has to ride out a blip, and the caller's adaptive
# bisection is the real recovery path for anything longer-lived.
_OUTER_RETRY_INITIAL_BACKOFF = 1.0
_OUTER_RETRY_MAX_BACKOFF = 8.0


class _BatchFailureClass(StrEnum):
    """How the batch retry loop must treat an exception from the LLM call."""

    PROPAGATE = "propagate"
    """Not a batch failure at all — re-raise so the caller's handler sees it."""

    FAIL_FAST = "fail_fast"
    """A re-send of the identical payload cannot help; fail the batch immediately."""

    RETRY = "retry"
    """Transport-shaped; an unchanged re-send may well succeed."""


def _classify_batch_failure(exc: Exception) -> _BatchFailureClass:
    """Decide how ``_consolidate_batch_with_llm`` should react to ``exc``.

    Before #3684 the loop caught bare ``Exception`` and retried everything on one
    ladder, on top of the provider's own. Three problems, in descending severity:

    1. ``ProviderRateLimitResetError`` is a *control signal*, not a failure: the
       provider told us when quota reopens, and ``execute_task`` turns it into a
       ``DeferOperation`` that reschedules the job for exactly then. Catching it
       here meant the defer never fired — the batch was reported failed, adaptive
       bisection re-hit the same quota wall on every sub-batch, and each memory
       ended up stamped ``consolidation_failed_at``. That flag is the exclusion
       predicate for pending consolidation and is cleared only by an explicit
       bank-wide reset, so a *transient* quota exhaustion permanently orphaned
       those facts. ``fact_extraction`` re-raises it for the same reason.
    2. A 401/403 is a permanent server misconfiguration. Same shape as (1): every
       memory in the bank would be marked failed because a key was wrong.
    3. Malformed or schema-invalid output is input-shaped. Consolidation pins
       ``llm_temperature_consolidation`` (0.0 by default) and this loop rebuilds a
       byte-identical payload, so a re-send asks a greedy decoder the same question
       and gets the same answer — the reporter measured twelve failures at the same
       character offset. Retrying still costs a full generation, and for a JSON
       parse error that is on top of the four the provider already burned.

    ``FAIL_FAST`` is not "give up": the batch is still reported failed, so the
    caller's adaptive bisection halves it and tries again. That path *does* vary
    the input, which is what an input-shaped failure needs. It is the identical
    re-send at the same batch size that has nothing to offer.
    """
    if isinstance(exc, ProviderRateLimitResetError):
        return _BatchFailureClass.PROPAGATE
    # Duck-typed rather than importing a provider SDK's error class: every SDK we
    # front (openai, anthropic) exposes the HTTP status this way.
    if getattr(exc, "status_code", None) in (401, 403):
        return _BatchFailureClass.PROPAGATE
    if isinstance(exc, json.JSONDecodeError | ValidationError | OutputTooLongError):
        return _BatchFailureClass.FAIL_FAST
    # Providers that surface an empty/unusable body flag their own retryability.
    if getattr(exc, "retryable", None) is False:
        return _BatchFailureClass.FAIL_FAST
    return _BatchFailureClass.RETRY


async def _consolidate_batch_with_llm(
    llm_config: Any,
    memories: list[dict[str, Any]],
    union_observations: "list[MemoryFact]",
    union_source_facts: "dict[str, MemoryFact]",
    config: Any,
    remaining_observation_slots: int | None = None,
    max_observations_per_scope: int = -1,
) -> _BatchLLMResult:
    """Single LLM call for a batch of facts against a pooled set of observations."""
    if config is None:
        raise ValueError("config is required for _consolidate_batch_with_llm")
    if union_observations:
        obs_list = _build_observations_for_llm(union_observations, union_source_facts)
        observations_text = json.dumps(obs_list, indent=2, ensure_ascii=False)
    else:
        observations_text = "[]"

    def _fact_line(m: dict[str, Any]) -> str:
        text = f"[{m['id']}] {m['text']}"
        temporal_parts = []
        if m.get("occurred_start"):
            temporal_parts.append(f"occurred_start={m['occurred_start']}")
        if m.get("occurred_end"):
            temporal_parts.append(f"occurred_end={m['occurred_end']}")
        if m.get("mentioned_at"):
            temporal_parts.append(f"mentioned_at={m['mentioned_at']}")
        if temporal_parts:
            text += f" ({', '.join(temporal_parts)})"
        return text

    facts_lines = "\n".join(_fact_line(m) for m in memories)

    # Build capacity note for the prompt when observation limit is configured
    observation_capacity_note: str | None = None
    if remaining_observation_slots is not None and max_observations_per_scope >= 0:
        if remaining_observation_slots == 0:
            observation_capacity_note = (
                f"OBSERVATION LIMIT REACHED ({max_observations_per_scope}/{max_observations_per_scope}). "
                "Only UPDATE or DELETE existing observations. Do NOT create new ones — "
                "merge new knowledge into existing observations via UPDATE."
            )
        elif remaining_observation_slots <= len(memories):
            observation_capacity_note = (
                f"This scope has {remaining_observation_slots} observation slot(s) remaining "
                f"(out of {max_observations_per_scope}). Prefer UPDATE over CREATE when possible."
            )

    # Split the prompt: a bank-agnostic system instruction (rules + input format +
    # decision guide + output format) that is byte-identical across batches AND
    # across banks, and a per-batch user message (mission + capacity note + facts +
    # existing observations). The split lets the system prefix be served from a
    # single Gemini context cache shared by every bank — the bank mission, capacity
    # note, and response_schema (all bank/batch-variable) are kept OUT of the
    # cached prefix so one cache serves all and it never busts within a run.
    system_prompt = build_consolidation_system_prompt(
        llm_output_language=config.llm_output_language if config is not None else None,
    )
    user_content = build_consolidation_input(
        facts_text=facts_lines,
        observations_text=observations_text,
        observations_mission=config.observations_mission,
        observation_capacity_note=observation_capacity_note,
    )

    # Opt into context caching of the stable system prefix when the provider
    # supports it (gemini/vertexai with the flag on). response_schema is NOT
    # passed to the fingerprint: it varies per batch (max_creates) but is not
    # part of the cached prefix, so keying on it would needlessly bust the cache.
    cached_prefix_name: str | None = None
    provider_impl = getattr(llm_config, "_provider_impl", None)
    if provider_impl is not None and provider_impl.supports_prompt_caching():
        try:
            cached_prefix_name = await provider_impl.get_or_create_cached_prefix(
                system_instruction=system_prompt,
            )
        except Exception:
            logger.exception("Consolidation cache prefix lookup failed; falling back to uncached call")
            cached_prefix_name = None

    # Use a constrained response model when observation limit is active
    response_model = _build_response_model(
        max_creates=remaining_observation_slots,
        supports_max_items=config.llm_supports_max_items,
    )

    max_attempts = config.consolidation_max_attempts
    inner_max_retries = config.consolidation_llm_max_retries
    last_exc: Exception | None = None
    attempts_made = 0
    # Pre-compute a stable identifier set for the batch so failure logs name the
    # exact memories whose consolidation is failing — without this, an opaque
    # "LLM batch call failed" line gives operators no way to find the offending
    # input until adaptive bisection narrows the batch down to a single memory.
    memory_ids = [str(m.get("id")) for m in memories]
    if len(memory_ids) <= 5:
        ids_label = ", ".join(memory_ids)
    else:
        ids_label = f"{', '.join(memory_ids[:3])}, ... +{len(memory_ids) - 3} more"
    batch_label = f"{len(memory_ids)} memories [{ids_label}]"
    for attempt in range(1, max_attempts + 1):
        attempts_made = attempt
        try:
            call_kwargs: dict[str, Any] = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "response_format": response_model,
                "temperature": config.llm_temperature_consolidation,
                "scope": "consolidation",
                # Resolved per operation (HINDSIGHT_API_LLM_STRICT_SCHEMA_CONSOLIDATION, falling
                # back to the global flag) so an operator can grammar-enforce consolidation's
                # structured output -- which narrows the raw-JSON failure mode behind #2668 --
                # without forcing strict schema on operations whose model can't satisfy it.
                "strict_schema": config.llm_strict_schema_consolidation,
            }
            # Only request an explicit output budget when configured. Left unset by default the key is
            # omitted, so each provider keeps its implicit default (backwards compatible). Operators on
            # providers with a low hidden cap (notably Bedrock imported models, which truncate structured
            # consolidation JSON) set HINDSIGHT_API_CONSOLIDATION_MAX_COMPLETION_TOKENS to fix it.
            if config.consolidation_max_completion_tokens is not None:
                call_kwargs["max_completion_tokens"] = config.consolidation_max_completion_tokens
            if inner_max_retries is not None:
                call_kwargs["max_retries"] = inner_max_retries
            if cached_prefix_name is not None:
                call_kwargs["cached_prefix"] = cached_prefix_name
            response: _ConsolidationBatchResponse = await llm_config.call(**call_kwargs)
            # Defensive truncation: some LLM providers may not enforce JSON schema max_length
            creates = response.creates
            if remaining_observation_slots is not None and remaining_observation_slots >= 0:
                if len(creates) > remaining_observation_slots:
                    logger.info(
                        f"[CONSOLIDATION] Truncating {len(creates)} creates to {remaining_observation_slots} "
                        f"(max_observations_per_scope={max_observations_per_scope})"
                    )
                    creates = creates[:remaining_observation_slots]
            updates = _dedupe_updates(response.updates, batch_label=batch_label)
            return _BatchLLMResult(
                creates=creates,
                updates=updates,
                deletes=response.deletes,
                obs_count=len(union_observations),
                prompt_chars=len(system_prompt) + len(user_content),
            )
        except Exception as exc:
            failure_class = _classify_batch_failure(exc)
            if failure_class is _BatchFailureClass.PROPAGATE:
                logger.warning(
                    f"[CONSOLIDATION] LLM batch call for {batch_label} raised a non-batch failure "
                    f"({type(exc).__name__}); propagating to the task handler rather than marking "
                    f"the memories failed: {exc}"
                )
                raise
            last_exc = exc
            if failure_class is _BatchFailureClass.FAIL_FAST:
                logger.warning(
                    f"[CONSOLIDATION] LLM batch call failed (attempt {attempt}/{max_attempts}) for "
                    f"{batch_label} with a non-retryable {type(exc).__name__}; not re-sending the "
                    f"identical payload: {exc}"
                )
                break
            logger.warning(
                f"[CONSOLIDATION] LLM batch call failed (attempt {attempt}/{max_attempts}) for {batch_label}: {exc}"
            )
            # Backoff on the outer ladder too. Without it a rate limit or a transient
            # overload was re-sent immediately, three times, defeating the point of the
            # provider's own backoff. Skipped after the final attempt — nothing follows it.
            if attempt < max_attempts:
                await asyncio.sleep(min(_OUTER_RETRY_INITIAL_BACKOFF * (2 ** (attempt - 1)), _OUTER_RETRY_MAX_BACKOFF))

    logger.error(
        f"[CONSOLIDATION] LLM batch call failed after {attempts_made}/{max_attempts} attempt(s) for "
        f"{batch_label}, skipping batch (the caller will bisect it). Last error: {last_exc}"
    )
    return _BatchLLMResult(
        obs_count=len(union_observations), prompt_chars=len(system_prompt) + len(user_content), failed=True
    )


async def _apply_create_observation(
    conn,
    memory_engine: "MemoryEngine",
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
    observation_text: str,
    embedding_str: str | None,
    tags: list[str] | None = None,
    event_date: datetime | None = None,
    occurred_start: datetime | None = None,
    occurred_end: datetime | None = None,
    mentioned_at: datetime | None = None,
    perf: ConsolidationPerfLog | None = None,
) -> dict[str, Any]:
    """Insert one observation from pre-embedded text.

    Runs on the caller's connection inside the caller's transaction, so it commits together
    with every other write derived from the same LLM response (#3876). The embedding is
    computed by the caller's prepare phase — a slow embedder must never pin a pooled
    connection, let alone one holding an open transaction.
    """
    now = datetime.now(timezone.utc)
    obs_event_date = event_date or now
    obs_occurred_start = occurred_start
    obs_occurred_end = occurred_end
    obs_mentioned_at = mentioned_at or now
    obs_tags = tags or []
    observation_id = uuid.uuid4()

    # Write the observation. A SQL store keeps it as a `memory_units` row (inline below, with the
    # search_vector the configured backend needs); a store that owns its rows takes it through
    # upsert_observation as a normal Observation-type memory carrying all of its own state.
    store = get_memories()
    # FOR SHARE liveness + INSERT share the batch's transaction, so a concurrent
    # delete cannot orphan the new observation between the check and the insert.
    live_source_memory_ids = await _filter_live_source_memories(conn, bank_id, source_memory_ids)
    if not live_source_memory_ids:
        logger.debug(f"Create skipped: all {len(source_memory_ids)} source memories were deleted concurrently")
        return {"action": "skipped", "reason": "sources_deleted"}
    source_memory_ids = live_source_memory_ids

    t0 = time.time()
    if not store.store_owned_for(bank_id):
        # Query varies based on text search backend.
        from ..schema import _is_oracle  # noqa: PLC0415

        config = get_config()
        if config.text_search_extension == "vchord":
            # VectorChord: manually tokenize and insert search_vector
            query = f"""
                INSERT INTO {fq_table("memory_units")} (
                    id, bank_id, text, fact_type, embedding, proof_count, source_memory_ids,
                    tags, event_date, occurred_start, occurred_end, mentioned_at, search_vector
                )
                VALUES ($1, $2, $3, 'observation', $4::vector, 1, $5, $6, $7, $8, $9, $10,
                        tokenize($3, 'llmlingua2')::bm25_catalog.bm25vector)
                RETURNING id
            """
        elif config.text_search_extension == "native" and not _is_oracle():
            # Native (PostgreSQL): search_vector is populated with to_tsvector()
            # using the configured native language dictionary, matching the batch
            # insert path in ops_postgresql.insert_facts_batch. On Oracle this falls
            # through to the no-search_vector branch below (Oracle maintains its text
            # index separately; to_tsvector/::regconfig is PG-only — see #3021).
            query = f"""
                INSERT INTO {fq_table("memory_units")} (
                    id, bank_id, text, fact_type, embedding, proof_count, source_memory_ids,
                    tags, event_date, occurred_start, occurred_end, mentioned_at, search_vector
                )
                VALUES ($1, $2, $3, 'observation', $4::vector, 1, $5, $6, $7, $8, $9, $10,
                        to_tsvector('{config.text_search_extension_native_language}'::regconfig, COALESCE($3, '')))
                RETURNING id
            """
        else:  # pg_textsearch, pgroonga, pg_search, and Oracle: base text columns / separate index
            query = f"""
                INSERT INTO {fq_table("memory_units")} (
                    id, bank_id, text, fact_type, embedding, proof_count, source_memory_ids,
                    tags, event_date, occurred_start, occurred_end, mentioned_at
                )
                VALUES ($1, $2, $3, 'observation', $4::vector, 1, $5, $6, $7, $8, $9, $10)
                RETURNING id
            """

        row = await conn.fetchrow(
            query,
            observation_id,
            bank_id,
            observation_text,
            embedding_str,
            source_memory_ids,
            obs_tags,
            obs_event_date,
            obs_occurred_start,
            obs_occurred_end,
            obs_mentioned_at,
        )
        created_id = row["id"]

        # Populate observation_sources junction table (Oracle only — PG uses native array ops).
        if memory_engine._backend.ops.uses_observation_sources_table and source_memory_ids:
            await conn.executemany(
                f"""
                INSERT INTO {fq_table("observation_sources")} (observation_id, source_id)
                VALUES ($1, $2)
                ON CONFLICT (observation_id, source_id) DO NOTHING
                """,
                [(observation_id, sid) for sid in dict.fromkeys(source_memory_ids)],
            )
    else:
        await store.upsert_observation(
            conn=conn,
            bank_id=bank_id,
            record=FactRecord(
                unit_id=str(observation_id),
                text=observation_text,
                embedding=embedding_str,
                fact_type="observation",
                tags=list(obs_tags),
                proof_count=1,
                source_memory_ids=[str(s) for s in source_memory_ids],
                event_date=obs_event_date,
                occurred_start=obs_occurred_start,
                occurred_end=obs_occurred_end,
                mentioned_at=obs_mentioned_at,
                created_at=now,
            ),
        )
        created_id = observation_id

    if perf:
        perf.record_timing("db_write", time.time() - t0)

    logger.debug(f"Created observation {observation_id} from {len(source_memory_ids)} memories (tags: {obs_tags})")

    return {"action": "created", "observation_id": str(created_id), "tags": obs_tags}
