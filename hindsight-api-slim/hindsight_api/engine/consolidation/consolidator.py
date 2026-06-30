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
from fnmatch import fnmatchcase
from itertools import combinations
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, field_validator

from ...config import get_config
from ...worker.stage import set_stage
from ..db_utils import acquire_with_retry
from ..llm_trace import (
    record_created_memory_ids,
    record_source_memory_ids,
    reset_trace_context,
    set_trace_context,
    trace_context_of,
)
from ..llm_wrapper import sanitize_llm_output
from ..memory_engine import Budget, fq_table
from ..retain import embedding_utils
from .prompts import (
    build_consolidation_input,
    build_consolidation_system_prompt,
)

if TYPE_CHECKING:
    from asyncpg import Connection

    from ...api.http import RequestContext
    from ..memory_engine import MemoryEngine
    from ..response_models import MemoryFact, RecallResult

logger = logging.getLogger(__name__)


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


_DEDUP_PROMPT = """You reconcile long-term memory observations. A NEW observation is about to be \
stored, and it is highly similar to an EXISTING one:

[NEW] {new}
[EXISTING] {existing}

If they assert the SAME fact (wording aside), respond action="merge" and provide `text`: a single \
observation that preserves EVERY detail from both. If they differ in ANY important detail — a \
number/quantity, a named entity or language, a negation, or a condition — respond action="keep"."""


def _dedup_active(config: Any) -> bool:
    """Whether create/update semantic dedup runs for this consolidation.

    Enabled when the resolved threshold is < 1.0, EXCEPT on Oracle: the merge path uses
    Postgres-only SQL (``unnest``/``array_agg``, ``UPDATE ... FROM``), so on Oracle dedup is
    skipped — it behaves exactly as it did before this feature, regardless of the configured
    threshold. This is why the feature can ship enabled-by-default without breaking Oracle.
    """
    if config is None or getattr(config, "consolidation_dedup_threshold", 1.0) >= 1.0:
        return False
    return get_config().database_backend != "oracle"


@dataclass
class _DedupOutcome:
    """Result of probing one observation against its in-scope neighbours.

    ``best_id`` is the nearest observation at/above the threshold (None if none),
    ``merged_text`` is the LLM-synthesized union text (set only when ``should_merge``).
    """

    best_id: str | None
    merged_text: str
    should_merge: bool


async def _dedup_adjudicate(
    conn: "Connection",
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
    """
    from ..search.retrieval import retrieve_semantic_bm25_combined

    threshold = config.consolidation_dedup_threshold
    if anchor_emb_str is None:
        embs = await embedding_utils.generate_embeddings_batch(memory_engine.embeddings, [anchor_text])
        if not embs:
            return _DedupOutcome(best_id=None, merged_text="", should_merge=False)
        anchor_emb_str = str(embs[0])
    tags_match = "all_strict" if tags else "any"
    grouped = await retrieve_semantic_bm25_combined(
        conn, anchor_emb_str, anchor_text, bank_id, ["observation"], _DEDUP_TOP_K, tags=tags, tags_match=tags_match
    )
    results = grouped.get("observation", ([], []))[0]
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

    decision: _DedupDecision = await dedup_llm_config.call(
        messages=[{"role": "user", "content": _DEDUP_PROMPT.format(new=anchor_text, existing=best_text)}],
        response_format=_DedupDecision,
        scope="consolidation_dedup",
    )
    if decision.action != "merge":
        return _DedupOutcome(best_id=best_id, merged_text="", should_merge=False)
    return _DedupOutcome(best_id=best_id, merged_text=decision.text.strip() or best_text, should_merge=True)


async def _dedup_reconcile_create(
    conn: "Connection",
    memory_engine: "MemoryEngine",
    bank_id: str,
    config: Any,
    dedup_llm_config: Any,
    create_text: str,
    create_source_ids: list[uuid.UUID],
    tags: list[str] | None,
) -> str | None:
    """Semantic dedup for a single CREATE (create-time, focused 1-by-1).

    On "merge", folds the new source facts + the synthesized text into the existing
    observation and returns its id (caller skips the CREATE). Returns None when there is
    no near twin or the LLM keeps them distinct.
    """
    outcome = await _dedup_adjudicate(
        conn, memory_engine, bank_id, config, dedup_llm_config, create_text, None, tags, exclude_id=None
    )
    if not outcome.should_merge or outcome.best_id is None:
        return None

    # Fold the new source facts into the twin and persist the merged text. We keep the twin's
    # existing embedding: the merged text is >= threshold similar, so the stored vector stays
    # representative and we avoid a re-embed + a dialect-specific vector UPDATE.
    search_vector_clause = (
        f",\n            search_vector = to_tsvector('{config.text_search_extension_native_language}'::regconfig, COALESCE($1, ''))"
        if config.text_search_extension == "native"
        else ""
    )
    await conn.execute(
        f"""
        UPDATE {fq_table("memory_units")}
        SET text = $1,
            source_memory_ids = (SELECT array_agg(DISTINCT e) FROM unnest(source_memory_ids || $2::uuid[]) e),
            proof_count = (SELECT count(DISTINCT e) FROM unnest(source_memory_ids || $2::uuid[]) e),
            updated_at = now(){search_vector_clause}
        WHERE id = $3::uuid
        """,
        outcome.merged_text,
        create_source_ids,
        uuid.UUID(outcome.best_id),
    )
    return outcome.best_id


async def _dedup_reconcile_update(
    conn: "Connection",
    memory_engine: "MemoryEngine",
    bank_id: str,
    config: Any,
    dedup_llm_config: Any,
    updated_id: str,
    updated_text: str,
    updated_emb_str: str | None,
    tags: list[str] | None,
) -> None:
    """Semantic dedup for an UPDATE (after the observation was rewritten + re-embedded).

    An UPDATE rewrites an observation's text and re-embeds it, so its vector can drift to
    within threshold of a DIFFERENT existing observation. The create-time guard never sees
    this (it only runs on CREATE), so without this the two persist as a near-duplicate pair —
    the measured residual-duplicate source. Probe the updated observation's new embedding
    against the others (excluding itself); on "merge", fold the just-updated observation's
    sources into the twin, persist the merged text, and DELETE the updated row. Unlike the
    CREATE path the row already exists, so reconciliation is a fold-and-delete, not a skip.
    """
    outcome = await _dedup_adjudicate(
        conn,
        memory_engine,
        bank_id,
        config,
        dedup_llm_config,
        updated_text,
        updated_emb_str,
        tags,
        exclude_id=updated_id,
    )
    if not outcome.should_merge or outcome.best_id is None:
        return

    # Fold the updated observation's sources into the twin (keeping the twin's embedding, as in
    # the create path) then delete the now-redundant updated row. The all_strict/any tag match
    # guarantees twin and updated share scope, so dropping the updated row's tags loses no
    # visibility. Temporal fields follow the surviving twin (minimal scope; matches create).
    search_vector_clause = (
        f",\n            search_vector = to_tsvector('{config.text_search_extension_native_language}'::regconfig, COALESCE($1, ''))"
        if config.text_search_extension == "native"
        else ""
    )
    await conn.execute(
        f"""
        UPDATE {fq_table("memory_units")} t
        SET text = $1,
            source_memory_ids = (
                SELECT array_agg(DISTINCT e) FROM unnest(t.source_memory_ids || u.source_memory_ids) e
            ),
            proof_count = (
                SELECT count(DISTINCT e) FROM unnest(t.source_memory_ids || u.source_memory_ids) e
            ),
            updated_at = now(){search_vector_clause}
        FROM {fq_table("memory_units")} u
        WHERE t.id = $2::uuid AND u.id = $3::uuid
        """,
        outcome.merged_text,
        uuid.UUID(outcome.best_id),
        uuid.UUID(updated_id),
    )
    await _execute_delete_action(conn, bank_id, updated_id)
    logger.info(
        "[CONSOLIDATION] dedup-merged updated observation %s into %s (cosine>=%.2f)",
        updated_id[:8],
        outcome.best_id[:8],
        config.consolidation_dedup_threshold,
    )


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
    """Parse the per-memory ``observation_scopes`` column from a DB row.

    asyncpg may return JSONB as a raw JSON string depending on driver settings;
    accept both that and a pre-parsed value.
    """
    raw = memory.get("observation_scopes")
    return json.loads(raw) if isinstance(raw, str) else raw


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
    return [frozenset(s) for s in parsed]  # explicit list[list[str]]


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

    Uses FOR SHARE to block concurrent deletes from removing a row between the
    check and the subsequent insert/update. Combined with the delete path running
    its stale-observation sweep *after* deleting the source row, this closes the
    race window where consolidation would otherwise produce an orphan observation.

    Oracle note: Oracle doesn't support FOR SHARE, so the SQL rewriter promotes
    it to FOR UPDATE. Oracle's MVCC consistent-read semantics make FOR SHARE
    unnecessary (the sweep runs AFTER deletion), but FOR UPDATE is more
    conservative and still correct.
    """
    if not source_memory_ids:
        return []
    rows = await conn.fetch(
        f"""
        SELECT id
        FROM {fq_table("memory_units")}
        WHERE id = ANY($1::uuid[]) AND bank_id = $2
        FOR SHARE
        """,
        source_memory_ids,
        bank_id,
    )
    live = {row["id"] for row in rows}
    return [mid for mid in source_memory_ids if mid in live]


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
    return await conn.fetchval(
        f"SELECT COUNT(*) FROM {fq_table('memory_units')} "
        f"WHERE bank_id = $1 AND fact_type = 'observation' AND tags @> $2::varchar[]",
        bank_id,
        tags,
    )


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
    for rule in _parse_scope_limit_rules(getattr(config, "observation_scope_limits", None)):
        if _scope_matches_globs(rule.globs, fact_tags):
            return rule.limit
    return config.max_observations_per_scope


def _build_response_model(max_creates: int | None = None) -> type[_ConsolidationBatchResponse]:
    """Build a response model, optionally constraining max creates via JSON schema."""
    if max_creates is None or max_creates < 0:
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


async def run_consolidation_job(
    memory_engine: "MemoryEngine",
    bank_id: str,
    request_context: "RequestContext",
    operation_id: str | None = None,
    observation_scopes: list[list[str]] | None = None,
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
            memory_engine, bank_id, request_context, config, llm_config, operation_id, observation_scopes
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

        # Build optional scope filter clause.  When observation_scopes is provided,
        # only process memories whose tags contain all tags in at least one scope.
        scope_clause = ""
        scope_params: list[Any] = [bank_id]
        if observation_scopes:
            or_parts: list[str] = []
            for scope_tags in observation_scopes:
                idx = len(scope_params) + 1
                or_parts.append(f"tags @> ${idx}::varchar[]")
                scope_params.append(scope_tags)
            scope_clause = " AND (" + " OR ".join(or_parts) + ")"

        # Count total unconsolidated memories for progress logging
        total_count = await conn.fetchval(
            f"""
            SELECT COUNT(*)
            FROM {fq_table("memory_units")}
            WHERE bank_id = $1
              AND consolidated_at IS NULL
              AND consolidation_failed_at IS NULL
              AND fact_type IN ('experience', 'world')
              {scope_clause}
            """,
            *scope_params,
        )

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
            pending = await count_conn.fetchval(
                f"""
                SELECT COUNT(*)
                FROM {fq_table("memory_units")}
                WHERE bank_id = $1
                  AND consolidated_at IS NULL
                  AND consolidation_failed_at IS NULL
                  AND fact_type IN ('experience', 'world')
                  {scope_clause}
                """,
                *scope_params,
            )
        return pending or 0

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

        # Fetch next batch of unconsolidated memories
        async with acquire_with_retry(pool) as conn:
            t0 = time.time()
            # scope_params[0] is bank_id; append fetch_limit after scope params
            fetch_params = list(scope_params) + [fetch_limit]
            limit_idx = len(fetch_params)
            memories = await conn.fetch(
                f"""
                SELECT id, text, fact_type, occurred_start, occurred_end, event_date, tags, mentioned_at,
                       observation_scopes
                FROM {fq_table("memory_units")}
                WHERE bank_id = $1
                  AND consolidated_at IS NULL
                  AND consolidation_failed_at IS NULL
                  AND fact_type IN ('experience', 'world')
                  {scope_clause}
                ORDER BY created_at ASC
                LIMIT ${limit_idx}
                """,
                *fetch_params,
            )
            perf.record_timing("fetch_memories", time.time() - t0)

        if not memories:
            break  # No more unconsolidated memories

        # Group memories by exact tag set before batching — security requirement:
        # memories with different tags must never share an LLM call.
        tag_groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for m in memories:
            tag_key = tuple(sorted(m.get("tags") or []))
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

                async with acquire_with_retry(pool) as conn:
                    obs_tags_list = _resolve_obs_tags_list(sub_batch[0]) if sub_batch else None

                    sub_deleted: int = 0
                    sub_llm_failed = False
                    if obs_tags_list:
                        sub_results: list[dict[str, Any]] = []
                        for obs_tags in obs_tags_list:
                            pass_results, pass_deleted, pass_failed = await _process_memory_batch(
                                conn=conn,
                                memory_engine=memory_engine,
                                llm_config=llm_config,
                                bank_id=bank_id,
                                memories=sub_batch,
                                request_context=request_context,
                                perf=batch_perf,
                                config=config,
                                obs_tags_override=obs_tags,
                            )
                            sub_deleted += pass_deleted
                            sub_llm_failed = sub_llm_failed or pass_failed
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
                            conn=conn,
                            memory_engine=memory_engine,
                            llm_config=llm_config,
                            bank_id=bank_id,
                            memories=sub_batch,
                            request_context=request_context,
                            perf=batch_perf,
                            config=config,
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

            async with acquire_with_retry(pool) as conn:
                if succeeded_ids:
                    await conn.executemany(
                        f"UPDATE {fq_table('memory_units')} SET consolidated_at = NOW() WHERE id = $1",
                        [(mem_id,) for mem_id in succeeded_ids],
                    )
                if failed_ids:
                    await conn.executemany(
                        f"UPDATE {fq_table('memory_units')} SET consolidation_failed_at = NOW() WHERE id = $1",
                        [(mem_id,) for mem_id in failed_ids],
                    )

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

            group_results = await asyncio.gather(*(_run_group(g, s) for g, s in zip(numbered_groups, group_scopes)))
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

    # Trigger mental model refreshes only on the final round (when all memories are processed).
    # If we hit the round limit and re-queued, skip MM refresh — the next round will handle it.
    if hit_round_limit:
        stats["mental_models_refreshed"] = 0
        logger.info(f"[CONSOLIDATION] bank={bank_id} skipping mental model refresh (round limit hit, re-queued)")
    else:
        set_stage("consolidation.refreshing_mental_models")
        await memory_engine._write_operation_progress(
            operation_id,
            stage="refreshing_mental_models",
            processed=stats["memories_processed"],
            total=await _progress_total(stats["memories_processed"]),
        )
        # SECURITY: Only refresh mental models with matching tags (or all if no tags were consolidated)
        mental_models_refreshed = await _trigger_mental_model_refreshes(
            memory_engine=memory_engine,
            bank_id=bank_id,
            request_context=request_context,
            consolidated_tags=list(consolidated_tags) if consolidated_tags else None,
            perf=perf,
        )
        stats["mental_models_refreshed"] = mental_models_refreshed

    perf.flush()

    return {"status": "completed", "bank_id": bank_id, **stats}


async def _trigger_mental_model_refreshes(
    memory_engine: "MemoryEngine",
    bank_id: str,
    request_context: "RequestContext",
    consolidated_tags: list[str] | None = None,
    perf: ConsolidationPerfLog | None = None,
) -> int:
    """
    Trigger refreshes for mental models with refresh_after_consolidation=true.

    SECURITY: Only triggers refresh for mental models whose tags overlap with the
    consolidated memory tags, preventing unnecessary refreshes across security boundaries.

    Args:
        memory_engine: MemoryEngine instance
        bank_id: Bank identifier
        request_context: Request context for authentication
        consolidated_tags: Tags from memories that were consolidated (None = refresh all)
        perf: Performance logging

    Returns:
        Number of mental models scheduled for refresh
    """
    pool = memory_engine._backend

    # Find mental models with refresh_after_consolidation=true that are actually stale.
    # The tag filter on the SELECT enforces the security boundary (never look outside the
    # relevant tag scope); compute_mental_model_is_stale then verifies that new memories
    # in the MM's scope really were ingested since its last refresh.
    async with acquire_with_retry(pool) as conn:
        if consolidated_tags:
            candidates = await conn.fetch(
                f"""
                SELECT id, name, tags, last_refreshed_at, trigger
                FROM {fq_table("mental_models")}
                WHERE bank_id = $1
                  AND (trigger->>'refresh_after_consolidation')::boolean = true
                  AND (
                    (tags IS NOT NULL AND tags != '{{}}' AND tags && $2::varchar[])
                    OR (tags IS NULL OR tags = '{{}}')
                  )
                """,
                bank_id,
                consolidated_tags,
            )
        else:
            candidates = await conn.fetch(
                f"""
                SELECT id, name, tags, last_refreshed_at, trigger
                FROM {fq_table("mental_models")}
                WHERE bank_id = $1
                  AND (trigger->>'refresh_after_consolidation')::boolean = true
                  AND (tags IS NULL OR tags = '{{}}')
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
            await memory_engine.submit_async_refresh_mental_model(
                bank_id=bank_id,
                mental_model_id=mental_model_id,
                request_context=request_context,
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
    conn: "Connection",
    memory_engine: "MemoryEngine",
    llm_config: Any,
    bank_id: str,
    memories: list[dict[str, Any]],
    request_context: "RequestContext",
    perf: ConsolidationPerfLog | None = None,
    config: Any = None,
    obs_tags_override: list[str] | None = None,
) -> tuple[list[dict[str, Any]], int, bool]:
    """
    Process a batch of memories in a single LLM call.

    Steps:
    1. Parallel recalls — one per fact (read-only; safe to parallelise)
    2. Union of retrieved observations across the batch (deduped by id)
    3. Single LLM call with all N facts + unioned observations
    4. Sequential action execution (writes remain serial for consistency)
    5. Returns one result dict per memory, in the same order as `memories`

    Per-fact security: action execution validates each learning_id against the
    observations that were recalled specifically for that fact, so cross-tag
    updates cannot occur.

    Args:
        obs_tags_override: When set, use these tags for observation recall and
            create/update instead of the memory's own tags. This enables multi-pass
            consolidation where a single memory can contribute to observations
            scoped at different tag levels (e.g., user-level vs session-level).
    """
    import asyncio

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
    per_fact_recalls = await asyncio.gather(*recall_tasks)
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
        current_count = await _count_observations_for_scope(conn, bank_id, fact_tags) if max_obs > 0 else 0
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

    # 4. Sequential execution of deletes / updates / creates
    # Deletes run first to free observation slots before creates consume them.
    # Track which memory indices participated so we can build per-memory results for stats
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

    # Execute deletes first to free observation slots before creates consume them
    deleted_count = 0
    for delete in llm_result.deletes:
        # Security: the observation must be present in the unioned recall
        if not any(str(obs.id) == delete.observation_id for obs in union_observations):
            logger.debug(
                f"Batch consolidation: rejected delete — observation {delete.observation_id} not in unioned recall"
            )
            continue
        await _execute_delete_action(conn=conn, bank_id=bank_id, observation_id=delete.observation_id)
        deleted_count += 1

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
        agg = _aggregate_source_fields(source_mems, tags=fact_tags)
        updated_emb_str = await _execute_update_action(
            conn=conn,
            memory_engine=memory_engine,
            bank_id=bank_id,
            source_memory_ids=[m["id"] for m in source_mems],
            observation_id=update.observation_id,
            new_text=update.text,
            observations=union_observations,
            source_fact_tags=agg.tags,
            source_occurred_start=agg.occurred_start,
            source_occurred_end=agg.occurred_end,
            source_mentioned_at=agg.mentioned_at,
            perf=perf,
        )
        for m in source_mems:
            per_memory_updated.add(str(m["id"]))
        # Reconcile the rewritten observation against its neighbours: the re-embed may have
        # drifted it into a near-twin of another existing observation (the residual-duplicate
        # source). updated_emb_str is None when the update was skipped — nothing to reconcile.
        if dedup_enabled and updated_emb_str is not None:
            await _dedup_reconcile_update(
                conn,
                memory_engine,
                bank_id,
                config,
                dedup_llm_config,
                update.observation_id,
                update.text,
                updated_emb_str,
                agg.tags,
            )

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

        # Semantic near-duplicate reconciliation: merge this CREATE into an existing
        # near-identical observation (LLM-adjudicated, 1-by-1) instead of inserting a dup.
        if dedup_enabled:
            merged_into = await _dedup_reconcile_create(
                conn, memory_engine, bank_id, config, dedup_llm_config, create.text, create_source_ids, agg.tags
            )
            if merged_into is not None:
                logger.info(
                    "[CONSOLIDATION] dedup-merged observation CREATE into %s (cosine>=%.2f)",
                    merged_into[:8],
                    config.consolidation_dedup_threshold,
                )
                for m in source_mems:
                    per_memory_created.add(str(m["id"]))
                continue

        await _execute_create_action(
            conn=conn,
            memory_engine=memory_engine,
            bank_id=bank_id,
            source_memory_ids=create_source_ids,
            text=create.text,
            source_fact_tags=agg.tags,
            event_date=agg.event_date,
            occurred_start=agg.occurred_start,
            occurred_end=agg.occurred_end,
            mentioned_at=agg.mentioned_at,
            perf=perf,
        )
        for m in source_mems:
            per_memory_created.add(str(m["id"]))

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
    await conn.execute(
        f"""
        INSERT INTO {fq_table("observation_history")} (observation_id, bank_id, content, changed_at)
        VALUES ($1, $2, $3::jsonb, now())
        """,
        obs_uuid,
        bank_id,
        json.dumps(asdict(snapshot)),
    )
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


async def _execute_update_action(
    conn: "Connection",
    memory_engine: "MemoryEngine",
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
    observation_id: str,
    new_text: str,
    observations: list["MemoryFact"],
    source_fact_tags: list[str] | None = None,
    source_occurred_start: datetime | None = None,
    source_occurred_end: datetime | None = None,
    source_mentioned_at: datetime | None = None,
    perf: ConsolidationPerfLog | None = None,
) -> str | None:
    """
    Update an existing observation.

    Extends source_memory_ids with all contributing memories, updates temporal fields
    (LEAST for occurred_start, GREATEST for occurred_end / mentioned_at), and merges tags.

    Returns the observation's freshly-computed embedding (pgvector literal) so the caller can
    run UPDATE-path dedup without re-embedding, or None when the update was skipped.
    """
    model = next((m for m in observations if str(m.id) == observation_id), None)
    if not model:
        logger.debug(f"Update skipped: observation {observation_id} not found in recall results")
        return

    live_source_memory_ids = await _filter_live_source_memories(conn, bank_id, source_memory_ids)
    if not live_source_memory_ids:
        logger.debug(
            f"Update skipped: all {len(source_memory_ids)} source memories for observation "
            f"{observation_id} were deleted concurrently"
        )
        return
    source_memory_ids = live_source_memory_ids

    from ...config import get_config

    history_entry = _ObservationHistorySnapshot(
        previous_text=model.text,
        previous_tags=list(model.tags or []),
        previous_occurred_start=model.occurred_start,
        previous_occurred_end=model.occurred_end,
        previous_mentioned_at=model.mentioned_at,
        new_source_memory_ids=[str(mid) for mid in source_memory_ids],
    )

    source_ids = list(model.source_fact_ids or []) + source_memory_ids

    # SECURITY: Merge source fact's tags into existing observation tags so all contributors can see it
    existing_tags = set(model.tags or [])
    source_tags = set(source_fact_tags or [])
    merged_tags = list(existing_tags | source_tags)

    t0 = time.time()
    embeddings = await embedding_utils.generate_embeddings_batch(memory_engine.embeddings, [new_text])
    embedding_str = str(embeddings[0]) if embeddings else None
    if perf:
        perf.record_timing("embedding", time.time() - t0)

    config = get_config()

    search_vector_clause = (
        f",\n            search_vector = to_tsvector('{config.text_search_extension_native_language}'::regconfig, COALESCE($1, ''))"
        if config.text_search_extension == "native"
        else ""
    )

    t0 = time.time()
    await conn.execute(
        f"""
        UPDATE {fq_table("memory_units")}
        SET text = $1,
            embedding = $2::vector,
            source_memory_ids = $3,
            proof_count = $4,
            tags = $9,
            updated_at = now(),
            occurred_start = LEAST(occurred_start, COALESCE($6, occurred_start)),
            occurred_end = GREATEST(occurred_end, COALESCE($7, occurred_end)),
            mentioned_at = GREATEST(mentioned_at, COALESCE($8, mentioned_at)){search_vector_clause}
        WHERE id = $5
        """,
        new_text,
        embedding_str,
        source_ids,
        len(source_ids),
        uuid.UUID(observation_id),
        source_occurred_start,
        source_occurred_end,
        source_mentioned_at,
        merged_tags,
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


async def _execute_create_action(
    conn: "Connection",
    memory_engine: "MemoryEngine",
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
    text: str,
    source_fact_tags: list[str] | None = None,
    event_date: datetime | None = None,
    occurred_start: datetime | None = None,
    occurred_end: datetime | None = None,
    mentioned_at: datetime | None = None,
    perf: ConsolidationPerfLog | None = None,
) -> None:
    """
    Create a new observation from one or more source memories.

    Tags are inherited from the source facts (determined algorithmically, not by LLM)
    to maintain visibility scope.
    """
    created = await _create_observation_directly(
        conn=conn,
        memory_engine=memory_engine,
        bank_id=bank_id,
        source_memory_ids=source_memory_ids,
        observation_text=text,
        tags=source_fact_tags or [],
        event_date=event_date,
        occurred_start=occurred_start,
        occurred_end=occurred_end,
        mentioned_at=mentioned_at,
        perf=perf,
    )
    # Map the new observation onto the consolidation trace as a produced memory.
    new_id = created.get("observation_id")
    if new_id:
        record_created_memory_ids([new_id])
    logger.debug(f"Created observation from {len(source_memory_ids)} source memories")


async def _execute_delete_action(
    conn: "Connection",
    bank_id: str,
    observation_id: str,
) -> None:
    """Delete a superseded or contradicted observation."""
    await conn.execute(
        f"DELETE FROM {fq_table('memory_units')} WHERE id = $1 AND bank_id = $2 AND fact_type = 'observation'",
        uuid.UUID(observation_id),
        bank_id,
    )
    logger.debug(f"Deleted observation {observation_id}")


async def _create_memory_links(
    conn: "Connection",
    memory_id: uuid.UUID,
    observation_id: uuid.UUID,
) -> None:
    """
    Placeholder for observation link creation.

    Observations do NOT get any memory_links copied from their source facts.
    Instead, retrieval uses source_memory_ids to traverse:
    - Entity connections: observation → source_memory_ids → unit_entities
    - Semantic similarity: observations have their own embeddings
    - Temporal proximity: observations have their own temporal fields

    This avoids data duplication and ensures observations are always
    connected via their source facts' relationships.

    The memory_id and observation_id parameters are kept for interface
    compatibility but no links are created.
    """
    # No links are created - observations rely on source_memory_ids for traversal
    pass


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
        llm_output_language=getattr(config, "llm_output_language", None),
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
    response_model = _build_response_model(max_creates=remaining_observation_slots)

    max_attempts = config.consolidation_max_attempts
    inner_max_retries = config.consolidation_llm_max_retries
    last_exc: Exception | None = None
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
        try:
            call_kwargs: dict[str, Any] = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "response_format": response_model,
                "scope": "consolidation",
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
            last_exc = exc
            logger.warning(
                f"[CONSOLIDATION] LLM batch call failed (attempt {attempt}/{max_attempts}) for {batch_label}: {exc}"
            )

    logger.error(
        f"[CONSOLIDATION] LLM batch call failed after {max_attempts} attempts for {batch_label}, "
        f"skipping batch. Last error: {last_exc}"
    )
    return _BatchLLMResult(
        obs_count=len(union_observations), prompt_chars=len(system_prompt) + len(user_content), failed=True
    )


async def _create_observation_directly(
    conn: "Connection",
    memory_engine: "MemoryEngine",
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
    observation_text: str,
    tags: list[str] | None = None,
    event_date: datetime | None = None,
    occurred_start: datetime | None = None,
    occurred_end: datetime | None = None,
    mentioned_at: datetime | None = None,
    perf: ConsolidationPerfLog | None = None,
) -> dict[str, Any]:
    """Create an observation from one or more source memories with pre-processed text."""
    live_source_memory_ids = await _filter_live_source_memories(conn, bank_id, source_memory_ids)
    if not live_source_memory_ids:
        logger.debug(f"Create skipped: all {len(source_memory_ids)} source memories were deleted concurrently")
        return {"action": "skipped", "reason": "sources_deleted"}
    source_memory_ids = live_source_memory_ids

    # Generate embedding for the observation (convert to string for pgvector)
    t0 = time.time()
    embeddings = await embedding_utils.generate_embeddings_batch(memory_engine.embeddings, [observation_text])
    embedding_str = str(embeddings[0]) if embeddings else None
    if perf:
        perf.record_timing("embedding", time.time() - t0)

    # Create the observation as a memory_unit
    now = datetime.now(timezone.utc)
    obs_event_date = event_date or now
    obs_occurred_start = occurred_start
    obs_occurred_end = occurred_end
    obs_mentioned_at = mentioned_at or now
    obs_tags = tags or []

    t0 = time.time()
    observation_id = uuid.uuid4()

    # Query varies based on text search backend
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
    elif config.text_search_extension == "native":
        # Native: search_vector is populated with to_tsvector() using the
        # configured native language dictionary, matching the batch insert
        # path in ops_postgresql.insert_facts_batch.
        query = f"""
            INSERT INTO {fq_table("memory_units")} (
                id, bank_id, text, fact_type, embedding, proof_count, source_memory_ids,
                tags, event_date, occurred_start, occurred_end, mentioned_at, search_vector
            )
            VALUES ($1, $2, $3, 'observation', $4::vector, 1, $5, $6, $7, $8, $9, $10,
                    to_tsvector('{config.text_search_extension_native_language}'::regconfig, COALESCE($3, '')))
            RETURNING id
        """
    else:  # pg_textsearch, pgroonga, pg_search: indexes operate on base text columns directly
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

    if perf:
        perf.record_timing("db_write", time.time() - t0)

    logger.debug(f"Created observation {observation_id} from {len(source_memory_ids)} memories (tags: {obs_tags})")

    return {"action": "created", "observation_id": str(row["id"]), "tags": obs_tags}
