"""
Entity extraction and resolution for memory system.

Uses spaCy for entity extraction and implements resolution logic
to disambiguate entities across memory units.
"""

import asyncio
import heapq
import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any, Final, cast

from .db_utils import acquire_with_retry
from .memory_engine import fq_table
from .retain.entity_labels import (
    build_labels_lookup as _build_labels_lookup_from_config,
)
from .retain.entity_labels import (
    is_label_entity as _is_label_entity,
)
from .retain.entity_labels import (
    parse_entity_labels as _parse_entity_labels,
)
from .retain.types import ResolvedEntity

logger = logging.getLogger(__name__)


@dataclass
class _EntityToCreate:
    """An entity that needs to be inserted (no matching candidate found)."""

    idx: int
    name: str
    event_date: datetime | None
    # Label entities (from entity_labels config) are never fuzzy-merged in-batch — their
    # canonical names are user-defined (e.g. "use:use-001") and must stay distinct (GH-1558).
    # Also stored on the row as entities.entity_kind so label rows stay out of the
    # partial trigram index (#3208).
    is_label: bool = False
    # False when the caller wrote this name literally: it is created as spelled and never
    # merged with a same-batch near-duplicate (#3479).
    resolve: bool = True


@dataclass
class _SimilarNamePair:
    """A pair of in-batch new-entity names judged similar enough to be the same entity."""

    name_a: str
    name_b: str


# The in-batch dedup pass is O(N^2) over the batch's *new* names. It is sub-millisecond for a
# normal retain (a handful of new entities) but scales quadratically — measured on the retain hot
# path: ~0.8ms at 100 names, ~5ms at 250, ~22ms at 500, ~81ms at 1000. So skip it past this many
# unique new names and log rather than silently degrade; the cap sits well above any realistic
# single-retain new-entity count while bounding the tail.
_INTRABATCH_MAX_NAMES = 250

# A pg_trgm "word" is a maximal run of alphanumerics (Unicode letters/digits, underscore excluded);
# everything else (space, punctuation, emoji) is a separator. This is why decoration variants like
# "Wren <emoji>" collapse to the same trigram set.
_TRGM_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _trigram_set(text: str) -> set[str]:
    """Trigrams of ``text`` the way PostgreSQL pg_trgm generates them: lowercase, split into words,
    pad each word with two leading + one trailing blank, and take every 3-char window."""
    trigrams: set[str] = set()
    for word in _TRGM_WORD.findall(text.lower()):
        padded = f"  {word} "
        for i in range(len(padded) - 2):
            trigrams.add(padded[i : i + 3])
    return trigrams


def _trigram_similarity(a: str, b: str) -> float:
    """pg_trgm ``similarity(a, b)`` computed in-memory — the Jaccard index of the trigram sets.

    Verified byte-for-byte against Postgres pg_trgm across emoji / accent / CJK / hyphen /
    apostrophe cases (issue #3107), so the merge cutoff calibrated on pg_trgm transfers exactly.
    Doing it in Python keeps the in-batch dedup off the retain transaction's DB connection and makes
    it backend-agnostic (Postgres, Oracle, and the pg_trgm-absent "full" fallback all behave alike).
    """
    ta, tb = _trigram_set(a), _trigram_set(b)
    intersection = len(ta & tb)
    union = len(ta) + len(tb) - intersection
    return intersection / union if union else 0.0


def _find_intrabatch_similar_pairs(names: list[str], threshold: float) -> list[_SimilarNamePair]:
    """Every pair of ``names`` whose in-memory trigram similarity meets ``threshold``. O(N^2) over a
    small, capped set of new names — pure CPU, no DB round-trip."""
    trigrams = [_trigram_set(n) for n in names]
    pairs: list[_SimilarNamePair] = []
    for i in range(len(names)):
        ti = trigrams[i]
        for j in range(i + 1, len(names)):
            tj = trigrams[j]
            intersection = len(ti & tj)
            union = len(ti) + len(tj) - intersection
            if union and intersection / union >= threshold:
                pairs.append(_SimilarNamePair(name_a=names[i], name_b=names[j]))
    return pairs


def _cluster_new_entity_names(
    rep_by_lower: dict[str, str],
    count_by_lower: dict[str, int],
    pairs: list[_SimilarNamePair],
) -> dict[str, str]:
    """Union-find the similar-name pairs into clusters and pick one canonical name each.

    Args:
        rep_by_lower: lowercase name -> a representative original-case spelling of it.
        count_by_lower: lowercase name -> how many mentions carry it (for canonical choice).
        pairs: name pairs judged similar (order/case irrelevant; compared lowercased).

    Returns:
        lowercase name -> canonical original-case name for its cluster. Singletons map to
        themselves, so the caller can look up every member uniformly.
    """
    parent: dict[str, str] = {nl: nl for nl in rep_by_lower}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    for pair in pairs:
        a, b = pair.name_a.lower(), pair.name_b.lower()
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    clusters: dict[str, list[str]] = {}
    for nl in rep_by_lower:
        clusters.setdefault(find(nl), []).append(nl)

    canonical_by_member: dict[str, str] = {}
    for members in clusters.values():
        # Canonical = most-mentioned, then shortest, then lexicographically smallest — a
        # deterministic pick that prefers the plainest spelling in the cluster.
        canonical_lower = min(members, key=lambda nl: (-count_by_lower[nl], len(rep_by_lower[nl]), rep_by_lower[nl]))
        canonical_name = rep_by_lower[canonical_lower]
        for nl in members:
            canonical_by_member[nl] = canonical_name
    return canonical_by_member


@dataclass
class _EntityStat:
    """Stat accumulation entry for a resolved entity (post-transaction update)."""

    entity_id: str
    event_date: datetime | None


@dataclass
class _EntityStatAgg:
    """Aggregated stats used when flushing pending updates."""

    count: int = 0
    max_date: datetime | None = None


# Sentinel distinguishing "key not in dict" from "key present with value None".
# Needed when merging event_date across duplicate unit rows: legacy two-tuple
# callers surface `None`, which must not clobber a real datetime from another
# caller for the same unit.
_SENTINEL_MISSING: Final = object()


def _later_date(a: datetime | None, b: datetime | None) -> datetime | None:
    """Return whichever of ``a`` / ``b`` is later (None loses to any datetime).

    Used to fold duplicate co-occurrence pairs across a retain batch: legacy
    two-tuple callers surface ``None``, which must not clobber a real
    datetime that arrived for the same pair from an aware caller.
    """
    if a is None:
        return b
    if b is None:
        return a
    return a if a > b else b


def _canonical_cooccurrence_pairs(entity_list: list[str]) -> Iterator[tuple[str, str]]:
    """Yield each distinct pair of ``entity_list`` as ``(a, b)`` with ``a < b``.

    Canonical ordering matches the entity_cooccurrences PK and check constraint.
    The pair is ordered into fresh locals rather than by swapping the loop
    variables: ``entity_id_1`` is the outer iterate, so swapping it would leak
    into the remaining inner iterations and build later pairs off the wrong
    element.
    """
    for i, entity_id_1 in enumerate(entity_list):
        for entity_id_2 in entity_list[i + 1 :]:
            if entity_id_1 == entity_id_2:
                continue
            yield (entity_id_1, entity_id_2) if entity_id_1 < entity_id_2 else (entity_id_2, entity_id_1)


@dataclass
class _CooccurrencePair:
    """A (entity_id_1, entity_id_2) pair observed in a retain batch (for post-txn flush)."""

    entity_id_1: str
    entity_id_2: str
    # When the two entities co-occurred in the source content. For real-time
    # retains this is ~now; for backfilled corpora it's the historical event
    # time, so the cooccurrence cache reflects the underlying knowledge
    # timeline instead of collapsing to the import moment.
    event_date: datetime | None = None


# Load spaCy model (singleton)
_nlp = None


# Candidates scored between cooperative yields to the event loop. Scoring is
# synchronous CPU (one SequenceMatcher per candidate, ~50µs), so a batch with a
# large candidate set would otherwise hold the loop thread for minutes — health
# probes time out and the orchestrator kills the worker mid-op (GH-3211).
# 256 candidates ≈ 13ms of work between yields.
_SCORING_YIELD_EVERY: Final = 256


def _cheap_rank_key(entity_text_lower: str, candidate: tuple[Any, str, Any, datetime | None, int | None]) -> tuple:
    """Ordering key (not a multi-value return) approximating match quality cheaply.

    Used only to truncate oversized candidate sets: the fuzzy strategies already
    cap and pre-rank in SQL by real similarity, so this is the backstop for sets
    built without a score (the "full" strategy's substring matching). Ranks an
    exact match first, then a close name length, then a well-established entity —
    all O(1) per candidate, unlike the SequenceMatcher pass it protects.
    """
    name_lower = candidate[1].lower()
    return (
        0 if name_lower == entity_text_lower else 1,
        abs(len(name_lower) - len(entity_text_lower)),
        -(candidate[4] or 0),
        candidate[1],
    )


class EntityResolver:
    """
    Resolves entities to canonical IDs with disambiguation.
    """

    def __init__(
        self,
        pool: Any,
        entity_lookup: str = "full",
        entity_resolution_batch_size: int = 100,
        intrabatch_merge_similarity: float = 0.5,
        entity_resolution_max_candidates: int = 200,
    ):
        """
        Initialize entity resolver.

        Args:
            pool: asyncpg connection pool
            entity_lookup: Lookup strategy — "full" loads all bank entities then
                matches in Python; "trigram" uses pg_trgm GIN index to fetch only
                similar candidates per entity name (much faster for large banks).
            entity_resolution_batch_size: Number of unique entity names to include
                in each pg_trgm candidate lookup query.
            intrabatch_merge_similarity: pg_trgm similarity at/above which two new
                names created by the same retain are merged into one entity.
            entity_resolution_max_candidates: Max candidates scored per entity
                mention. Scoring is a synchronous SequenceMatcher call per
                candidate, so an unbounded candidate set turns one resolution
                batch into minutes of event-loop-blocking CPU (GH-3211).
        """
        self.pool = pool
        self.entity_lookup = entity_lookup
        if entity_resolution_batch_size < 1:
            raise ValueError("entity_resolution_batch_size must be >= 1")
        self.entity_resolution_batch_size = entity_resolution_batch_size
        self._intrabatch_merge_similarity = intrabatch_merge_similarity
        if entity_resolution_max_candidates < 1:
            raise ValueError("entity_resolution_max_candidates must be >= 1")
        self.entity_resolution_max_candidates = entity_resolution_max_candidates
        self._pg_trgm_checked = False
        # Backend-specific operations — accessed via pool.ops (Django pattern).
        self._ops = pool.ops if pool is not None else None
        # Keyed by asyncio task id so concurrent retain batches never mix their
        # pending updates.  flush_pending_stats() pops only the calling task's items.
        self._pending_stats: dict[int, list[_EntityStat]] = {}
        self._pending_cooccurrences: dict[int, list[_CooccurrencePair]] = {}

    def _task_key(self) -> int:
        """Return a unique key for the current asyncio task (or 0 for non-task context)."""
        task = asyncio.current_task()
        return id(task) if task is not None else 0

    def discard_pending_stats(self) -> None:
        """
        Discard accumulated entity stats and co-occurrence counts for the current task.

        Call this on any exception path between resolve_entities_batch /
        link_units_to_entities_batch and flush_pending_stats() to prevent the
        per-task dicts from growing unbounded when tasks fail before flushing.
        Safe to call even if no entries exist for the current task.
        """
        key = self._task_key()
        self._pending_stats.pop(key, None)
        self._pending_cooccurrences.pop(key, None)

    async def flush_pending_stats(self) -> None:
        """
        Flush accumulated entity stats and co-occurrence counts for the current task.

        Must be called AFTER the retain transaction commits.  Pops only the items
        accumulated by the calling asyncio task so concurrent retain batches never
        flush each other's uncommitted entity IDs.
        """
        if self.pool is None:
            return

        key = self._task_key()
        stats = self._pending_stats.pop(key, [])
        cooccurrences = self._pending_cooccurrences.pop(key, [])

        if not stats and not cooccurrences:
            return

        async with acquire_with_retry(self.pool) as conn:
            if stats:
                # Aggregate: sum counts and find max date per entity_id.
                agg: dict[str, _EntityStatAgg] = defaultdict(_EntityStatAgg)
                for s in stats:
                    entry = agg[s.entity_id]
                    entry.count += 1
                    if s.event_date is not None:
                        entry.max_date = s.event_date if entry.max_date is None else max(entry.max_date, s.event_date)

                # Sort by entity_id so all concurrent workers acquire row locks in
                # the same order — prevents circular lock dependencies (deadlocks).
                rows = sorted((eid, a.count, a.max_date) for eid, a in agg.items())
                await conn.executemany(
                    f"""
                    UPDATE {fq_table("entities")} SET
                        mention_count = mention_count + $2,
                        last_seen     = GREATEST(last_seen, $3)
                    WHERE id = $1::uuid
                    """,
                    rows,
                )

            if cooccurrences:
                # Aggregate per (entity_id_1, entity_id_2): count occurrences and
                # keep the latest event_date we saw. Using GREATEST(...) in the SQL
                # already handles merging against the existing row; here we fold
                # the batch so executemany doesn't send the same pair twice.
                coo_agg: dict[tuple[str, str], tuple[int, datetime | None]] = {}
                for c in cooccurrences:
                    pair = (c.entity_id_1, c.entity_id_2)
                    prev_count, prev_date = coo_agg.get(pair, (0, None))
                    coo_agg[pair] = (prev_count + 1, _later_date(prev_date, c.event_date))

                now = datetime.now(UTC)
                # Sort by (entity_id_1, entity_id_2) for consistent lock ordering.
                await conn.executemany(
                    f"""
                    INSERT INTO {fq_table("entity_cooccurrences")}
                        (entity_id_1, entity_id_2, cooccurrence_count, last_cooccurred)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (entity_id_1, entity_id_2)
                    DO UPDATE SET
                        cooccurrence_count = {fq_table("entity_cooccurrences")}.cooccurrence_count + EXCLUDED.cooccurrence_count,
                        last_cooccurred    = GREATEST({fq_table("entity_cooccurrences")}.last_cooccurred, EXCLUDED.last_cooccurred)
                    """,
                    sorted((e1, e2, count, event_date or now) for (e1, e2), (count, event_date) in coo_agg.items()),
                )

    @staticmethod
    def _build_labels_lookup(entity_labels: list | None) -> set[str]:
        """Build a set of valid 'key:value' entity label strings for fast lookup."""
        return _build_labels_lookup_from_config(entity_labels)

    @staticmethod
    def _chunked(values: list[str], size: int) -> list[list[str]]:
        """Split values into fixed-size batches."""
        return [values[i : i + size] for i in range(0, len(values), size)]

    @staticmethod
    def _label_texts(entity_texts: list[str], taxonomy_lookup: set[str] | None, labels_cfg) -> set[str]:
        """Subset of entity_texts that are label entities (resolved by exact match only).

        Only gate on the config, not on the lookup set: text/map groups have no
        fixed vocabulary, so a config with only those groups builds an EMPTY
        lookup — its labels are classified by key prefix inside
        ``is_label_entity``, and gating on the set would miss them entirely.
        """
        if not labels_cfg:
            return set()
        return {t for t in entity_texts if _is_label_entity(t, labels_cfg, taxonomy_lookup or set())}

    async def resolve_entities_batch(
        self,
        bank_id: str,
        entities_data: list[dict],
        context: str,
        unit_event_date,
        conn=None,
        entity_labels: list | None = None,
    ) -> list[ResolvedEntity]:
        """
        Resolve multiple entities in batch (MUCH faster than sequential).

        Groups entities by type, queries candidates in bulk, and resolves
        all entities with minimal DB queries.

        Args:
            bank_id: bank ID
            entities_data: List of dicts with 'text', 'type', 'nearby_entities'
            context: Context where entities appear
            unit_event_date: When this unit was created
            conn: Optional connection to use (if None, acquires from pool)

        Each mention may carry ``"resolve": False`` to opt out of resolution. The
        default, True, treats a name as a *guess* at which entity is meant, so
        similar existing entities are scored on name similarity + co-occurrence +
        recency and the best above threshold is reused. False takes the name
        literally: an existing entity is reused only when its canonical name
        matches case-insensitively, any other name creates its own entity, and it
        is never merged with a same-batch near-duplicate. Callers who authored the
        names deliberately want False (#3479) — resolution would otherwise let
        what the graph already believes outscore, and silently discard, their
        correction. It is per mention because retain resolves caller-supplied and
        extracted names in one batch, and only the caller's half is authoritative.

        Returns:
            Resolved entity identities (id + stored canonical name) in the same
            order as input.
        """
        if not entities_data:
            return []

        taxonomy_lookup = self._build_labels_lookup(entity_labels)
        labels_cfg = _parse_entity_labels(entity_labels)
        if conn is None:
            async with acquire_with_retry(self.pool) as conn:
                return await self._resolve_entities_batch_impl(
                    conn, bank_id, entities_data, context, unit_event_date, taxonomy_lookup, labels_cfg
                )
        else:
            return await self._resolve_entities_batch_impl(
                conn, bank_id, entities_data, context, unit_event_date, taxonomy_lookup, labels_cfg
            )

    async def _resolve_entities_batch_impl(
        self,
        conn,
        bank_id: str,
        entities_data: list[dict],
        context: str,
        unit_event_date,
        taxonomy_lookup: set[str] | None = None,
        labels_cfg=None,
    ) -> list[ResolvedEntity]:
        # `entities_data and` matters: an empty batch must fall through to the normal strategy
        # dispatch (which the pg_trgm auto-detection hangs off), not take the shortcut vacuously.
        if entities_data and not any(e.get("resolve", True) for e in entities_data):
            # Nothing in this batch resolves, so the trigram/UTL_MATCH probe and the
            # co-occurrence fetch would both be dead work. _resolve_from_candidates routes every
            # mention straight to its find-or-create path, which matches on LOWER(canonical_name)
            # equality. A *mixed* batch still probes — the per-mention check below skips the
            # literal names when scoring, which costs a little wasted lookup but keeps the
            # common all-resolving case on one code path.
            return await self._resolve_from_candidates(
                conn,
                bank_id,
                entities_data,
                unit_event_date,
                all_candidates={},
                cooccurrence_map={},
                taxonomy_lookup=taxonomy_lookup,
                labels_cfg=labels_cfg,
            )
        if self.entity_lookup == "trigram":
            # Route to backend-specific fuzzy strategy.
            # Non-PG backends (Oracle) use UTL_MATCH instead of pg_trgm.
            backend_strategy = self._ops.get_entity_resolution_strategy()
            if backend_strategy == "oracle_fuzzy":
                return await self._resolve_entities_batch_oracle_fuzzy(
                    conn, bank_id, entities_data, unit_event_date, taxonomy_lookup, labels_cfg
                )
            # Auto-detect pg_trgm availability on first call and fall back to
            # "full" strategy if the extension is not installed.  See #626.
            if not self._pg_trgm_checked:
                self._pg_trgm_checked = True
                has_trgm = await conn.fetchval("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm')")
                if not has_trgm:
                    logger.warning(
                        "pg_trgm extension is not available — falling back to 'full' "
                        "entity lookup strategy. Install pg_trgm for faster entity "
                        "resolution on large banks. See: "
                        "https://github.com/vectorize-io/hindsight/issues/626"
                    )
                    self.entity_lookup = "full"
                    return await self._resolve_entities_batch_full(
                        conn, bank_id, entities_data, unit_event_date, taxonomy_lookup, labels_cfg
                    )
            return await self._resolve_entities_batch_trigram(
                conn, bank_id, entities_data, unit_event_date, taxonomy_lookup, labels_cfg
            )
        return await self._resolve_entities_batch_full(
            conn, bank_id, entities_data, unit_event_date, taxonomy_lookup, labels_cfg
        )

    async def _resolve_entities_batch_full(
        self,
        conn,
        bank_id: str,
        entities_data: list[dict],
        unit_event_date,
        taxonomy_lookup: set[str] | None = None,
        labels_cfg=None,
    ) -> list[ResolvedEntity]:
        """Original strategy: load all bank entities then match in Python."""
        # Query ALL candidates for this bank
        all_entities = await conn.fetch(
            f"""
            SELECT canonical_name, id, metadata, last_seen, mention_count
            FROM {fq_table("entities")}
            WHERE bank_id = $1
            """,
            bank_id,
        )

        # Build entity ID to name mapping for co-occurrence lookups
        entity_id_to_name = {row["id"]: row["canonical_name"].lower() for row in all_entities}

        # Query ALL co-occurrences for this bank's entities in one query
        # This builds a map of entity_id -> set of co-occurring entity names
        all_cooccurrences = await conn.fetch(
            f"""
            SELECT ec.entity_id_1, ec.entity_id_2, ec.cooccurrence_count
            FROM {fq_table("entity_cooccurrences")} ec
            WHERE ec.entity_id_1 IN (SELECT id FROM {fq_table("entities")} WHERE bank_id = $1)
               OR ec.entity_id_2 IN (SELECT id FROM {fq_table("entities")} WHERE bank_id = $1)
            """,
            bank_id,
        )

        # Build co-occurrence map: entity_id -> set of co-occurring entity names (lowercase)
        cooccurrence_map: dict[str, set[str]] = {}
        for row in all_cooccurrences:
            eid1, eid2 = row["entity_id_1"], row["entity_id_2"]
            # Add both directions
            if eid1 not in cooccurrence_map:
                cooccurrence_map[eid1] = set()
            if eid2 not in cooccurrence_map:
                cooccurrence_map[eid2] = set()
            # Map to canonical names for comparison with nearby_entities
            if eid2 in entity_id_to_name:
                cooccurrence_map[eid1].add(entity_id_to_name[eid2])
            if eid1 in entity_id_to_name:
                cooccurrence_map[eid2].add(entity_id_to_name[eid1])

        # Build candidate map for each entity text
        all_candidates = {}  # Maps entity_text -> list of candidates
        entity_texts = list(set(e["text"] for e in entities_data))

        for entity_text in entity_texts:
            matching = []
            entity_text_lower = entity_text.lower()
            for row in all_entities:
                canonical_name = row["canonical_name"]
                ent_id = row["id"]
                metadata = row["metadata"]
                last_seen = row["last_seen"]
                mention_count = row["mention_count"]
                canonical_lower = canonical_name.lower()
                # Match if exact or substring match
                if (
                    entity_text_lower == canonical_lower
                    or entity_text_lower in canonical_lower
                    or canonical_lower in entity_text_lower
                ):
                    matching.append((ent_id, canonical_name, metadata, last_seen, mention_count))
            all_candidates[entity_text] = matching

        return await self._resolve_from_candidates(
            conn,
            bank_id,
            entities_data,
            unit_event_date,
            all_candidates,
            cooccurrence_map,
            taxonomy_lookup,
            labels_cfg,
        )

    async def _resolve_entities_batch_trigram(
        self,
        conn,
        bank_id: str,
        entities_data: list[dict],
        unit_event_date,
        taxonomy_lookup: set[str] | None = None,
        labels_cfg=None,
    ) -> list[ResolvedEntity]:
        """
        Trigram strategy: fetch only similar candidates per entity name using pg_trgm.

        Instead of loading all bank entities (O(N)), uses a GIN trigram index to fetch
        only the small set of candidates that are textually similar to each input name.
        Reduces DB data transfer from 165K rows to ~5-20 rows per entity.
        """
        entity_texts = list(set(e["text"] for e in entities_data))

        # Label entities resolve by exact match only (their canonical names are
        # user-defined and must not be fuzzy-merged). Probing them via the trigram
        # index only returns similar-but-distinct label values that are always
        # discarded, and that wasted work grows with the number of values a label
        # accumulates. Resolve label texts with an exact lookup on the unique
        # (bank_id, LOWER(canonical_name)) index and only fuzzy-match the rest.
        label_set = self._label_texts(entity_texts, taxonomy_lookup, labels_cfg)
        label_texts = [t for t in entity_texts if t in label_set]
        fuzzy_texts = [t for t in entity_texts if t not in label_set]

        rows = []

        # Exact, index-only lookup for label texts.
        for entity_text_batch in self._chunked(label_texts, self.entity_resolution_batch_size):
            rows.extend(
                await conn.fetch(
                    f"""
                    SELECT e.id, e.canonical_name, e.metadata, e.last_seen, e.mention_count,
                           q.query_text
                    FROM unnest($2::text[]) AS q(query_text)
                    JOIN {fq_table("entities")} e ON (
                        e.bank_id = $1
                        AND LOWER(e.canonical_name) = LOWER(q.query_text)
                    )
                    """,
                    bank_id,
                    entity_text_batch,
                )
            )

        # Fetch candidates for the remaining texts in bounded batches.
        # Uses the GIN trigram index on LOWER(canonical_name) for case-insensitive
        # similarity lookup. Previous version also had LIKE '%...' substring fallbacks,
        # but those forced full sequential scans of the entities table and caused
        # TimeoutErrors on banks with 10k+ entities. The pg_trgm similarity threshold
        # that governs the `%` operator is applied once at pool-connection setup
        # (HINDSIGHT_API_ENTITY_TRGM_SIMILARITY_THRESHOLD), so it is not toggled here.
        # ``entity_kind != 'label'`` matches the predicate of the partial trigram
        # index (label rows are exact-match-only, so they can never be a
        # legitimate fuzzy result — without the filter they only inflate the
        # candidate set and get discarded in the bitmap recheck, #3208). The
        # clause must textually match the index predicate for the planner to
        # choose the partial index, so it stays inside the LATERAL's WHERE
        # alongside the `%` operator rather than moving out to the outer join.
        #
        # The LATERAL keeps only the best `max_candidates` per query text: on a bank
        # with many near-identical names a single probe can otherwise return
        # thousands of rows, and every one of them costs a SequenceMatcher call in
        # _resolve_from_candidates (GH-3211). Ranking by pg_trgm similarity — which
        # the index scan computes anyway — keeps the truncation at the noise end.
        for entity_text_batch in self._chunked(fuzzy_texts, self.entity_resolution_batch_size):
            rows.extend(
                await conn.fetch(
                    f"""
                    SELECT c.id, c.canonical_name, c.metadata, c.last_seen, c.mention_count,
                           q.query_text
                    FROM unnest($2::text[]) AS q(query_text)
                    CROSS JOIN LATERAL (
                        SELECT e.id, e.canonical_name, e.metadata, e.last_seen, e.mention_count
                        FROM {fq_table("entities")} e
                        WHERE e.bank_id = $1
                          AND e.entity_kind != 'label'
                          AND LOWER(e.canonical_name) % LOWER(q.query_text)
                        ORDER BY similarity(LOWER(e.canonical_name), LOWER(q.query_text)) DESC, e.id
                        LIMIT $3
                    ) c
                    """,
                    bank_id,
                    entity_text_batch,
                    self.entity_resolution_max_candidates,
                )
            )

        # Group candidates by query_text
        all_candidates: dict[str, list] = {t: [] for t in entity_texts}
        candidate_ids: set = set()
        for row in rows:
            query_text = row["query_text"]
            all_candidates[query_text].append(
                (row["id"], row["canonical_name"], row["metadata"], row["last_seen"], row["mention_count"])
            )
            candidate_ids.add(row["id"])

        # Fetch co-occurrences only for the candidate entities (not all bank entities)
        cooccurrence_map: dict[str, set[str]] = {}
        if candidate_ids:
            candidate_id_list = list(candidate_ids)
            cooc_rows = await conn.fetch(
                f"""
                SELECT ec.entity_id_1, ec.entity_id_2
                FROM {fq_table("entity_cooccurrences")} ec
                WHERE ec.entity_id_1 = ANY($1::uuid[])
                   OR ec.entity_id_2 = ANY($1::uuid[])
                """,
                candidate_id_list,
            )
            # Build name lookup for co-occurrence mapping
            id_to_name = {
                row["id"]: row["canonical_name"].lower()
                for cands in all_candidates.values()
                for row in [{"id": c[0], "canonical_name": c[1]} for c in cands]
            }
            for row in cooc_rows:
                eid1, eid2 = row["entity_id_1"], row["entity_id_2"]
                if eid1 not in cooccurrence_map:
                    cooccurrence_map[eid1] = set()
                if eid2 not in cooccurrence_map:
                    cooccurrence_map[eid2] = set()
                if eid2 in id_to_name:
                    cooccurrence_map[eid1].add(id_to_name[eid2])
                if eid1 in id_to_name:
                    cooccurrence_map[eid2].add(id_to_name[eid1])

        return await self._resolve_from_candidates(
            conn,
            bank_id,
            entities_data,
            unit_event_date,
            all_candidates,
            cooccurrence_map,
            taxonomy_lookup,
            labels_cfg,
        )

    async def _resolve_entities_batch_oracle_fuzzy(
        self,
        conn: Any,
        bank_id: str,
        entities_data: list[dict],
        unit_event_date: datetime | None,
        taxonomy_lookup: set[str] | None = None,
        labels_cfg=None,
    ) -> list[ResolvedEntity]:
        """
        Oracle strategy: fetch similar candidates using UTL_MATCH.JARO_WINKLER_SIMILARITY.

        Replaces pg_trgm for Oracle backends. Uses JSON_TABLE to expand the
        entity text list into rows (Oracle equivalent of PG's unnest), then
        joins with a Jaro-Winkler threshold of 70/100 (≈ pg_trgm 0.15).
        Falls back to the "full" strategy if UTL_MATCH is unavailable.
        """
        entity_texts = list(set(e["text"] for e in entities_data))
        entities_table = fq_table("entities")

        # Label entities resolve by exact match only, so the fuzzy Jaro-Winkler
        # join only returns similar-but-distinct label values that are always
        # discarded. Resolve label texts with an exact lookup on the unique
        # (bank_id, LOWER(canonical_name)) index and only fuzzy-match the rest.
        label_set = self._label_texts(entity_texts, taxonomy_lookup, labels_cfg)
        label_texts = [t for t in entity_texts if t in label_set]
        fuzzy_texts = [t for t in entity_texts if t not in label_set]

        try:
            # Batch entity texts into bounded sub-queries using JSON_TABLE to
            # expand the list into rows. UTL_MATCH.JARO_WINKLER_SIMILARITY
            # returns 0-100; threshold 70 ≈ pg_trgm similarity 0.15.
            # Bounded batches mirror the PG trigram path so very wide retain
            # batches don't time out a single JOIN on large banks.
            rows = []
            for entity_text_batch in self._chunked(label_texts, self.entity_resolution_batch_size):
                rows.extend(
                    await conn.fetch(
                        f"""
                        SELECT e.id, e.canonical_name, e.metadata, e.last_seen, e.mention_count,
                               q.query_text
                        FROM JSON_TABLE($2, '$[*]' COLUMNS (query_text VARCHAR2(4000) PATH '$')) q
                        JOIN {entities_table} e ON (
                            e.bank_id = $1
                            AND LOWER(e.canonical_name) = LOWER(q.query_text)
                        )
                        """,
                        bank_id,
                        json.dumps(entity_text_batch),
                    )
                )
            # Only the best `max_candidates` per query text are returned: each
            # candidate costs a synchronous SequenceMatcher call downstream, so an
            # unbounded fuzzy match set blocks the event loop for minutes
            # (GH-3211). Ranking by the same Jaro-Winkler score the join already
            # computes keeps the truncation at the noise end.
            for entity_text_batch in self._chunked(fuzzy_texts, self.entity_resolution_batch_size):
                rows.extend(
                    await conn.fetch(
                        f"""
                        SELECT id, canonical_name, metadata, last_seen, mention_count, query_text
                        FROM (
                            SELECT e.id, e.canonical_name, e.metadata, e.last_seen, e.mention_count,
                                   q.query_text,
                                   ROW_NUMBER() OVER (
                                       PARTITION BY q.query_text
                                       ORDER BY UTL_MATCH.JARO_WINKLER_SIMILARITY(
                                           LOWER(e.canonical_name), LOWER(q.query_text)
                                       ) DESC, e.id
                                   ) AS rn
                            FROM JSON_TABLE($2, '$[*]' COLUMNS (query_text VARCHAR2(4000) PATH '$')) q
                            JOIN {entities_table} e ON (
                                e.bank_id = $1
                                AND e.entity_kind != 'label'
                                AND UTL_MATCH.JARO_WINKLER_SIMILARITY(LOWER(e.canonical_name), LOWER(q.query_text)) > 70
                            )
                        )
                        WHERE rn <= $3
                        """,
                        bank_id,
                        json.dumps(entity_text_batch),
                        self.entity_resolution_max_candidates,
                    )
                )
        except Exception as e:
            # UTL_MATCH may not be available (ORA-06550, ORA-00904, etc.)
            # Catch broadly because Oracle error types vary depending on driver.
            # Fall back to the "full" strategy which works on any backend.
            logger.warning(
                "UTL_MATCH.JARO_WINKLER_SIMILARITY not available on Oracle — "
                "falling back to 'full' entity lookup strategy. Error: %s",
                e,
            )
            self.entity_lookup = "full"
            return await self._resolve_entities_batch_full(conn, bank_id, entities_data, unit_event_date)

        # Group candidates by query_text (same structure as trigram strategy)
        all_candidates: dict[str, list] = {t: [] for t in entity_texts}
        candidate_ids: set = set()
        for row in rows:
            query_text = row["query_text"]
            all_candidates[query_text].append(
                (row["id"], row["canonical_name"], row["metadata"], row["last_seen"], row["mention_count"])
            )
            candidate_ids.add(row["id"])

        # Fetch co-occurrences only for the candidate entities (not all bank entities)
        cooccurrence_map: dict[str, set[str]] = {}
        if candidate_ids:
            candidate_id_list = list(candidate_ids)
            cooc_rows = await conn.fetch(
                f"""
                SELECT ec.entity_id_1, ec.entity_id_2
                FROM {fq_table("entity_cooccurrences")} ec
                WHERE ec.entity_id_1 = ANY($1::uuid[])
                   OR ec.entity_id_2 = ANY($1::uuid[])
                """,
                candidate_id_list,
            )
            # Build name lookup for co-occurrence mapping
            id_to_name = {
                row["id"]: row["canonical_name"].lower()
                for cands in all_candidates.values()
                for row in [{"id": c[0], "canonical_name": c[1]} for c in cands]
            }
            for row in cooc_rows:
                eid1, eid2 = row["entity_id_1"], row["entity_id_2"]
                if eid1 not in cooccurrence_map:
                    cooccurrence_map[eid1] = set()
                if eid2 not in cooccurrence_map:
                    cooccurrence_map[eid2] = set()
                if eid2 in id_to_name:
                    cooccurrence_map[eid1].add(id_to_name[eid2])
                if eid1 in id_to_name:
                    cooccurrence_map[eid2].add(id_to_name[eid1])

        return await self._resolve_from_candidates(
            conn,
            bank_id,
            entities_data,
            unit_event_date,
            all_candidates,
            cooccurrence_map,
            taxonomy_lookup,
            labels_cfg,
        )

    def _intrabatch_canonical_map(self, entities_to_create: list[_EntityToCreate]) -> dict[str, str]:
        """Map each non-label new name (lowercased) to its cluster's canonical spelling.

        Uses in-memory trigram similarity (``_trigram_similarity``, verified equal to Postgres
        pg_trgm), so it is backend-agnostic — no DB round-trip on the retain hot path, and it runs
        identically on PostgreSQL, Oracle, and the pg_trgm-absent "full" fallback. Label entities
        are excluded so distinct label values stay separate (GH-1558).
        """
        rep_by_lower: dict[str, str] = {}
        count_by_lower: dict[str, int] = {}
        for e in entities_to_create:
            if e.is_label or not e.resolve:
                continue
            name_lower = e.name.lower()
            rep_by_lower.setdefault(name_lower, e.name)
            count_by_lower[name_lower] = count_by_lower.get(name_lower, 0) + 1

        if len(rep_by_lower) < 2:
            return {}  # nothing to compare
        if len(rep_by_lower) > _INTRABATCH_MAX_NAMES:
            logger.warning(
                "Skipping in-batch entity dedup: %d unique new names exceeds the %d cap "
                "(O(N^2) trigram comparison); same-batch surface variants may not be merged.",
                len(rep_by_lower),
                _INTRABATCH_MAX_NAMES,
            )
            return {}
        pairs = _find_intrabatch_similar_pairs(list(rep_by_lower.values()), self._intrabatch_merge_similarity)
        if not pairs:
            return {}
        return _cluster_new_entity_names(rep_by_lower, count_by_lower, pairs)

    async def _resolve_from_candidates(
        self,
        conn,
        bank_id: str,
        entities_data: list[dict],
        unit_event_date,
        all_candidates: dict[str, list],
        cooccurrence_map: dict[str, set[str]],
        taxonomy_lookup: set[str] | None = None,
        labels_cfg=None,
    ) -> list[ResolvedEntity]:
        """Shared scoring + upsert logic used by every lookup strategy.

        A mention carrying ``"resolve": False`` skips the scoring entirely and takes the
        find-or-create path below, which matches an existing row on ``LOWER(canonical_name)``
        equality and inserts one otherwise.
        """

        # Resolve each entity using pre-fetched candidates. A slot stays None
        # only if find-or-create fails to produce a row for a mention (a DB
        # inconsistency); it surfaces as a clear error at the reassert boundary
        # rather than a silent NOT NULL violation deeper in Phase 2.
        resolved: list[ResolvedEntity | None] = [None] * len(entities_data)
        entities_to_update: list[_EntityStat] = []
        entities_to_create: list[_EntityToCreate] = []
        # Candidates scored since the last yield, counted across mentions so a
        # batch of many small candidate sets yields as often as one large set.
        scored_since_yield = 0

        for idx, entity_data in enumerate(entities_data):
            entity_text = entity_data["text"]
            nearby_entities = entity_data.get("nearby_entities", [])
            # Use per-entity date if available, otherwise fall back to batch-level date
            entity_event_date = entity_data.get("event_date", unit_event_date)

            # Per mention, not per batch: retain resolves the caller's entities and the
            # extractor's in one pass, and only the caller's are meant literally (#3479).
            resolve = entity_data.get("resolve", True)
            candidates = all_candidates.get(entity_text, [])

            # Backstop truncation for candidate sets that were not capped at the
            # source (the "full" strategy matches substrings in Python). The fuzzy
            # strategies already return at most this many rows per query text, so
            # this is normally a no-op.
            if len(candidates) > self.entity_resolution_max_candidates:
                logger.debug(
                    "Truncating %d candidates to %d for entity text %r",
                    len(candidates),
                    self.entity_resolution_max_candidates,
                    entity_text,
                )
                entity_text_lower_for_rank = entity_text.lower()
                candidates = heapq.nsmallest(
                    self.entity_resolution_max_candidates,
                    candidates,
                    key=lambda c: _cheap_rank_key(entity_text_lower_for_rank, c),
                )

            # Label entities (from entity_labels config) use exact matching only.
            # Their canonical names are user-defined (e.g., "use:use-001"),
            # so fuzzy resolution must NOT merge distinct label values that
            # happen to be textually similar (GH-1558). Don't gate on the
            # lookup set — it is empty for text/map-only configs, whose labels
            # classify by key prefix (see _label_texts).
            is_label = bool(labels_cfg and _is_label_entity(entity_text, labels_cfg, taxonomy_lookup or set()))

            if not resolve or not candidates:
                # Nothing to score against — or the caller named the entity literally, so
                # similarity must not get a vote. Either way the find-or-create pass below
                # reuses an identically-named row and otherwise inserts this exact name.
                entities_to_create.append(
                    _EntityToCreate(
                        idx=idx, name=entity_text, event_date=entity_event_date, is_label=is_label, resolve=resolve
                    )
                )
                continue

            if is_label:
                # Exact case-insensitive match only for label entities
                exact_match: ResolvedEntity | None = None
                entity_text_lower = entity_text.lower()
                for candidate_id, canonical_name, metadata, last_seen, mention_count in candidates:
                    if canonical_name.lower() == entity_text_lower:
                        exact_match = ResolvedEntity(
                            entity_id=candidate_id, canonical_name=canonical_name, entity_kind="label"
                        )
                        break
                if exact_match:
                    resolved[idx] = exact_match
                    entities_to_update.append(
                        _EntityStat(entity_id=exact_match.entity_id, event_date=entity_event_date)
                    )
                else:
                    entities_to_create.append(
                        _EntityToCreate(idx=idx, name=entity_text, event_date=entity_event_date, is_label=True)
                    )
                continue

            # Score candidates
            best_candidate: ResolvedEntity | None = None
            best_score = 0.0

            nearby_entity_set = {e["text"].lower() for e in nearby_entities if e["text"] != entity_text}

            for candidate_id, canonical_name, metadata, last_seen, mention_count in candidates:
                # Hand the loop back periodically so /health (and every other task
                # on this worker) still gets scheduled while a wide batch scores.
                # Counted before the label skip below, so a candidate list that is
                # entirely labels still yields — the skip runs _is_label_entity per
                # row, which is cheap but not free.
                scored_since_yield += 1
                if scored_since_yield >= _SCORING_YIELD_EVERY:
                    scored_since_yield = 0
                    await asyncio.sleep(0)

                # A label row can never be a fuzzy-match target (#1558): the
                # trigram/UTL_MATCH probes exclude them in SQL via entity_kind,
                # but the "full" fallback strategy loads every bank entity, so
                # a textually-close label value could still outscore the 0.6
                # threshold here (e.g. "topic empathy" vs "topic:empathy").
                if labels_cfg and _is_label_entity(canonical_name, labels_cfg, taxonomy_lookup or set()):
                    continue

                score = 0.0

                # 1. Name similarity (0-0.5)
                name_similarity = SequenceMatcher(None, entity_text.lower(), canonical_name.lower()).ratio()
                score += name_similarity * 0.5

                # 2. Co-occurring entities (0-0.3)
                if nearby_entity_set:
                    co_entities = cooccurrence_map.get(candidate_id, set())
                    overlap = len(nearby_entity_set & co_entities)
                    co_entity_score = overlap / len(nearby_entity_set)
                    score += co_entity_score * 0.3

                # 3. Temporal proximity (0-0.2)
                if last_seen and entity_event_date:
                    # Normalize timezone awareness for comparison
                    event_date_utc = (
                        entity_event_date if entity_event_date.tzinfo else entity_event_date.replace(tzinfo=UTC)
                    )
                    last_seen_utc = last_seen if last_seen.tzinfo else last_seen.replace(tzinfo=UTC)
                    days_diff = abs((event_date_utc - last_seen_utc).total_seconds() / 86400)
                    if days_diff < 7:
                        temporal_score = max(0, 1.0 - (days_diff / 7))
                        score += temporal_score * 0.2

                if score > best_score:
                    best_score = score
                    best_candidate = ResolvedEntity(entity_id=candidate_id, canonical_name=canonical_name)

            # Apply unified threshold
            threshold = 0.6

            if best_score > threshold and best_candidate is not None:
                resolved[idx] = best_candidate
                entities_to_update.append(_EntityStat(entity_id=best_candidate.entity_id, event_date=entity_event_date))
            else:
                entities_to_create.append(
                    _EntityToCreate(idx=idx, name=entity_data["text"], event_date=entity_event_date, is_label=is_label)
                )

        # Existing entities: IDs already known from the candidate SELECT above.
        # No in-transaction UPDATE — mention_count/last_seen are stats deferred to
        # flush_pending_stats() which the orchestrator calls after the transaction.
        pending: list[_EntityStat] = list(entities_to_update)

        # New entities: INSERT with DO NOTHING to avoid row locks on concurrent races.
        # ON CONFLICT DO NOTHING returns nothing for rows that conflicted; we handle
        # that rare case with a fallback SELECT.
        if entities_to_create:
            # Fuzzy-cluster the NON-label names about to be created so same-batch surface
            # variants (case/emoji/suffix/typo of one name) collapse to a single entity. Without
            # this, resolution only compares against already-persisted rows, so the first sighting
            # of each variant in a batch always creates a distinct entity (issue #3107). Labels are
            # excluded and keep exact grouping, and so are names the caller wrote literally:
            # "Alice" and "Alice Smith" listed side by side are two entities because they were
            # written as two (#3479).
            canonical_by_member = self._intrabatch_canonical_map(entities_to_create)

            @dataclass
            class _NameGroup:
                name: str
                event_date: datetime | None
                is_label: bool
                indices: list[int] = field(default_factory=list)

            groups: dict[str, _NameGroup] = {}
            for e in entities_to_create:
                # Non-label variants fold into their cluster's canonical name; everything else
                # (labels, singletons) keys on itself, preserving the prior exact-match behavior.
                canonical = canonical_by_member.get(e.name.lower(), e.name)
                key = canonical.lower()
                group = groups.get(key)
                if group is None:
                    # Labels key on themselves and the dedup pass only clusters
                    # non-label names, so the first member's is_label holds for
                    # every member of the group.
                    group = _NameGroup(name=canonical, event_date=e.event_date, is_label=e.is_label)
                    groups[key] = group
                elif e.event_date is not None and (group.event_date is None or e.event_date < group.event_date):
                    # Keep the earliest event_date across the cluster ("first seen").
                    group.event_date = e.event_date
                group.indices.append(e.idx)

            # Sort by lowercase name for deterministic ordering.
            sorted_groups = sorted(groups.items())
            entity_names = [g.name for _, g in sorted_groups]
            entity_dates = [g.event_date for _, g in sorted_groups]
            entity_kinds = ["label" if g.is_label else "regular" for _, g in sorted_groups]
            # Stored canonical name per lowercase key, so a resurrected parent
            # keeps the name it was created/matched with rather than a fallback.
            canonical_by_name = {name_lower: g.name for name_lower, g in sorted_groups}

            # INSERT ... ON CONFLICT DO NOTHING — no row lock on already-existing entities.
            # mention_count starts at 0 here; flush_pending_stats() is the sole source of
            # truth for mention counting (one stat per original mention in the batch).
            entities_table = fq_table("entities")

            id_by_name = await self._ops.bulk_insert_entities(
                conn,
                entities_table,
                bank_id,
                entity_names,
                entity_dates,
                entity_kinds,
            )

            # Fallback SELECT for names that conflicted (another worker won the race).
            #
            # IMPORTANT: we must let the database do the lowercasing on BOTH sides of the
            # comparison.  Python's str.lower() and PostgreSQL's LOWER() differ for some
            # Unicode characters — most notably Turkish İ (U+0130):
            #   Python:     'İstanbul'.lower()  == 'i\u0307stanbul'  (i + combining dot, 2 chars)
            #   PostgreSQL: LOWER('İstanbul')   == 'istanbul'         (plain i, 1 char)
            # Passing a Python-lowercased name to "LOWER(canonical_name) = ANY($2::text[])"
            # would fail to match the stored entity, leaving entity_id as None and causing
            # a NOT NULL constraint violation on unit_entities.entity_id.
            missing_original = [g.name for name_lower, g in sorted_groups if name_lower not in id_by_name]
            if missing_original:
                existing_rows = await self._ops.fetch_missing_entity_ids(
                    conn,
                    entities_table,
                    bank_id,
                    missing_original,
                )
                for row in existing_rows:
                    id_by_name[row["name_lower"]] = row["id"]
                    canonical_by_name[row["name_lower"]] = row["canonical_name"]
                    # Also index by Python's lower() of the original input name so the
                    # assignment loop (which uses Python-lowercased keys) finds it even
                    # when Python and the database produce different lowercase strings.
                    if "input_name" in row:
                        input_name_lower = row["input_name"].lower()
                        id_by_name[input_name_lower] = row["id"]
                        canonical_by_name[input_name_lower] = row["canonical_name"]

            # Assign entity IDs back and queue one stat per original mention so that
            # flush_pending_stats() increments mention_count by the true mention count,
            # not just 1 per unique name.
            for name_lower, g in sorted_groups:
                entity_id = id_by_name.get(name_lower)
                if entity_id:
                    canonical_name = canonical_by_name.get(name_lower, g.name)
                    kind = "label" if g.is_label else "regular"
                    for original_idx in g.indices:
                        resolved[original_idx] = ResolvedEntity(
                            entity_id=entity_id, canonical_name=canonical_name, entity_kind=kind
                        )
                        pending.append(_EntityStat(entity_id=str(entity_id), event_date=g.event_date))

        # Accumulate into the resolver's pending list; the orchestrator flushes
        # these with await entity_resolver.flush_pending_stats() after the txn.
        key = self._task_key()
        self._pending_stats.setdefault(key, []).extend(pending)

        missing = [i for i, entity in enumerate(resolved) if entity is None]
        if missing:
            raise RuntimeError(
                f"Entity resolution produced no row for {len(missing)} mention(s) "
                f"(indices {missing[:5]}); refusing to link units to a missing parent."
            )
        return cast(list[ResolvedEntity], resolved)

    async def reassert_entities_batch(
        self,
        bank_id: str,
        resolved_entities: list[ResolvedEntity],
        conn,
    ) -> None:
        """Lock (and, if pruned, re-create) resolved parents before linking units.

        Phase-1 resolution and the Phase-2 ``unit_entities`` insert run on
        different transactions. In the gap, ``prune_orphan_entities`` can delete
        a just-resolved parent — it legitimately has no ``unit_entities`` row
        yet — and the Phase-2 FK insert then fails, dropping the whole batch as
        non-retryable (silent memory loss, #2662).

        Called on the Phase-2 connection immediately before
        ``link_units_to_entities_batch``, this locks the parents that still
        exist (so the pruner blocks until we commit) and re-inserts any that
        already vanished, in one round-trip. An entity referenced by a live unit
        is by definition not an orphan, so resurrecting it is correct.
        """
        # Deduplicate by id and lock in a stable order so concurrent reasserts
        # acquire row locks consistently (same convention as bulk_insert_links).
        seen: set[str] = set()
        unique: list[ResolvedEntity] = []
        for entity in sorted(resolved_entities, key=lambda e: e.entity_id):
            if entity.entity_id in seen:
                continue
            seen.add(entity.entity_id)
            unique.append(entity)

        if not unique:
            return

        await self._ops.bulk_reassert_entities(
            conn,
            fq_table("entities"),
            bank_id,
            [entity.entity_id for entity in unique],
            [entity.canonical_name for entity in unique],
            [entity.entity_kind for entity in unique],
        )

    async def link_units_to_entities_batch(
        self,
        unit_entity_pairs: list[tuple[str, str]] | list[tuple[str, str, datetime | None]],
        conn=None,
        bank_id: str | None = None,
    ):
        """
        Link multiple memory units to entities in batch (MUCH faster than sequential).

        Also updates co-occurrence cache for entities that appear in the same unit.

        Args:
            unit_entity_pairs: List of (unit_id, entity_id) or
                (unit_id, entity_id, event_date) tuples. When `event_date` is
                supplied, ``entity_cooccurrences.last_cooccurred`` for pairs
                observed in that unit advances to the event time instead of
                ``now()``, which matters for backfilled corpora where ingest
                time is a single spike unrelated to the underlying timeline.
                Legacy two-tuples remain accepted.
            conn: Optional connection to use (if None, acquires from pool)
        """
        if not unit_entity_pairs:
            return

        # Normalize to 3-tuples internally so downstream code doesn't branch.
        normalized: list[tuple[str, str, datetime | None]] = [
            (t[0], t[1], t[2] if len(t) >= 3 else None)  # type: ignore[misc]
            for t in unit_entity_pairs
        ]

        if conn is None:
            async with acquire_with_retry(self.pool) as conn:
                return await self._link_units_to_entities_batch_impl(conn, normalized, bank_id)
        else:
            return await self._link_units_to_entities_batch_impl(conn, normalized, bank_id)

    async def record_unit_entity_postings(
        self,
        unit_entity_pairs: list[tuple[str, str]] | list[tuple[str, str, datetime | None]],
        bank_id: str | None = None,
        txn=None,
    ):
        """Store-owned variant of :meth:`link_units_to_entities_batch` that touches NO
        Postgres connection.

        For a memories store that OWNS its memory rows (an external backend), the unit→entity
        posting is recorded by the store — ``record_unit_entities`` ignores the ``conn`` — and
        the co-occurrence update only accumulates in memory for the post-transaction flush.
        Neither needs a database transaction, so the retain orchestrator can run the posting in
        its connection-free store phase and never hold the data-plane connection across the
        object-store write. NOT for the Postgres store, whose posting is a real ``unit_entities``
        INSERT that requires the connection.

        ``txn`` is the caller's write-group handle. For a store that keeps the posting on the
        memory, this re-writes rows the same write-group just created, so it belongs to that
        group — see :meth:`MemoriesExtension.record_unit_entities`.
        """
        if not unit_entity_pairs:
            return

        normalized: list[tuple[str, str, datetime | None]] = [
            (t[0], t[1], t[2] if len(t) >= 3 else None)  # type: ignore[misc]
            for t in unit_entity_pairs
        ]
        return await self._link_units_to_entities_batch_impl(None, normalized, bank_id, txn=txn)

    async def _link_units_to_entities_batch_impl(
        self, conn, unit_entity_pairs: list[tuple[str, str, datetime | None]], bank_id: str | None = None, txn=None
    ):
        # Sorted bulk insert to prevent deadlocks from inconsistent lock ordering
        # across concurrent transactions on the unit_entities unique index.
        sorted_pairs = sorted(unit_entity_pairs, key=lambda t: (t[0], t[1]))
        unit_ids = [p[0] for p in sorted_pairs]
        entity_ids = [p[1] for p in sorted_pairs]

        # The unit→entity posting belongs to whoever stores the memory, so the
        # memories store records it. Co-occurrence below is separate and unaffected:
        # it references only `entities`, which stays in Postgres either way, and is
        # read by the entity-graph endpoint and by resolution's disambiguation signal.
        from .memories import get_memories

        await get_memories().record_unit_entities(
            conn=conn,
            ops=self._ops,
            fq_table=fq_table,
            bank_id=bank_id,
            unit_ids=unit_ids,
            entity_ids=entity_ids,
            txn=txn,
        )

        # Build maps keyed by unit_id:
        #   unit_to_entities: entity set per unit (for the co-occurrence cross-product)
        #   unit_event_date:  event time per unit (propagated onto every pair from that unit)
        # When a unit shows up more than once with conflicting event_dates (legacy
        # callers passing None interleaved with aware callers), prefer the first
        # non-None value so we don't accidentally erase an explicit timestamp.
        unit_to_entities: dict[str, set[str]] = {}
        unit_event_date: dict[str, datetime | None] = {}
        for unit_id, entity_id, event_date in unit_entity_pairs:
            unit_to_entities.setdefault(unit_id, set()).add(entity_id)
            if event_date is not None and unit_event_date.get(unit_id) is None:
                unit_event_date[unit_id] = event_date
            elif unit_id not in unit_event_date:
                unit_event_date[unit_id] = event_date

        # Update co-occurrences for all pairs in each unit. Carry the unit's
        # event_date onto every pair so the flush step can stamp
        # `last_cooccurred` with the correct time.
        cooccurrence_pairs: dict[tuple[str, str], datetime | None] = {}
        for unit_id, entity_ids in unit_to_entities.items():
            entity_list = list(entity_ids)
            event_date = unit_event_date.get(unit_id)
            for key in _canonical_cooccurrence_pairs(entity_list):
                prev = cooccurrence_pairs.get(key, _SENTINEL_MISSING)
                if prev is _SENTINEL_MISSING:
                    cooccurrence_pairs[key] = event_date
                else:
                    cooccurrence_pairs[key] = _later_date(prev, event_date)

        # Accumulate co-occurrence pairs for post-transaction flush.
        # The actual INSERT/UPDATE is deferred to flush_pending_stats() to avoid
        # row-level lock contention (ON CONFLICT DO UPDATE inside a long transaction
        # serialises concurrent writers on popular entity pairs).
        if cooccurrence_pairs:
            key = self._task_key()
            self._pending_cooccurrences.setdefault(key, []).extend(
                _CooccurrencePair(entity_id_1=e1, entity_id_2=e2, event_date=ed)
                for (e1, e2), ed in cooccurrence_pairs.items()
            )
