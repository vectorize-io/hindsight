"""
Entity extraction and resolution for memory system.

Uses spaCy for entity extraction and implements resolution logic
to disambiguate entities across memory units.
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import SequenceMatcher

import asyncpg

from .db_utils import acquire_with_retry
from .memory_engine import fq_table
from .retain.entity_labels import build_labels_lookup as _build_labels_lookup_from_config

logger = logging.getLogger(__name__)


@dataclass
class _EntityToCreate:
    """An entity that needs to be inserted (no matching candidate found)."""

    idx: int
    name: str
    event_date: datetime | None


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


@dataclass
class _CooccurrencePair:
    """A (entity_id_1, entity_id_2) pair observed in a retain batch (for post-txn flush)."""

    entity_id_1: str
    entity_id_2: str


# Load spaCy model (singleton)
_nlp = None


class EntityResolver:
    """
    Resolves entities to canonical IDs with disambiguation.
    """

    def __init__(self, pool: asyncpg.Pool, entity_lookup: str = "full"):
        """
        Initialize entity resolver.

        Args:
            pool: asyncpg connection pool
            entity_lookup: Lookup strategy — "full" loads all bank entities then
                matches in Python; "trigram" uses pg_trgm GIN index to fetch only
                similar candidates per entity name (much faster for large banks).
        """
        self.pool = pool
        self.entity_lookup = entity_lookup
        self._pg_trgm_checked = False
        # Keyed by asyncio task id so concurrent retain batches never mix their
        # pending updates.  flush_pending_stats() pops only the calling task's items.
        self._pending_stats: dict[int, list[_EntityStat]] = {}
        self._pending_cooccurrences: dict[int, list[_CooccurrencePair]] = {}

    def _task_key(self) -> int:
        """Return a unique key for the current asyncio task (or 0 for non-task context)."""
        task = asyncio.current_task()
        return id(task) if task is not None else 0

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
                # Aggregate: count occurrences per (entity_id_1, entity_id_2) pair.
                coo_agg: dict[tuple[str, str], int] = {}
                for c in cooccurrences:
                    pair = (c.entity_id_1, c.entity_id_2)
                    coo_agg[pair] = coo_agg.get(pair, 0) + 1

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
                    sorted((e1, e2, count, now) for (e1, e2), count in coo_agg.items()),
                )

    @staticmethod
    def _build_labels_lookup(entity_labels: list | None) -> set[str]:
        """Build a set of valid 'key:value' entity label strings for fast lookup."""
        return _build_labels_lookup_from_config(entity_labels)

    async def resolve_entities_batch(
        self,
        bank_id: str,
        entities_data: list[dict],
        context: str,
        unit_event_date,
        conn=None,
        entity_labels: list | None = None,
    ) -> list[str]:
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

        Returns:
            List of entity IDs in same order as input
        """
        if not entities_data:
            return []

        taxonomy_lookup = self._build_labels_lookup(entity_labels)
        if conn is None:
            async with acquire_with_retry(self.pool) as conn:
                return await self._resolve_entities_batch_impl(
                    conn, bank_id, entities_data, context, unit_event_date, taxonomy_lookup
                )
        else:
            return await self._resolve_entities_batch_impl(
                conn, bank_id, entities_data, context, unit_event_date, taxonomy_lookup
            )

    async def _resolve_entities_batch_impl(
        self,
        conn,
        bank_id: str,
        entities_data: list[dict],
        context: str,
        unit_event_date,
        taxonomy_lookup: set[str] | None = None,
    ) -> list[str]:
        if self.entity_lookup == "trigram":
            # Auto-detect pg_trgm availability on first call and fall back to
            # "full" strategy if the extension is not installed.  See #626.
            if not self._pg_trgm_checked:
                self._pg_trgm_checked = True
                has_trgm = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm')"
                )
                if not has_trgm:
                    logger.warning(
                        "pg_trgm extension is not available — falling back to 'full' "
                        "entity lookup strategy. Install pg_trgm for faster entity "
                        "resolution on large banks. See: "
                        "https://github.com/vectorize-io/hindsight/issues/626"
                    )
                    self.entity_lookup = "full"
                    return await self._resolve_entities_batch_full(
                        conn, bank_id, entities_data, unit_event_date
                    )
            return await self._resolve_entities_batch_trigram(conn, bank_id, entities_data, unit_event_date)
        return await self._resolve_entities_batch_full(conn, bank_id, entities_data, unit_event_date)

    async def _resolve_entities_batch_full(
        self, conn, bank_id: str, entities_data: list[dict], unit_event_date
    ) -> list[str]:
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
            conn, bank_id, entities_data, unit_event_date, all_candidates, cooccurrence_map
        )

    async def _resolve_entities_batch_trigram(
        self, conn, bank_id: str, entities_data: list[dict], unit_event_date
    ) -> list[str]:
        """
        Trigram strategy: fetch only similar candidates per entity name using pg_trgm.

        Instead of loading all bank entities (O(N)), uses a GIN trigram index to fetch
        only the small set of candidates that are textually similar to each input name.
        Reduces DB data transfer from 165K rows to ~5-20 rows per entity.
        """
        entity_texts = list(set(e["text"] for e in entities_data))

        # Fetch candidates for all unique entity texts in a single batched query.
        # The trigram % operator uses the GIN index; the substring conditions cover
        # exact prefix/suffix matches that trigrams might miss at low similarity.
        rows = await conn.fetch(
            f"""
            SELECT DISTINCT ON (e.id)
                e.id, e.canonical_name, e.metadata, e.last_seen, e.mention_count,
                q.query_text
            FROM unnest($2::text[]) AS q(query_text)
            JOIN {fq_table("entities")} e ON (
                e.bank_id = $1
                AND (
                    e.canonical_name % q.query_text
                    OR LOWER(e.canonical_name) LIKE '%' || LOWER(q.query_text) || '%'
                    OR LOWER(q.query_text) LIKE '%' || LOWER(e.canonical_name) || '%'
                )
            )
            """,
            bank_id,
            entity_texts,
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
            conn, bank_id, entities_data, unit_event_date, all_candidates, cooccurrence_map
        )

    async def _resolve_from_candidates(
        self,
        conn,
        bank_id: str,
        entities_data: list[dict],
        unit_event_date,
        all_candidates: dict[str, list],
        cooccurrence_map: dict[str, set[str]],
    ) -> list[str]:
        """Shared scoring + upsert logic used by both lookup strategies."""

        # Resolve each entity using pre-fetched candidates
        entity_ids = [None] * len(entities_data)
        entities_to_update: list[_EntityStat] = []
        entities_to_create: list[_EntityToCreate] = []

        for idx, entity_data in enumerate(entities_data):
            entity_text = entity_data["text"]
            nearby_entities = entity_data.get("nearby_entities", [])
            # Use per-entity date if available, otherwise fall back to batch-level date
            entity_event_date = entity_data.get("event_date", unit_event_date)

            candidates = all_candidates.get(entity_text, [])

            if not candidates:
                # Will create new entity
                entities_to_create.append(_EntityToCreate(idx=idx, name=entity_text, event_date=entity_event_date))
                continue

            # Score candidates
            best_candidate = None
            best_score = 0.0

            nearby_entity_set = {e["text"].lower() for e in nearby_entities if e["text"] != entity_text}

            for candidate_id, canonical_name, metadata, last_seen, mention_count in candidates:
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
                    best_candidate = candidate_id

            # Apply unified threshold
            threshold = 0.6

            if best_score > threshold:
                entity_ids[idx] = best_candidate
                entities_to_update.append(_EntityStat(entity_id=best_candidate, event_date=entity_event_date))
            else:
                entities_to_create.append(
                    _EntityToCreate(idx=idx, name=entity_data["text"], event_date=entity_event_date)
                )

        # Existing entities: IDs already known from the candidate SELECT above.
        # No in-transaction UPDATE — mention_count/last_seen are stats deferred to
        # flush_pending_stats() which the orchestrator calls after the transaction.
        pending: list[_EntityStat] = list(entities_to_update)

        # New entities: INSERT with DO NOTHING to avoid row locks on concurrent races.
        # ON CONFLICT DO NOTHING returns nothing for rows that conflicted; we handle
        # that rare case with a fallback SELECT.
        if entities_to_create:
            # Group by lowercase name — deduplicate within the batch.
            @dataclass
            class _NameGroup:
                name: str
                event_date: datetime | None
                indices: list[int] = field(default_factory=list)

            groups: dict[str, _NameGroup] = {}
            for e in entities_to_create:
                name_lower = e.name.lower()
                if name_lower not in groups:
                    groups[name_lower] = _NameGroup(name=e.name, event_date=e.event_date)
                groups[name_lower].indices.append(e.idx)

            # Sort by lowercase name for deterministic ordering.
            sorted_groups = sorted(groups.items())
            entity_names = [g.name for _, g in sorted_groups]
            entity_dates = [g.event_date for _, g in sorted_groups]

            # INSERT ... ON CONFLICT DO NOTHING — no row lock on already-existing entities.
            # mention_count starts at 0 here; flush_pending_stats() is the sole source of
            # truth for mention counting (one stat per original mention in the batch).
            inserted_rows = await conn.fetch(
                f"""
                INSERT INTO {fq_table("entities")} (bank_id, canonical_name, first_seen, last_seen, mention_count)
                SELECT $1, name, COALESCE(event_date, now()), COALESCE(event_date, now()), 0
                FROM unnest($2::text[], $3::timestamptz[]) AS t(name, event_date)
                ON CONFLICT (bank_id, LOWER(canonical_name))
                DO NOTHING
                RETURNING id, LOWER(canonical_name) AS name_lower
                """,
                bank_id,
                entity_names,
                entity_dates,
            )
            id_by_name: dict[str, str] = {row["name_lower"]: row["id"] for row in inserted_rows}

            # Fallback SELECT for names that conflicted (another worker won the race).
            #
            # IMPORTANT: we must let PostgreSQL do the lowercasing on BOTH sides of the
            # comparison.  Python's str.lower() and PostgreSQL's LOWER() differ for some
            # Unicode characters — most notably Turkish İ (U+0130):
            #   Python:     'İstanbul'.lower()  == 'i\u0307stanbul'  (i + combining dot, 2 chars)
            #   PostgreSQL: LOWER('İstanbul')   == 'istanbul'         (plain i, 1 char)
            # Passing a Python-lowercased name to "LOWER(canonical_name) = ANY($2::text[])"
            # would fail to match the stored entity, leaving entity_id as None and causing
            # a NOT NULL constraint violation on unit_entities.entity_id.
            #
            # Fix: pass the original (mixed-case) input names and use
            # "LOWER(canonical_name) = ANY(SELECT LOWER(n) FROM unnest($2) AS n)" so
            # PostgreSQL lowercases both sides identically.  The query also returns the
            # original input_name so we can index id_by_name by Python's lower() of that
            # name, which is what the assignment loop below uses as its lookup key.
            missing_original = [g.name for name_lower, g in sorted_groups if name_lower not in id_by_name]
            if missing_original:
                existing_rows = await conn.fetch(
                    f"""
                    SELECT e.id, LOWER(e.canonical_name) AS name_lower, inputs.input_name
                    FROM {fq_table("entities")} e
                    JOIN (
                        SELECT LOWER(n) AS input_name_lower, n AS input_name
                        FROM unnest($2::text[]) AS n
                    ) AS inputs ON LOWER(e.canonical_name) = inputs.input_name_lower
                    WHERE e.bank_id = $1
                    """,
                    bank_id,
                    missing_original,
                )
                for row in existing_rows:
                    id_by_name[row["name_lower"]] = row["id"]
                    # Also index by Python's lower() of the original input name so the
                    # assignment loop (which uses Python-lowercased keys) finds it even
                    # when Python and PostgreSQL produce different lowercase strings.
                    id_by_name[row["input_name"].lower()] = row["id"]

            # Assign entity IDs back and queue one stat per original mention so that
            # flush_pending_stats() increments mention_count by the true mention count,
            # not just 1 per unique name.
            for name_lower, g in sorted_groups:
                entity_id = id_by_name.get(name_lower)
                if entity_id:
                    for original_idx in g.indices:
                        entity_ids[original_idx] = entity_id
                        pending.append(_EntityStat(entity_id=entity_id, event_date=g.event_date))

        # Accumulate into the resolver's pending list; the orchestrator flushes
        # these with await entity_resolver.flush_pending_stats() after the txn.
        key = self._task_key()
        self._pending_stats.setdefault(key, []).extend(pending)

        return entity_ids

    async def resolve_entity(
        self,
        bank_id: str,
        entity_text: str,
        context: str,
        nearby_entities: list[dict],
        unit_event_date,
    ) -> str:
        """
        Resolve an entity to a canonical entity ID.

        Args:
            bank_id: bank ID (entities are scoped to agents)
            entity_text: Entity text ("Alice", "Google", etc.)
            context: Context where entity appears
            nearby_entities: Other entities in the same unit
            unit_event_date: When this unit was created

        Returns:
            Entity ID (creates new entity if needed)
        """
        async with acquire_with_retry(self.pool) as conn:
            # Find candidate entities with similar name
            candidates = await conn.fetch(
                f"""
                SELECT id, canonical_name, metadata, last_seen
                FROM {fq_table("entities")}
                WHERE bank_id = $1
                  AND (
                    canonical_name ILIKE $2
                    OR canonical_name ILIKE $3
                    OR $2 ILIKE canonical_name || '%%'
                  )
                ORDER BY mention_count DESC
                """,
                bank_id,
                entity_text,
                f"%{entity_text}%",
            )

            if not candidates:
                # New entity - create it
                return await self._create_entity(conn, bank_id, entity_text, unit_event_date)

            # Score candidates based on:
            # 1. Name similarity
            # 2. Context overlap (TODO: could use embeddings)
            # 3. Co-occurring entities
            # 4. Temporal proximity

            best_candidate = None
            best_score = 0.0
            best_name_similarity = 0.0

            nearby_entity_set = {e["text"].lower() for e in nearby_entities if e["text"] != entity_text}

            for row in candidates:
                candidate_id = row["id"]
                canonical_name = row["canonical_name"]
                metadata = row["metadata"]
                last_seen = row["last_seen"]
                score = 0.0

                # 1. Name similarity (0-1)
                name_similarity = SequenceMatcher(None, entity_text.lower(), canonical_name.lower()).ratio()
                score += name_similarity * 0.5

                # 2. Co-occurring entities (0-0.5)
                # Get entities that co-occurred with this candidate before
                # Use the materialized co-occurrence cache for fast lookup
                co_entity_rows = await conn.fetch(
                    f"""
                    SELECT e.canonical_name, ec.cooccurrence_count
                    FROM {fq_table("entity_cooccurrences")} ec
                    JOIN {fq_table("entities")} e ON (
                        CASE
                            WHEN ec.entity_id_1 = $1 THEN ec.entity_id_2
                            WHEN ec.entity_id_2 = $1 THEN ec.entity_id_1
                        END = e.id
                    )
                    WHERE ec.entity_id_1 = $1 OR ec.entity_id_2 = $1
                    """,
                    candidate_id,
                )
                co_entities = {r["canonical_name"].lower() for r in co_entity_rows}

                # Check overlap with nearby entities
                overlap = len(nearby_entity_set & co_entities)
                if nearby_entity_set:
                    co_entity_score = overlap / len(nearby_entity_set)
                    score += co_entity_score * 0.3

                # 3. Temporal proximity (0-0.2)
                if last_seen:
                    days_diff = abs((unit_event_date - last_seen).total_seconds() / 86400)
                    if days_diff < 7:  # Within a week
                        temporal_score = max(0, 1.0 - (days_diff / 7))
                        score += temporal_score * 0.2

                if score > best_score:
                    best_score = score
                    best_candidate = candidate_id
                    best_name_similarity = name_similarity

            # Threshold for considering it the same entity
            threshold = 0.6

            if best_score > threshold:
                # Update entity
                await conn.execute(
                    f"""
                    UPDATE {fq_table("entities")}
                    SET mention_count = mention_count + 1,
                        last_seen = $1
                    WHERE id = $2
                    """,
                    unit_event_date,
                    best_candidate,
                )
                return best_candidate
            else:
                # Not confident - create new entity
                return await self._create_entity(conn, bank_id, entity_text, unit_event_date)

    async def _create_entity(
        self,
        conn,
        bank_id: str,
        entity_text: str,
        event_date,
    ) -> str:
        """
        Create a new entity or get existing one if it already exists.

        Uses INSERT ... ON CONFLICT to handle race conditions where
        two concurrent transactions try to create the same entity.

        Args:
            conn: Database connection
            bank_id: bank ID
            entity_text: Entity text
            event_date: When first seen

        Returns:
            Entity ID
        """
        entity_id = await conn.fetchval(
            f"""
            INSERT INTO {fq_table("entities")} (bank_id, canonical_name, first_seen, last_seen, mention_count)
            VALUES ($1, $2, COALESCE($3, now()), COALESCE($4, now()), 1)
            ON CONFLICT (bank_id, LOWER(canonical_name))
            DO UPDATE SET
                mention_count = {fq_table("entities")}.mention_count + 1,
                last_seen = EXCLUDED.last_seen
            RETURNING id
            """,
            bank_id,
            entity_text,
            event_date,
            event_date,
        )
        return entity_id

    async def link_unit_to_entity(self, unit_id: str, entity_id: str):
        """
        Link a memory unit to an entity.
        Also updates co-occurrence cache with other entities in the same unit.

        Args:
            unit_id: Memory unit ID
            entity_id: Entity ID
        """
        async with acquire_with_retry(self.pool) as conn:
            # Insert unit-entity link
            await conn.execute(
                f"""
                INSERT INTO {fq_table("unit_entities")} (unit_id, entity_id)
                VALUES ($1, $2)
                ON CONFLICT DO NOTHING
                """,
                unit_id,
                entity_id,
            )

            # Update co-occurrence cache: find other entities in this unit
            rows = await conn.fetch(
                f"""
                SELECT entity_id
                FROM {fq_table("unit_entities")}
                WHERE unit_id = $1 AND entity_id != $2
                """,
                unit_id,
                entity_id,
            )

            other_entities = [row["entity_id"] for row in rows]

            # Update co-occurrences for each pair
            for other_entity_id in other_entities:
                await self._update_cooccurrence(conn, entity_id, other_entity_id)

    async def _update_cooccurrence(self, conn, entity_id_1: str, entity_id_2: str):
        """
        Update the co-occurrence cache for two entities.

        Uses CHECK constraint ordering (entity_id_1 < entity_id_2) to avoid duplicates.

        Args:
            conn: Database connection
            entity_id_1: First entity ID
            entity_id_2: Second entity ID
        """
        # Ensure consistent ordering (smaller UUID first)
        if entity_id_1 > entity_id_2:
            entity_id_1, entity_id_2 = entity_id_2, entity_id_1

        await conn.execute(
            f"""
            INSERT INTO {fq_table("entity_cooccurrences")} (entity_id_1, entity_id_2, cooccurrence_count, last_cooccurred)
            VALUES ($1, $2, 1, NOW())
            ON CONFLICT (entity_id_1, entity_id_2)
            DO UPDATE SET
                cooccurrence_count = {fq_table("entity_cooccurrences")}.cooccurrence_count + 1,
                last_cooccurred = NOW()
            """,
            entity_id_1,
            entity_id_2,
        )

    async def link_units_to_entities_batch(self, unit_entity_pairs: list[tuple[str, str]], conn=None):
        """
        Link multiple memory units to entities in batch (MUCH faster than sequential).

        Also updates co-occurrence cache for entities that appear in the same unit.

        Args:
            unit_entity_pairs: List of (unit_id, entity_id) tuples
            conn: Optional connection to use (if None, acquires from pool)
        """
        if not unit_entity_pairs:
            return

        if conn is None:
            async with acquire_with_retry(self.pool) as conn:
                return await self._link_units_to_entities_batch_impl(conn, unit_entity_pairs)
        else:
            return await self._link_units_to_entities_batch_impl(conn, unit_entity_pairs)

    async def _link_units_to_entities_batch_impl(self, conn, unit_entity_pairs: list[tuple[str, str]]):
        # Batch insert all unit-entity links
        await conn.executemany(
            f"""
            INSERT INTO {fq_table("unit_entities")} (unit_id, entity_id)
            VALUES ($1, $2)
            ON CONFLICT DO NOTHING
            """,
            unit_entity_pairs,
        )

        # Build map of unit -> entities for co-occurrence calculation
        # Use sets to avoid duplicate entities in the same unit
        unit_to_entities = {}
        for unit_id, entity_id in unit_entity_pairs:
            if unit_id not in unit_to_entities:
                unit_to_entities[unit_id] = set()
            unit_to_entities[unit_id].add(entity_id)

        # Update co-occurrences for all pairs in each unit
        cooccurrence_pairs = set()  # Use set to avoid duplicates
        for unit_id, entity_ids in unit_to_entities.items():
            entity_list = list(entity_ids)  # Convert set to list for iteration
            # For each pair of entities in this unit, create co-occurrence
            for i, entity_id_1 in enumerate(entity_list):
                for entity_id_2 in entity_list[i + 1 :]:
                    # Skip if same entity (shouldn't happen with set, but be safe)
                    if entity_id_1 == entity_id_2:
                        continue
                    # Ensure consistent ordering (entity_id_1 < entity_id_2)
                    if entity_id_1 > entity_id_2:
                        entity_id_1, entity_id_2 = entity_id_2, entity_id_1
                    cooccurrence_pairs.add((entity_id_1, entity_id_2))

        # Accumulate co-occurrence pairs for post-transaction flush.
        # The actual INSERT/UPDATE is deferred to flush_pending_stats() to avoid
        # row-level lock contention (ON CONFLICT DO UPDATE inside a long transaction
        # serialises concurrent writers on popular entity pairs).
        if cooccurrence_pairs:
            key = self._task_key()
            self._pending_cooccurrences.setdefault(key, []).extend(
                _CooccurrencePair(entity_id_1=e1, entity_id_2=e2) for e1, e2 in cooccurrence_pairs
            )

    async def get_units_by_entity(self, entity_id: str, limit: int = 100) -> list[str]:
        """
        Get all units that mention an entity.

        Args:
            entity_id: Entity ID
            limit: Max results

        Returns:
            List of unit IDs
        """
        async with acquire_with_retry(self.pool) as conn:
            rows = await conn.fetch(
                f"""
                SELECT unit_id
                FROM {fq_table("unit_entities")}
                WHERE entity_id = $1
                ORDER BY unit_id
                LIMIT $2
                """,
                entity_id,
                limit,
            )
            return [row["unit_id"] for row in rows]

    async def get_entity_by_text(
        self,
        bank_id: str,
        entity_text: str,
    ) -> str | None:
        """
        Find an entity by text (for query resolution).

        Args:
            bank_id: bank ID
            entity_text: Entity text to search for

        Returns:
            Entity ID if found, None otherwise
        """
        async with acquire_with_retry(self.pool) as conn:
            row = await conn.fetchrow(
                f"""
                SELECT id FROM {fq_table("entities")}
                WHERE bank_id = $1
                  AND canonical_name ILIKE $2
                ORDER BY mention_count DESC
                LIMIT 1
                """,
                bank_id,
                entity_text,
            )

            return row["id"] if row else None
