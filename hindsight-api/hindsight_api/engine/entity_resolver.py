"""
Entity extraction and resolution for memory system.

Uses spaCy for entity extraction and implements resolution logic
to disambiguate entities across memory units.
"""

from datetime import UTC, datetime
from difflib import SequenceMatcher

import asyncpg

from .db_utils import acquire_with_retry
from .memory_engine import fq_table

# Load spaCy model (singleton)
_nlp = None


class EntityResolver:
    """
    Resolves entities to canonical IDs with disambiguation.
    """

    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize entity resolver.

        Args:
            pool: asyncpg connection pool
        """
        self.pool = pool

    async def resolve_entities_batch(
        self,
        bank_id: str,
        entities_data: list[dict],
        context: str,
        unit_event_date,
        conn=None,
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

        if conn is None:
            async with acquire_with_retry(self.pool) as conn:
                return await self._resolve_entities_batch_impl(conn, bank_id, entities_data, context, unit_event_date)
        else:
            return await self._resolve_entities_batch_impl(conn, bank_id, entities_data, context, unit_event_date)

    async def _resolve_entities_batch_impl(
        self, conn, bank_id: str, entities_data: list[dict], context: str, unit_event_date
    ) -> list[str]:
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

        # Resolve each entity using pre-fetched candidates
        entity_ids = [None] * len(entities_data)
        entities_to_update = []  # (entity_id, event_date)
        entities_to_create = []  # (idx, entity_data, event_date)

        for idx, entity_data in enumerate(entities_data):
            entity_text = entity_data["text"]
            nearby_entities = entity_data.get("nearby_entities", [])
            # Use per-entity date if available, otherwise fall back to batch-level date
            entity_event_date = entity_data.get("event_date", unit_event_date)

            candidates = all_candidates.get(entity_text, [])

            if not candidates:
                # Will create new entity
                entities_to_create.append((idx, entity_data, entity_event_date))
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
                entities_to_update.append((best_candidate, entity_event_date))
            else:
                entities_to_create.append((idx, entity_data, entity_event_date))

        # Batch update existing entities
        if entities_to_update:
            await conn.executemany(
                f"""
                UPDATE {fq_table("entities")} SET
                    mention_count = mention_count + 1,
                    last_seen = $2
                WHERE id = $1::uuid
                """,
                entities_to_update,
            )

        # Batch create new entities using COPY + INSERT for maximum speed
        # This handles duplicates via ON CONFLICT and returns all IDs
        if entities_to_create:
            # Group entities by canonical name (lowercase) to handle duplicates within batch
            # For duplicates, we only insert once and reuse the ID
            unique_entities = {}  # lowercase_name -> (entity_data, event_date, [indices])
            for idx, entity_data, event_date in entities_to_create:
                name_lower = entity_data["text"].lower()
                if name_lower not in unique_entities:
                    unique_entities[name_lower] = (entity_data, event_date, [idx])
                else:
                    # Same entity appears multiple times - add index to list
                    unique_entities[name_lower][2].append(idx)

            # Batch insert unique entities and get their IDs
            # Use a single query with unnest for speed
            entity_names = []
            entity_dates = []
            indices_map = []  # Maps result index -> list of original indices

            for name_lower, (entity_data, event_date, indices) in unique_entities.items():
                entity_names.append(entity_data["text"])
                entity_dates.append(event_date)
                indices_map.append(indices)

            # Batch INSERT ... ON CONFLICT with RETURNING
            # This is much faster than individual inserts
            rows = await conn.fetch(
                f"""
                INSERT INTO {fq_table("entities")} (bank_id, canonical_name, first_seen, last_seen, mention_count)
                SELECT $1, name, event_date, event_date, 1
                FROM unnest($2::text[], $3::timestamptz[]) AS t(name, event_date)
                ON CONFLICT (bank_id, LOWER(canonical_name))
                DO UPDATE SET
                    mention_count = {fq_table("entities")}.mention_count + 1,
                    last_seen = EXCLUDED.last_seen
                RETURNING id
                """,
                bank_id,
                entity_names,
                entity_dates,
            )

            # Map returned IDs back to original indices
            for result_idx, row in enumerate(rows):
                entity_id = row["id"]
                for original_idx in indices_map[result_idx]:
                    entity_ids[original_idx] = entity_id

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
            VALUES ($1, $2, $3, $4, 1)
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

        # Batch update co-occurrences
        if cooccurrence_pairs:
            now = datetime.now(UTC)
            await conn.executemany(
                f"""
                INSERT INTO {fq_table("entity_cooccurrences")} (entity_id_1, entity_id_2, cooccurrence_count, last_cooccurred)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (entity_id_1, entity_id_2)
                DO UPDATE SET
                    cooccurrence_count = {fq_table("entity_cooccurrences")}.cooccurrence_count + 1,
                    last_cooccurred = EXCLUDED.last_cooccurred
                """,
                [(e1, e2, 1, now) for e1, e2 in cooccurrence_pairs],
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
