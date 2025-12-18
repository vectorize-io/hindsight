"""
Link creation utilities for temporal, semantic, and entity links.
"""

import logging
import time
from datetime import UTC, datetime, timedelta
from uuid import UUID

from ..memory_engine import fq_table
from .types import EntityLink

logger = logging.getLogger(__name__)


def _normalize_datetime(dt):
    """Normalize datetime to be timezone-aware (UTC) for consistent comparison."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        # Naive datetime - assume UTC
        return dt.replace(tzinfo=UTC)
    return dt


def compute_temporal_links(
    new_units: dict,
    candidates: list,
    time_window_hours: int = 24,
) -> list:
    """
    Compute temporal links between new units and candidate neighbors.

    This is a pure function that takes query results and returns link tuples,
    making it easy to test without database access.

    Args:
        new_units: Dict mapping unit_id (str) to event_date (datetime)
        candidates: List of dicts with 'id' and 'event_date' keys (candidate neighbors)
        time_window_hours: Time window in hours for temporal links

    Returns:
        List of tuples: (from_unit_id, to_unit_id, 'temporal', weight, None)
    """
    if not new_units:
        return []

    links = []
    for unit_id, unit_event_date in new_units.items():
        # Normalize unit_event_date for consistent comparison
        unit_event_date_norm = _normalize_datetime(unit_event_date)

        # Calculate time window bounds with overflow protection
        try:
            time_lower = unit_event_date_norm - timedelta(hours=time_window_hours)
        except OverflowError:
            time_lower = datetime.min.replace(tzinfo=UTC)
        try:
            time_upper = unit_event_date_norm + timedelta(hours=time_window_hours)
        except OverflowError:
            time_upper = datetime.max.replace(tzinfo=UTC)

        # Filter candidates within this unit's time window
        matching_neighbors = [
            (row["id"], row["event_date"])
            for row in candidates
            if time_lower <= _normalize_datetime(row["event_date"]) <= time_upper
        ][:10]  # Limit to top 10

        for recent_id, recent_event_date in matching_neighbors:
            # Calculate temporal proximity weight
            time_diff_hours = abs(
                (unit_event_date_norm - _normalize_datetime(recent_event_date)).total_seconds() / 3600
            )
            weight = max(0.3, 1.0 - (time_diff_hours / time_window_hours))
            links.append((unit_id, str(recent_id), "temporal", weight, None))

    return links


def compute_temporal_query_bounds(
    new_units: dict,
    time_window_hours: int = 24,
) -> tuple:
    """
    Compute the min/max date bounds for querying temporal neighbors.

    Args:
        new_units: Dict mapping unit_id (str) to event_date (datetime)
        time_window_hours: Time window in hours

    Returns:
        Tuple of (min_date, max_date) with overflow protection
    """
    if not new_units:
        return None, None

    # Normalize all dates to be timezone-aware to avoid comparison issues
    all_dates = [_normalize_datetime(d) for d in new_units.values()]

    try:
        min_date = min(all_dates) - timedelta(hours=time_window_hours)
    except OverflowError:
        min_date = datetime.min.replace(tzinfo=UTC)

    try:
        max_date = max(all_dates) + timedelta(hours=time_window_hours)
    except OverflowError:
        max_date = datetime.max.replace(tzinfo=UTC)

    return min_date, max_date


def _log(log_buffer, message, level="info"):
    """Helper to log to buffer if available, otherwise use logger.

    Args:
        log_buffer: Buffer to append messages to (for main output)
        message: The log message
        level: 'info', 'debug', 'warning', or 'error'. Debug messages are not added to buffer.
    """
    if level == "debug":
        # Debug messages only go to logger, not to buffer
        logger.debug(message)
        return

    if log_buffer is not None:
        log_buffer.append(message)
    else:
        if level == "info":
            logger.info(message)
        else:
            logger.log(logging.WARNING if level == "warning" else logging.ERROR, message)


async def extract_entities_batch_optimized(
    entity_resolver,
    conn,
    bank_id: str,
    unit_ids: list[str],
    sentences: list[str],
    context: str,
    fact_dates: list,
    llm_entities: list[list[dict]],
    log_buffer: list[str] = None,
) -> list[tuple]:
    """
    Process LLM-extracted entities for ALL facts in batch.

    Uses entities provided by the LLM (no spaCy needed), then resolves
    and links them in bulk.

    Args:
        entity_resolver: EntityResolver instance for entity resolution
        conn: Database connection
        agent_id: bank IDentifier
        unit_ids: List of unit IDs
        sentences: List of fact sentences
        context: Context string
        fact_dates: List of fact dates
        llm_entities: List of entity lists from LLM extraction
        log_buffer: Optional buffer for logging

    Returns:
        List of tuples for batch insertion: (from_unit_id, to_unit_id, link_type, weight, entity_id)
    """
    try:
        # Step 1: Convert LLM entities to the format expected by entity resolver
        substep_start = time.time()
        all_entities = []
        for entity_list in llm_entities:
            # Convert List[Entity] or List[dict] to List[Dict] format
            formatted_entities = []
            for ent in entity_list:
                # Handle both Entity objects and dicts
                if hasattr(ent, "text"):
                    # Entity objects only have 'text', default type to 'CONCEPT'
                    formatted_entities.append({"text": ent.text, "type": "CONCEPT"})
                elif isinstance(ent, dict):
                    formatted_entities.append({"text": ent.get("text", ""), "type": ent.get("type", "CONCEPT")})
            all_entities.append(formatted_entities)

        total_entities = sum(len(ents) for ents in all_entities)
        _log(
            log_buffer,
            f"  [6.1] Process LLM entities: {total_entities} entities from {len(sentences)} facts in {time.time() - substep_start:.3f}s",
            level="debug",
        )

        # Step 2: Resolve entities in BATCH (much faster!)
        substep_start = time.time()
        step_6_2_start = time.time()

        # [6.2.1] Prepare all entities for batch resolution
        substep_6_2_1_start = time.time()
        all_entities_flat = []
        entity_to_unit = []  # Maps flat index to (unit_id, local_index)

        for unit_id, entities, fact_date in zip(unit_ids, all_entities, fact_dates):
            if not entities:
                continue

            for local_idx, entity in enumerate(entities):
                all_entities_flat.append(
                    {
                        "text": entity["text"],
                        "type": entity["type"],
                        "nearby_entities": entities,
                    }
                )
                entity_to_unit.append((unit_id, local_idx, fact_date))
        _log(
            log_buffer,
            f"    [6.2.1] Prepare entities: {len(all_entities_flat)} entities in {time.time() - substep_6_2_1_start:.3f}s",
            level="debug",
        )

        # Resolve ALL entities in one batch call
        if all_entities_flat:
            # [6.2.2] Batch resolve entities - single call with per-entity dates
            substep_6_2_2_start = time.time()

            # Add per-entity dates to entity data for batch resolution
            for idx, (unit_id, local_idx, fact_date) in enumerate(entity_to_unit):
                all_entities_flat[idx]["event_date"] = fact_date

            # Resolve ALL entities in ONE batch call (much faster than sequential buckets)
            # INSERT ... ON CONFLICT handles any race conditions at the DB level
            resolved_entity_ids = await entity_resolver.resolve_entities_batch(
                bank_id=bank_id,
                entities_data=all_entities_flat,
                context=context,
                unit_event_date=None,  # Not used when per-entity dates provided
                conn=conn,  # Use main transaction connection
            )

            _log(
                log_buffer,
                f"    [6.2.2] Resolve entities: {len(all_entities_flat)} entities in single batch in {time.time() - substep_6_2_2_start:.3f}s",
                level="debug",
            )

            # [6.2.3] Create unit-entity links in BATCH
            substep_6_2_3_start = time.time()
            # Map resolved entities back to units and collect all (unit, entity) pairs
            unit_to_entity_ids = {}
            unit_entity_pairs = []
            for idx, (unit_id, local_idx, fact_date) in enumerate(entity_to_unit):
                if unit_id not in unit_to_entity_ids:
                    unit_to_entity_ids[unit_id] = []

                entity_id = resolved_entity_ids[idx]
                unit_to_entity_ids[unit_id].append(entity_id)
                unit_entity_pairs.append((unit_id, entity_id))

            # Batch insert all unit-entity links (MUCH faster!)
            await entity_resolver.link_units_to_entities_batch(unit_entity_pairs, conn=conn)
            _log(
                log_buffer,
                f"    [6.2.3] Create unit-entity links (batched): {len(unit_entity_pairs)} links in {time.time() - substep_6_2_3_start:.3f}s",
                level="debug",
            )

            _log(
                log_buffer,
                f"  [6.2] Entity resolution (batched): {len(all_entities_flat)} entities resolved in {time.time() - step_6_2_start:.3f}s",
                level="debug",
            )
        else:
            unit_to_entity_ids = {}
            _log(
                log_buffer,
                f"  [6.2] Entity resolution (batched): 0 entities in {time.time() - step_6_2_start:.3f}s",
                level="debug",
            )

        # Step 3: Create entity links between units that share entities
        substep_start = time.time()
        # Collect all unique entity IDs
        all_entity_ids = set()
        for entity_ids in unit_to_entity_ids.values():
            all_entity_ids.update(entity_ids)

        _log(log_buffer, f"  [6.3] Creating entity links for {len(all_entity_ids)} unique entities...", level="debug")

        # Find all units that reference these entities (ONE batched query)
        entity_to_units = {}
        if all_entity_ids:
            query_start = time.time()
            import uuid

            entity_id_list = [uuid.UUID(eid) if isinstance(eid, str) else eid for eid in all_entity_ids]
            rows = await conn.fetch(
                f"""
                SELECT entity_id, unit_id
                FROM {fq_table("unit_entities")}
                WHERE entity_id = ANY($1::uuid[])
                """,
                entity_id_list,
            )
            _log(
                log_buffer,
                f"      [6.3.1] Query unit_entities: {len(rows)} rows in {time.time() - query_start:.3f}s",
                level="debug",
            )

            # Group by entity_id
            group_start = time.time()
            for row in rows:
                entity_id = row["entity_id"]
                if entity_id not in entity_to_units:
                    entity_to_units[entity_id] = []
                entity_to_units[entity_id].append(row["unit_id"])
            _log(log_buffer, f"      [6.3.2] Group by entity_id: {time.time() - group_start:.3f}s", level="debug")

        # Create bidirectional links between units that share entities
        # OPTIMIZATION: Limit links per entity to avoid N² explosion
        # Only link each new unit to the most recent MAX_LINKS_PER_ENTITY units
        MAX_LINKS_PER_ENTITY = 50  # Limit to prevent explosion when entity appears in many facts
        link_gen_start = time.time()
        links: list[EntityLink] = []
        new_unit_set = set(unit_ids)  # Units from this batch

        def to_uuid(val) -> UUID:
            return UUID(val) if isinstance(val, str) else val

        for entity_id, units_with_entity in entity_to_units.items():
            entity_uuid = to_uuid(entity_id)
            # Separate new units (from this batch) and existing units
            new_units = [u for u in units_with_entity if str(u) in new_unit_set or u in new_unit_set]
            existing_units = [u for u in units_with_entity if str(u) not in new_unit_set and u not in new_unit_set]

            # Link new units to each other (within batch) - also limited
            # For very common entities, limit within-batch links too
            new_units_to_link = (
                new_units[-MAX_LINKS_PER_ENTITY:] if len(new_units) > MAX_LINKS_PER_ENTITY else new_units
            )
            for i, unit_id_1 in enumerate(new_units_to_link):
                for unit_id_2 in new_units_to_link[i + 1 :]:
                    links.append(
                        EntityLink(
                            from_unit_id=to_uuid(unit_id_1), to_unit_id=to_uuid(unit_id_2), entity_id=entity_uuid
                        )
                    )
                    links.append(
                        EntityLink(
                            from_unit_id=to_uuid(unit_id_2), to_unit_id=to_uuid(unit_id_1), entity_id=entity_uuid
                        )
                    )

            # Link new units to LIMITED existing units (most recent)
            existing_to_link = existing_units[-MAX_LINKS_PER_ENTITY:]  # Take most recent
            for new_unit in new_units:
                for existing_unit in existing_to_link:
                    links.append(
                        EntityLink(
                            from_unit_id=to_uuid(new_unit), to_unit_id=to_uuid(existing_unit), entity_id=entity_uuid
                        )
                    )
                    links.append(
                        EntityLink(
                            from_unit_id=to_uuid(existing_unit), to_unit_id=to_uuid(new_unit), entity_id=entity_uuid
                        )
                    )

        _log(
            log_buffer, f"      [6.3.3] Generate {len(links)} links: {time.time() - link_gen_start:.3f}s", level="debug"
        )
        _log(
            log_buffer,
            f"  [6.3] Entity link creation: {len(links)} links for {len(all_entity_ids)} unique entities in {time.time() - substep_start:.3f}s",
            level="debug",
        )

        return links

    except Exception as e:
        logger.error(f"Failed to extract entities in batch: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


async def create_temporal_links_batch_per_fact(
    conn,
    bank_id: str,
    unit_ids: list[str],
    time_window_hours: int = 24,
    log_buffer: list[str] = None,
) -> int:
    """
    Create temporal links for multiple units, each with their own event_date.

    Queries the event_date for each unit from the database and creates temporal
    links based on individual dates (supports per-fact dating).

    Args:
        conn: Database connection
        agent_id: bank IDentifier
        unit_ids: List of unit IDs
        time_window_hours: Time window in hours for temporal links
        log_buffer: Optional buffer for logging

    Returns:
        Number of temporal links created
    """
    if not unit_ids:
        return 0

    try:
        import time as time_mod

        # Get the event_date for each new unit
        fetch_dates_start = time_mod.time()
        rows = await conn.fetch(
            f"""
            SELECT id, event_date
            FROM {fq_table("memory_units")}
            WHERE id::text = ANY($1)
            """,
            unit_ids,
        )
        new_units = {str(row["id"]): row["event_date"] for row in rows}
        _log(
            log_buffer,
            f"      [7.1] Fetch event_dates for {len(unit_ids)} units: {time_mod.time() - fetch_dates_start:.3f}s",
        )

        # Fetch ALL potential temporal neighbors in ONE query (much faster!)
        # Get time range across all units with overflow protection
        min_date, max_date = compute_temporal_query_bounds(new_units, time_window_hours)

        fetch_neighbors_start = time_mod.time()
        all_candidates = await conn.fetch(
            f"""
            SELECT id, event_date
            FROM {fq_table("memory_units")}
            WHERE bank_id = $1
              AND event_date BETWEEN $2 AND $3
              AND id::text != ALL($4)
            ORDER BY event_date DESC
            """,
            bank_id,
            min_date,
            max_date,
            unit_ids,
        )
        _log(
            log_buffer,
            f"      [7.2] Fetch {len(all_candidates)} candidate neighbors (1 query): {time_mod.time() - fetch_neighbors_start:.3f}s",
        )

        # Filter and create links in memory (much faster than N queries)
        link_gen_start = time_mod.time()
        links = compute_temporal_links(new_units, all_candidates, time_window_hours)

        # Also compute temporal links WITHIN the new batch (new units to each other)
        if len(new_units) > 1:
            # Convert new_units dict to candidate format for within-batch linking
            new_unit_items = list(new_units.items())
            for i, (unit_id, event_date) in enumerate(new_unit_items):
                unit_event_date_norm = _normalize_datetime(event_date)

                # Compare with other new units (only those after this one to avoid duplicates)
                for j in range(i + 1, len(new_unit_items)):
                    other_id, other_event_date = new_unit_items[j]
                    other_event_date_norm = _normalize_datetime(other_event_date)

                    # Check if within time window
                    time_diff_hours = abs((unit_event_date_norm - other_event_date_norm).total_seconds() / 3600)
                    if time_diff_hours <= time_window_hours:
                        weight = max(0.3, 1.0 - (time_diff_hours / time_window_hours))
                        # Create bidirectional links
                        links.append((unit_id, other_id, "temporal", weight, None))
                        links.append((other_id, unit_id, "temporal", weight, None))

        _log(log_buffer, f"      [7.3] Generate {len(links)} temporal links: {time_mod.time() - link_gen_start:.3f}s")

        if links:
            insert_start = time_mod.time()
            await conn.executemany(
                f"""
                INSERT INTO {fq_table("memory_links")} (from_unit_id, to_unit_id, link_type, weight, entity_id)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING
                """,
                links,
            )
            _log(log_buffer, f"      [7.4] Insert {len(links)} temporal links: {time_mod.time() - insert_start:.3f}s")

        return len(links)

    except Exception as e:
        logger.error(f"Failed to create temporal links: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


async def create_semantic_links_batch(
    conn,
    bank_id: str,
    unit_ids: list[str],
    embeddings: list[list[float]],
    top_k: int = 5,
    threshold: float = 0.7,
    log_buffer: list[str] = None,
) -> int:
    """
    Create semantic links for multiple units efficiently.

    For each unit, finds similar units and creates links.

    Args:
        conn: Database connection
        agent_id: bank IDentifier
        unit_ids: List of unit IDs
        embeddings: List of embedding vectors
        top_k: Number of top similar units to link
        threshold: Minimum similarity threshold
        log_buffer: Optional buffer for logging

    Returns:
        Number of semantic links created
    """
    if not unit_ids or not embeddings:
        return 0

    try:
        import time as time_mod

        import numpy as np

        # Fetch ALL existing units with embeddings in ONE query
        fetch_start = time_mod.time()
        all_existing = await conn.fetch(
            f"""
            SELECT id, embedding
            FROM {fq_table("memory_units")}
            WHERE bank_id = $1
              AND embedding IS NOT NULL
              AND id::text != ALL($2)
            """,
            bank_id,
            unit_ids,
        )
        _log(
            log_buffer,
            f"      [8.1] Fetch {len(all_existing)} existing embeddings (1 query): {time_mod.time() - fetch_start:.3f}s",
        )

        # Convert to numpy for vectorized similarity computation
        compute_start = time_mod.time()
        all_links = []

        if all_existing:
            # Convert existing embeddings to numpy array
            existing_ids = [str(row["id"]) for row in all_existing]
            # Stack embeddings as 2D array: (num_embeddings, embedding_dim)
            embedding_arrays = []
            for row in all_existing:
                raw_emb = row["embedding"]
                # Handle different pgvector formats
                if isinstance(raw_emb, str):
                    # Parse string format: "[1.0, 2.0, ...]"
                    import json

                    emb = np.array(json.loads(raw_emb), dtype=np.float32)
                elif isinstance(raw_emb, (list, tuple)):
                    emb = np.array(raw_emb, dtype=np.float32)
                else:
                    # Try direct conversion (works for numpy arrays, pgvector objects, etc.)
                    emb = np.array(raw_emb, dtype=np.float32)

                # Ensure it's 1D
                if emb.ndim != 1:
                    raise ValueError(f"Expected 1D embedding, got shape {emb.shape}")
                embedding_arrays.append(emb)

            if not embedding_arrays:
                existing_embeddings = np.array([])
            elif len(embedding_arrays) == 1:
                # Single embedding: reshape to (1, dim)
                existing_embeddings = embedding_arrays[0].reshape(1, -1)
            else:
                # Multiple embeddings: vstack
                existing_embeddings = np.vstack(embedding_arrays)

            # For each new unit, compute similarities with ALL existing units
            for unit_id, new_embedding in zip(unit_ids, embeddings):
                new_emb_array = np.array(new_embedding)

                # Compute cosine similarities (dot product for normalized vectors)
                similarities = np.dot(existing_embeddings, new_emb_array)

                # Find top-k above threshold
                # Get indices of similarities above threshold
                above_threshold = np.where(similarities >= threshold)[0]

                if len(above_threshold) > 0:
                    # Sort by similarity (descending) and take top-k
                    sorted_indices = above_threshold[np.argsort(-similarities[above_threshold])][:top_k]

                    for idx in sorted_indices:
                        similar_id = existing_ids[idx]
                        # Clamp to [0, 1] to handle floating point precision issues
                        similarity = float(min(1.0, max(0.0, similarities[idx])))
                        all_links.append((unit_id, similar_id, "semantic", similarity, None))

        # Also compute similarities WITHIN the new batch (new units to each other)
        # Apply the same top_k limit per unit as we do for existing units
        if len(unit_ids) > 1:
            new_embeddings_matrix = np.array(embeddings)

            for i, unit_id in enumerate(unit_ids):
                # Compute similarities with all OTHER new units
                other_indices = [j for j in range(len(unit_ids)) if j != i]
                if not other_indices:
                    continue

                other_embeddings = new_embeddings_matrix[other_indices]
                similarities = np.dot(other_embeddings, new_embeddings_matrix[i])

                # Find top-k above threshold (same logic as existing units)
                above_threshold = np.where(similarities >= threshold)[0]

                if len(above_threshold) > 0:
                    # Sort by similarity (descending) and take top-k
                    sorted_local_indices = above_threshold[np.argsort(-similarities[above_threshold])][:top_k]

                    for local_idx in sorted_local_indices:
                        other_idx = other_indices[local_idx]
                        other_id = unit_ids[other_idx]
                        # Clamp to [0, 1] to handle floating point precision issues
                        similarity = float(min(1.0, max(0.0, similarities[local_idx])))
                        all_links.append((unit_id, other_id, "semantic", similarity, None))

        _log(
            log_buffer,
            f"      [8.2] Compute similarities & generate {len(all_links)} semantic links: {time_mod.time() - compute_start:.3f}s",
        )

        if all_links:
            insert_start = time_mod.time()
            await conn.executemany(
                f"""
                INSERT INTO {fq_table("memory_links")} (from_unit_id, to_unit_id, link_type, weight, entity_id)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING
                """,
                all_links,
            )
            _log(
                log_buffer, f"      [8.3] Insert {len(all_links)} semantic links: {time_mod.time() - insert_start:.3f}s"
            )

        return len(all_links)

    except Exception as e:
        logger.error(f"Failed to create semantic links: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


async def insert_entity_links_batch(conn, links: list[EntityLink], chunk_size: int = 50000):
    """
    Insert all entity links using COPY to temp table + INSERT for maximum speed.

    Uses PostgreSQL COPY (via copy_records_to_table) for bulk loading,
    then INSERT ... ON CONFLICT from temp table. This is the fastest
    method for bulk inserts with conflict handling.

    Args:
        conn: Database connection
        links: List of EntityLink objects
        chunk_size: Number of rows per batch (default 50000)
    """
    if not links:
        return

    import time as time_mod

    total_start = time_mod.time()

    # Create temp table for bulk loading
    create_start = time_mod.time()
    await conn.execute("""
        CREATE TEMP TABLE IF NOT EXISTS _temp_entity_links (
            from_unit_id uuid,
            to_unit_id uuid,
            link_type text,
            weight float,
            entity_id uuid
        ) ON COMMIT DROP
    """)
    logger.debug(f"      [9.1] Create temp table: {time_mod.time() - create_start:.3f}s")

    # Clear any existing data in temp table
    truncate_start = time_mod.time()
    await conn.execute("TRUNCATE _temp_entity_links")
    logger.debug(f"      [9.2] Truncate temp table: {time_mod.time() - truncate_start:.3f}s")

    # Convert EntityLink objects to tuples for COPY
    convert_start = time_mod.time()
    records = []
    for link in links:
        records.append((link.from_unit_id, link.to_unit_id, link.link_type, link.weight, link.entity_id))
    logger.debug(f"      [9.3] Convert {len(records)} records: {time_mod.time() - convert_start:.3f}s")

    # Bulk load using COPY (fastest method)
    copy_start = time_mod.time()
    await conn.copy_records_to_table(
        "_temp_entity_links",
        records=records,
        columns=["from_unit_id", "to_unit_id", "link_type", "weight", "entity_id"],
    )
    logger.debug(f"      [9.4] COPY {len(records)} records to temp table: {time_mod.time() - copy_start:.3f}s")

    # Insert from temp table with ON CONFLICT (single query for all rows)
    insert_start = time_mod.time()
    await conn.execute(f"""
        INSERT INTO {fq_table("memory_links")} (from_unit_id, to_unit_id, link_type, weight, entity_id)
        SELECT from_unit_id, to_unit_id, link_type, weight, entity_id
        FROM _temp_entity_links
        ON CONFLICT (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING
    """)
    logger.debug(f"      [9.5] INSERT from temp table: {time_mod.time() - insert_start:.3f}s")
    logger.debug(f"      [9.TOTAL] Entity links batch insert: {time_mod.time() - total_start:.3f}s")


async def create_causal_links_batch(
    conn,
    unit_ids: list[str],
    causal_relations_per_fact: list[list[dict]],
) -> int:
    """
    Create causal links between facts based on LLM-extracted causal relationships.

    Args:
        conn: Database connection
        unit_ids: List of unit IDs (in same order as causal_relations_per_fact)
        causal_relations_per_fact: List of causal relations for each fact.
            Each element is a list of dicts with:
            - target_fact_index: Index into unit_ids for the target fact
            - relation_type: "causes", "caused_by", "enables", or "prevents"
            - strength: Float in [0.0, 1.0] representing relationship strength

    Returns:
        Number of causal links created

    Causal link types:
    - "causes": This fact directly causes the target fact (forward causation)
    - "caused_by": This fact was caused by the target fact (backward causation)
    - "enables": This fact enables/allows the target fact (enablement)
    - "prevents": This fact prevents/blocks the target fact (prevention)
    """
    if not unit_ids or not causal_relations_per_fact:
        return 0

    try:
        import time as time_mod

        create_start = time_mod.time()

        # Build links list
        links = []
        for fact_idx, causal_relations in enumerate(causal_relations_per_fact):
            if not causal_relations:
                continue

            from_unit_id = unit_ids[fact_idx]

            for relation in causal_relations:
                target_idx = relation["target_fact_index"]
                relation_type = relation["relation_type"]
                strength = relation.get("strength", 1.0)

                # Validate relation_type - must match database constraint
                valid_types = {"causes", "caused_by", "enables", "prevents"}
                if relation_type not in valid_types:
                    logger.error(
                        f"Invalid relation_type '{relation_type}' (type: {type(relation_type).__name__}) "
                        f"from fact {fact_idx}. Must be one of: {valid_types}. "
                        f"Relation data: {relation}"
                    )
                    continue

                # Validate target index
                if target_idx < 0 or target_idx >= len(unit_ids):
                    logger.warning(f"Invalid target_fact_index {target_idx} in causal relation from fact {fact_idx}")
                    continue

                to_unit_id = unit_ids[target_idx]

                # Don't create self-links
                if from_unit_id == to_unit_id:
                    continue

                # Add the causal link
                # link_type is the relation_type (e.g., "causes", "caused_by")
                # weight is the strength of the relationship
                links.append((from_unit_id, to_unit_id, relation_type, strength, None))

        if links:
            insert_start = time_mod.time()
            try:
                await conn.executemany(
                    f"""
                    INSERT INTO {fq_table("memory_links")} (from_unit_id, to_unit_id, link_type, weight, entity_id)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING
                    """,
                    links,
                )
            except Exception as db_error:
                # Log the actual data being inserted for debugging
                logger.error(f"Database insert failed for causal links. Error: {db_error}")
                logger.error(f"Attempted to insert {len(links)} links. First few:")
                for i, link in enumerate(links[:3]):
                    logger.error(
                        f"  Link {i}: from={link[0]}, to={link[1]}, type='{link[2]}' (repr={repr(link[2])}), weight={link[3]}, entity={link[4]}"
                    )
                raise

        return len(links)

    except Exception as e:
        logger.error(f"Failed to create causal links: {str(e)}")
        import traceback

        traceback.print_exc()
        raise
