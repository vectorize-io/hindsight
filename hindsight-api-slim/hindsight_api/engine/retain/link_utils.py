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

# Sentinel UUID used in the unique index to represent NULL entity_id
_NIL_ENTITY_UUID = "00000000-0000-0000-0000-000000000000"

# Maximum number of temporal links to keep per unit (from_unit_id).
# Retrieval only reads top 10-20 per unit via LATERAL join, so keeping
# more is wasted storage and write amplification.
MAX_TEMPORAL_LINKS_PER_UNIT = 20


def _cap_links_per_unit(links: list[tuple], max_per_unit: int = MAX_TEMPORAL_LINKS_PER_UNIT) -> list[tuple]:
    """Keep only the top-N links per from_unit_id, ranked by weight descending.

    Args:
        links: List of (from_unit_id, to_unit_id, link_type, weight, entity_id) tuples.
        max_per_unit: Maximum number of links to retain per from_unit_id.

    Returns:
        Filtered list of link tuples.
    """
    if not links:
        return links

    # Group by from_unit_id (index 0)
    groups: dict[str, list[tuple]] = {}
    for link in links:
        key = str(link[0])
        if key not in groups:
            groups[key] = []
        groups[key].append(link)

    # For each group, sort by weight (index 3) descending and keep top N
    result: list[tuple] = []
    for group_links in groups.values():
        group_links.sort(key=lambda lnk: lnk[3], reverse=True)
        result.extend(group_links[:max_per_unit])

    return result


async def _bulk_insert_links(
    conn,
    links: list[tuple],
    bank_id: str = "",
    chunk_size: int = 5000,
    skip_exists_check: bool = False,
) -> None:
    """Insert links into memory_links using sorted bulk INSERT FROM unnest().

    Sorting by (from_unit_id, to_unit_id) ensures all concurrent transactions
    acquire index locks in the same order, eliminating circular-wait deadlocks.

    A single INSERT ... SELECT FROM unnest() is also faster than executemany
    (one round-trip vs N), and acquires all locks within one statement execution
    rather than interleaving with other transactions between rows.

    Args:
        conn: Database connection (must be inside a transaction).
        links: List of (from_unit_id, to_unit_id, link_type, weight, entity_id) tuples.
        bank_id: Bank identifier stored on memory_links for fast filtering.
        chunk_size: Max rows per INSERT statement to avoid query timeouts on
                    very large tables (100M+ rows).
        skip_exists_check: Skip WHERE EXISTS checks on memory_units. Use when
                    all referenced unit IDs are guaranteed to exist (e.g., within
                    the same transaction that inserted them).
    """
    if not links:
        return

    # Sort by (from_unit_id, to_unit_id) to guarantee consistent lock ordering
    # across concurrent transactions — prevents deadlocks.
    sorted_links = sorted(links, key=lambda lnk: (str(lnk[0]), str(lnk[1])))

    from_ids = [lnk[0] for lnk in sorted_links]
    to_ids = [lnk[1] for lnk in sorted_links]
    types = [lnk[2] for lnk in sorted_links]
    weights = [lnk[3] for lnk in sorted_links]
    entity_ids = [lnk[4] for lnk in sorted_links]

    exists_clause = ""
    if not skip_exists_check:
        exists_clause = (
            f"WHERE EXISTS (SELECT 1 FROM {fq_table('memory_units')} mu WHERE mu.id = f)"
            f"  AND EXISTS (SELECT 1 FROM {fq_table('memory_units')} mu WHERE mu.id = t)"
        )

    for chunk_start in range(0, len(sorted_links), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(sorted_links))
        await conn.execute(
            f"""
            INSERT INTO {fq_table("memory_links")}
                (from_unit_id, to_unit_id, link_type, weight, entity_id, bank_id)
            SELECT f, t, tp, w, e, $6
            FROM unnest($1::uuid[], $2::uuid[], $3::text[], $4::float8[], $5::uuid[])
                AS t(f, t, tp, w, e)
            {exists_clause}
            ON CONFLICT (from_unit_id, to_unit_id, link_type,
                         COALESCE(entity_id, '{_NIL_ENTITY_UUID}'::uuid))
            DO NOTHING
            """,
            from_ids[chunk_start:chunk_end],
            to_ids[chunk_start:chunk_end],
            types[chunk_start:chunk_end],
            weights[chunk_start:chunk_end],
            entity_ids[chunk_start:chunk_end],
            bank_id,
            timeout=300,
        )


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
        # Units without event_date can't form temporal links
        if unit_event_date is None:
            continue
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

    return _cap_links_per_unit(links)


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
    # Filter out None values — units without event_date can't form temporal links
    all_dates = [_normalize_datetime(d) for d in new_units.values() if d is not None]

    if not all_dates:
        return None, None

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


def _prepare_entities_for_resolution(
    unit_ids: list[str],
    sentences: list[str],
    fact_dates: list,
    llm_entities: list[list[dict]],
    log_buffer: list[str] = None,
) -> tuple[list[dict], list[list[dict]], list[tuple]]:
    """
    Convert LLM entities into the flat format expected by entity resolver.

    Returns:
        Tuple of (all_entities_flat, all_entities, entity_to_unit) where:
        - all_entities_flat: flat list of entity dicts ready for resolve_entities_batch
        - all_entities: per-unit formatted entity lists
        - entity_to_unit: maps flat index to (unit_id, local_index, fact_date)
    """
    substep_start = time.time()
    all_entities = []
    for entity_list in llm_entities:
        formatted_entities = []
        for ent in entity_list:
            if hasattr(ent, "text"):
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

    substep_start = time.time()
    all_entities_flat = []
    entity_to_unit: list[tuple] = []

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
        f"    [6.2.1] Prepare entities: {len(all_entities_flat)} entities in {time.time() - substep_start:.3f}s",
        level="debug",
    )

    # Attach per-entity dates
    for idx, (_unit_id, _local_idx, fact_date) in enumerate(entity_to_unit):
        all_entities_flat[idx]["event_date"] = fact_date

    return all_entities_flat, all_entities, entity_to_unit


async def resolve_entities_only(
    entity_resolver,
    conn,
    bank_id: str,
    unit_ids: list[str],
    sentences: list[str],
    context: str,
    fact_dates: list,
    llm_entities: list[list[dict]],
    log_buffer: list[str] = None,
    entity_labels: list | None = None,
) -> tuple[list[str], list[tuple], dict[str, list[str]]]:
    """
    Phase 1 of entity processing: resolve entity names to canonical IDs.

    Runs the expensive read-heavy trigram search, co-occurrence fetch, and scoring
    OUTSIDE the main write transaction.  Also INSERTs new entities (idempotent
    DO NOTHING) so that IDs are available for the subsequent write phase.

    Args:
        entity_resolver: EntityResolver instance
        conn: Database connection (separate from the main write transaction)
        bank_id: Bank identifier
        unit_ids: Placeholder unit IDs (used only for grouping, not yet inserted)
        sentences: Fact texts
        context: Context string
        fact_dates: Per-fact dates
        llm_entities: Per-fact entity lists from LLM extraction
        log_buffer: Optional logging buffer
        entity_labels: Optional entity label taxonomy

    Returns:
        Tuple of (resolved_entity_ids, entity_to_unit, unit_to_entity_ids) where:
        - resolved_entity_ids: list of entity IDs in same order as flattened entities
        - entity_to_unit: maps flat index to (unit_id, local_index, fact_date)
        - unit_to_entity_ids: maps unit_id to list of resolved entity IDs
    """
    all_entities_flat, _all_entities, entity_to_unit = _prepare_entities_for_resolution(
        unit_ids, sentences, fact_dates, llm_entities, log_buffer
    )

    if not all_entities_flat:
        _log(log_buffer, "  [6.2] Entity resolution (batched): 0 entities", level="debug")
        return [], [], {}

    step_start = time.time()
    resolved_entity_ids = await entity_resolver.resolve_entities_batch(
        bank_id=bank_id,
        entities_data=all_entities_flat,
        context=context,
        unit_event_date=None,
        conn=conn,
        entity_labels=entity_labels,
    )
    _log(
        log_buffer,
        f"    [6.2.2] Resolve entities: {len(all_entities_flat)} entities in single batch in {time.time() - step_start:.3f}s",
        level="debug",
    )

    # Build unit_to_entity_ids mapping
    unit_to_entity_ids: dict[str, list[str]] = {}
    for idx, (unit_id, _local_idx, _fact_date) in enumerate(entity_to_unit):
        if unit_id not in unit_to_entity_ids:
            unit_to_entity_ids[unit_id] = []
        unit_to_entity_ids[unit_id].append(resolved_entity_ids[idx])

    _log(
        log_buffer,
        f"  [6.2] Entity resolution (batched): {len(all_entities_flat)} entities resolved in {time.time() - step_start:.3f}s",
        level="debug",
    )

    return resolved_entity_ids, entity_to_unit, unit_to_entity_ids


async def build_entity_links_from_resolved(
    entity_resolver,
    conn,
    bank_id: str,
    unit_ids: list[str],
    resolved_entity_ids: list[str],
    entity_to_unit: list[tuple],
    unit_to_entity_ids: dict[str, list[str]],
    log_buffer: list[str] = None,
    skip_unit_entities_insert: bool = False,
) -> list["EntityLink"]:
    """
    Build entity links between units that share entities.

    Queries unit_entities to find which existing units share entities with the
    new units, then generates EntityLink objects for UI graph visualization.

    Args:
        entity_resolver: EntityResolver instance
        conn: Database connection
        bank_id: Bank identifier
        unit_ids: Actual unit IDs (must already be inserted in the DB)
        resolved_entity_ids: Entity IDs from resolve_entities_only
        entity_to_unit: Mapping from resolve_entities_only
        unit_to_entity_ids: Mapping from resolve_entities_only
        log_buffer: Optional logging buffer
        skip_unit_entities_insert: If True, skip unit_entities INSERT (already done in Phase 2)

    Returns:
        List of EntityLink objects for batch insertion
    """
    if not resolved_entity_ids:
        return []

    if not skip_unit_entities_insert:
        # Insert unit-entity links (used in fallback path where Phase 2 didn't do this)
        substep_start = time.time()
        unit_entity_pairs = []
        for idx, (unit_id, _local_idx, _fact_date) in enumerate(entity_to_unit):
            unit_entity_pairs.append((unit_id, resolved_entity_ids[idx]))

        await entity_resolver.link_units_to_entities_batch(unit_entity_pairs, conn=conn)
        _log(
            log_buffer,
            f"    [6.2.3] Create unit-entity links (batched): {len(unit_entity_pairs)} links in {time.time() - substep_start:.3f}s",
            level="debug",
        )

    # Build entity links between units that share entities
    substep_start = time.time()
    all_entity_ids = set()
    for entity_ids_list in unit_to_entity_ids.values():
        all_entity_ids.update(entity_ids_list)

    _log(log_buffer, f"  [6.3] Creating entity links for {len(all_entity_ids)} unique entities...", level="debug")

    MAX_LINKS_PER_ENTITY = 10

    entity_to_units = {}
    if all_entity_ids:
        query_start = time.time()
        import uuid

        entity_id_list = [uuid.UUID(eid) if isinstance(eid, str) else eid for eid in all_entity_ids]
        # Use LATERAL with LIMIT to cap rows fetched per entity at the SQL level,
        # avoiding transfer of thousands of rows for high-cardinality entities.
        rows = await conn.fetch(
            f"""
            SELECT e.entity_id, n.unit_id
            FROM unnest($1::uuid[]) AS e(entity_id)
            CROSS JOIN LATERAL (
                SELECT ue.unit_id
                FROM {fq_table("unit_entities")} ue
                WHERE ue.entity_id = e.entity_id
                ORDER BY ue.unit_id DESC
                LIMIT $2
            ) n
            """,
            entity_id_list,
            MAX_LINKS_PER_ENTITY + len(unit_ids),  # room for new units + existing cap
        )
        _log(
            log_buffer,
            f"      [6.3.1] Query unit_entities (LATERAL): {len(rows)} rows in {time.time() - query_start:.3f}s",
            level="debug",
        )

        group_start = time.time()
        for row in rows:
            entity_id = row["entity_id"]
            if entity_id not in entity_to_units:
                entity_to_units[entity_id] = []
            entity_to_units[entity_id].append(row["unit_id"])
        _log(log_buffer, f"      [6.3.2] Group by entity_id: {time.time() - group_start:.3f}s", level="debug")
    link_gen_start = time.time()
    links: list[EntityLink] = []
    new_unit_set = set(unit_ids)

    def to_uuid(val) -> UUID:
        return UUID(val) if isinstance(val, str) else val

    for entity_id, units_with_entity in entity_to_units.items():
        entity_uuid = to_uuid(entity_id)
        new_units = [u for u in units_with_entity if str(u) in new_unit_set or u in new_unit_set]
        existing_units = [u for u in units_with_entity if str(u) not in new_unit_set and u not in new_unit_set]

        new_units_to_link = new_units[-MAX_LINKS_PER_ENTITY:] if len(new_units) > MAX_LINKS_PER_ENTITY else new_units
        for i, unit_id_1 in enumerate(new_units_to_link):
            for unit_id_2 in new_units_to_link[i + 1 :]:
                links.append(
                    EntityLink(from_unit_id=to_uuid(unit_id_1), to_unit_id=to_uuid(unit_id_2), entity_id=entity_uuid)
                )
                links.append(
                    EntityLink(from_unit_id=to_uuid(unit_id_2), to_unit_id=to_uuid(unit_id_1), entity_id=entity_uuid)
                )

        existing_to_link = existing_units[-MAX_LINKS_PER_ENTITY:]
        for new_unit in new_units:
            for existing_unit in existing_to_link:
                links.append(
                    EntityLink(from_unit_id=to_uuid(new_unit), to_unit_id=to_uuid(existing_unit), entity_id=entity_uuid)
                )
                links.append(
                    EntityLink(from_unit_id=to_uuid(existing_unit), to_unit_id=to_uuid(new_unit), entity_id=entity_uuid)
                )

    _log(log_buffer, f"      [6.3.3] Generate {len(links)} links: {time.time() - link_gen_start:.3f}s", level="debug")
    _log(
        log_buffer,
        f"  [6.3] Entity link creation: {len(links)} links for {len(all_entity_ids)} unique entities in {time.time() - substep_start:.3f}s",
        level="debug",
    )

    return links


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
        bank_id: Bank identifier
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
            SELECT id, event_date, fact_type
            FROM {fq_table("memory_units")}
            WHERE id::text = ANY($1)
            """,
            unit_ids,
        )
        new_units = {str(row["id"]): (row["event_date"], row["fact_type"]) for row in rows}
        _log(
            log_buffer,
            f"      [7.1] Fetch event_dates for {len(unit_ids)} units: {time_mod.time() - fetch_dates_start:.3f}s",
        )

        # Use LATERAL push-down to fetch only top-N temporal neighbors per new unit,
        # avoiding transfer of the entire time-window result set (could be 50k+ rows).
        fetch_neighbors_start = time_mod.time()

        # Build arrays of new unit IDs, event dates, and fact types for the LATERAL query
        new_unit_entries = [(uid, edate, ftype) for uid, (edate, ftype) in new_units.items() if edate is not None]
        if new_unit_entries:
            import uuid as uuid_mod

            lateral_unit_ids = [
                uuid_mod.UUID(uid) if isinstance(uid, str) else uid for uid in [e[0] for e in new_unit_entries]
            ]
            lateral_event_dates = [_normalize_datetime(e[1]) for e in new_unit_entries]
            lateral_fact_types = [e[2] for e in new_unit_entries]
            # Bidirectional index scan: instead of scanning all units in the 24h
            # window (O(N) — 164k rows at scale) and sorting by proximity, we scan
            # the nearest K units in each direction using the B-tree index on
            # (bank_id, fact_type, event_date). This reads only 2×K rows per probe
            # regardless of bank size — 120x faster at 164k units (0.6ms vs 74ms).
            TEMPORAL_LATERAL_BATCH = 500
            half_limit = MAX_TEMPORAL_LINKS_PER_UNIT  # fetch K in each direction, take top K combined
            mu = fq_table("memory_units")
            rows = []
            for batch_start in range(0, len(new_unit_entries), TEMPORAL_LATERAL_BATCH):
                batch_end = batch_start + TEMPORAL_LATERAL_BATCH
                batch_rows = await conn.fetch(
                    f"""
                    SELECT from_id, id, event_date, time_diff_hours FROM (
                        SELECT src.unit_id::text AS from_id, combined.*,
                               ROW_NUMBER() OVER (
                                   PARTITION BY src.unit_id
                                   ORDER BY combined.time_diff_hours
                               ) AS rn
                        FROM unnest($1::uuid[], $2::timestamptz[], $3::text[])
                             AS src(unit_id, event_date, fact_type)
                        CROSS JOIN LATERAL (
                            -- Scan backward (older events) using index order
                            (SELECT mu.id, mu.event_date,
                                    ABS(EXTRACT(EPOCH FROM mu.event_date - src.event_date)) / 3600.0 AS time_diff_hours
                             FROM {mu} mu
                             WHERE mu.bank_id = $4
                               AND mu.fact_type = src.fact_type
                               AND mu.event_date <= src.event_date
                               AND mu.id != src.unit_id
                             ORDER BY mu.event_date DESC
                             LIMIT $5)
                            UNION ALL
                            -- Scan forward (newer events) using index order
                            (SELECT mu.id, mu.event_date,
                                    ABS(EXTRACT(EPOCH FROM mu.event_date - src.event_date)) / 3600.0 AS time_diff_hours
                             FROM {mu} mu
                             WHERE mu.bank_id = $4
                               AND mu.fact_type = src.fact_type
                               AND mu.event_date > src.event_date
                               AND mu.id != src.unit_id
                             ORDER BY mu.event_date ASC
                             LIMIT $5)
                        ) combined
                    ) ranked
                    WHERE rn <= $5
                    """,
                    lateral_unit_ids[batch_start:batch_end],
                    lateral_event_dates[batch_start:batch_end],
                    lateral_fact_types[batch_start:batch_end],
                    bank_id,
                    half_limit,
                )
                rows.extend(batch_rows)
        else:
            rows = []

        _log(
            log_buffer,
            f"      [7.2] Fetch {len(rows)} candidate neighbors (LATERAL): {time_mod.time() - fetch_neighbors_start:.3f}s",
        )

        # Build links directly from the LATERAL results (already per-unit limited)
        link_gen_start = time_mod.time()
        links = []
        for row in rows:
            time_diff_h = float(row["time_diff_hours"])
            weight = max(0.3, 1.0 - (time_diff_h / time_window_hours))
            links.append((row["from_id"], str(row["id"]), "temporal", weight, None))

        # Also compute temporal links WITHIN the new batch (new units to each other)
        if len(new_units) > 1:
            # Convert new_units dict to candidate format for within-batch linking
            new_unit_items = list(new_units.items())
            for i, (unit_id, (event_date, fact_type)) in enumerate(new_unit_items):
                if event_date is None:
                    continue  # Skip units without event_date for temporal linking
                unit_event_date_norm = _normalize_datetime(event_date)

                # Compare with other new units (only those after this one to avoid duplicates)
                for j in range(i + 1, len(new_unit_items)):
                    other_id, (other_event_date, other_fact_type) = new_unit_items[j]
                    if other_event_date is None:
                        continue  # Skip units without event_date
                    if fact_type != other_fact_type:
                        continue  # Only link facts of the same type
                    other_event_date_norm = _normalize_datetime(other_event_date)

                    # Check if within time window
                    time_diff_hours = abs((unit_event_date_norm - other_event_date_norm).total_seconds() / 3600)
                    if time_diff_hours <= time_window_hours:
                        weight = max(0.3, 1.0 - (time_diff_hours / time_window_hours))
                        # Create bidirectional links
                        links.append((unit_id, other_id, "temporal", weight, None))
                        links.append((other_id, unit_id, "temporal", weight, None))

        # Cap temporal links per unit to avoid write amplification;
        # retrieval only reads top 10-20 per unit anyway.
        links = _cap_links_per_unit(links)

        _log(log_buffer, f"      [7.3] Generate {len(links)} temporal links: {time_mod.time() - link_gen_start:.3f}s")

        if links:
            insert_start = time_mod.time()
            await _bulk_insert_links(conn, links, bank_id=bank_id, skip_exists_check=True)
            _log(log_buffer, f"      [7.4] Insert {len(links)} temporal links: {time_mod.time() - insert_start:.3f}s")

        return len(links)

    except Exception as e:
        logger.error(f"Failed to create temporal links: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


async def compute_semantic_links_ann(
    conn,
    bank_id: str,
    unit_ids: list[str],
    embeddings: list[list[float]],
    fact_types: list[str] | None = None,
    top_k: int = 50,
    threshold: float = 0.7,
    log_buffer: list[str] = None,
) -> list[tuple]:
    """
    Phase 1: ANN search for semantic neighbors among existing units.

    Runs on a separate connection OUTSIDE the write transaction to avoid
    holding locks during expensive HNSW index probes. Uses a temp table +
    LATERAL join to batch all probes in a single query.

    Queries are split by fact_type so PostgreSQL uses the per-bank partial
    HNSW indexes (idx_mu_emb_worl_*, idx_mu_emb_expr_*). Without the
    fact_type filter, the planner falls back to sequential scan (~50x slower).

    Args:
        conn: Database connection (separate from write transaction, autocommit)
        bank_id: Bank identifier
        unit_ids: Placeholder unit IDs (real IDs not yet created)
        embeddings: Embedding vectors for each unit
        fact_types: Per-unit fact types (same length as unit_ids). Used to
            query only the matching HNSW index per seed.
        top_k: Max neighbors per unit
        threshold: Minimum cosine similarity
        log_buffer: Optional logging buffer

    Returns:
        List of (from_id, to_id, "semantic", similarity, None) tuples
        where from_id uses placeholder IDs.
    """
    if not unit_ids or not embeddings:
        return []

    import time as time_mod
    import uuid as uuid_mod

    ann_start = time_mod.time()
    links = []

    # Lower ef_search for retain ANN — default 400 is tuned for recall precision
    # but at 164k units each HNSW probe takes 94ms. ef_search=60 gives 2.7ms/probe
    # (35x faster) with sufficient accuracy for top-50 semantic link creation.
    # Reset after to avoid polluting the connection pool for recall queries.
    await conn.execute("SET hnsw.ef_search = 60")

    logger.debug(f"[ANN] Starting: {len(unit_ids)} seeds, top_k={top_k}")

    # Build per-unit fact_types (default to 'world' if not provided)
    if fact_types is None:
        fact_types = ["world"] * len(unit_ids)

    # No exclude_uuids — large exclusion lists (8k+ UUIDs) force PostgreSQL to
    # sequential-scan every HNSW probe result against the array, destroying
    # performance (67s for 8k seeds). Self-links are harmless (ON CONFLICT DO
    # NOTHING handles duplicates in memory_links).
    t_setup = time_mod.time()
    await conn.execute("CREATE TEMP TABLE IF NOT EXISTS _ann_seeds (unit_id text, emb_text text, fact_type text)")
    await conn.execute("TRUNCATE _ann_seeds")

    records = [
        (uid, emb if isinstance(emb, str) else str(emb), ft) for uid, emb, ft in zip(unit_ids, embeddings, fact_types)
    ]
    await conn.copy_records_to_table("_ann_seeds", records=records, columns=["unit_id", "emb_text", "fact_type"])
    logger.debug(f"[ANN] Temp table setup: {time_mod.time() - t_setup:.3f}s ({len(records)} seeds)")

    # Run one ANN query per fact_type so each uses the right HNSW index.
    rows = []
    active_types = set(fact_types)
    for fact_type in active_types:
        t_query = time_mod.time()
        seed_count = sum(1 for ft in fact_types if ft == fact_type)
        logger.debug(f"[ANN] Querying fact_type={fact_type}: {seed_count} seeds")
        ft_rows = await conn.fetch(
            f"""
            SELECT s.unit_id       AS from_id,
                   n.id::text      AS to_id,
                   n.similarity
            FROM _ann_seeds s
            CROSS JOIN LATERAL (
                SELECT mu.id,
                       1 - (mu.embedding <=> s.emb_text::vector) AS similarity
                FROM {fq_table("memory_units")} mu
                WHERE mu.bank_id = $1
                  AND mu.fact_type = $2
                  AND mu.embedding IS NOT NULL
                ORDER BY mu.embedding <=> s.emb_text::vector
                LIMIT $3
            ) n
            WHERE s.fact_type = $2
            """,
            bank_id,
            fact_type,
            top_k,
            timeout=300,  # ANN on large banks can take minutes
        )
        logger.debug(f"[ANN] fact_type={fact_type}: {len(ft_rows)} rows in {time_mod.time() - t_query:.3f}s")
        rows.extend(ft_rows)

    # Clean up temp table (no ON COMMIT DROP since we're not in a transaction)
    await conn.execute("DROP TABLE IF EXISTS _ann_seeds")

    # Reset ef_search to default so the pooled connection doesn't affect recall queries
    await conn.execute("RESET hnsw.ef_search")

    for row in rows:
        sim = float(min(1.0, max(0.0, row["similarity"])))
        if sim >= threshold:
            links.append((row["from_id"], row["to_id"], "semantic", sim, None))

    _log(
        log_buffer,
        f"      [8.1] ANN search (Phase 1): {len(unit_ids)} units → {len(links)} links in {time_mod.time() - ann_start:.3f}s",
    )

    return links


def compute_semantic_links_within_batch(
    unit_ids: list[str],
    embeddings: list[list[float]],
    top_k: int = 50,
    threshold: float = 0.7,
) -> list[tuple]:
    """
    Compute semantic links between units within the same batch (no DB needed).

    Uses numpy dot product on embeddings already in memory — instant.

    Args:
        unit_ids: Unit IDs (real IDs from insert_facts_batch)
        embeddings: Embedding vectors
        top_k: Max neighbors per unit
        threshold: Minimum cosine similarity

    Returns:
        List of (from_id, to_id, "semantic", similarity, None) tuples
    """
    if len(unit_ids) < 2:
        return []

    import numpy as np

    links = []
    new_embeddings_matrix = np.array(embeddings)

    for i, unit_id in enumerate(unit_ids):
        other_indices = [j for j in range(len(unit_ids)) if j != i]
        if not other_indices:
            continue

        other_embeddings = new_embeddings_matrix[other_indices]
        similarities = np.dot(other_embeddings, new_embeddings_matrix[i])

        above_threshold = np.where(similarities >= threshold)[0]
        if len(above_threshold) > 0:
            sorted_local_indices = above_threshold[np.argsort(-similarities[above_threshold])][:top_k]
            for local_idx in sorted_local_indices:
                other_idx = other_indices[local_idx]
                other_id = unit_ids[other_idx]
                similarity = float(min(1.0, max(0.0, similarities[local_idx])))
                links.append((unit_id, other_id, "semantic", similarity, None))

    return links


async def create_semantic_links_batch(
    conn,
    bank_id: str,
    unit_ids: list[str],
    embeddings: list[list[float]],
    top_k: int = 50,
    threshold: float = 0.7,
    log_buffer: list[str] = None,
    pre_computed_ann_links: list[tuple] | None = None,
) -> int:
    """
    Phase 2: Create semantic links (within-batch + pre-computed ANN results).

    Within-batch similarities are computed in Python (numpy, instant).
    ANN results from Phase 1 are passed in via pre_computed_ann_links and
    inserted alongside the within-batch links.

    Args:
        conn: Database connection (inside write transaction)
        bank_id: Bank identifier
        unit_ids: Real unit IDs (from insert_facts_batch)
        embeddings: Embedding vectors
        top_k: Max neighbors per unit
        threshold: Minimum cosine similarity
        log_buffer: Optional logging buffer
        pre_computed_ann_links: ANN results from Phase 1 (already remapped to real IDs)

    Returns:
        Number of semantic links created
    """
    if not unit_ids or not embeddings:
        return 0

    try:
        import time as time_mod

        all_links = []

        # Within-batch similarities (numpy, no DB)
        batch_start = time_mod.time()
        within_batch_links = compute_semantic_links_within_batch(unit_ids, embeddings, top_k, threshold)
        all_links.extend(within_batch_links)
        _log(
            log_buffer,
            f"      [8.1] Within-batch semantic: {len(within_batch_links)} links in {time_mod.time() - batch_start:.3f}s",
        )

        # Add pre-computed ANN links from Phase 1
        if pre_computed_ann_links:
            all_links.extend(pre_computed_ann_links)
            _log(
                log_buffer,
                f"      [8.2] Pre-computed ANN: {len(pre_computed_ann_links)} links",
            )

        if all_links:
            insert_start = time_mod.time()
            await _bulk_insert_links(conn, all_links, bank_id=bank_id)
            _log(
                log_buffer, f"      [8.3] Insert {len(all_links)} semantic links: {time_mod.time() - insert_start:.3f}s"
            )

        return len(all_links)

    except Exception as e:
        logger.error(f"Failed to create semantic links: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


async def insert_entity_links_batch(conn, links: list[EntityLink], bank_id: str, chunk_size: int = 5000):
    """
    Insert entity links into memory_links via sorted bulk INSERT FROM unnest().

    Args:
        conn: Database connection
        links: List of EntityLink objects
        bank_id: Bank identifier (stored directly on memory_links for fast filtering)
        chunk_size: Number of rows per INSERT chunk (default 5000)
    """
    if not links:
        return

    import time as time_mod

    total_start = time_mod.time()
    tuples = [(link.from_unit_id, link.to_unit_id, link.link_type, link.weight, link.entity_id) for link in links]
    await _bulk_insert_links(conn, tuples, bank_id=bank_id, chunk_size=chunk_size)
    logger.debug(
        f"      [9.TOTAL] Entity links batch insert ({len(tuples)} rows): {time_mod.time() - total_start:.3f}s"
    )


async def create_causal_links_batch(
    conn,
    bank_id: str,
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
            - relation_type: "caused_by"
            - strength: Float in [0.0, 1.0] representing relationship strength

    Returns:
        Number of causal links created

    Causal link type:
    - "caused_by": This fact was caused by the target fact
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

                # Validate relation_type - only "caused_by" is supported (DB constraint)
                valid_types = {"caused_by"}
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
            await _bulk_insert_links(conn, links, bank_id=bank_id, skip_exists_check=True)
            logger.debug(f"      [10.1] Insert {len(links)} causal links: {time_mod.time() - insert_start:.3f}s")

        return len(links)

    except Exception as e:
        logger.error(f"Failed to create causal links: {str(e)}")
        import traceback

        traceback.print_exc()
        raise
