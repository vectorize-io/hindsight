"""Graph-shaped reads and the link-maintenance passes, in SQL.

Everything here is a query over the *joins* around `memory_units` rather than
over the memories themselves: `unit_entities` (which entities a memory mentions)
and `memory_links` (memory-to-memory temporal/semantic/causal edges).

Two groups of callers:

* **The graph view.** :func:`graph_units`, :func:`graph_entity_rows` and
  :func:`graph_direct_links` return raw rows; the engine still owns the
  filtering, the observation inheritance, the derived entity edges, the
  colouring and the response assembly. These functions answer only "which
  memories", "which entity postings" and "which stored edges".
* **The graph-maintenance job.** :func:`enqueue_relink_victims` runs inside the
  delete transaction; :func:`relink_pass`, :func:`prune_orphan_entities` and
  :func:`prune_stale_cooccurrences` are the three reconciliation passes the job
  drives. The job keeps the orchestration (pass ordering, the deadlock retry
  around the sweeps, the timing log); each function here does the pass's work.

:func:`entity_memory_counts` and :func:`entities_for_units` are the two entity
postings reads that are not part of the graph view but read the same join table.

A store whose links travel inside the memory has nothing to relink and no join
table to sweep, which is why these are methods on the interface at all: it
answers them with zeroes rather than with SQL.
"""

from __future__ import annotations

import logging
import uuid as uuid_module
from collections.abc import Callable
from typing import Any

from ...db.base import DatabaseConnection
from ...retain.link_utils import (
    MAX_TEMPORAL_LINKS_PER_UNIT,
    _bulk_insert_links,
    _normalize_datetime,
    compute_semantic_links_ann,
)

logger = logging.getLogger(__name__)

# Mirrors the ``top_k`` default in ``compute_semantic_links_ann`` at retain
# time. If you change one, change the other — otherwise victims would either
# never reach the cap (probe returns less than the cap) or stay perpetually
# under it (cap is higher than retain creates).
MAX_SEMANTIC_LINKS_PER_UNIT = 50

# Worker fetches this many rows per relink-loop iteration. Bounds
# per-iteration probe/insert latency so a 10k-row backlog doesn't hold a
# worker slot for minutes. Chosen so the typical iteration runs in well
# under 1s.
_DRAIN_BATCH_SIZE = 50

# Defensive guard against runaway relink loops — at _DRAIN_BATCH_SIZE units per
# iteration that's 500k targets, far beyond any realistic single-bank backlog.
_RELINK_ITERATION_CAP = 10000

# Cap at 10k edges — the UI can't usefully render more, and uncapped queries
# on highly-connected graphs (e.g. 1000 nodes with 500k+ edges) are too slow.
_GRAPH_MAX_EDGES = 10000

# Columns the graph view renders: nodes take id/text/date/context/entities,
# the table rows take the rest, and `source_memory_ids` is what lets the caller
# inherit an observation's links and entities from the facts behind it.
_GRAPH_UNIT_COLUMNS = (
    "id, text, event_date, context, occurred_start, occurred_end, mentioned_at, "
    "document_id, chunk_id, fact_type, tags, created_at, proof_count, source_memory_ids"
)

# DataAccessOps is stateless, so one shared instance is enough. Everything in this
# package is Postgres by construction — it is the Postgres store's SQL — so the
# functions that take a bare `conn` (no backend, no ops passed) resolve the
# Postgres ops directly rather than trying to sniff a dialect off the connection,
# which a raw asyncpg connection cannot answer.
_PG_OPS: Any = None


def _ops_for(conn: DatabaseConnection) -> Any:
    """The Postgres ``DataAccessOps`` — this whole module is the Postgres store."""
    global _PG_OPS
    if _PG_OPS is None:
        from ...db import create_data_access_ops

        _PG_OPS = create_data_access_ops("postgresql")
    return _PG_OPS


def _as_uuids(unit_ids: list) -> list:
    """Coerce a mixed list of uuid strings / UUIDs to UUIDs for a ``uuid[]`` bind."""
    return [uuid_module.UUID(uid) if isinstance(uid, str) else uid for uid in unit_ids]


# ---------------------------------------------------------------- graph view


async def graph_units(
    *,
    conn: DatabaseConnection,
    fq_table: Callable[[str], str],
    bank_id: str,
    unit_ids: list[str] | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Memory nodes for the graph view.

    ``unit_ids=None`` returns the bank's units, newest first, up to ``limit``.
    A list returns exactly those units (no ``limit``: an explicit id set is
    already bounded, and truncating it would silently drop requested nodes).
    """
    conditions: list[str] = []
    params: list[Any] = []

    if bank_id:
        params.append(bank_id)
        conditions.append(f"bank_id = ${len(params)}")

    if unit_ids is not None:
        if not unit_ids:
            return []
        params.append(_as_uuids(unit_ids))
        conditions.append(f"id = ANY(${len(params)}::uuid[])")
        limit_clause = ""
    else:
        params.append(limit)
        limit_clause = f"LIMIT ${len(params)}"

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

    rows = await conn.fetch(
        f"""
        SELECT {_GRAPH_UNIT_COLUMNS}
        FROM {fq_table("memory_units")}
        {where_clause}
        ORDER BY mentioned_at DESC NULLS LAST, event_date DESC
        {limit_clause}
        """,
        *params,
    )
    return [dict(row) for row in rows]


async def graph_entity_rows(
    *,
    conn: DatabaseConnection,
    fq_table: Callable[[str], str],
    bank_id: str,
    unit_ids: list[str],
) -> list[dict[str, Any]]:
    """``(unit_id, entity_id, canonical_name)`` rows for the graph view's entity edges.

    Direct `unit_entities` postings only. An observation's entities are inherited
    from its source memories by the caller, which is why the ids it passes here
    are the visible units *plus* their source memories.

    Scoped by unit id rather than by bank: the ids already came from a
    bank-scoped :func:`graph_units`, and `unit_entities` carries no bank column.
    """
    if not unit_ids:
        return []

    rows = await conn.fetch(
        f"""
        SELECT ue.unit_id, e.id AS entity_id, e.canonical_name
        FROM {fq_table("unit_entities")} ue
        JOIN {fq_table("entities")} e ON ue.entity_id = e.id
        WHERE ue.unit_id = ANY($1::uuid[])
        ORDER BY ue.unit_id
        """,
        _as_uuids(unit_ids),
    )
    return [dict(row) for row in rows]


async def graph_direct_links(
    *,
    conn: DatabaseConnection,
    fq_table: Callable[[str], str],
    bank_id: str,
    unit_ids: list[str],
) -> list[dict[str, Any]]:
    """Memory-to-memory edges with *both* endpoints in ``unit_ids``.

    Entity edges are derived by the caller from `unit_entities` so we don't
    materialize them in `memory_links` anymore (dropped in migration
    e9b2c7d1f3a4) — no link_type filter is needed. ``entity_name`` is selected as
    NULL so the row shape matches the derived edges the caller mixes these with.

    Pass the visible units *and* the source memories they inherit from: the
    caller copies a source memory's links onto the observations built on it.
    """
    if not unit_ids:
        return []

    rows = await conn.fetch(
        f"""
        SELECT ml.from_unit_id,
               ml.to_unit_id,
               ml.link_type,
               ml.weight,
               NULL::text AS entity_name
        FROM {fq_table("memory_links")} ml
        WHERE ml.from_unit_id = ANY($1::uuid[])
          AND ml.to_unit_id = ANY($1::uuid[])
        ORDER BY ml.weight DESC NULLS LAST
        LIMIT $2
        """,
        _as_uuids(unit_ids),
        _GRAPH_MAX_EDGES,
    )
    return [dict(row) for row in rows]


# ------------------------------------------------------------ entity postings


async def entity_memory_counts(
    *,
    conn: DatabaseConnection,
    fq_table: Callable[[str], str],
    bank_id: str,
    entity_ids: list[str] | None = None,
) -> dict[str, int]:
    """Live memory count per entity id, for the entities in ``bank_id``.

    The GROUP BY is what makes this an orphan test: an entity with no surviving
    `unit_entities` row produces no group, so it is simply absent from the
    result rather than present with a zero.

    Scoped through ``memory_units.bank_id`` — `unit_entities` has no bank column,
    and joining is what keeps the count to *live* memories (deleted units take
    their postings with them via ON DELETE CASCADE).
    """
    params: list[Any] = [bank_id]
    entity_filter = ""
    if entity_ids is not None:
        if not entity_ids:
            return {}
        params.append(_as_uuids(entity_ids))
        entity_filter = f"AND ue.entity_id = ANY(${len(params)}::uuid[])"

    rows = await conn.fetch(
        f"""
        SELECT ue.entity_id, COUNT(*) AS n
        FROM {fq_table("unit_entities")} ue
        JOIN {fq_table("memory_units")} mu ON mu.id = ue.unit_id
        WHERE mu.bank_id = $1
        {entity_filter}
        GROUP BY ue.entity_id
        """,
        *params,
    )
    return {str(row["entity_id"]): int(row["n"]) for row in rows}


def _entity_rows_for_units_sql(
    fq_table: Callable[[str], str],
    ops: Any,
    unit_ids_placeholder: int,
) -> str:
    """SQL SELECT producing ``(unit_id, entity_id, canonical_name)`` rows for
    the given unit IDs.

    Direct rows come from ``unit_entities``. Observations rarely carry
    direct rows there; their entity association lives transitively through
    their source memories (``source_memory_ids`` on PG, the
    ``observation_sources`` junction on Oracle). When an observation has
    no direct entity rows the SELECT inherits its source memories'
    entities, so the result is the same set callers would get from
    ``get_memory_unit``.

    ``unit_ids_placeholder`` is the 1-based parameter index that holds the
    ``uuid[]`` of unit IDs. The placeholder is referenced twice — both
    sides of the UNION need it — so callers should not reuse the slot.
    """
    ue = fq_table("unit_entities")
    ents = fq_table("entities")
    mu = fq_table("memory_units")
    p = unit_ids_placeholder

    direct = (
        f"SELECT ue.unit_id, e.id AS entity_id, e.canonical_name "
        f"FROM {ue} ue "
        f"JOIN {ents} e ON e.id = ue.entity_id "
        f"WHERE ue.unit_id = ANY(${p}::uuid[])"
    )

    if ops.uses_observation_sources_table:
        os_t = fq_table("observation_sources")
        inherited = (
            f"SELECT os.observation_id AS unit_id, e.id AS entity_id, e.canonical_name "
            f"FROM {os_t} os "
            f"JOIN {ue} src_ue ON src_ue.unit_id = os.source_id "
            f"JOIN {ents} e ON e.id = src_ue.entity_id "
            f"WHERE os.observation_id = ANY(${p}::uuid[]) "
            f"AND NOT EXISTS (SELECT 1 FROM {ue} d WHERE d.unit_id = os.observation_id)"
        )
    else:
        inherited = (
            f"SELECT obs.id AS unit_id, e.id AS entity_id, e.canonical_name "
            f"FROM {mu} obs "
            f"CROSS JOIN LATERAL unnest(obs.source_memory_ids) AS src_id "
            f"JOIN {ue} src_ue ON src_ue.unit_id = src_id "
            f"JOIN {ents} e ON e.id = src_ue.entity_id "
            f"WHERE obs.id = ANY(${p}::uuid[]) "
            f"AND obs.fact_type = 'observation' "
            f"AND obs.source_memory_ids IS NOT NULL "
            f"AND NOT EXISTS (SELECT 1 FROM {ue} d WHERE d.unit_id = obs.id)"
        )

    return f"({direct}) UNION ({inherited})"


async def entities_for_units(
    *,
    conn: DatabaseConnection,
    fq_table: Callable[[str], str],
    bank_id: str,
    unit_ids: list[str],
) -> dict[str, list[str]]:
    """The entity ids each unit carries, keyed by unit id.

    Observations inherit their source memories' entities when they carry no
    direct postings of their own — see :func:`_entity_rows_for_units_sql`. Units
    with no entities are absent rather than mapped to an empty list.
    """
    if not unit_ids:
        return {}

    rows = await conn.fetch(
        _entity_rows_for_units_sql(fq_table, _ops_for(conn), unit_ids_placeholder=1),
        _as_uuids(unit_ids),
    )

    # UNION already de-duplicates whole rows, but a unit can reach the same
    # entity through more than one source memory, so dedupe per unit while
    # preserving the order the rows arrived in.
    by_unit: dict[str, list[str]] = {}
    for row in rows:
        unit_key = str(row["unit_id"])
        entity_id = str(row["entity_id"])
        bucket = by_unit.setdefault(unit_key, [])
        if entity_id not in bucket:
            bucket.append(entity_id)
    return by_unit


async def entity_map_for_units(
    *,
    conn: DatabaseConnection,
    fq_table: Callable[[str], str],
    bank_id: str,
    unit_ids: list[str],
) -> dict[str, list[dict[str, str]]]:
    """``{unit_id: [{entity_id, canonical_name}]}`` — the recall/curation shape.

    The named twin of :func:`entities_for_units`: recall renders the entity name
    on each fact, so it needs the label, not just the id. Observation-via-source
    inheritance and the per-unit dedupe are identical.
    """
    if not unit_ids:
        return {}

    rows = await conn.fetch(
        _entity_rows_for_units_sql(fq_table, _ops_for(conn), unit_ids_placeholder=1),
        _as_uuids(unit_ids),
    )
    by_unit: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        unit_key = str(row["unit_id"])
        entity_id = str(row["entity_id"])
        bucket = by_unit.setdefault(unit_key, [])
        if not any(existing["entity_id"] == entity_id for existing in bucket):
            bucket.append({"entity_id": entity_id, "canonical_name": row["canonical_name"]})
    return by_unit


# --------------------------------------------------------------- maintenance


async def enqueue_relink_victims(
    *,
    conn: DatabaseConnection,
    fq_table: Callable[[str], str],
    bank_id: str,
    deleted_unit_ids: list,
) -> int:
    """Enqueue surviving units whose outgoing temporal/semantic links pointed at
    ``deleted_unit_ids`` for later link top-up.

    Must run inside the same transaction that deletes the units, *before* the
    cascade fires — once the rows are gone, the join that finds the victims
    returns nothing.

    Args:
        conn: Database connection inside the active delete transaction.
        fq_table: Schema-qualifying table-name resolver.
        bank_id: Bank owning the deleted units.
        deleted_unit_ids: Memory_unit IDs about to be (or being) deleted.

    Returns:
        Number of distinct victim units enqueued (after dedup against rows
        already in the queue).
    """
    if not deleted_unit_ids:
        return 0

    ops = _ops_for(conn)
    deleted_uuids = _as_uuids(deleted_unit_ids)
    deleted_str_set = {str(uid) for uid in deleted_uuids}

    # Find units (other than the ones being deleted) that have an outgoing
    # temporal/semantic link pointing at a doomed unit. Entity links are
    # intentionally excluded — they're scheduled for removal and would only
    # add noise to the recompute job.
    victim_rows = await conn.fetch(
        f"""
        SELECT DISTINCT from_unit_id
        FROM {fq_table("memory_links")}
        WHERE to_unit_id = ANY($1::uuid[])
          AND bank_id = $2
          AND link_type IN ('temporal', 'semantic')
        """,
        deleted_uuids,
        bank_id,
    )

    victim_ids = [row["from_unit_id"] for row in victim_rows if str(row["from_unit_id"]) not in deleted_str_set]

    if not victim_ids:
        return 0

    await ops.enqueue_graph_maintenance(
        conn,
        fq_table("graph_maintenance_queue"),
        bank_id,
        victim_ids,
    )

    logger.debug(
        f"[GRAPH_MAINT] Enqueued {len(victim_ids)} relink victims in "
        f"bank={bank_id} (deleted {len(deleted_unit_ids)} units)"
    )
    return len(victim_ids)


async def relink_pass(
    *,
    backend: Any,
    fq_table: Callable[[str], str],
    bank_id: str,
    config: Any,
) -> dict:
    """Drain ``graph_maintenance_queue`` for ``bank_id``, topping up lost links.

    Per-iteration loop: claim → top up → commit. We rely on submit-time
    dedup to keep at most one job per bank running, so no need for
    SKIP LOCKED.

    Takes ``backend`` rather than a connection because the loop spans several
    transactions — one per claimed batch, plus a separate connection for the ANN
    probe — so it has to acquire its own.

    ``config`` is the caller's resolved configuration. The Postgres pass takes
    its caps from retain's link_utils (so relink and retain agree on what "full"
    means) and never reads it; it is accepted so a store that *does* tune its
    relinking gets it.

    Returns:
        ``{"relink_units_processed": int, "relink_links_added": int}``.
    """
    del config  # accepted for symmetry with stores that tune their own relinking
    ops = backend.ops

    units_processed = 0
    links_added = 0
    iterations = 0
    while True:
        from ...memory_engine import acquire_with_retry

        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                unit_ids = await ops.claim_graph_maintenance_batch(
                    conn,
                    fq_table("graph_maintenance_queue"),
                    bank_id,
                    _DRAIN_BATCH_SIZE,
                )
                if not unit_ids:
                    break

                links_added += await _relink_batch(conn, fq_table, bank_id, unit_ids, ops, backend)

        units_processed += len(unit_ids)
        iterations += 1

        if iterations > _RELINK_ITERATION_CAP:
            # Defensive guard against runaway loops — at 50 units/iter that's
            # 500k targets, far beyond any realistic single-bank backlog.
            logger.error(
                f"[GRAPH_MAINT] bank={bank_id} hit iteration cap ({iterations}); aborting relink "
                f"(units_processed={units_processed}, links_added={links_added})"
            )
            break

    return {"relink_units_processed": units_processed, "relink_links_added": links_added}


async def _relink_batch(
    conn: DatabaseConnection,
    fq_table: Callable[[str], str],
    bank_id: str,
    victim_ids: list[str],
    ops: Any,
    backend: Any,
) -> int:
    """Top up temporal/semantic links for a batch of victim units. Returns rows inserted."""
    # Load each victim's metadata. Victims whose units were deleted between
    # enqueue and now silently drop out — exactly the no-op behaviour we want
    # for stale queue rows.
    victim_uuids = [uuid_module.UUID(vid) for vid in victim_ids]
    victim_rows = await conn.fetch(
        f"""
        SELECT id::text AS id, event_date, fact_type, embedding::text AS embedding
        FROM {fq_table("memory_units")}
        WHERE id = ANY($1::uuid[])
          AND bank_id = $2
          AND fact_type IN ('experience', 'world')
        """,
        victim_uuids,
        bank_id,
    )

    if not victim_rows:
        return 0

    alive_uuids = [uuid_module.UUID(row["id"]) for row in victim_rows]

    # Count current outgoing temporal/semantic links per victim so we only
    # probe for the ones genuinely below cap. Saves the bulk of the work when
    # most victims still have plenty of links.
    count_rows = await conn.fetch(
        f"""
        SELECT from_unit_id, link_type, COUNT(*) AS cnt
        FROM {fq_table("memory_links")}
        WHERE from_unit_id = ANY($1::uuid[])
          AND bank_id = $2
          AND link_type IN ('temporal', 'semantic')
        GROUP BY from_unit_id, link_type
        """,
        alive_uuids,
        bank_id,
    )
    counts: dict[tuple[str, str], int] = {}
    for row in count_rows:
        counts[(str(row["from_unit_id"]), row["link_type"])] = int(row["cnt"])

    # --- Temporal top-up ---
    temporal_needs = [r for r in victim_rows if counts.get((r["id"], "temporal"), 0) < MAX_TEMPORAL_LINKS_PER_UNIT]
    new_links: list[tuple] = []

    if temporal_needs:
        lateral_unit_ids = [uuid_module.UUID(r["id"]) for r in temporal_needs if r["event_date"] is not None]
        lateral_event_dates = [
            _normalize_datetime(r["event_date"]) for r in temporal_needs if r["event_date"] is not None
        ]
        lateral_fact_types = [r["fact_type"] for r in temporal_needs if r["event_date"] is not None]

        if lateral_unit_ids:
            rows = await ops.fetch_temporal_neighbors(
                conn,
                fq_table("memory_units"),
                bank_id,
                lateral_unit_ids,
                lateral_event_dates,
                lateral_fact_types,
                MAX_TEMPORAL_LINKS_PER_UNIT,
            )
            for row in rows:
                time_diff_h = float(row["time_diff_hours"])
                # Mirror the 24h window enforced at retain time. The bidirectional
                # index scan returns the K closest neighbours regardless of
                # window, so we filter here.
                if time_diff_h > 24:
                    continue
                weight = max(0.3, 1.0 - (time_diff_h / 24))
                new_links.append((row["from_id"], str(row["id"]), "temporal", weight, None))

    # --- Semantic top-up ---
    # ANN must run on its own connection: it opens a nested transaction with
    # SET LOCAL hnsw.ef_search + CREATE TEMP TABLE ON COMMIT DROP, and nesting
    # that inside our current write transaction would commit our writes early.
    semantic_needs = [
        r
        for r in victim_rows
        if counts.get((r["id"], "semantic"), 0) < MAX_SEMANTIC_LINKS_PER_UNIT and r["embedding"] is not None
    ]
    if semantic_needs:
        from ...memory_engine import acquire_with_retry

        seed_ids = [r["id"] for r in semantic_needs]
        seed_embs = [r["embedding"] for r in semantic_needs]
        seed_ftypes = [r["fact_type"] for r in semantic_needs]
        async with acquire_with_retry(backend) as ann_conn:
            try:
                ann_links = await compute_semantic_links_ann(
                    ann_conn,
                    bank_id,
                    seed_ids,
                    seed_embs,
                    fact_types=seed_ftypes,
                )
                # Strip self-links (rare but possible because the ANN probe
                # has no exclude list — see the comment in compute_semantic_links_ann).
                ann_links = [lnk for lnk in ann_links if lnk[0] != lnk[1]]
                new_links.extend(ann_links)
            except Exception as e:
                # ANN uses PG-specific HNSW syntax; on dialects/configs where
                # it isn't available we still want the temporal top-up to land.
                logger.warning(f"[GRAPH_MAINT] Semantic top-up failed for bank={bank_id}: {type(e).__name__}: {e}")

    if not new_links:
        return 0

    await _bulk_insert_links(
        conn,
        new_links,
        bank_id=bank_id,
        skip_exists_check=False,
        ops=ops,
    )
    return len(new_links)


async def prune_orphan_entities(
    *,
    conn: DatabaseConnection,
    fq_table: Callable[[str], str],
    bank_id: str,
) -> int:
    """Delete ``entities`` rows in the bank with no remaining ``unit_entities``
    references. Returns the number pruned.

    FK ON DELETE CASCADE on ``entity_cooccurrences`` then removes any
    cooccurrence row pointing at the pruned entities — which is why this runs
    before :func:`prune_stale_cooccurrences` rather than after.

    A bank-wide single-statement delete, cheap when there's nothing to do. It is
    idempotent (rerunning only deletes what is still orphaned), so the caller is
    free to retry the whole transaction on deadlock.
    """
    ops = _ops_for(conn)
    return await ops.prune_orphan_entities(
        conn,
        fq_table("entities"),
        fq_table("unit_entities"),
        bank_id,
    )


async def prune_stale_cooccurrences(
    *,
    conn: DatabaseConnection,
    fq_table: Callable[[str], str],
    bank_id: str,
) -> int:
    """Delete cooccurrence rows no current memory witnesses. Returns the count.

    Defensive sweep for rows where both endpoints still exist but no current
    memory_unit references both of them — the cooccurrence was real at the time
    it was recorded, but every unit that witnessed it has since been deleted.
    :func:`prune_orphan_entities` cascades the *missing-entity* case via FK; this
    pass catches the *stale-count* case it cannot see.

    Like the orphan prune, a bank-wide idempotent sweep backed by indexes, so
    it's cheap when there's nothing to do and safe for the caller to retry.
    """
    ops = _ops_for(conn)
    return await ops.prune_stale_cooccurrences(
        conn,
        fq_table("entity_cooccurrences"),
        fq_table("unit_entities"),
        fq_table("entities"),
        bank_id,
    )


__all__ = [
    "MAX_SEMANTIC_LINKS_PER_UNIT",
    "enqueue_relink_victims",
    "entities_for_units",
    "entity_map_for_units",
    "entity_memory_counts",
    "graph_direct_links",
    "graph_entity_rows",
    "graph_units",
    "prune_orphan_entities",
    "prune_stale_cooccurrences",
    "relink_pass",
]
