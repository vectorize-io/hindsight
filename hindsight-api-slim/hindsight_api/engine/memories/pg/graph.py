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
* **The graph-maintenance job.** :func:`enqueue_relink_victims` and
  :func:`enqueue_entity_prune_candidates` run inside the delete transaction;
  :func:`relink_pass` and :func:`entity_prune_pass` are the two drain loops the
  job drives. The job keeps the orchestration (pass ordering, the time budget,
  the timing log); each function here does the pass's work.

:func:`entity_memory_counts` and :func:`entities_for_units` are the two entity
postings reads that are not part of the graph view but read the same join table.

A store whose links travel inside the memory has nothing to relink and no join
table to sweep, which is why these are methods on the interface at all: it
answers them with zeroes rather than with SQL.
"""

from __future__ import annotations

import logging
import time
import uuid as uuid_module
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ....config import get_config
from ...db.base import DatabaseConnection
from ...retain.link_utils import (
    MAX_TEMPORAL_LINKS_PER_UNIT,
    _bulk_insert_links,
    _normalize_datetime,
    compute_semantic_links_ann,
)
from ..base import EntityPrunePassResult, RelinkPassResult

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

# Candidate entities claimed per entity-prune iteration.
#
# The binding cost is the cooccurrence prune: a candidate drags in every
# cooccurrence pair it appears in, and the statement builds a set of currently
# live pairs (#3367) seeded from those candidates' units to judge them against.
# Measured on a deliberately dense fixture (100k entities, 1.5M unit_entities,
# 2.86M cooccurrences, endpoints holding 150-400 postings each):
#
#     batch  50 →  16-65ms     <- here
#     batch 500 →  265s        <- the planner flips to a per-row plan and the
#                                 statement blows the 60s command timeout
#
# So the batch size, not the bank size, is what has to stay bounded — and it has
# to stay small enough that the planner keeps choosing the hash anti-join.
# Raising it trades away three orders of magnitude of margin; don't, without
# re-measuring against a bank with hub entities.
#
# Also stays under Oracle's 1000-element IN-list limit, since ops_oracle expands
# ``= ANY(...)`` into an explicit list.
_ENTITY_PRUNE_BATCH_SIZE = 50

# Deadlock retries per entity-prune batch. The batch is idempotent, so a retry
# only re-deletes what is still dead; a handful of attempts clears the
# contention window a concurrent retain opens.
_PRUNE_BATCH_MAX_RETRIES = 3

# Unit ids per candidate-lookup round-trip when enqueueing. Bounded by Oracle's
# 1000-element IN-list limit (ops_oracle expands ``= ANY(...)`` into a literal
# list), which a bulk delete would otherwise blow straight through.
_ENQUEUE_LOOKUP_CHUNK = 500

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


def _ops_for(conn: DatabaseConnection) -> Any:
    """The ``DataAccessOps`` matching the connection's SQL dialect.

    This is the SQL memories store, and SQL means Postgres *or* Oracle — the two
    speak different dialects (Oracle inherits entity links through the
    ``observation_sources`` junction, Postgres through ``source_memory_ids``
    arrays), so the ops must follow the connection rather than assume Postgres.
    The ops go by ``conn.backend_type`` — the connection objects carry the dialect
    but not the backend's ``ops`` handle, so resolve through the per-dialect cache
    of ``create_data_access_ops`` (a dict lookup after the first call, and the same
    instance the backend holds). The default covers callers that hand in a bare
    asyncpg connection with no dialect to report.
    """
    from ...db import create_data_access_ops

    return create_data_access_ops(getattr(conn, "backend_type", "postgresql"))


def _as_uuids(unit_ids: list) -> list:
    """Coerce a mixed list of uuid strings / UUIDs to UUIDs for a ``uuid[]`` bind."""
    return [uuid_module.UUID(uid) if isinstance(uid, str) else uid for uid in unit_ids]


# ---------------------------------------------------------------- graph view


def _observations_via_source_match(
    fq_table: Callable[[str], str],
    ops: Any,
    source_column: str,
    source_placeholder: int,
    bank_placeholder: int | None,
) -> str:
    """A predicate matching observations whose *sources* satisfy ``<col> = $n``.

    Observations carry no `document_id` / `chunk_id` of their own; the link to a
    source row lives in `source_memory_ids` (native array) or the
    `observation_sources` junction, depending on the dialect.
    """
    if ops.uses_observation_sources_table:
        bank_clause = f" AND src.bank_id = ${bank_placeholder}" if bank_placeholder else ""
        return (
            f"id IN (SELECT os.observation_id "
            f"FROM {fq_table('observation_sources')} os "
            f"JOIN {fq_table('memory_units')} src ON src.id = os.source_id "
            f"WHERE src.{source_column} = ${source_placeholder}{bank_clause})"
        )
    bank_clause = f" AND bank_id = ${bank_placeholder}" if bank_placeholder else ""
    return (
        f"source_memory_ids && (SELECT array_agg(id) "
        f"FROM {fq_table('memory_units')} "
        f"WHERE {source_column} = ${source_placeholder}{bank_clause})"
    )


async def graph_units(
    *,
    conn: DatabaseConnection,
    fq_table: Callable[[str], str],
    bank_id: str | None = None,
    fact_type: str | None = None,
    search_query: str | None = None,
    document_id: str | None = None,
    chunk_id: str | None = None,
    tags: list[str] | None = None,
    tags_match: str = "all_strict",
    limit: int = 1000,
) -> dict[str, Any]:
    """Memory nodes for the graph view, plus the total matching count.

    Returns ``{"units": [...], "total": int}``: ``units`` is the page (newest
    first, capped at ``limit``); ``total`` is how many match the filters, which
    the UI shows alongside the page. ``document_id`` / ``chunk_id`` also match an
    observation whose *sources* carry them, since observations have neither of
    their own.
    """
    from ...search.tags import build_tags_where_clause_simple

    ops = _ops_for(conn)
    conditions: list[str] = []
    params: list[Any] = []

    bank_placeholder: int | None = None
    if bank_id:
        params.append(bank_id)
        bank_placeholder = len(params)
        conditions.append(f"bank_id = ${bank_placeholder}")

    if fact_type:
        params.append(fact_type)
        conditions.append(f"fact_type = ${len(params)}")

    if document_id:
        params.append(document_id)
        obs = _observations_via_source_match(fq_table, ops, "document_id", len(params), bank_placeholder)
        conditions.append(f"(document_id = ${len(params)} OR (fact_type = 'observation' AND {obs}))")

    if chunk_id:
        params.append(chunk_id)
        obs = _observations_via_source_match(fq_table, ops, "chunk_id", len(params), bank_placeholder)
        conditions.append(f"(chunk_id = ${len(params)} OR (fact_type = 'observation' AND {obs}))")

    if search_query:
        params.append(f"%{search_query}%")
        conditions.append(f"(text ILIKE ${len(params)} OR context ILIKE ${len(params)})")

    if tags:
        tag_clause = build_tags_where_clause_simple(tags, len(params) + 1, match=tags_match)
        if tag_clause:
            conditions.append(tag_clause.removeprefix("AND "))
            params.append(tags)
    elif tags_match == "exact":
        # Exact match with no tags is the "global" scope: rows carrying no tags at
        # all. (Other modes treat empty tags as "no filter".)
        conditions.append("(tags IS NULL OR tags = '{}')")

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

    total_row = await conn.fetchrow(
        f"SELECT COUNT(*) AS total FROM {fq_table('memory_units')} {where_clause}",
        *params,
    )
    total = total_row["total"] if total_row else 0

    params.append(limit)
    rows = await conn.fetch(
        f"""
        SELECT {_GRAPH_UNIT_COLUMNS}
        FROM {fq_table("memory_units")}
        {where_clause}
        ORDER BY mentioned_at DESC NULLS LAST, event_date DESC
        LIMIT ${len(params)}
        """,
        *params,
    )
    return {"units": [dict(row) for row in rows], "total": total}


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


async def resolve_entity_names(
    *,
    conn: DatabaseConnection,
    fq_table: Callable[[str], str],
    bank_id: str,
    entity_ids: list[str],
) -> dict[str, str]:
    """``{entity_id: canonical_name}`` for the given ids, scoped to ``bank_id``.

    The label half of :func:`entity_map_for_units`, for a backend that already
    carries a unit's entity ids on the recalled result: recall builds the
    unit->entity map from those ids and needs only the names, so this resolves the
    ``entities`` registry once without re-fetching any memory. Bank-scoped like the
    sibling registry reads (the ``entities`` table has a ``bank_id`` column).

    Ids that don't parse as UUIDs are dropped rather than raised — a malformed id
    on a store's result payload must not turn into a DB error mid-recall — and ids
    with no registry row (or in another bank) are simply absent from the result.
    """
    if not entity_ids:
        return {}
    # Bind ``uuid.UUID`` objects for the ``uuid[]`` param (repo convention, see
    # ``_as_uuids``), but coerce defensively: skip anything unparseable instead of
    # letting the whole resolve raise.
    uuids: list = []
    for raw in {str(e) for e in entity_ids}:
        try:
            uuids.append(uuid_module.UUID(raw))
        except (ValueError, AttributeError, TypeError):
            continue
    if not uuids:
        return {}
    rows = await conn.fetch(
        f"SELECT id, canonical_name FROM {fq_table('entities')} WHERE id = ANY($1::uuid[]) AND bank_id = $2",
        uuids,
        bank_id,
    )
    return {str(row["id"]): row["canonical_name"] for row in rows}


# --------------------------------------------------------------- maintenance


async def enqueue_relink_victims(
    *,
    conn: DatabaseConnection,
    fq_table: Callable[[str], str],
    bank_id: str,
    affected_unit_ids: list,
    include_affected_units: bool = False,
) -> int:
    """Enqueue surviving units whose outgoing temporal/semantic links pointed at
    ``affected_unit_ids`` for later link top-up.

    Must run inside the same transaction that drops those links, *before* the
    delete (or cascade) fires — once the rows are gone, the join that finds the
    victims returns nothing.

    Args:
        conn: Database connection inside the active transaction.
        fq_table: Schema-qualifying table-name resolver.
        bank_id: Bank owning the affected units.
        affected_unit_ids: Memory_unit IDs whose incident temporal/semantic links
            are about to be (or are being) removed.
        include_affected_units: Also enqueue ``affected_unit_ids`` themselves — for
            an edit that deletes a unit's links but leaves the unit live, so its own
            outgoing adjacency is rebuilt too. One combined insert keeps the queue's
            sorted lock ordering intact.

    Returns:
        Number of distinct victim units enqueued (after dedup against rows
        already in the queue).
    """
    if not affected_unit_ids:
        return 0

    ops = _ops_for(conn)
    affected_uuids = _as_uuids(affected_unit_ids)
    affected_str_set = {str(uid) for uid in affected_uuids}

    # Find units (other than the affected ones) that have an outgoing
    # temporal/semantic link pointing at an affected unit. Entity links are
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
        affected_uuids,
        bank_id,
    )

    victim_ids = {row["from_unit_id"] for row in victim_rows if str(row["from_unit_id"]) not in affected_str_set}
    if include_affected_units:
        victim_ids.update(affected_uuids)

    if not victim_ids:
        return 0

    await ops.enqueue_graph_maintenance(
        conn,
        fq_table("graph_maintenance_queue"),
        bank_id,
        list(victim_ids),
    )

    logger.debug(
        f"[GRAPH_MAINT] Enqueued {len(victim_ids)} relink victims in "
        f"bank={bank_id} ({len(affected_unit_ids)} units affected)"
    )
    return len(victim_ids)


async def relink_pass(
    *,
    backend: Any,
    fq_table: Callable[[str], str],
    bank_id: str,
    config: Any,
    deadline: float | None = None,
) -> RelinkPassResult:
    """Drain ``graph_maintenance_queue`` for ``bank_id``, topping up lost links.

    Per-iteration loop: claim → top up → commit. We rely on at most one job per
    bank running, so no need for SKIP LOCKED. Submit-time dedup alone does NOT
    give that — it only inspects 'pending' rows — so the guarantee comes from
    ``claim_tasks``, which refuses to claim a graph_maintenance row for a bank
    that already has one in flight (``graph_maintenance_bank_serialization_sql``,
    #3230). Without it these claims convoy: they lock queue rows ``FOR UPDATE``
    with no ``SKIP LOCKED``, so a second run blocks on the first while holding a
    worker slot.

    Takes ``backend`` rather than a connection because the loop spans several
    transactions — one per claimed batch, plus a separate connection for the ANN
    probe — so it has to acquire its own.

    ``config`` is the caller's resolved configuration. The Postgres pass takes
    its caps from retain's link_utils (so relink and retain agree on what "full"
    means) and never reads it; it is accepted so a store that *does* tune its
    relinking gets it.

    ``deadline`` is a ``time.monotonic()`` value past which no new batch is
    claimed. Each batch commits before the next is claimed, so stopping early
    keeps the work already done and leaves the rest queued for the next run.

    Returns:
        A :class:`RelinkPassResult`. ``queue_exhausted`` is False when the
        deadline (or the iteration cap) stopped the drain with rows still queued.
    """
    del config  # accepted for symmetry with stores that tune their own relinking
    ops = backend.ops

    units_processed = 0
    links_added = 0
    iterations = 0
    drained = True
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            drained = False
            break

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
            drained = False
            break

    return RelinkPassResult(
        units_processed=units_processed,
        links_added=links_added,
        queue_exhausted=drained,
    )


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
                    threshold=get_config().semantic_link_min_similarity,
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


async def enqueue_entity_prune_candidates(
    *,
    conn: DatabaseConnection,
    fq_table: Callable[[str], str],
    bank_id: str,
    affected_unit_ids: list,
) -> int:
    """Enqueue the entities ``affected_unit_ids`` reference as prune candidates.

    Must run inside the same transaction that removes those units (or their
    ``unit_entities`` rows), *before* the delete or cascade fires — afterwards
    there is no posting left to read the entity ids from, and the entity is
    stranded as an orphan nothing will ever look at again.

    Enqueueing an entity that turns out to still be referenced is free: the
    drain re-checks and keeps it. Over-enqueueing is always the safe direction.

    Returns:
        Number of candidate entities enqueued.
    """
    if not affected_unit_ids:
        return 0

    ops = _ops_for(conn)
    queue_table = fq_table("entity_maintenance_queue")
    ue_table = fq_table("unit_entities")
    unit_uuids = _as_uuids(list(affected_unit_ids))

    # Chunked because a bulk delete can hand in thousands of unit ids and the
    # lookup binds them with `= ANY(...)`, which ops_oracle expands into a
    # literal IN list — Oracle caps those at 1000 elements.
    enqueued = 0
    for start in range(0, len(unit_uuids), _ENQUEUE_LOOKUP_CHUNK):
        enqueued += await ops.enqueue_entity_maintenance(
            conn,
            queue_table,
            ue_table,
            bank_id,
            unit_uuids[start : start + _ENQUEUE_LOOKUP_CHUNK],
        )
    return enqueued


@dataclass
class _PruneBatch:
    """One entity-prune iteration's counters (avoids a bare tuple return)."""

    claimed: int
    orphan_entities_pruned: int
    stale_cooccurrences_pruned: int


async def entity_prune_pass(
    *,
    backend: Any,
    fq_table: Callable[[str], str],
    bank_id: str,
    deadline: float | None = None,
) -> EntityPrunePassResult:
    """Drain ``entity_maintenance_queue`` for ``bank_id``, pruning what died.

    Per-iteration loop: claim → prune → commit, mirroring :func:`relink_pass`.
    Each iteration does two deletes over the claimed batch:

    1. **Orphan entities** — candidates with no remaining ``unit_entities`` row.
       FK ON DELETE CASCADE on ``entity_cooccurrences`` takes their cooccurrence
       rows with them, which is why this runs first.
    2. **Stale cooccurrences** — pairs incident to a surviving candidate where
       both entities still exist but no current unit witnesses them together.
       The cooccurrence was real when recorded; every unit that saw it has since
       been deleted. The FK cascade above cannot see this case.

    Both deletes are scoped to the claimed batch. They used to be bank-wide
    statements re-run on every invocation — the orphan prune probing once per
    entity in the bank, the cooccurrence prune evaluating an INTERSECT per
    cooccurrence row in the bank — so their cost tracked the size of the bank
    rather than the size of the delete, and past a few million rows they could
    no longer finish inside asyncpg's command timeout. The job then failed on
    every run, forever, on exactly the banks that most needed it (#3222).

    Committing per batch is what makes the pass resumable: work already done
    stays done when ``deadline`` cuts the drain short or the task dies, and the
    next run picks up the remaining queue rows.

    Args:
        backend: Database backend — the loop spans a transaction per batch, so
            it acquires its own connections.
        fq_table: Schema-qualifier for table names.
        bank_id: Bank to drain.
        deadline: ``time.monotonic()`` value past which no new batch is claimed.
            ``None`` drains to empty.

    Returns:
        An :class:`EntityPrunePassResult`. ``queue_exhausted`` is False when the
        deadline stopped the drain with rows still queued.
    """
    from ...db_utils import retry_with_backoff
    from ...memory_engine import acquire_with_retry

    examined = 0
    orphans_pruned = 0
    stale_pruned = 0
    drained = True

    while True:
        if deadline is not None and time.monotonic() >= deadline:
            drained = False
            break

        async def _run_batch() -> _PruneBatch:
            async with acquire_with_retry(backend) as conn:
                async with conn.transaction():
                    ops = backend.ops
                    entity_ids = await ops.claim_entity_maintenance_batch(
                        conn,
                        fq_table("entity_maintenance_queue"),
                        bank_id,
                        _ENTITY_PRUNE_BATCH_SIZE,
                    )
                    if not entity_ids:
                        return _PruneBatch(claimed=0, orphan_entities_pruned=0, stale_cooccurrences_pruned=0)

                    orphaned = await ops.prune_orphan_entities(
                        conn,
                        fq_table("entities"),
                        fq_table("unit_entities"),
                        bank_id,
                        entity_ids,
                    )
                    # The orphan prune above cascades cooccurrences via FK. This
                    # second delete catches the *stale-count* case: both entities
                    # still exist but no current unit witnesses them together.
                    stale = await ops.prune_stale_cooccurrences(
                        conn,
                        fq_table("entity_cooccurrences"),
                        fq_table("unit_entities"),
                        entity_ids,
                    )
                    return _PruneBatch(
                        claimed=len(entity_ids),
                        orphan_entities_pruned=orphaned,
                        stale_cooccurrences_pruned=stale,
                    )

        # Retry the batch on deadlock. Both deletes take their row locks in the
        # same order the concurrent retain writers do (entity id for the entity
        # upsert, (entity_id_1, entity_id_2) for the cooccurrence upsert), so a
        # cycle should not form on Postgres at all; this stays as the backstop
        # for the paths that ordering can't cover — the FK cascade out of the
        # orphan prune, and Oracle, whose DELETE can't carry the ordered-lock
        # CTE. Both deletes are idempotent, so re-running the batch is safe.
        #
        # Deliberately narrower than the budget the bank-wide sweep used (8).
        # `retry_with_backoff` treats a TimeoutError as transient, which was
        # ruinous while the statement was O(bank): a sweep that could never
        # finish inside the command timeout was re-run nine times, burning ten
        # minutes of a worker slot per task attempt (#3222). A bounded batch
        # that times out is not slow work, it is a sick database — retry a few
        # times and let the failure surface.
        batch = await retry_with_backoff(_run_batch, max_retries=_PRUNE_BATCH_MAX_RETRIES)
        if batch.claimed == 0:
            break

        examined += batch.claimed
        orphans_pruned += batch.orphan_entities_pruned
        stale_pruned += batch.stale_cooccurrences_pruned

    return EntityPrunePassResult(
        entities_examined=examined,
        orphan_entities_pruned=orphans_pruned,
        stale_cooccurrences_pruned=stale_pruned,
        queue_exhausted=drained,
    )


__all__ = [
    "MAX_SEMANTIC_LINKS_PER_UNIT",
    "enqueue_entity_prune_candidates",
    "enqueue_relink_victims",
    "entities_for_units",
    "entity_map_for_units",
    "entity_memory_counts",
    "entity_prune_pass",
    "graph_direct_links",
    "graph_entity_rows",
    "graph_units",
    "relink_pass",
    "resolve_entity_names",
]
