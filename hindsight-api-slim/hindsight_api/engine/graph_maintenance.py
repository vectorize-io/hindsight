"""Async graph maintenance after document/unit deletes.

Three reconciliation passes run together on every worker invocation:

1. **Relink top-up.** Drain ``graph_maintenance_queue`` (units whose
   outgoing temporal/semantic links lost a neighbour to a delete). For
   each, count current outgoing links per type; if below cap, run the
   same probes retain uses (:func:`fetch_temporal_neighbors`,
   :func:`compute_semantic_links_ann`) and insert the missing links.
   ``bulk_insert_links`` has ``ON CONFLICT DO NOTHING`` on the uniqueness
   key, so we can re-probe freely and the DB de-dupes.

2. **Orphan entity prune.** Delete ``entities`` rows in the bank that no
   longer have any ``unit_entities`` references. FK ON DELETE CASCADE on
   ``entity_cooccurrences`` then removes any cooccurrence row pointing
   at the pruned entities.

3. **Stale cooccurrence prune.** Defensive sweep for cooccurrence rows
   where both endpoints still exist but no current memory_unit references
   both of them — the cooccurrence was real at the time it was recorded,
   but every unit that witnessed it has since been deleted.

All three passes run on every invocation. The queue is the only source
of work for pass 1; passes 2 and 3 are bank-wide sweeps backed by indexes
on ``entities(bank_id)`` and ``unit_entities(entity_id)``, so they're
cheap when there's nothing to do.

The worker dedupes on bank: a second job for the same bank is dropped
while one is pending. Once processing starts, a new job becomes the
*next* pending slot — so work enqueued during processing gets picked up
by the follow-up run.
"""

from __future__ import annotations

import logging
import time
import uuid as uuid_module
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..models import RequestContext
from .db.base import DatabaseConnection
from .retain.link_utils import (
    MAX_TEMPORAL_LINKS_PER_UNIT,
    _bulk_insert_links,
    _normalize_datetime,
    compute_semantic_links_ann,
)
from .schema import fq_table

if TYPE_CHECKING:
    from .memory_engine import MemoryEngine

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


@dataclass
class JobResult:
    """Counters surfaced to the worker dispatcher and operation result."""

    relink_units_processed: int = 0
    relink_links_added: int = 0
    orphan_entities_pruned: int = 0
    stale_cooccurrences_pruned: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "relink_units_processed": self.relink_units_processed,
            "relink_links_added": self.relink_links_added,
            "orphan_entities_pruned": self.orphan_entities_pruned,
            "stale_cooccurrences_pruned": self.stale_cooccurrences_pruned,
        }


async def enqueue_relink_victims(
    conn: DatabaseConnection,
    bank_id: str,
    deleted_unit_ids: list[str],
    ops: Any,
) -> int:
    """Enqueue surviving units whose outgoing temporal/semantic links pointed at
    ``deleted_unit_ids`` for later link top-up.

    Must run inside the same transaction that deletes the units, *before* the
    cascade fires — once the rows are gone, the join that finds the victims
    returns nothing.

    Args:
        conn: Database connection inside the active delete transaction.
        bank_id: Bank owning the deleted units.
        deleted_unit_ids: Memory_unit IDs about to be (or being) deleted.
        ops: ``DataAccessOps`` instance, supplies the dialect-specific
            bulk-insert path.

    Returns:
        Number of distinct victim units enqueued (after dedup against rows
        already in the queue).
    """
    if not deleted_unit_ids:
        return 0

    deleted_uuids = [uuid_module.UUID(uid) if isinstance(uid, str) else uid for uid in deleted_unit_ids]
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


async def run_graph_maintenance_job(
    memory_engine: "MemoryEngine",
    bank_id: str,
    request_context: RequestContext,
    operation_id: str | None = None,
) -> dict[str, int]:
    """Run all maintenance passes for ``bank_id`` until the relink queue is
    drained, then sweep entities and cooccurrences once.

    Returns:
        Per-pass counters from :class:`JobResult`.
    """
    del request_context  # accepted for symmetry with other run_*_job helpers
    backend = await memory_engine._get_backend()
    ops = backend.ops

    result = JobResult()
    job_start = time.time()

    # --- Pass 1: relink ---
    # Per-iteration loop: claim → top up → commit. We rely on submit-time
    # dedup to keep at most one job per bank running, so no need for
    # SKIP LOCKED.
    iterations = 0
    while True:
        from .memory_engine import acquire_with_retry

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

                result.relink_links_added += await _relink_batch(conn, bank_id, unit_ids, ops, backend)

        result.relink_units_processed += len(unit_ids)
        iterations += 1

        if iterations > 10000:
            # Defensive guard against runaway loops — at 50 units/iter that's
            # 500k targets, far beyond any realistic single-bank backlog.
            logger.error(
                f"[GRAPH_MAINT] bank={bank_id} hit iteration cap ({iterations}); aborting relink ({result.as_dict()})"
            )
            break

    # --- Pass 2 & 3: entity / cooccurrence sweeps ---
    # Bank-wide single-statement deletes. Cheap when there's nothing to do.
    from .memory_engine import acquire_with_retry

    async with acquire_with_retry(backend) as conn:
        async with conn.transaction():
            result.orphan_entities_pruned = await ops.prune_orphan_entities(
                conn,
                fq_table("entities"),
                fq_table("unit_entities"),
                bank_id,
            )
            # The orphan prune above cascades cooccurrences via FK. The
            # explicit cooccurrence pass below catches the *stale-count*
            # case: both entities still exist but no current unit witnesses
            # them together.
            result.stale_cooccurrences_pruned = await ops.prune_stale_cooccurrences(
                conn,
                fq_table("entity_cooccurrences"),
                fq_table("unit_entities"),
                fq_table("entities"),
                bank_id,
            )

    elapsed = time.time() - job_start
    logger.info(
        f"[GRAPH_MAINT] bank={bank_id} done: {result.as_dict()}, elapsed={elapsed:.2f}s, operation_id={operation_id}"
    )
    return result.as_dict()


async def _relink_batch(
    conn: DatabaseConnection,
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
        from .memory_engine import acquire_with_retry

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
