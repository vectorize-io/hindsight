"""Async graph maintenance after document/unit deletes.

Three reconciliation passes run together on every worker invocation:

1. **Relink top-up.** Drain ``graph_maintenance_queue`` (units whose
   outgoing temporal/semantic links lost a neighbour to a delete). For
   each, count current outgoing links per type; if below cap, run the
   same probes retain uses and insert the missing links.

2. **Orphan entity prune.** Delete ``entities`` rows in the bank that no
   longer have any live memory references. FK ON DELETE CASCADE on
   ``entity_cooccurrences`` then removes any cooccurrence row pointing
   at the pruned entities.

3. **Stale cooccurrence prune.** Defensive sweep for cooccurrence rows
   where both endpoints still exist but no current memory references
   both of them — the cooccurrence was real at the time it was recorded,
   but every unit that witnessed it has since been deleted.

Each pass is work the *memories store* owns, because each is a query over
`memory_links`, `unit_entities` and `entities` — the slice the store carves
out. This module orchestrates them (drain the queue, wrap the sweep in a
deadlock-retry) and asks the store to do the part that touches storage. A store
whose links travel inside its memories has no `memory_links` to dangle and no
join table to sweep, so its relink and cooccurrence passes are no-ops and the
job simply prunes the orphan `entities` rows, which stay in Postgres regardless.

The worker dedupes on bank: a second job for the same bank is dropped
while one is pending. Once processing starts, a new job becomes the
*next* pending slot — so work enqueued during processing gets picked up
by the follow-up run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..models import RequestContext
from .db.base import DatabaseConnection

# Re-exported for callers and tests that import the link caps from here; the caps
# themselves live with the retain-time link builders the relink pass mirrors.
from .retain.link_utils import MAX_TEMPORAL_LINKS_PER_UNIT  # noqa: F401
from .schema import fq_table

if TYPE_CHECKING:
    from .memory_engine import MemoryEngine

logger = logging.getLogger(__name__)

# Mirrors the ``top_k`` default in ``compute_semantic_links_ann`` at retain
# time. If you change one, change the other — otherwise victims would either
# never reach the cap (probe returns less than the cap) or stay perpetually
# under it (cap is higher than retain creates).
#
# Kept here as well as in the store's Postgres relink pass because the relink
# tests import it from this module; the two must not drift.
MAX_SEMANTIC_LINKS_PER_UNIT = 50

# Retry budget for the idempotent Pass 2/3 entity/cooccurrence sweep. Higher
# than db_utils' default (3) because the sweep has no client waiting on it and
# is safe to rerun, so we'd rather spend a longer jittered-backoff tail than
# drop a maintenance pass and leak stale graph rows (see run_graph_maintenance_job).
_SWEEP_MAX_RETRIES = 8


@dataclass
class _SweepCounts:
    """Prune counts returned by the Pass 2/3 sweep (avoids a bare tuple return)."""

    orphan_entities_pruned: int
    stale_cooccurrences_pruned: int


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
    ops=None,
) -> int:
    """Enqueue surviving units whose outgoing links pointed at ``deleted_unit_ids``.

    Must run inside the same transaction that deletes the units, *before* the
    cascade fires — once the rows are gone, the join that finds the victims
    returns nothing.

    Delegated to the memories store: finding the victims is a `memory_links`
    query, and a store whose links are inline has none, so it returns 0 and the
    relink pass has nothing to do. ``ops`` is accepted for the callers that still
    pass it and ignored — the store resolves what it needs from ``conn``.

    Returns:
        Number of distinct victim units enqueued (0 for a store with no links).
    """
    if not deleted_unit_ids:
        return 0

    from .memories import get_memories

    return await get_memories().enqueue_relink_victims(
        conn=conn,
        fq_table=fq_table,
        bank_id=bank_id,
        deleted_unit_ids=deleted_unit_ids,
    )


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
    from ..config import get_config
    from .memories import get_memories

    backend = await memory_engine._get_backend()
    store = get_memories()
    config = get_config()

    result = JobResult()
    job_start = time.time()

    # --- Pass 1: relink ---
    # The store owns the whole drain loop: it is a claim → top-up → commit over
    # its own link table, so how it batches and re-probes is its business. A
    # store with no links returns an empty dict and this is a no-op.
    relink = await store.relink_pass(backend=backend, fq_table=fq_table, bank_id=bank_id, config=config)
    result.relink_units_processed = relink.get("relink_units_processed", 0)
    result.relink_links_added = relink.get("relink_links_added", 0)

    # --- Pass 2 & 3: entity / cooccurrence sweeps ---
    # Bank-wide single-statement deletes. Cheap when there's nothing to do.
    #
    # Unlike Pass 1's queue claim, these DELETEs aren't protected by any
    # consistent lock-ordering guarantee: the stale-cooccurrence prune scans
    # entity_cooccurrences via a join/NOT EXISTS plan, while retain's concurrent
    # cooccurrence upserts (entity_resolver._flush_pending) lock the same rows in
    # sorted (entity_id_1, entity_id_2) order. When a sweep and a concurrent
    # upsert touch overlapping rows in opposite orders, Postgres detects a
    # genuine circular wait and aborts one side with DeadlockDetectedError. Both
    # prunes are idempotent bank-wide sweeps — rerunning only deletes what's
    # still stale — so retrying the whole transaction on deadlock is safe.
    #
    # The prunes themselves are the store's: the orphan-`entities` sweep applies
    # to every store (that registry stays in Postgres), while the cooccurrence
    # sweep is a no-op for a store that never wrote `unit_entities`.
    from .db_utils import retry_with_backoff
    from .memory_engine import acquire_with_retry

    async def _run_sweep() -> _SweepCounts:
        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                orphan_pruned = await store.prune_orphan_entities(conn=conn, fq_table=fq_table, bank_id=bank_id)
                # The orphan prune above cascades cooccurrences via FK. The
                # explicit cooccurrence pass below catches the *stale-count*
                # case: both entities still exist but no current unit witnesses
                # them together.
                stale_pruned = await store.prune_stale_cooccurrences(conn=conn, fq_table=fq_table, bank_id=bank_id)
                return _SweepCounts(orphan_entities_pruned=orphan_pruned, stale_cooccurrences_pruned=stale_pruned)

    # A larger retry budget than the default (3): this is idempotent background
    # maintenance with no client waiting on it, so a longer retry tail costs
    # nothing, whereas a dropped sweep silently leaks orphan entities / stale
    # cooccurrences until the next run. With jittered backoff a single sweep
    # contending against continuous retain upserts effectively never exhausts
    # this budget (each retry independently clears with high probability).
    sweep = await retry_with_backoff(_run_sweep, max_retries=_SWEEP_MAX_RETRIES)
    result.orphan_entities_pruned = sweep.orphan_entities_pruned
    result.stale_cooccurrences_pruned = sweep.stale_cooccurrences_pruned

    elapsed = time.time() - job_start
    logger.info(
        f"[GRAPH_MAINT] bank={bank_id} done: {result.as_dict()}, elapsed={elapsed:.2f}s, operation_id={operation_id}"
    )
    return result.as_dict()
