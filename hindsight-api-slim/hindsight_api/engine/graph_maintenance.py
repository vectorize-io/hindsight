"""Async graph maintenance after document/unit deletes.

Two queue-driven passes run together on every worker invocation:

1. **Relink top-up.** Drain ``graph_maintenance_queue`` (units whose
   outgoing temporal/semantic links lost a neighbour to a delete). For
   each, count current outgoing links per type; if below cap, run the
   same probes retain uses and insert the missing links.

2. **Entity prune.** Drain ``entity_maintenance_queue`` (entities a delete
   may have stranded). Per batch: delete the candidates no ``unit_entities``
   row references any more — FK ON DELETE CASCADE on ``entity_cooccurrences``
   takes their cooccurrence rows with them — then delete the cooccurrence rows
   incident to the survivors that no current memory witnesses, the stale-count
   case the cascade cannot see.

Both passes are *queued work*, not sweeps. Pass 2 used to be two bank-wide
statements re-evaluated on every invocation whether or not anything had
changed, so its cost tracked the size of the bank instead of the size of the
delete; on a multi-million-row bank neither statement could finish inside
asyncpg's command timeout and the job failed on every run, forever (#3222).
Both queues are now filled inside the deleting transaction, so each run only
looks at what that delete actually touched.

Each pass is work the *memories store* owns, because each is a query over
`memory_links`, `unit_entities` and `entities` — the slice the store carves
out. This module orchestrates them (pass ordering, the time budget, the timing
log) and asks the store to do the part that touches storage. A store whose
links travel inside its memories has no `memory_links` to dangle and no join
table to sweep, so both passes are no-ops for it.

The worker dedupes on bank: a second job for the same bank is dropped
while one is pending. Once processing starts, a new job becomes the
*next* pending slot — so work enqueued during processing gets picked up
by the follow-up run.

That follow-up run is *deferred*, not parallel: ``claim_tasks`` will not claim a
graph_maintenance row for a bank that already has one in flight (#3230). Two
concurrent runs would do no extra work anyway — each is this same bank-wide
sweep — while convoying on each other's row locks and holding a worker slot
each.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..models import RequestContext
from .db.base import DatabaseConnection

# Re-exported for callers and tests that import the link caps from here; the caps
# themselves live with the link builders the relink pass mirrors — the temporal one
# with the retain-time builders, the semantic one with the store's relink pass — so
# there is a single definition of each and the two cannot drift.
from .memories.pg.graph import MAX_SEMANTIC_LINKS_PER_UNIT  # noqa: F401
from .retain.link_utils import MAX_TEMPORAL_LINKS_PER_UNIT  # noqa: F401
from .schema import fq_table

if TYPE_CHECKING:
    from .memory_engine import MemoryEngine

logger = logging.getLogger(__name__)

# Wall-clock budget for one graph_maintenance run. Both passes commit per batch,
# so hitting the budget is not a failure: it stops claiming new work, reports
# what it did, and the follow-up run resumes from the queue rows still there.
# A backlog (a large delete, or the one-time seed the entity queue ships with)
# then converges over several runs instead of holding a worker slot for as long
# as it takes — the failure mode #3222 describes, where the whole run was
# cancelled and every batch's work was retried from scratch.
_JOB_TIME_BUDGET_SECONDS = 240.0


@dataclass
class JobResult:
    """Counters surfaced to the worker dispatcher and operation result."""

    relink_units_processed: int = 0
    relink_links_added: int = 0
    entities_examined: int = 0
    orphan_entities_pruned: int = 0
    stale_cooccurrences_pruned: int = 0
    # False when the time budget stopped a drain with work still queued. The
    # caller re-submits so the backlog keeps draining without waiting for the
    # next delete to trigger a run.
    queues_drained: bool = True

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "relink_units_processed": self.relink_units_processed,
            "relink_links_added": self.relink_links_added,
            "entities_examined": self.entities_examined,
            "orphan_entities_pruned": self.orphan_entities_pruned,
            "stale_cooccurrences_pruned": self.stale_cooccurrences_pruned,
            "queues_drained": self.queues_drained,
        }


async def enqueue_relink_victims(
    conn: DatabaseConnection,
    bank_id: str,
    affected_unit_ids: list[str],
    include_affected_units: bool = False,
) -> int:
    """Enqueue surviving units whose outgoing temporal/semantic links pointed at
    ``affected_unit_ids`` for later link top-up.

    Must run inside the same transaction that drops those links, *before* the
    delete (or cascade) fires — once the rows are gone, the join that finds the
    victims returns nothing.

    ``include_affected_units`` covers the case where the affected units are NOT
    being removed: an edit deletes every link incident to the edited unit but
    leaves it live, so the unit needs its own outgoing adjacency rebuilt too.
    Passing it for a unit that will be gone at commit is harmless but pointless
    — the drain skips queue rows with no live unit — so callers should only set
    it when the unit survives the transaction.

    Delegated to the memories store: finding the victims is a `memory_links`
    query, and a store whose links are inline has none, so it returns 0 and the
    relink pass has nothing to do. The store resolves the dialect it needs from
    ``conn``.

    Args:
        conn: Database connection inside the active transaction.
        bank_id: Bank owning the affected units.
        affected_unit_ids: Memory_unit IDs whose incident temporal/semantic
            links are about to be (or are being) removed.
        include_affected_units: Also enqueue ``affected_unit_ids`` themselves,
            for callers that leave them live.

    Returns:
        Number of distinct victim units enqueued (0 for a store with no links).
    """
    if not affected_unit_ids:
        return 0

    from .memories import get_memories

    return await get_memories().enqueue_relink_victims(
        conn=conn,
        fq_table=fq_table,
        bank_id=bank_id,
        affected_unit_ids=affected_unit_ids,
        include_affected_units=include_affected_units,
    )


async def enqueue_entity_prune_candidates(
    conn: DatabaseConnection,
    bank_id: str,
    affected_unit_ids: list[str],
) -> int:
    """Enqueue the entities ``affected_unit_ids`` reference as prune candidates.

    Must run inside the same transaction that removes those units (or replaces
    their entity postings), *before* the delete or cascade fires: afterwards the
    ``unit_entities`` rows naming the entities are gone, and an entity nothing
    points at is an orphan nothing will ever look at again.

    Pair this with :func:`enqueue_relink_victims` at every delete site. They
    capture different things — that one records the *survivors* whose links now
    dangle, this one the *entities* the doomed units were holding up — and
    neither substitutes for the other. A site that deletes units without calling
    this leaks orphan entities and stale cooccurrences until something else
    happens to enqueue the same entity.

    Over-enqueueing costs nothing: the drain re-checks each candidate and keeps
    the ones still referenced.

    Delegated to the memories store: a store that never wrote ``unit_entities``
    has no postings to lose and returns 0.

    Args:
        conn: Database connection inside the active transaction.
        bank_id: Bank owning the affected units.
        affected_unit_ids: Memory_unit IDs whose entity postings are about to
            be (or are being) removed.

    Returns:
        Number of candidate entities enqueued (0 for a store with no postings).
    """
    if not affected_unit_ids:
        return 0

    from .memories import get_memories

    return await get_memories().enqueue_entity_prune_candidates(
        conn=conn,
        fq_table=fq_table,
        bank_id=bank_id,
        affected_unit_ids=affected_unit_ids,
    )


async def run_graph_maintenance_job(
    memory_engine: "MemoryEngine",
    bank_id: str,
    request_context: RequestContext,
    operation_id: str | None = None,
) -> dict[str, int | bool]:
    """Drain both maintenance queues for ``bank_id``, within a time budget.

    Returns:
        Per-pass counters from :class:`JobResult`. ``queues_drained`` is False
        when the budget ran out with work still queued — the caller re-submits.
    """
    from ..config import get_config
    from .memories import get_memories

    backend = await memory_engine._get_backend()
    store = get_memories()
    config = get_config()

    result = JobResult()
    job_start = time.time()
    deadline = time.monotonic() + _JOB_TIME_BUDGET_SECONDS

    # --- Pass 1: relink ---
    # The store owns the whole drain loop: it is a claim → top-up → commit over
    # its own link table, so how it batches and re-probes is its business — including
    # the #3034 serialisation (the claim takes queue rows FOR UPDATE in (bank_id,
    # unit_id) order against a concurrent re-enqueue), which lives in the store's
    # claim (`ops.claim_graph_maintenance_batch`). A store with no links returns an
    # empty dict and this is a no-op.
    relink = await store.relink_pass(
        backend=backend, fq_table=fq_table, bank_id=bank_id, config=config, deadline=deadline
    )
    result.relink_units_processed = relink.get("relink_units_processed", 0)
    result.relink_links_added = relink.get("relink_links_added", 0)

    # --- Pass 2: entity prune ---
    # Same shape as Pass 1 and owned by the store for the same reason: a
    # claim → prune → commit loop over `entities` / `unit_entities` /
    # `entity_cooccurrences`, including the ordered locking that keeps its
    # deletes from cycling against retain's concurrent entity and cooccurrence
    # upserts. A store that never wrote `unit_entities` returns an empty dict
    # and this is a no-op. Runs after the relink pass so the remaining budget
    # is whatever Pass 1 left.
    prune = await store.entity_prune_pass(backend=backend, fq_table=fq_table, bank_id=bank_id, deadline=deadline)
    result.entities_examined = prune.get("entities_examined", 0)
    result.orphan_entities_pruned = prune.get("orphan_entities_pruned", 0)
    result.stale_cooccurrences_pruned = prune.get("stale_cooccurrences_pruned", 0)
    result.queues_drained = relink.get("relink_queue_exhausted", True) and prune.get("entity_queue_exhausted", True)

    elapsed = time.time() - job_start
    if not result.queues_drained:
        # Not a failure: every batch committed, so the remaining queue rows are
        # simply the next run's work. Logged at WARNING because a bank that keeps
        # landing here is producing maintenance faster than one run absorbs it,
        # which is worth seeing before it becomes a backlog.
        logger.warning(
            f"[GRAPH_MAINT] bank={bank_id} hit the {_JOB_TIME_BUDGET_SECONDS:.0f}s budget with work still "
            f"queued; committed {result.as_dict()} in {elapsed:.2f}s"
        )
        # Chain a follow-up so the backlog converges without waiting for the next
        # delete to trigger a run. Only under a real queue: a synchronous task
        # backend (tests, embedded) runs submitted tasks inline, so this would
        # recurse one stack frame per budget window instead of scheduling. There
        # the caller is already the loop — it gets the remaining rows on its next
        # call. No force_sweep: the submit's pre-check inspects both queues, so
        # this only lands an operation while work is genuinely left.
        from .task_backend import SyncTaskBackend

        if not isinstance(memory_engine._task_backend, SyncTaskBackend):
            await memory_engine.submit_async_graph_maintenance(bank_id=bank_id, request_context=request_context)

    logger.info(
        f"[GRAPH_MAINT] bank={bank_id} done: {result.as_dict()}, elapsed={elapsed:.2f}s, operation_id={operation_id}"
    )
    return result.as_dict()
