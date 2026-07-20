"""
System performance test runner.

Thin orchestrator that runs existing benchmark suites (retain, recall) with
fixed scale configurations and collects structured JSON results.

Each suite is completely independent — uses its own engine, bank, and cleanup.

Usage:
    # Run all suites at default (small) scale
    uv run perf-test

    # Run specific suite
    uv run perf-test --suite retain
    uv run perf-test --suite recall
    uv run perf-test --suite graph-maintenance
    uv run perf-test --suite graph-maintenance-contention   # deadlock gate (#2529)
    uv run perf-test --suite stats

    # Configurable scale
    uv run perf-test --scale tiny      # ~10s, CI smoke test
    uv run perf-test --scale small     # ~30s, default
    uv run perf-test --scale medium    # ~2min
    uv run perf-test --scale large     # ~10min

    # Prod-simulation for the stats suite (~500k units / ~18M links, bulk-loaded)
    uv run perf-test --suite stats --scale huge

    # Save results as JSON
    uv run perf-test --output results.json
"""

import argparse
import asyncio
import json
import random
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# Reuse battle-tested building blocks from existing benchmarks
from benchmarks.perf.recall_perf import (
    FACT_TEMPLATES,
    _augment_query_with_temporal,
    _build_engine,
    _fill_template,
    _insert_synthetic_observations,
    _make_fact_callback,
    _RRFReranker,
    _wait_for_operation,
)

console = Console()

# ---------------------------------------------------------------------------
# Scale configuration
# ---------------------------------------------------------------------------

SCALES: dict[str, dict[str, int]] = {
    "tiny": {
        "retain_items": 20,
        "recall_bank_size": 20,
        "recall_iterations": 5,
        "recall_concurrency": 1,
        "consolidation_items": 20,
        "graph_maintenance_bank_size": 20,
        "graph_contention_entities": 20,
        "graph_contention_pairs": 60,
        "graph_contention_upsert_workers": 4,
        "graph_contention_sweep_workers": 2,
        "graph_contention_rounds": 15,
        "stats_bank_size": 20,
    },
    "small": {
        "retain_items": 200,
        "recall_bank_size": 200,
        "recall_iterations": 20,
        "recall_concurrency": 4,
        "consolidation_items": 200,
        "graph_maintenance_bank_size": 200,
        "graph_contention_entities": 60,
        "graph_contention_pairs": 400,
        "graph_contention_upsert_workers": 8,
        "graph_contention_sweep_workers": 2,
        "graph_contention_rounds": 25,
        "stats_bank_size": 200,
    },
    "medium": {
        "retain_items": 1_000,
        "recall_bank_size": 1_000,
        "recall_iterations": 50,
        "recall_concurrency": 8,
        "consolidation_items": 1_000,
        "graph_maintenance_bank_size": 1_000,
        "graph_contention_entities": 120,
        "graph_contention_pairs": 1_500,
        "graph_contention_upsert_workers": 10,
        "graph_contention_sweep_workers": 2,
        "graph_contention_rounds": 35,
        "stats_bank_size": 1_000,
    },
    "large": {
        "retain_items": 5_000,
        "recall_bank_size": 5_000,
        "recall_iterations": 100,
        "recall_concurrency": 16,
        "consolidation_items": 5_000,
        # Past the seqscan→HNSW crossover (~10k units) so this suite exercises
        # the per-bank partial HNSW index path, not just the small-bank exact
        # scan. Verified: at 15k real-embedding units the ANN probe is planned
        # as an Index Scan on idx_mu_emb_*. medium (1k) stays in the exact-scan
        # regime, so the two scales cover both planner paths.
        "graph_maintenance_bank_size": 15_000,
        # Hot cooccurrence set big enough that both the sweep DELETE and the
        # sorted upserts lock many rows at once — wide overlap → reliable
        # opposite-order deadlocks under the higher worker fan-out.
        "graph_contention_entities": 200,
        "graph_contention_pairs": 4_000,
        "graph_contention_upsert_workers": 16,
        "graph_contention_sweep_workers": 2,
        "graph_contention_rounds": 45,
        # Large, entity-dense bank so the unit_entities→memory_units rollup join
        # in _compute_bank_stats is exercised at a size where its cost shows.
        "stats_bank_size": 15_000,
    },
    # Prod-simulation scale for the `stats` suite only. The numbers mirror a real
    # deployed bank: ~500k units and ~17.8M *physical* memory_links rows
    # (semantic + temporal + caused_by — entity links are NOT stored, they are
    # derived at query time from unit_entities). At this size the real retain
    # pipeline is infeasible, so `stats` bulk-loads via COPY (see
    # _bulk_populate_stats_bank). The non-stats keys fall back to `large` sizing
    # so a full `--scale huge` run doesn't try to retain 500k items.
    "huge": {
        "retain_items": 5_000,
        "recall_bank_size": 5_000,
        "recall_iterations": 10,
        "recall_concurrency": 4,
        "consolidation_items": 5_000,
        "graph_maintenance_bank_size": 15_000,
        # Contention keys kept for parity; `huge` targets the stats suite only.
        "graph_contention_entities": 120,
        "graph_contention_pairs": 1_500,
        "graph_contention_upsert_workers": 10,
        "graph_contention_sweep_workers": 2,
        "graph_contention_rounds": 35,
        "stats_bank_size": 500_000,  # unused by the bulk path; kept for key parity
        "stats_units": 500_000,
        "stats_semantic_links": 9_460_147,
        "stats_temporal_links": 8_344_084,
        "stats_caused_by_links": 30_015,
        # Derived entity-link total to reproduce (unit_entities rollup, capped at
        # LEAST(n-1, 10) per entity). Not stored as memory_links rows.
        "stats_entity_links": 110_881,
    },
}

# Fraction of the populated bank deleted to generate relink victims for the
# graph_maintenance suite. Mirrors the issue's "delete a handful of units, then
# top up the surviving units' links" workload (see #1919).
GRAPH_MAINTENANCE_DELETE_PCT = 0.1

# graph-maintenance-contention pass/fail gate (#2529). The discriminating metric
# is the *escape rate*: of the DeadlockDetectedErrors the sweep hits, what
# fraction escaped run_graph_maintenance_job and dropped a maintenance pass.
# Without the retry wrap EVERY deadlock escapes (rate ≈ 1.0); with the jittered
# retry_with_backoff the sweep effectively never drops a pass under a realistic
# single-sweep load (rate ≈ 0). A vanishingly small tail can still slip through
# under deliberately brutal multi-sweep synthetic load if the retry budget is
# exhausted, so we gate on the rate rather than a hard zero. A run that regresses
# the fix jumps straight back to ≈1.0, so 0.5 cleanly separates the two.
GRAPH_CONTENTION_ESCAPE_RATE_THRESHOLD = 0.5

# Recall queries that exercise different retrieval strategies
RECALL_QUERIES = [
    "database migration",
    "performance regression",
    "Alice Chen deployment",
    "Kubernetes monitoring",
    "security incident review",
    "API integration testing",
    "data pipeline processing",
    "infrastructure scaling",
]


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


@dataclass
class PercentileStats:
    p50: float
    p95: float
    p99: float
    mean: float
    min: float
    max: float
    count: int

    @staticmethod
    def from_samples(samples: list[float]) -> "PercentileStats":
        if not samples:
            return PercentileStats(p50=0, p95=0, p99=0, mean=0, min=0, max=0, count=0)
        s = sorted(samples)
        n = len(s)

        def pct(p: float) -> float:
            idx = min(int(p / 100 * n), n - 1)
            return s[idx]

        return PercentileStats(
            p50=pct(50),
            p95=pct(95),
            p99=pct(99),
            mean=statistics.mean(samples),
            min=min(samples),
            max=max(samples),
            count=n,
        )


@dataclass
class RetainResult:
    total_items: int
    total_duration_seconds: float
    throughput_items_per_sec: float


@dataclass
class RecallResult:
    bank_size: int
    concurrency: int
    latency: PercentileStats
    throughput_queries_per_sec: float
    phase_timings: dict[str, PercentileStats] = field(default_factory=dict)


@dataclass
class ConsolidationResult:
    total_items: int
    memories_processed: int
    observations_created: int
    observations_updated: int
    observations_merged: int
    skipped: int
    total_duration_seconds: float
    throughput_memories_per_sec: float


@dataclass
class GraphMaintenanceResult:
    bank_size: int
    deleted_units: int
    victims_enqueued: int
    relink_units_processed: int
    relink_links_added: int
    orphan_entities_pruned: int
    stale_cooccurrences_pruned: int
    total_duration_seconds: float
    throughput_units_per_sec: float
    ms_per_victim: float
    # Where the wall-clock goes inside the relink pass — the focus of #1919.
    semantic_ann_seconds: float
    semantic_ann_calls: int
    temporal_seconds: float
    temporal_calls: int


@dataclass
class _ContentionCounters:
    """Mutable tallies shared across the contention suite's upsert/sweep workers."""

    upserts: int = 0
    upsert_deadlocks: int = 0
    attempted: int = 0
    deadlocks: int = 0
    dropped: int = 0
    succeeded: int = 0


@dataclass
class GraphContentionResult:
    """Result of the graph-maintenance-contention suite (#2529).

    The headline metric is ``sweep_passes_dropped``: maintenance passes that
    raised ``DeadlockDetectedError`` out of ``run_graph_maintenance_job`` and
    were silently lost. ``sweep_deadlocks_observed`` counts raw deadlocks at the
    ``prune_stale_cooccurrences`` call — it stays > 0 on *both* main and the
    fix (the contention is identical), but ``sweep_passes_dropped`` collapses
    from > 0 to 0 once the sweep is wrapped in ``retry_with_backoff``. That
    delta is the fix's measurable effect.
    """

    contention_entities: int
    contention_pairs: int
    upsert_workers: int
    sweep_workers: int
    total_upserts: int
    upsert_deadlocks: int
    sweep_passes_attempted: int
    sweep_deadlocks_observed: int
    sweep_passes_dropped: int
    sweep_passes_succeeded: int
    # dropped / deadlocks_observed — the gate metric. ≈1.0 unprotected, ≈0 fixed.
    sweep_deadlock_escape_rate: float
    sweep_latency: PercentileStats
    total_duration_seconds: float


@dataclass
class StatsResult:
    bank_size: int
    concurrency: int
    total_units: int
    total_links: int
    total_entities: int
    # Cache-miss path — engine._compute_bank_stats, the raw aggregation queries
    # (node/link/ops/doc counts + the unit_entities→memory_units entity rollup).
    cold_latency: PercentileStats
    cold_throughput_per_sec: float
    # Cache-hit path — engine.get_bank_stats, what a polling client actually
    # sees once the short-TTL per-process cache is warm.
    warm_latency: PercentileStats
    cache_speedup: float


@dataclass
class SuiteResult:
    name: str
    duration_seconds: float
    success: bool
    error: str | None = None
    retain: RetainResult | None = None
    recall: RecallResult | None = None
    consolidation: ConsolidationResult | None = None
    graph_maintenance: GraphMaintenanceResult | None = None
    graph_contention: GraphContentionResult | None = None
    stats: StatsResult | None = None


@dataclass
class PerfTestResults:
    timestamp: str
    scale: str
    git_sha: str
    suites: list[SuiteResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_git_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _attach_mock_callback(engine: Any) -> None:
    """Attach recall_perf's mock fact extraction callback to an engine."""
    callback, _ = _make_fact_callback()
    engine._retain_llm_config.set_response_callback(callback)
    engine._llm_config.set_response_callback(callback)


async def _populate_bank(engine: Any, bank_id: str, size: int, event_date: str | None = None) -> None:
    """Populate a bank with synthetic data using mock LLM + async retain.

    When *event_date* (YYYY-MM-DD) is given, every item is stamped with that
    same date so all memories cluster into one narrow time range — the dense
    temporal zone the recall-temporal suite uses to stress the temporal arm
    (mirrors ``recall_perf.py generate --event-date``).
    """
    from hindsight_api.models import RequestContext
    from hindsight_api.worker.poller import WorkerPoller

    _attach_mock_callback(engine)

    contents = [{"content": _fill_template(FACT_TEMPLATES[i % len(FACT_TEMPLATES)])} for i in range(size)]
    if event_date:
        for item in contents:
            item["event_date"] = event_date

    result = await engine.submit_async_retain(
        bank_id=bank_id,
        contents=contents,
        request_context=RequestContext(),
    )
    operation_id = result["operation_id"]

    pool = await engine._get_pool()
    poller = WorkerPoller(
        backend=engine._backend,
        worker_id="perf-test-worker",
        executor=engine.execute_task,
        poll_interval_ms=200,
        max_slots=8,
    )
    poller_task = asyncio.create_task(poller.run())

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        progress.add_task(f"Populating bank ({size:,} items)…")
        await _wait_for_operation(pool, operation_id)

    await poller.shutdown_graceful(timeout=60.0)
    poller_task.cancel()
    try:
        await poller_task
    except asyncio.CancelledError:
        pass


# Average entities mentioned per unit when bulk-loading the stats bank — mirrors
# the handful of entities the mock fact callback attaches per unit during real
# retain, so the unit_entities table reaches a prod-like row count.
STATS_ENTITY_MENTIONS_PER_UNIT = 3

# All memory_links indexes are dropped before the bulk COPY and recreated after.
# Loading ~18M rows into an unindexed table + one bulk index build is far faster
# than paying per-row btree maintenance on every insert (the dominant cost of the
# load). Every index is restored afterward, so the table is left in its normal
# shape — including idx_memory_links_bank_id_link_type, which the /stats link
# GROUP BY relies on for its index-only scan, and the unique index, so this is
# safe to run even against a shared database.
_STATS_BULK_DROP_INDEXES = (
    "idx_memory_links_unique",
    "idx_memory_links_from_unit",
    "idx_memory_links_to_unit",
    "idx_memory_links_entity",
    "idx_memory_links_bank_id_link_type",
)


async def _bulk_populate_stats_bank(
    engine: Any,
    bank_id: str,
    *,
    units: int,
    semantic_links: int,
    temporal_links: int,
    caused_by_links: int,
    entity_link_target: int,
) -> None:
    """Bulk-load a prod-scale bank for the stats suite via COPY (no LLM/embeddings).

    Rows are shaped exactly as the real retain pipeline leaves them so
    _compute_bank_stats sees a faithful workload:

    * memory_units — `units` rows, NULL embeddings (stats never reads them), a
      realistic experience/world/observation fact_type mix.
    * memory_links — `semantic + temporal + caused_by` *physical* rows. Entity
      links are deliberately NOT inserted here: on the deployed schema the
      entity_id column is 100% NULL and the entity total is derived at query
      time. Pairs are generated collision-free so the unique index never trips.
    * entities / unit_entities — sized so the LEAST(n-1, 10) rollup in
      _compute_bank_stats reconstructs ~`entity_link_target` entity links: each
      of `entity_link_target` "shared" entities is attached to exactly 2 units
      (→ 1 derived link each), with the remaining mentions as singletons (→ 0)
      to reach a prod-like ~`units * STATS_ENTITY_MENTIONS_PER_UNIT` row count.

    FK triggers on memory_links/unit_entities are disabled during the load (this
    is a throwaway perf bank), so COPY doesn't validate ~35M edge endpoints.
    """
    from hindsight_api.engine.memory_engine import get_current_schema

    pool = await engine._get_pool()
    schema = get_current_schema()

    # Bulk maintenance (multi-million-row COPY, unique-index rebuild, VACUUM)
    # scales with the *whole* table, not just this bank, and can exceed the
    # engine's OLTP command_timeout when other large banks already exist. Give
    # these one-off fixture ops their own generous ceiling so a pre-existing
    # bank can't trip an (empty-message) asyncio.TimeoutError mid-load.
    bulk_timeout = 7200.0

    unit_ids = [uuid.uuid4() for _ in range(units)]
    base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def _fact_type(i: int) -> str:
        m = i % 20
        if m < 11:
            return "experience"  # 55%
        if m < 18:
            return "world"  # 35%
        return "observation"  # 10%

    def unit_records():
        for i, uid in enumerate(unit_ids):
            # Spread event_date across ~a year so date indexes look realistic.
            yield (uid, bank_id, f"perf stats unit {i}", base_date + timedelta(minutes=i % 525_600), _fact_type(i))

    # Entity plan: `entity_link_target` shared entities (2 units each → 1 derived
    # link) + singleton entities padding out to the target unit_entities count.
    shared_entities = max(0, entity_link_target)
    total_unit_entities = max(2 * shared_entities, units * STATS_ENTITY_MENTIONS_PER_UNIT)
    singleton_entities = total_unit_entities - 2 * shared_entities
    shared_eids = [uuid.uuid4() for _ in range(shared_entities)]
    singleton_eids = [uuid.uuid4() for _ in range(singleton_entities)]
    half = max(1, units // 2)

    def entity_records():
        for k, eid in enumerate(shared_eids):
            yield (eid, f"perf_entity_s_{k}", bank_id)
        for k, eid in enumerate(singleton_eids):
            yield (eid, f"perf_entity_x_{k}", bank_id)

    def unit_entity_records():
        for j, eid in enumerate(shared_eids):
            yield (unit_ids[j % units], eid)
            yield (unit_ids[(j + half) % units], eid)
        for k, eid in enumerate(singleton_eids):
            yield (unit_ids[k % units], eid)

    def link_records(link_type: str, count: int):
        # Collision-free distinct (from, to) pairs: spread `from` across leading
        # units, `to` across the rest skipping self. link_type is part of the
        # unique index, so semantic/temporal/caused_by never collide with each
        # other. COUNT(*) GROUP BY link_type is indifferent to the distribution.
        span = units - 1
        for k in range(count):
            f = k // span
            r = k % span
            t = r if r < f else r + 1
            yield (unit_ids[f], unit_ids[t], link_type, bank_id)

    def _q(table: str) -> str:
        return f'"{schema}".{table}' if schema else table

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        progress.add_task(
            f"Bulk-loading {units:,} units, "
            f"{semantic_links + temporal_links + caused_by_links:,} links, "
            f"{total_unit_entities:,} unit-entities…"
        )
        async with pool.acquire() as conn:
            # Every pooled connection carries a server-side statement_timeout
            # (the engine's runaway-query safety net, default 600s). The unique-
            # index rebuild and VACUUM over millions of rows legitimately exceed
            # it, so disable it on this connection for the load and restore it
            # after (PG cancels with "canceling statement due to statement
            # timeout" otherwise — which is exactly how the first 18M run died).
            prev_stmt_timeout = await conn.fetchval("SHOW statement_timeout")
            await conn.execute("SET statement_timeout = 0")
            await conn.copy_records_to_table(
                "entities",
                records=entity_records(),
                columns=["id", "canonical_name", "bank_id"],
                schema_name=schema,
                timeout=bulk_timeout,
            )
            await conn.copy_records_to_table(
                "memory_units",
                records=unit_records(),
                columns=["id", "bank_id", "text", "event_date", "fact_type"],
                schema_name=schema,
                timeout=bulk_timeout,
            )

            dropped: list[str] = []
            triggers_disabled = False
            try:
                for idx in _STATS_BULK_DROP_INDEXES:
                    await conn.execute(f"DROP INDEX IF EXISTS {_q(idx)}", timeout=bulk_timeout)
                    dropped.append(idx)
                try:
                    await conn.execute(f"ALTER TABLE {_q('memory_links')} DISABLE TRIGGER ALL", timeout=bulk_timeout)
                    await conn.execute(f"ALTER TABLE {_q('unit_entities')} DISABLE TRIGGER ALL", timeout=bulk_timeout)
                    triggers_disabled = True
                except Exception as exc:  # noqa: BLE001 — best effort; fall back to validated COPY
                    console.print(f"  [yellow]Could not disable FK triggers ({exc}); COPY will validate FKs[/yellow]")

                for link_type, count in (
                    ("semantic", semantic_links),
                    ("temporal", temporal_links),
                    ("caused_by", caused_by_links),
                ):
                    if count > 0:
                        await conn.copy_records_to_table(
                            "memory_links",
                            records=link_records(link_type, count),
                            columns=["from_unit_id", "to_unit_id", "link_type", "bank_id"],
                            schema_name=schema,
                            timeout=bulk_timeout,
                        )
                await conn.copy_records_to_table(
                    "unit_entities",
                    records=unit_entity_records(),
                    columns=["unit_id", "entity_id"],
                    schema_name=schema,
                    timeout=bulk_timeout,
                )
            finally:
                if triggers_disabled:
                    await conn.execute(f"ALTER TABLE {_q('memory_links')} ENABLE TRIGGER ALL", timeout=bulk_timeout)
                    await conn.execute(f"ALTER TABLE {_q('unit_entities')} ENABLE TRIGGER ALL", timeout=bulk_timeout)
                # Recreate the dropped indexes so the bank matches the prod schema
                # (the planner sees the same index set when the query runs).
                if "idx_memory_links_unique" in dropped:
                    await conn.execute(
                        f"CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_links_unique ON {_q('memory_links')} "
                        "(from_unit_id, to_unit_id, link_type, "
                        "COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid))",
                        timeout=bulk_timeout,
                    )
                if "idx_memory_links_from_unit" in dropped:
                    await conn.execute(
                        f"CREATE INDEX IF NOT EXISTS idx_memory_links_from_unit ON {_q('memory_links')} (from_unit_id)",
                        timeout=bulk_timeout,
                    )
                if "idx_memory_links_to_unit" in dropped:
                    await conn.execute(
                        f"CREATE INDEX IF NOT EXISTS idx_memory_links_to_unit ON {_q('memory_links')} (to_unit_id)",
                        timeout=bulk_timeout,
                    )
                if "idx_memory_links_entity" in dropped:
                    await conn.execute(
                        f"CREATE INDEX IF NOT EXISTS idx_memory_links_entity ON {_q('memory_links')} (entity_id)",
                        timeout=bulk_timeout,
                    )
                if "idx_memory_links_bank_id_link_type" in dropped:
                    # The index the /stats link-count GROUP BY uses — restore it
                    # so the measured query plans against the prod index set.
                    await conn.execute(
                        f"CREATE INDEX IF NOT EXISTS idx_memory_links_bank_id_link_type "
                        f"ON {_q('memory_links')} (bank_id, link_type)",
                        timeout=bulk_timeout,
                    )
            # VACUUM ANALYZE (not just ANALYZE): refresh planner stats AND set the
            # visibility map. Without the latter, the link_type COUNT can't use an
            # index-only scan (every tuple needs a heap visibility check), so a
            # fresh COPY measures a full heap scan that prod — where autovacuum
            # keeps the vm set — never pays. On an 18M-row bank this is the
            # difference between a ~1.6s seq scan and a ~1.0s index-only scan.
            await conn.execute(f"VACUUM (ANALYZE) {_q('memory_units')}", timeout=bulk_timeout)
            await conn.execute(f"VACUUM (ANALYZE) {_q('memory_links')}", timeout=bulk_timeout)
            await conn.execute(f"VACUUM (ANALYZE) {_q('unit_entities')}", timeout=bulk_timeout)
            # Restore the connection's statement_timeout before it returns to the
            # pool so later acquirers keep the runaway-query safety net.
            await conn.execute(f"SET statement_timeout = '{prev_stmt_timeout}'")


# ---------------------------------------------------------------------------
# Suite: retain
# ---------------------------------------------------------------------------


async def run_retain_suite(scale_cfg: dict[str, int]) -> SuiteResult:
    """Measure retain throughput with mock LLM — embedding + DB write speed."""
    from hindsight_api.models import RequestContext

    total_items = scale_cfg["retain_items"]
    bank_id = f"perf-retain-{uuid.uuid4().hex[:8]}"

    console.print(f"\n[bold cyan]Suite: retain[/bold cyan]  items={total_items}  bank={bank_id}")

    engine = _build_engine(disable_observations=True)
    await engine.initialize()
    _attach_mock_callback(engine)

    contents = [{"content": _fill_template(FACT_TEMPLATES[i % len(FACT_TEMPLATES)])} for i in range(total_items)]
    request_context = RequestContext()

    t0 = time.perf_counter()
    await engine.retain_batch_async(
        bank_id=bank_id,
        contents=contents,
        request_context=request_context,
    )
    duration = time.perf_counter() - t0
    throughput = total_items / duration

    await engine.delete_bank(bank_id=bank_id, request_context=request_context)
    await engine.close()

    retain_result = RetainResult(
        total_items=total_items,
        total_duration_seconds=round(duration, 3),
        throughput_items_per_sec=round(throughput, 2),
    )

    # Print summary
    table = Table(title="Retain Throughput")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Items", str(total_items))
    table.add_row("Duration", f"{duration:.3f}s")
    table.add_row("Throughput", f"{throughput:.2f} items/s")
    console.print(table)

    return SuiteResult(name="retain", duration_seconds=round(duration, 3), success=True, retain=retain_result)


# ---------------------------------------------------------------------------
# Suite: recall
# ---------------------------------------------------------------------------


async def run_recall_suite(scale_cfg: dict[str, int]) -> SuiteResult:
    """Measure recall latency/throughput with pre-populated bank."""
    from hindsight_api.engine.memory_engine import Budget
    from hindsight_api.models import RequestContext

    bank_size = scale_cfg["recall_bank_size"]
    iterations = scale_cfg["recall_iterations"]
    concurrency = scale_cfg["recall_concurrency"]
    bank_id = f"perf-recall-{uuid.uuid4().hex[:8]}"

    console.print(
        f"\n[bold cyan]Suite: recall[/bold cyan]  "
        f"bank_size={bank_size}  iterations={iterations}  concurrency={concurrency}  bank={bank_id}"
    )

    engine = _build_engine()
    await engine.initialize()

    # Use RRF reranker to isolate DB performance from cross-encoder CPU cost
    engine._cross_encoder_reranker = _RRFReranker()

    # Populate bank using recall_perf's synthetic data patterns
    await _populate_bank(engine, bank_id, bank_size)

    request_context = RequestContext()
    durations: list[float] = []
    all_phase_timings: dict[str, list[float]] = {}

    async def recall_one(query: str) -> float:
        t0 = time.perf_counter()
        result = await engine.recall_async(
            bank_id=bank_id,
            query=query,
            budget=Budget.HIGH,
            max_tokens=4096,
            enable_trace=True,
            request_context=request_context,
            _quiet=True,
        )
        elapsed = time.perf_counter() - t0
        if result.trace:
            summary = result.trace.get("summary", {})
            for pm in summary.get("phase_metrics", []):
                all_phase_timings.setdefault(pm["phase_name"], []).append(pm["duration_seconds"])
        return elapsed

    # Run recall iterations in parallel batches
    remaining = iterations
    query_idx = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running recall…", total=iterations)
        while remaining > 0:
            batch_size = min(concurrency, remaining)
            queries = [RECALL_QUERIES[(query_idx + i) % len(RECALL_QUERIES)] for i in range(batch_size)]
            query_idx += batch_size
            batch = await asyncio.gather(*[recall_one(q) for q in queries])
            durations.extend(batch)
            remaining -= batch_size
            progress.advance(task, batch_size)

    suite_duration = sum(durations)
    throughput = iterations / (suite_duration / concurrency) if suite_duration > 0 else 0

    await engine.delete_bank(bank_id=bank_id, request_context=RequestContext())
    await engine.close()

    latency_stats = PercentileStats.from_samples(durations)
    phase_stats = {name: PercentileStats.from_samples(times) for name, times in all_phase_timings.items()}

    recall_result = RecallResult(
        bank_size=bank_size,
        concurrency=concurrency,
        latency=latency_stats,
        throughput_queries_per_sec=round(throughput, 2),
        phase_timings=phase_stats,
    )

    # Print summary
    table = Table(title="Recall Latency")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Bank size", f"{bank_size:,}")
    table.add_row("Iterations", str(iterations))
    table.add_row("Concurrency", str(concurrency))
    table.add_row("Throughput", f"{throughput:.2f} queries/s")
    table.add_row("Mean", f"{latency_stats.mean:.3f}s")
    table.add_row("p50", f"{latency_stats.p50:.3f}s")
    table.add_row("p95", f"{latency_stats.p95:.3f}s")
    table.add_row("p99", f"{latency_stats.p99:.3f}s")
    table.add_row("Min", f"{latency_stats.min:.3f}s")
    table.add_row("Max", f"{latency_stats.max:.3f}s")
    console.print(table)

    if phase_stats:
        phase_table = Table(title="Per-Step Timing Breakdown")
        phase_table.add_column("Step", style="cyan")
        phase_table.add_column("Mean", style="green", justify="right")
        phase_table.add_column("p50", style="green", justify="right")
        phase_table.add_column("p95", style="yellow", justify="right")
        phase_table.add_column("Max", style="red", justify="right")

        sorted_phases = sorted(phase_stats.items(), key=lambda x: x[1].mean, reverse=True)
        for name, ps in sorted_phases:
            phase_table.add_row(name, f"{ps.mean:.3f}s", f"{ps.p50:.3f}s", f"{ps.p95:.3f}s", f"{ps.max:.3f}s")
        console.print(phase_table)

    return SuiteResult(name="recall", duration_seconds=round(suite_duration, 3), success=True, recall=recall_result)


# ---------------------------------------------------------------------------
# Suite: recall-with-observations
# ---------------------------------------------------------------------------


async def run_recall_with_observations_suite(scale_cfg: dict[str, int]) -> SuiteResult:
    """Measure recall latency/throughput with pre-populated bank including synthetic observations."""
    from hindsight_api.engine.memory_engine import Budget
    from hindsight_api.models import RequestContext

    bank_size = scale_cfg["recall_bank_size"]
    iterations = scale_cfg["recall_iterations"]
    concurrency = scale_cfg["recall_concurrency"]
    bank_id = f"perf-recall-obs-{uuid.uuid4().hex[:8]}"

    console.print(
        f"\n[bold cyan]Suite: recall-with-observations[/bold cyan]  "
        f"bank_size={bank_size}  iterations={iterations}  concurrency={concurrency}  bank={bank_id}"
    )

    engine = _build_engine()
    await engine.initialize()

    # Use RRF reranker to isolate DB performance from cross-encoder CPU cost
    engine._cross_encoder_reranker = _RRFReranker()

    # Populate bank with facts then insert synthetic observations (1 per fact)
    await _populate_bank(engine, bank_id, bank_size)

    pool = await engine._get_pool()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        progress.add_task("Inserting synthetic observations…")
        n_obs = await _insert_synthetic_observations(pool, bank_id)
    console.print(f"  Inserted {n_obs:,} observations")

    request_context = RequestContext()
    durations: list[float] = []
    all_phase_timings: dict[str, list[float]] = {}

    async def recall_one(query: str) -> float:
        t0 = time.perf_counter()
        result = await engine.recall_async(
            bank_id=bank_id,
            query=query,
            budget=Budget.HIGH,
            max_tokens=4096,
            enable_trace=True,
            request_context=request_context,
            _quiet=True,
        )
        elapsed = time.perf_counter() - t0
        if result.trace:
            summary = result.trace.get("summary", {})
            for pm in summary.get("phase_metrics", []):
                all_phase_timings.setdefault(pm["phase_name"], []).append(pm["duration_seconds"])
        return elapsed

    # Run recall iterations in parallel batches
    remaining = iterations
    query_idx = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running recall (with observations)…", total=iterations)
        while remaining > 0:
            batch_size = min(concurrency, remaining)
            queries = [RECALL_QUERIES[(query_idx + i) % len(RECALL_QUERIES)] for i in range(batch_size)]
            query_idx += batch_size
            batch = await asyncio.gather(*[recall_one(q) for q in queries])
            durations.extend(batch)
            remaining -= batch_size
            progress.advance(task, batch_size)

    suite_duration = sum(durations)
    throughput = iterations / (suite_duration / concurrency) if suite_duration > 0 else 0

    await engine.delete_bank(bank_id=bank_id, request_context=RequestContext())
    await engine.close()

    latency_stats = PercentileStats.from_samples(durations)
    phase_stats = {name: PercentileStats.from_samples(times) for name, times in all_phase_timings.items()}

    recall_result = RecallResult(
        bank_size=bank_size,
        concurrency=concurrency,
        latency=latency_stats,
        throughput_queries_per_sec=round(throughput, 2),
        phase_timings=phase_stats,
    )

    # Print summary
    table = Table(title="Recall Latency (with observations)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Bank size (facts)", f"{bank_size:,}")
    table.add_row("Observations", f"{n_obs:,}")
    table.add_row("Iterations", str(iterations))
    table.add_row("Concurrency", str(concurrency))
    table.add_row("Throughput", f"{throughput:.2f} queries/s")
    table.add_row("Mean", f"{latency_stats.mean:.3f}s")
    table.add_row("p50", f"{latency_stats.p50:.3f}s")
    table.add_row("p95", f"{latency_stats.p95:.3f}s")
    table.add_row("p99", f"{latency_stats.p99:.3f}s")
    table.add_row("Min", f"{latency_stats.min:.3f}s")
    table.add_row("Max", f"{latency_stats.max:.3f}s")
    console.print(table)

    if phase_stats:
        phase_table = Table(title="Per-Step Timing Breakdown (with observations)")
        phase_table.add_column("Step", style="cyan")
        phase_table.add_column("Mean", style="green", justify="right")
        phase_table.add_column("p50", style="green", justify="right")
        phase_table.add_column("p95", style="yellow", justify="right")
        phase_table.add_column("Max", style="red", justify="right")

        sorted_phases = sorted(phase_stats.items(), key=lambda x: x[1].mean, reverse=True)
        for name, ps in sorted_phases:
            phase_table.add_row(name, f"{ps.mean:.3f}s", f"{ps.p50:.3f}s", f"{ps.p95:.3f}s", f"{ps.max:.3f}s")
        console.print(phase_table)

    return SuiteResult(
        name="recall-with-observations",
        duration_seconds=round(suite_duration, 3),
        success=True,
        recall=recall_result,
    )


# ---------------------------------------------------------------------------
# Suite: recall-temporal
# ---------------------------------------------------------------------------

# All memories are stamped with this date and every query is augmented with a
# 1-day window on it, so the temporal entry-point scan matches (near-)all rows
# — the dense-temporal-zone regime that degraded in PR #1958 / was bounded in
# #1983. The specific date is arbitrary; only the clustering matters.
RECALL_TEMPORAL_EVENT_DATE = "2025-01-15"


async def run_recall_temporal_suite(scale_cfg: dict[str, int]) -> SuiteResult:
    """Measure recall latency/throughput while forcing the temporal retrieval arm."""
    from hindsight_api.engine.memory_engine import Budget
    from hindsight_api.models import RequestContext

    bank_size = scale_cfg["recall_bank_size"]
    iterations = scale_cfg["recall_iterations"]
    concurrency = scale_cfg["recall_concurrency"]
    bank_id = f"perf-recall-temporal-{uuid.uuid4().hex[:8]}"

    console.print(
        f"\n[bold cyan]Suite: recall-temporal[/bold cyan]  "
        f"bank_size={bank_size}  iterations={iterations}  concurrency={concurrency}  "
        f"event_date={RECALL_TEMPORAL_EVENT_DATE}  bank={bank_id}"
    )

    engine = _build_engine()
    await engine.initialize()

    # Use RRF reranker to isolate DB performance from cross-encoder CPU cost
    engine._cross_encoder_reranker = _RRFReranker()

    # Cluster all memories on one date so the temporal entry-point scan is stressed
    await _populate_bank(engine, bank_id, bank_size, event_date=RECALL_TEMPORAL_EVENT_DATE)

    request_context = RequestContext()
    durations: list[float] = []
    all_phase_timings: dict[str, list[float]] = {}

    async def recall_one(query: str) -> float:
        # Append "on January 15, 2025" so the query analyzer extracts a 1-day
        # window and the temporal arm fires against the clustered memories.
        temporal_query = _augment_query_with_temporal(query, RECALL_TEMPORAL_EVENT_DATE)
        t0 = time.perf_counter()
        result = await engine.recall_async(
            bank_id=bank_id,
            query=temporal_query,
            budget=Budget.HIGH,
            max_tokens=4096,
            enable_trace=True,
            request_context=request_context,
            _quiet=True,
        )
        elapsed = time.perf_counter() - t0
        if result.trace:
            summary = result.trace.get("summary", {})
            for pm in summary.get("phase_metrics", []):
                all_phase_timings.setdefault(pm["phase_name"], []).append(pm["duration_seconds"])
        return elapsed

    # Run recall iterations in parallel batches
    remaining = iterations
    query_idx = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running recall (temporal)…", total=iterations)
        while remaining > 0:
            batch_size = min(concurrency, remaining)
            queries = [RECALL_QUERIES[(query_idx + i) % len(RECALL_QUERIES)] for i in range(batch_size)]
            query_idx += batch_size
            batch = await asyncio.gather(*[recall_one(q) for q in queries])
            durations.extend(batch)
            remaining -= batch_size
            progress.advance(task, batch_size)

    suite_duration = sum(durations)
    throughput = iterations / (suite_duration / concurrency) if suite_duration > 0 else 0

    await engine.delete_bank(bank_id=bank_id, request_context=RequestContext())
    await engine.close()

    latency_stats = PercentileStats.from_samples(durations)
    phase_stats = {name: PercentileStats.from_samples(times) for name, times in all_phase_timings.items()}

    recall_result = RecallResult(
        bank_size=bank_size,
        concurrency=concurrency,
        latency=latency_stats,
        throughput_queries_per_sec=round(throughput, 2),
        phase_timings=phase_stats,
    )

    # Print summary
    table = Table(title="Recall Latency (temporal arm forced)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Bank size", f"{bank_size:,}")
    table.add_row("Event date", RECALL_TEMPORAL_EVENT_DATE)
    table.add_row("Iterations", str(iterations))
    table.add_row("Concurrency", str(concurrency))
    table.add_row("Throughput", f"{throughput:.2f} queries/s")
    table.add_row("Mean", f"{latency_stats.mean:.3f}s")
    table.add_row("p50", f"{latency_stats.p50:.3f}s")
    table.add_row("p95", f"{latency_stats.p95:.3f}s")
    table.add_row("p99", f"{latency_stats.p99:.3f}s")
    table.add_row("Min", f"{latency_stats.min:.3f}s")
    table.add_row("Max", f"{latency_stats.max:.3f}s")
    console.print(table)

    if phase_stats:
        phase_table = Table(title="Per-Step Timing Breakdown (temporal)")
        phase_table.add_column("Step", style="cyan")
        phase_table.add_column("Mean", style="green", justify="right")
        phase_table.add_column("p50", style="green", justify="right")
        phase_table.add_column("p95", style="yellow", justify="right")
        phase_table.add_column("Max", style="red", justify="right")

        sorted_phases = sorted(phase_stats.items(), key=lambda x: x[1].mean, reverse=True)
        for name, ps in sorted_phases:
            phase_table.add_row(name, f"{ps.mean:.3f}s", f"{ps.p50:.3f}s", f"{ps.p95:.3f}s", f"{ps.max:.3f}s")
        console.print(phase_table)

    return SuiteResult(
        name="recall-temporal",
        duration_seconds=round(suite_duration, 3),
        success=True,
        recall=recall_result,
    )


# ---------------------------------------------------------------------------
# Suite: consolidation
# ---------------------------------------------------------------------------


def _make_consolidation_callback() -> tuple:
    """
    Return (callback, call_counter) for mock consolidation LLM.

    For the consolidation scope, returns a response that creates one observation
    per fact in the batch.  For retain scopes, delegates to the standard fact
    extraction callback so we can populate the bank normally.
    """
    fact_callback, fact_counter = _make_fact_callback()
    consolidation_counter = [0]

    def callback(messages: list[dict], scope: str):
        if scope == "consolidation":
            consolidation_counter[0] += 1
            # Parse the fact IDs from the prompt to build realistic create actions.
            # The prompt contains lines like "[<uuid>] fact text"
            import re

            prompt_text = messages[-1]["content"] if messages else ""
            fact_ids = re.findall(r"\[([0-9a-f-]{36})\]", prompt_text)

            from hindsight_api.engine.consolidation.consolidator import (
                _ConsolidationBatchResponse,
                _CreateAction,
            )

            creates = [
                _CreateAction(
                    text=f"Observation from fact {fid[:8]}",
                    source_fact_ids=[fid],
                )
                for fid in fact_ids
            ]
            return _ConsolidationBatchResponse(creates=creates, updates=[], deletes=[])
        # Delegate all other scopes to the retain callback
        return fact_callback(messages, scope)

    return callback, fact_counter, consolidation_counter


async def run_consolidation_suite(scale_cfg: dict[str, int]) -> SuiteResult:
    """Measure consolidation throughput with mock LLM — DB + embedding overhead."""
    import os

    from hindsight_api.engine.consolidation.consolidator import run_consolidation_job
    from hindsight_api.models import RequestContext

    total_items = scale_cfg["consolidation_items"]
    bank_id = f"perf-consolidation-{uuid.uuid4().hex[:8]}"

    console.print(f"\n[bold cyan]Suite: consolidation[/bold cyan]  items={total_items}  bank={bank_id}")

    # Enable observations so consolidation has work to do
    os.environ["HINDSIGHT_API_ENABLE_OBSERVATIONS"] = "true"
    from hindsight_api.config import clear_config_cache

    clear_config_cache()

    engine = _build_engine()
    await engine.initialize()

    # Set up mock callback that handles both retain and consolidation scopes
    callback, _, consolidation_counter = _make_consolidation_callback()
    engine._retain_llm_config.set_response_callback(callback)
    engine._llm_config.set_response_callback(callback)
    engine._consolidation_llm_config.set_response_callback(callback)

    # Populate bank with facts using synchronous batch retain.
    # This queues consolidation tasks (since observations are enabled) but
    # no worker is running so they just sit in the queue — we run consolidation
    # explicitly below.
    contents = [{"content": _fill_template(FACT_TEMPLATES[i % len(FACT_TEMPLATES)])} for i in range(total_items)]
    request_context = RequestContext()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        progress.add_task(f"Populating bank ({total_items:,} items)…")
        await engine.retain_batch_async(
            bank_id=bank_id,
            contents=contents,
            request_context=request_context,
        )

    # Run consolidation
    request_context = RequestContext()
    t0 = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        progress.add_task(f"Running consolidation ({total_items:,} items)…")
        result = await run_consolidation_job(
            memory_engine=engine,
            bank_id=bank_id,
            request_context=request_context,
        )

    duration = time.perf_counter() - t0
    memories_processed = result.get("memories_processed", 0)
    throughput = memories_processed / duration if duration > 0 else 0

    await engine.delete_bank(bank_id=bank_id, request_context=request_context)
    await engine.close()

    consolidation_result = ConsolidationResult(
        total_items=total_items,
        memories_processed=memories_processed,
        observations_created=result.get("observations_created", 0),
        observations_updated=result.get("observations_updated", 0),
        observations_merged=result.get("observations_merged", 0),
        skipped=result.get("skipped", 0),
        total_duration_seconds=round(duration, 3),
        throughput_memories_per_sec=round(throughput, 2),
    )

    # Print summary
    table = Table(title="Consolidation Throughput")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Bank items", f"{total_items:,}")
    table.add_row("Memories processed", str(memories_processed))
    table.add_row("Duration", f"{duration:.3f}s")
    table.add_row("Throughput", f"{throughput:.2f} memories/s")
    table.add_row("Observations created", str(result.get("observations_created", 0)))
    table.add_row("Observations updated", str(result.get("observations_updated", 0)))
    table.add_row("Observations merged", str(result.get("observations_merged", 0)))
    table.add_row("Skipped", str(result.get("skipped", 0)))
    console.print(table)

    return SuiteResult(
        name="consolidation",
        duration_seconds=round(duration, 3),
        success=True,
        consolidation=consolidation_result,
    )


# ---------------------------------------------------------------------------
# Suite: graph-maintenance
# ---------------------------------------------------------------------------


@dataclass
class _GraphMaintTimers:
    """Accumulates time spent in the two relink probes (#1919).

    ``run_graph_maintenance_job`` runs them deep inside its own connections and
    transactions, so the only seam that doesn't perturb the path under test is
    wrapping the functions it calls. We patch the symbol the graph_maintenance
    module resolves (``compute_semantic_links_ann``) and the bound ops method
    (``fetch_temporal_neighbors``), tallying wall-clock and call counts.
    """

    semantic_seconds: float = 0.0
    semantic_calls: int = 0
    temporal_seconds: float = 0.0
    temporal_calls: int = 0


@dataclass
class _InstrumentedJob:
    """Result of one instrumented maintenance run: the job's counter dict plus probe timings."""

    result: dict[str, int]
    timers: _GraphMaintTimers


async def _run_graph_maintenance_instrumented(engine: Any, bank_id: str, request_context: Any) -> _InstrumentedJob:
    """Run the maintenance job with the two relink probes timed."""
    from hindsight_api.engine import graph_maintenance as gm
    from hindsight_api.engine.graph_maintenance import run_graph_maintenance_job

    timers = _GraphMaintTimers()

    backend = await engine._get_backend()
    ops = backend.ops

    orig_ann = gm.compute_semantic_links_ann
    orig_temporal = ops.fetch_temporal_neighbors

    async def timed_ann(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return await orig_ann(*args, **kwargs)
        finally:
            timers.semantic_seconds += time.perf_counter() - t0
            timers.semantic_calls += 1

    async def timed_temporal(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return await orig_temporal(*args, **kwargs)
        finally:
            timers.temporal_seconds += time.perf_counter() - t0
            timers.temporal_calls += 1

    gm.compute_semantic_links_ann = timed_ann
    ops.fetch_temporal_neighbors = timed_temporal
    try:
        result = await run_graph_maintenance_job(
            memory_engine=engine,
            bank_id=bank_id,
            request_context=request_context,
        )
    finally:
        gm.compute_semantic_links_ann = orig_ann
        ops.fetch_temporal_neighbors = orig_temporal

    return _InstrumentedJob(result=result, timers=timers)


async def _delete_units_and_enqueue(engine: Any, bank_id: str, deleted_ids: list[str]) -> int:
    """Delete ``deleted_ids`` and enqueue their surviving neighbours as relink victims.

    Mirrors the capture-then-delete order ``delete_memory_unit`` uses: victims must
    be found before the cascade removes the links that identify them.
    Returns the queue depth (distinct victims enqueued) afterwards.
    """
    import uuid as uuid_module

    from hindsight_api.engine.graph_maintenance import enqueue_relink_victims
    from hindsight_api.engine.memory_engine import acquire_with_retry
    from hindsight_api.engine.schema import fq_table

    backend = await engine._get_backend()
    ops = backend.ops
    deleted_uuids = [uuid_module.UUID(uid) for uid in deleted_ids]

    async with acquire_with_retry(backend) as conn:
        async with conn.transaction():
            await enqueue_relink_victims(conn, bank_id, deleted_ids, ops=ops)
            await conn.execute(
                f"DELETE FROM {fq_table('memory_units')} WHERE id = ANY($1::uuid[]) AND bank_id = $2",
                deleted_uuids,
                bank_id,
            )

    pool = await engine._get_pool()
    depth = await pool.fetchval(
        f"SELECT COUNT(*) FROM {fq_table('graph_maintenance_queue')} WHERE bank_id = $1",
        bank_id,
    )
    return int(depth or 0)


async def run_graph_maintenance_suite(scale_cfg: dict[str, int]) -> SuiteResult:
    """Measure graph_maintenance (relink + entity/cooccurrence sweep) after deletes.

    Background maintenance is supposed to be cheap, but #1919 reports the
    semantic-ANN relink pass taking 10–20s per batch on ~1k-unit banks. This
    suite reproduces that path: populate a bank with real embeddings + links,
    delete a fraction of units to enqueue relink victims, then run the job and
    break the wall-clock down by probe so the bottleneck is visible.
    """
    from hindsight_api.engine.schema import fq_table
    from hindsight_api.models import RequestContext

    bank_size = scale_cfg["graph_maintenance_bank_size"]
    bank_id = f"perf-graphmaint-{uuid.uuid4().hex[:8]}"

    console.print(f"\n[bold cyan]Suite: graph-maintenance[/bold cyan]  bank_size={bank_size}  bank={bank_id}")

    engine = _build_engine(disable_observations=True)
    await engine.initialize()

    # Populate with the real retain pipeline so units get real embeddings and
    # the temporal/semantic links the relink pass tops up.
    await _populate_bank(engine, bank_id, bank_size)

    pool = await engine._get_pool()
    # Source memories are the only relink-eligible units (experience/world).
    src_rows = await pool.fetch(
        f"""
        SELECT id::text AS id
        FROM {fq_table("memory_units")}
        WHERE bank_id = $1 AND fact_type IN ('experience', 'world')
        ORDER BY id
        """,
        bank_id,
    )
    src_ids = [r["id"] for r in src_rows]
    n_delete = max(1, int(len(src_ids) * GRAPH_MAINTENANCE_DELETE_PCT))
    # Evenly spread the deletions across the id space so victims are drawn from
    # across the bank rather than one cluster.
    step = max(1, len(src_ids) // n_delete)
    deleted_ids = src_ids[::step][:n_delete]

    request_context = RequestContext()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        progress.add_task(f"Deleting {len(deleted_ids):,} units + enqueuing victims…")
        victims_enqueued = await _delete_units_and_enqueue(engine, bank_id, deleted_ids)

    console.print(f"  Deleted {len(deleted_ids):,} units → {victims_enqueued:,} relink victims enqueued")

    t0 = time.perf_counter()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        progress.add_task(f"Running graph_maintenance ({victims_enqueued:,} victims)…")
        instrumented = await _run_graph_maintenance_instrumented(engine, bank_id, request_context)
    duration = time.perf_counter() - t0

    result = instrumented.result
    timers = instrumented.timers
    relink_processed = result.get("relink_units_processed", 0)
    relink_added = result.get("relink_links_added", 0)
    throughput = relink_processed / duration if duration > 0 else 0
    ms_per_victim = (duration * 1000 / relink_processed) if relink_processed else 0.0

    await engine.delete_bank(bank_id=bank_id, request_context=request_context)
    await engine.close()

    gm_result = GraphMaintenanceResult(
        bank_size=bank_size,
        deleted_units=len(deleted_ids),
        victims_enqueued=victims_enqueued,
        relink_units_processed=relink_processed,
        relink_links_added=relink_added,
        orphan_entities_pruned=result.get("orphan_entities_pruned", 0),
        stale_cooccurrences_pruned=result.get("stale_cooccurrences_pruned", 0),
        total_duration_seconds=round(duration, 3),
        throughput_units_per_sec=round(throughput, 2),
        ms_per_victim=round(ms_per_victim, 2),
        semantic_ann_seconds=round(timers.semantic_seconds, 3),
        semantic_ann_calls=timers.semantic_calls,
        temporal_seconds=round(timers.temporal_seconds, 3),
        temporal_calls=timers.temporal_calls,
    )

    # Print summary
    table = Table(title="Graph Maintenance")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Bank size", f"{bank_size:,}")
    table.add_row("Deleted units", f"{len(deleted_ids):,}")
    table.add_row("Victims enqueued", f"{victims_enqueued:,}")
    table.add_row("Victims processed", f"{relink_processed:,}")
    table.add_row("Links added", f"{relink_added:,}")
    table.add_row("Duration", f"{duration:.3f}s")
    table.add_row("Throughput", f"{throughput:.2f} victims/s")
    table.add_row("Latency / victim", f"{ms_per_victim:.2f} ms")
    table.add_row("Orphan entities pruned", f"{gm_result.orphan_entities_pruned:,}")
    table.add_row("Stale cooccurrences pruned", f"{gm_result.stale_cooccurrences_pruned:,}")
    console.print(table)

    # Probe breakdown — the #1919 investigation. Shows how much of the job is
    # the semantic ANN relink vs the temporal probe vs everything else.
    other = max(0.0, duration - timers.semantic_seconds - timers.temporal_seconds)
    breakdown = Table(title="Relink Probe Breakdown")
    breakdown.add_column("Probe", style="cyan")
    breakdown.add_column("Total", style="green", justify="right")
    breakdown.add_column("Calls", justify="right")
    breakdown.add_column("Avg/call", justify="right")
    breakdown.add_column("% of job", justify="right")

    def _row(label: str, secs: float, calls: int) -> None:
        avg = f"{secs / calls:.3f}s" if calls else "—"
        pct = f"{secs / duration * 100:.1f}%" if duration > 0 else "—"
        breakdown.add_row(label, f"{secs:.3f}s", str(calls), avg, pct)

    _row("semantic ANN", timers.semantic_seconds, timers.semantic_calls)
    _row("temporal", timers.temporal_seconds, timers.temporal_calls)
    breakdown.add_row(
        "other (claim/count/insert/sweep)",
        f"{other:.3f}s",
        "—",
        "—",
        f"{other / duration * 100:.1f}%" if duration > 0 else "—",
    )
    console.print(breakdown)

    return SuiteResult(
        name="graph-maintenance",
        duration_seconds=round(duration, 3),
        success=True,
        graph_maintenance=gm_result,
    )


# ---------------------------------------------------------------------------
# Suite: graph-maintenance-contention  (#2529)
# ---------------------------------------------------------------------------
#
# The graph-maintenance suite above runs run_graph_maintenance_job in ISOLATION
# — populate, delete, one pass, no concurrent writer. That is exactly why
# continuous perf never caught #2529: its Pass 2/3 cooccurrence sweep
# (prune_stale_cooccurrences, an unordered DELETE scan) only deadlocks when it
# overlaps retain's concurrent cooccurrence upsert (entity_resolver
# ._flush_pending, which locks rows in sorted (entity_id_1, entity_id_2) order).
# With no concurrent upsert there is no lock cycle, so the isolated suite can't
# see it, and a dropped maintenance pass shows up in prod only as slow graph
# bloat over time — never as a latency number.
#
# This suite closes that gap: it drives both sides at once and measures how many
# maintenance passes the deadlock silently drops. On main that number is > 0;
# with #2529's retry_with_backoff wrap it is 0 while the deadlocks still happen
# (and get absorbed) — which is the whole point.


async def _seed_contention_fixture(engine: Any, bank_id: str, n_entities: int, n_pairs: int) -> list[tuple]:
    """Seed ``n_entities`` entities plus ``n_pairs`` *stale* cooccurrences among them.

    Every entity is pinned by its own dedicated keeper unit (a unit_entities row),
    so ``prune_orphan_entities`` leaves them alone — but because no single unit
    references two contention entities, every seeded cooccurrence is the exact
    "both entities exist, no current unit witnesses them together" stale case
    that ``prune_stale_cooccurrences`` targets. Returns the sorted pair list.
    """
    from hindsight_api.engine.schema import fq_table

    pool = await engine._get_pool()
    ent_ids = [uuid.uuid4() for _ in range(n_entities)]
    unit_ids = [uuid.uuid4() for _ in range(n_entities)]

    await pool.executemany(
        f"""
        INSERT INTO {fq_table("entities")} (id, bank_id, canonical_name, first_seen, last_seen, mention_count)
        VALUES ($1, $2, $3, NOW(), NOW(), 1)
        """,
        [(eid, bank_id, f"contention-entity-{i}") for i, eid in enumerate(ent_ids)],
    )
    await pool.executemany(
        f"""
        INSERT INTO {fq_table("memory_units")} (id, bank_id, text, fact_type, event_date, created_at, updated_at)
        VALUES ($1, $2, $3, 'experience', NOW(), NOW(), NOW())
        """,
        [(uid, bank_id, f"contention keeper unit {i}") for i, uid in enumerate(unit_ids)],
    )
    await pool.executemany(
        f"INSERT INTO {fq_table('unit_entities')} (unit_id, entity_id) VALUES ($1, $2)",
        list(zip(unit_ids, ent_ids, strict=True)),
    )

    # Distinct sorted pairs (entity_cooccurrence_order_check pins id_1 < id_2).
    rng = random.Random(1234)
    target = min(n_pairs, n_entities * (n_entities - 1) // 2)
    pairs: set[tuple] = set()
    while len(pairs) < target:
        a, b = rng.sample(ent_ids, 2)
        pairs.add((a, b) if a < b else (b, a))
    pair_list = sorted(pairs)

    await pool.executemany(
        f"""
        INSERT INTO {fq_table("entity_cooccurrences")} (entity_id_1, entity_id_2, cooccurrence_count, last_cooccurred)
        VALUES ($1, $2, 1, NOW())
        ON CONFLICT (entity_id_1, entity_id_2) DO NOTHING
        """,
        pair_list,
    )
    return pair_list


async def run_graph_maintenance_contention_suite(scale_cfg: dict[str, int]) -> SuiteResult:
    """Provoke the #2529 cooccurrence-sweep deadlock under concurrent retain load.

    Seeds a hot set of stale cooccurrence pairs, then runs two workloads at once:

    * ``upsert_workers`` tasks re-upserting those pairs in sorted
      ``(entity_id_1, entity_id_2)`` order — a faithful copy of retain's
      ``entity_resolver._flush_pending`` cooccurrence path (same SQL, same lock
      order).
    * ``sweep_workers`` tasks repeatedly calling ``run_graph_maintenance_job``,
      whose Pass 2/3 ``prune_stale_cooccurrences`` DELETEs those same rows in
      an unordered join-scan order.

    Overlapping rows locked in opposite orders → Postgres aborts one side with
    ``DeadlockDetectedError``. When the sweep is the victim, the pass is dropped.
    The suite counts dropped passes: > 0 on main, 0 with the fix.
    """
    from asyncpg.exceptions import DeadlockDetectedError
    from hindsight_api.engine.graph_maintenance import run_graph_maintenance_job
    from hindsight_api.engine.schema import fq_table
    from hindsight_api.models import RequestContext

    n_entities = scale_cfg["graph_contention_entities"]
    n_pairs = scale_cfg["graph_contention_pairs"]
    upsert_workers = scale_cfg["graph_contention_upsert_workers"]
    sweep_workers = scale_cfg["graph_contention_sweep_workers"]
    rounds = scale_cfg["graph_contention_rounds"]
    bank_id = f"perf-graphcont-{uuid.uuid4().hex[:8]}"

    console.print(
        f"\n[bold cyan]Suite: graph-maintenance-contention[/bold cyan]  "
        f"entities={n_entities}  pairs={n_pairs}  upsert_workers={upsert_workers}  "
        f"sweep_workers={sweep_workers}  bank={bank_id}"
    )

    engine = _build_engine(disable_observations=True)
    await engine.initialize()
    await engine.get_bank_profile(bank_id=bank_id, request_context=RequestContext())

    pair_list = await _seed_contention_fixture(engine, bank_id, n_entities, n_pairs)
    console.print(f"  Seeded {n_entities:,} entities + {len(pair_list):,} stale cooccurrence pairs")

    pool = await engine._get_pool()
    request_context = RequestContext()

    # Mirror entity_resolver._flush_pending's cooccurrence upsert exactly.
    upsert_sql = f"""
        INSERT INTO {fq_table("entity_cooccurrences")} (entity_id_1, entity_id_2, cooccurrence_count, last_cooccurred)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (entity_id_1, entity_id_2)
        DO UPDATE SET
            cooccurrence_count = {fq_table("entity_cooccurrences")}.cooccurrence_count + EXCLUDED.cooccurrence_count,
            last_cooccurred    = GREATEST({fq_table("entity_cooccurrences")}.last_cooccurred, EXCLUDED.last_cooccurred)
    """

    counters = _ContentionCounters()
    sweep_latencies: list[float] = []

    # Count raw deadlocks at the prune call — fires on BOTH branches (the fix
    # retries around it, so it still passes through here on every attempt),
    # proving the contention is real rather than that the load vanished.
    backend = await engine._get_backend()
    ops = backend.ops
    orig_prune = ops.prune_stale_cooccurrences

    async def counting_prune(*args: Any, **kwargs: Any) -> Any:
        try:
            return await orig_prune(*args, **kwargs)
        except DeadlockDetectedError:
            counters.deadlocks += 1
            raise

    async def _upsert_worker(seed: int) -> None:
        rng = random.Random(seed)
        subset_k = max(1, int(len(pair_list) * 0.7))
        for _ in range(rounds):
            subset = sorted(rng.sample(pair_list, subset_k))  # sorted → retain's lock order
            now = datetime.now(timezone.utc)
            rows = [(a, b, 1, now) for (a, b) in subset]
            while True:
                try:
                    async with pool.acquire() as conn:
                        async with conn.transaction():
                            await conn.executemany(upsert_sql, rows)
                    counters.upserts += 1
                    break
                except DeadlockDetectedError:
                    # retain retries at a higher level; keep the load flowing.
                    counters.upsert_deadlocks += 1

    async def _sweep_worker(stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            counters.attempted += 1
            t0 = time.perf_counter()
            try:
                await run_graph_maintenance_job(engine, bank_id, request_context)
            except DeadlockDetectedError:
                # The escaping deadlock #2529 fixes — the maintenance pass is lost.
                counters.dropped += 1
            else:
                counters.succeeded += 1
                sweep_latencies.append(time.perf_counter() - t0)

    ops.prune_stale_cooccurrences = counting_prune
    stop_event = asyncio.Event()
    t0 = time.perf_counter()
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            progress.add_task(f"Contending ({upsert_workers} upsert × {sweep_workers} sweep)…")
            sweep_tasks = [asyncio.create_task(_sweep_worker(stop_event)) for _ in range(sweep_workers)]
            await asyncio.gather(*(_upsert_worker(1000 + i) for i in range(upsert_workers)))
            # Let sweeps drain any final overlap, then wind them down.
            await asyncio.sleep(0.5)
            stop_event.set()
            await asyncio.gather(*sweep_tasks)
    finally:
        ops.prune_stale_cooccurrences = orig_prune
    duration = time.perf_counter() - t0

    await engine.delete_bank(bank_id=bank_id, request_context=request_context)
    await engine.close()

    # Escape rate = fraction of contention events that cost a maintenance pass.
    # Denominator is max(observed, dropped) so a drop that bypasses the prune
    # counter (e.g. a deadlock in the orphan-prune FK cascade) still registers
    # as an escape instead of dividing by zero into a false 0%.
    denom = max(counters.deadlocks, counters.dropped)
    escape_rate = counters.dropped / denom if denom else 0.0

    result = GraphContentionResult(
        contention_entities=n_entities,
        contention_pairs=len(pair_list),
        upsert_workers=upsert_workers,
        sweep_workers=sweep_workers,
        total_upserts=counters.upserts,
        upsert_deadlocks=counters.upsert_deadlocks,
        sweep_passes_attempted=counters.attempted,
        sweep_deadlocks_observed=counters.deadlocks,
        sweep_passes_dropped=counters.dropped,
        sweep_passes_succeeded=counters.succeeded,
        sweep_deadlock_escape_rate=round(escape_rate, 3),
        sweep_latency=PercentileStats.from_samples(sweep_latencies),
        total_duration_seconds=round(duration, 3),
    )

    # Hollow-run guard: did the contention workload actually run? Guard on the
    # workloads executing (upserts committed, sweeps attempted) — NOT on seeing
    # deadlocks. A correct source-level fix (sorted FOR UPDATE lock ordering in
    # prune_stale_cooccurrences) *eliminates* the deadlock, so 0 observed is a
    # healthy pass, not a hollow one; gating on deadlocks>0 would wrongly fail
    # the better fix. A genuinely hollow run (broken seed / crashed workers)
    # shows up as no upserts or no sweeps.
    ran_contention = counters.upserts > 0 and counters.attempted > 0
    if not ran_contention:
        console.print(
            "  [yellow]WARNING: contention workload did not run (no upserts/sweeps) — "
            "seed or worker failure, result inconclusive.[/yellow]"
        )
    elif counters.deadlocks == 0:
        console.print(
            "  [green]No deadlocks reproduced — sweep lock ordering prevented the cycle "
            "at the source (not merely retried).[/green]"
        )

    # Healthy either way — deadlocks eliminated (0 observed) or retried away
    # (observed > 0, ~none escape). The gate is the same: no pass was dropped.
    success = ran_contention and escape_rate <= GRAPH_CONTENTION_ESCAPE_RATE_THRESHOLD

    table = Table(title="Graph Maintenance — Contention (#2529)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Contention entities", f"{n_entities:,}")
    table.add_row("Stale pairs", f"{len(pair_list):,}")
    table.add_row("Upsert / sweep workers", f"{upsert_workers} / {sweep_workers}")
    table.add_row("Upserts committed", f"{counters.upserts:,}")
    table.add_row("Upsert-side deadlocks", f"{counters.upsert_deadlocks:,}")
    table.add_row("Sweep passes attempted", f"{counters.attempted:,}")
    table.add_row("Deadlocks observed (at prune)", f"{counters.deadlocks:,}")
    dropped_style = "red" if counters.dropped else "green"
    table.add_row("[bold]Sweep passes DROPPED[/bold]", f"[{dropped_style}]{counters.dropped:,}[/{dropped_style}]")
    rate_style = "red" if escape_rate > GRAPH_CONTENTION_ESCAPE_RATE_THRESHOLD else "green"
    table.add_row("[bold]Deadlock escape rate[/bold]", f"[{rate_style}]{escape_rate:.0%}[/{rate_style}]")
    table.add_row("Sweep passes succeeded", f"{counters.succeeded:,}")
    table.add_row("Sweep latency p50/p95", f"{result.sweep_latency.p50:.3f}s / {result.sweep_latency.p95:.3f}s")
    table.add_row("Duration", f"{duration:.2f}s")
    console.print(table)

    return SuiteResult(
        name="graph-maintenance-contention",
        duration_seconds=round(duration, 3),
        success=success,
        graph_contention=result,
    )


# ---------------------------------------------------------------------------
# Suite: stats
# ---------------------------------------------------------------------------


async def run_stats_suite(scale_cfg: dict[str, int]) -> SuiteResult:
    """Measure the /stats endpoint (get_bank_stats) on a populated bank.

    /stats is polled by the control plane and CLI, so it gets hammered far more
    often than retain/recall. Each cache miss runs a handful of bank-scoped
    aggregations — node counts by fact_type, link counts by link_type, and a
    unit_entities→memory_units rollup to reconstruct the entity-link total — so
    the cost grows with unit/entity density.

    The suite runs with the per-process result cache **disabled** (TTL=0) so the
    headline numbers are the true uncached aggregation — the latency a server
    configured with ``HINDSIGHT_API_BANK_STATS_CACHE_TTL_SECONDS=0`` actually
    serves on every poll. It then re-enables the cache for a second pass to
    quantify what that cache buys:

    * uncached — cache off, every call pays the full aggregation (the metric the
      "disable the cache" decision is about).
    * cached — cache on and warmed, what a steady polling client would see.

    Population is either the real retain pipeline (small ``stats_bank_size``
    scales) or, for the prod-simulation ``huge`` scale, a direct COPY bulk-load
    (``_bulk_populate_stats_bank``) since retain can't reach ~500k units / ~18M
    links in reasonable time.
    """
    from hindsight_api.engine.bank_stats_cache import BankStatsCache
    from hindsight_api.engine.schema import fq_table
    from hindsight_api.models import RequestContext

    bulk = "stats_semantic_links" in scale_cfg
    iterations = scale_cfg["recall_iterations"]
    concurrency = scale_cfg["recall_concurrency"]
    bank_id = f"perf-stats-{uuid.uuid4().hex[:8]}"

    if bulk:
        bank_size = scale_cfg["stats_units"]
        descr = (
            f"units={bank_size:,}  links≈"
            f"{scale_cfg['stats_semantic_links'] + scale_cfg['stats_temporal_links'] + scale_cfg['stats_caused_by_links']:,}"
        )
    else:
        bank_size = scale_cfg["stats_bank_size"]
        descr = f"bank_size={bank_size:,}"

    console.print(
        f"\n[bold cyan]Suite: stats[/bold cyan]  "
        f"{descr}  iterations={iterations}  concurrency={concurrency}  bank={bank_id}"
    )

    engine = _build_engine(disable_observations=True)
    await engine.initialize()

    # Disable the result cache on the server so the uncached pass measures the
    # real aggregation (mirrors HINDSIGHT_API_BANK_STATS_CACHE_TTL_SECONDS=0).
    def _disable_stats_cache() -> None:
        engine._bank_stats_cache = BankStatsCache(ttl_seconds=0, max_entries=0)

    _disable_stats_cache()

    if bulk:
        await _bulk_populate_stats_bank(
            engine,
            bank_id,
            units=scale_cfg["stats_units"],
            semantic_links=scale_cfg["stats_semantic_links"],
            temporal_links=scale_cfg["stats_temporal_links"],
            caused_by_links=scale_cfg["stats_caused_by_links"],
            entity_link_target=scale_cfg["stats_entity_links"],
        )
    else:
        # Real retain pipeline so units, links and entities all exist — the
        # entity rollup join is the part that scales with the bank.
        await _populate_bank(engine, bank_id, bank_size)

    request_context = RequestContext()
    pool = await engine._get_pool()

    units_row = await pool.fetchrow(
        f"SELECT COUNT(*) AS count FROM {fq_table('memory_units')} WHERE bank_id = $1",
        bank_id,
    )
    links_row = await pool.fetchrow(
        f"SELECT COUNT(*) AS count FROM {fq_table('memory_links')} WHERE bank_id = $1",
        bank_id,
    )
    # Derived entity-link total — the same unit_entities rollup _compute_bank_stats
    # reports as link_counts["entity"] (entity edges aren't stored; each entity
    # contributes LEAST(units_sharing - 1, 10) links). This is the prod-comparable
    # "entity" number, not the raw distinct-entity count.
    entities_row = await pool.fetchrow(
        f"""
        WITH per_entity AS (
            SELECT ue.entity_id, COUNT(*) AS n
            FROM {fq_table("unit_entities")} ue
            JOIN {fq_table("memory_units")} mu ON mu.id = ue.unit_id
            WHERE mu.bank_id = $1
            GROUP BY ue.entity_id
        )
        SELECT COALESCE(SUM(LEAST(n - 1, 10)), 0)::bigint AS count FROM per_entity
        """,
        bank_id,
    )
    total_units = int(units_row["count"]) if units_row else 0
    total_links = int(links_row["count"]) if links_row else 0
    total_entities = int(entities_row["count"]) if entities_row else 0

    async def _run_batches(label: str, call: "Callable[[], Any]") -> list[float]:
        durations: list[float] = []

        async def time_one() -> float:
            t0 = time.perf_counter()
            await call()
            return time.perf_counter() - t0

        remaining = iterations
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(label, total=iterations)
            while remaining > 0:
                batch = min(concurrency, remaining)
                durations.extend(await asyncio.gather(*[time_one() for _ in range(batch)]))
                remaining -= batch
                progress.advance(task, batch)
        return durations

    # Uncached: cache is disabled, so every get_bank_stats pays the full
    # aggregation — the endpoint latency with the cache turned off.
    cold_durations = await _run_batches(
        "Running stats (cache disabled / uncached)…",
        lambda: engine.get_bank_stats(bank_id, request_context=request_context),
    )
    # Cached: enable + warm the cache, then measure what a polling client sees.
    engine._bank_stats_cache = BankStatsCache(ttl_seconds=300.0, max_entries=16)
    await engine.get_bank_stats(bank_id, request_context=request_context)  # prime
    warm_durations = await _run_batches(
        "Running stats (cache enabled / warm)…",
        lambda: engine.get_bank_stats(bank_id, request_context=request_context),
    )
    _disable_stats_cache()  # leave the engine as the suite found it

    await engine.delete_bank(bank_id=bank_id, request_context=request_context)
    await engine.close()

    cold_stats = PercentileStats.from_samples(cold_durations)
    warm_stats = PercentileStats.from_samples(warm_durations)
    cold_total = sum(cold_durations)
    cold_throughput = iterations / (cold_total / concurrency) if cold_total > 0 else 0
    cache_speedup = cold_stats.p50 / warm_stats.p50 if warm_stats.p50 > 0 else 0

    stats_result = StatsResult(
        bank_size=bank_size,
        concurrency=concurrency,
        total_units=total_units,
        total_links=total_links,
        total_entities=total_entities,
        cold_latency=cold_stats,
        cold_throughput_per_sec=round(cold_throughput, 2),
        warm_latency=warm_stats,
        cache_speedup=round(cache_speedup, 1),
    )

    table = Table(title="Bank Stats Latency (cache disabled)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row(
        "Units / physical links / entity links (derived)",
        f"{total_units:,} / {total_links:,} / {total_entities:,}",
    )
    table.add_row("Iterations / concurrency", f"{iterations} / {concurrency}")
    table.add_row("Uncached throughput", f"{cold_throughput:.2f} calls/s")
    table.add_row("Uncached mean", f"{cold_stats.mean:.3f}s")
    table.add_row("Uncached p50 / p95 / p99", f"{cold_stats.p50:.3f}s / {cold_stats.p95:.3f}s / {cold_stats.p99:.3f}s")
    table.add_row("Cached mean", f"{warm_stats.mean:.4f}s")
    table.add_row("Cached p50 / p95 / p99", f"{warm_stats.p50:.4f}s / {warm_stats.p95:.4f}s / {warm_stats.p99:.4f}s")
    table.add_row("Cache speedup (p50)", f"{cache_speedup:.1f}x")
    console.print(table)

    return SuiteResult(
        name="stats",
        duration_seconds=round(cold_total, 3),
        success=True,
        stats=stats_result,
    )


# ---------------------------------------------------------------------------
# Registry and orchestrator
# ---------------------------------------------------------------------------

SUITES = {
    "retain": run_retain_suite,
    "recall": run_recall_suite,
    "recall-with-observations": run_recall_with_observations_suite,
    "recall-temporal": run_recall_temporal_suite,
    "consolidation": run_consolidation_suite,
    "graph-maintenance": run_graph_maintenance_suite,
    "graph-maintenance-contention": run_graph_maintenance_contention_suite,
    "stats": run_stats_suite,
}


async def run(scale: str, suite_names: list[str]) -> PerfTestResults:
    """Run selected suites and collect results."""
    scale_cfg = SCALES[scale]
    git_sha = _get_git_sha()

    console.print("\n[bold]System Performance Test[/bold]")
    console.print(f"  Scale  : {scale}")
    console.print(f"  Suites : {', '.join(suite_names)}")
    console.print(f"  Git SHA: {git_sha}")

    results = PerfTestResults(
        timestamp=datetime.now(timezone.utc).isoformat(),
        scale=scale,
        git_sha=git_sha,
    )

    t_total = time.perf_counter()

    for name in suite_names:
        runner = SUITES[name]
        try:
            suite_result = await runner(scale_cfg)
        except Exception as e:
            console.print(f"\n[bold red]Suite {name} failed: {e}[/bold red]")
            suite_result = SuiteResult(name=name, duration_seconds=0, success=False, error=str(e))
        results.suites.append(suite_result)

    total_duration = time.perf_counter() - t_total
    failed = [s for s in results.suites if not s.success]
    if failed:
        console.print(f"\n[bold red]{len(failed)} suite(s) failed in {total_duration:.1f}s[/bold red]")
    else:
        console.print(f"\n[bold green]All suites completed in {total_duration:.1f}s[/bold green]")

    return results


def _serialize(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    return obj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="System performance test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scale",
        choices=list(SCALES),
        default="small",
        help="Test scale (default: small)",
    )
    parser.add_argument(
        "--suite",
        choices=list(SUITES),
        action="append",
        dest="suites",
        help="Run specific suite (can be repeated; default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write JSON results to file",
    )

    args = parser.parse_args()
    suite_names = args.suites or list(SUITES)

    results = asyncio.run(run(args.scale, suite_names))
    results_dict = _serialize(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results_dict, f, indent=2)
        console.print(f"\n[dim]Results written to {args.output}[/dim]")

    # Print unified summary table
    _print_summary(results)

    # Exit with failure if any suite failed
    if any(not s.success for s in results.suites):
        raise SystemExit(1)


def _print_summary(results: PerfTestResults) -> None:
    """Print a unified summary table across all suites."""
    console.print(f"\n[bold]{'=' * 60}[/bold]")
    console.print(f"[bold]Performance Report[/bold]  scale={results.scale}  sha={results.git_sha}  {results.timestamp}")
    console.print(f"[bold]{'=' * 60}[/bold]")

    table = Table(title="Results Summary", show_lines=True)
    table.add_column("Suite", style="bold cyan")
    table.add_column("Status", justify="center")
    table.add_column("Metric", style="white")
    table.add_column("Value", style="green", justify="right")
    table.add_column("p50", justify="right")
    table.add_column("p95", justify="right")
    table.add_column("p99", justify="right")

    for suite in results.suites:
        status = "[green]PASS[/green]" if suite.success else "[red]FAIL[/red]"

        if suite.retain:
            r = suite.retain
            table.add_row(
                suite.name,
                status,
                "throughput",
                f"{r.throughput_items_per_sec} items/s",
                "",
                "",
                "",
            )
            table.add_row(
                "",
                "",
                "duration",
                f"{r.total_duration_seconds}s",
                "",
                "",
                "",
            )
            table.add_row(
                "",
                "",
                "items",
                str(r.total_items),
                "",
                "",
                "",
            )

        if suite.recall:
            rc = suite.recall
            lat = rc.latency
            table.add_row(
                suite.name,
                status,
                "throughput",
                f"{rc.throughput_queries_per_sec} q/s",
                "",
                "",
                "",
            )
            table.add_row(
                "",
                "",
                "latency",
                f"mean={lat.mean:.3f}s",
                f"{lat.p50:.3f}s",
                f"{lat.p95:.3f}s",
                f"{lat.p99:.3f}s",
            )
            table.add_row(
                "",
                "",
                "bank/concurrency",
                f"{rc.bank_size:,} / {rc.concurrency}",
                "",
                "",
                "",
            )
            # Phase breakdown — top 5 by mean
            sorted_phases = sorted(rc.phase_timings.items(), key=lambda x: x[1].mean, reverse=True)
            for phase_name, ps in sorted_phases[:5]:
                table.add_row(
                    "",
                    "",
                    f"  {phase_name}",
                    f"mean={ps.mean:.3f}s",
                    f"{ps.p50:.3f}s",
                    f"{ps.p95:.3f}s",
                    f"{ps.p99:.3f}s",
                )

        if suite.consolidation:
            c = suite.consolidation
            table.add_row(
                suite.name,
                status,
                "throughput",
                f"{c.throughput_memories_per_sec} mem/s",
                "",
                "",
                "",
            )
            table.add_row(
                "",
                "",
                "duration",
                f"{c.total_duration_seconds}s",
                "",
                "",
                "",
            )
            table.add_row(
                "",
                "",
                "processed/total",
                f"{c.memories_processed:,} / {c.total_items:,}",
                "",
                "",
                "",
            )
            table.add_row(
                "",
                "",
                "created/updated/merged/skipped",
                f"{c.observations_created}/{c.observations_updated}/{c.observations_merged}/{c.skipped}",
                "",
                "",
                "",
            )

        if suite.graph_maintenance:
            g = suite.graph_maintenance
            table.add_row(
                suite.name,
                status,
                "throughput",
                f"{g.throughput_units_per_sec} victims/s",
                "",
                "",
                "",
            )
            table.add_row("", "", "duration", f"{g.total_duration_seconds}s", "", "", "")
            table.add_row("", "", "latency/victim", f"{g.ms_per_victim} ms", "", "", "")
            table.add_row(
                "",
                "",
                "victims (enq/proc)",
                f"{g.victims_enqueued:,} / {g.relink_units_processed:,}",
                "",
                "",
                "",
            )
            table.add_row(
                "",
                "",
                "  semantic ANN",
                f"{g.semantic_ann_seconds}s ({g.semantic_ann_calls} calls)",
                "",
                "",
                "",
            )
            table.add_row(
                "",
                "",
                "  temporal",
                f"{g.temporal_seconds}s ({g.temporal_calls} calls)",
                "",
                "",
                "",
            )

        if suite.graph_contention:
            gc = suite.graph_contention
            drop_val = f"[red]{gc.sweep_passes_dropped}[/red]" if gc.sweep_passes_dropped else "[green]0[/green]"
            table.add_row(suite.name, status, "sweep passes dropped", drop_val, "", "", "")
            rate_red = gc.sweep_deadlock_escape_rate > GRAPH_CONTENTION_ESCAPE_RATE_THRESHOLD
            rate_val = f"[{'red' if rate_red else 'green'}]{gc.sweep_deadlock_escape_rate:.0%}[/]"
            table.add_row("", "", "deadlock escape rate", rate_val, "", "", "")
            table.add_row("", "", "deadlocks observed", f"{gc.sweep_deadlocks_observed:,}", "", "", "")
            table.add_row(
                "",
                "",
                "passes (ok/attempted)",
                f"{gc.sweep_passes_succeeded:,} / {gc.sweep_passes_attempted:,}",
                "",
                "",
                "",
            )
            table.add_row(
                "",
                "",
                "sweep latency",
                f"mean={gc.sweep_latency.mean:.3f}s",
                f"{gc.sweep_latency.p50:.3f}s",
                f"{gc.sweep_latency.p95:.3f}s",
                f"{gc.sweep_latency.p99:.3f}s",
            )
            table.add_row(
                "", "", "upserts (ok/deadlocked)", f"{gc.total_upserts:,} / {gc.upsert_deadlocks:,}", "", "", ""
            )

        if suite.stats:
            st = suite.stats
            table.add_row(
                suite.name,
                status,
                "uncached latency",
                f"mean={st.cold_latency.mean:.3f}s",
                f"{st.cold_latency.p50:.3f}s",
                f"{st.cold_latency.p95:.3f}s",
                f"{st.cold_latency.p99:.3f}s",
            )
            table.add_row(
                "",
                "",
                "cached latency",
                f"mean={st.warm_latency.mean:.4f}s",
                f"{st.warm_latency.p50:.4f}s",
                f"{st.warm_latency.p95:.4f}s",
                f"{st.warm_latency.p99:.4f}s",
            )
            table.add_row("", "", "uncached throughput", f"{st.cold_throughput_per_sec} calls/s", "", "", "")
            table.add_row("", "", "cache speedup (p50)", f"{st.cache_speedup}x", "", "", "")
            table.add_row(
                "",
                "",
                "units/phys-links/entity-links",
                f"{st.total_units:,} / {st.total_links:,} / {st.total_entities:,}",
                "",
                "",
                "",
            )

        if not suite.success:
            table.add_row(suite.name, status, "error", suite.error or "unknown", "", "", "")

    console.print(table)


if __name__ == "__main__":
    main()
