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

    # Configurable scale
    uv run perf-test --scale tiny      # ~10s, CI smoke test
    uv run perf-test --scale small     # ~30s, default
    uv run perf-test --scale medium    # ~2min
    uv run perf-test --scale large     # ~10min

    # Save results as JSON
    uv run perf-test --output results.json
"""

import argparse
import asyncio
import json
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# Reuse battle-tested building blocks from existing benchmarks
from benchmarks.perf.recall_perf import (
    FACT_TEMPLATES,
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
    },
    "small": {
        "retain_items": 200,
        "recall_bank_size": 200,
        "recall_iterations": 20,
        "recall_concurrency": 4,
        "consolidation_items": 200,
    },
    "medium": {
        "retain_items": 1_000,
        "recall_bank_size": 1_000,
        "recall_iterations": 50,
        "recall_concurrency": 8,
        "consolidation_items": 1_000,
    },
    "large": {
        "retain_items": 5_000,
        "recall_bank_size": 5_000,
        "recall_iterations": 100,
        "recall_concurrency": 16,
        "consolidation_items": 5_000,
    },
}

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
class SuiteResult:
    name: str
    duration_seconds: float
    success: bool
    error: str | None = None
    retain: RetainResult | None = None
    recall: RecallResult | None = None
    consolidation: ConsolidationResult | None = None


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


async def _populate_bank(engine: Any, bank_id: str, size: int) -> None:
    """Populate a bank with synthetic data using mock LLM + async retain."""
    from hindsight_api.models import RequestContext
    from hindsight_api.worker.poller import WorkerPoller

    _attach_mock_callback(engine)

    contents = [{"content": _fill_template(FACT_TEMPLATES[i % len(FACT_TEMPLATES)])} for i in range(size)]

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
# Registry and orchestrator
# ---------------------------------------------------------------------------

SUITES = {
    "retain": run_retain_suite,
    "recall": run_recall_suite,
    "recall-with-observations": run_recall_with_observations_suite,
    "consolidation": run_consolidation_suite,
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

        if not suite.success:
            table.add_row(suite.name, status, "error", suite.error or "unknown", "", "", "")

    console.print(table)


if __name__ == "__main__":
    main()
