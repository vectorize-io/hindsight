"""Latency harness for temporal extraction.

Two things are measured, because they answer different questions:

* **solo** — one call at a time. This is the pure cost of the code path, and
  the number to optimise.
* **burst** — N calls issued simultaneously on one asyncio event loop, which is
  how recall actually calls it (``retrieve_all_fact_types_parallel`` invokes
  ``extract_temporal_constraint`` inline in an ``async def``). Because the work
  is CPU-bound Python, the loop serialises it, so a caller's observed latency
  includes everything queued ahead of it. This is the number the p99 gate uses:
  it is what a concurrent client actually experiences.

Run standalone::

    uv run python -m tests.query_analyzer_bench --concurrency 32 64 128
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime

from tests.query_analyzer_corpus import REFERENCE_DATE, build_corpus, build_perf_workload


@dataclass
class LatencyStats:
    label: str
    samples: list[float] = field(default_factory=list)

    def add(self, ms: float) -> None:
        self.samples.append(ms)

    @property
    def n(self) -> int:
        return len(self.samples)

    def pct(self, p: float) -> float:
        if not self.samples:
            return float("nan")
        ordered = sorted(self.samples)
        # Nearest-rank percentile; index clamped into range.
        idx = min(len(ordered) - 1, max(0, int(round(p / 100.0 * len(ordered) + 0.5)) - 1))
        return ordered[idx]

    @property
    def mean(self) -> float:
        return statistics.mean(self.samples) if self.samples else float("nan")

    @property
    def max(self) -> float:
        return max(self.samples) if self.samples else float("nan")

    def row(self) -> str:
        return (
            f"  {self.label:<34} n={self.n:>6}  mean={self.mean:7.3f}  "
            f"p50={self.pct(50):7.3f}  p95={self.pct(95):7.3f}  "
            f"p99={self.pct(99):7.3f}  max={self.max:8.3f}"
        )


AnalyzeFn = Callable[[str, datetime], object]


def _default_fn() -> AnalyzeFn:
    from hindsight_api.engine.search.temporal_extraction import extract_temporal_constraint

    from hindsight_api.engine.query_analyzer import DateparserQueryAnalyzer

    analyzer = DateparserQueryAnalyzer()
    analyzer.load()

    def call(query: str, ref: datetime):
        return extract_temporal_constraint(query, reference_date=ref, analyzer=analyzer)

    return call


def warmup(fn: AnalyzeFn, queries: Iterable[str], rounds: int = 2) -> None:
    for _ in range(rounds):
        for q in queries:
            fn(q, REFERENCE_DATE)


def measure_solo(fn: AnalyzeFn, queries: list[str], repeats: int = 3) -> LatencyStats:
    """One call at a time — the pure code-path cost."""
    stats = LatencyStats("solo")
    for _ in range(repeats):
        for q in queries:
            t = time.perf_counter()
            fn(q, REFERENCE_DATE)
            stats.add((time.perf_counter() - t) * 1000.0)
    return stats


def measure_by_category(fn: AnalyzeFn, repeats: int = 3) -> list[LatencyStats]:
    """Per-branch breakdown, so a regression can be attributed to a rule class."""
    by_cat: dict[str, LatencyStats] = {}
    corpus = build_corpus()
    for _ in range(repeats):
        for query, category in corpus:
            stats = by_cat.setdefault(category, LatencyStats(category))
            t = time.perf_counter()
            fn(query, REFERENCE_DATE)
            stats.add((time.perf_counter() - t) * 1000.0)
    return sorted(by_cat.values(), key=lambda s: -s.pct(99))


async def _burst_once(fn: AnalyzeFn, queries: list[str], concurrency: int, stats: LatencyStats, offset: int) -> None:
    """Issue `concurrency` calls simultaneously; record each one's own latency.

    Every coroutine timestamps immediately before its own call, so the recorded
    latency is submit-to-complete for that caller: the loop is already busy with
    the callers ahead of it, and that queueing delay is real client-visible
    latency, not an artefact.
    """
    start_barrier = asyncio.Event()

    async def one(idx: int) -> None:
        await start_barrier.wait()
        # Rotate by round so a burst smaller than the workload still sweeps the
        # whole mix over successive rounds; otherwise burst@32 would only ever
        # see workload[:32] and the gate would measure one biased slice.
        q = queries[(offset + idx) % len(queries)]
        t = time.perf_counter()
        fn(q, REFERENCE_DATE)
        stats.add((time.perf_counter() - t) * 1000.0)

    tasks = [asyncio.create_task(one(i)) for i in range(concurrency)]
    await asyncio.sleep(0)  # let every task reach the barrier
    start_barrier.set()
    await asyncio.gather(*tasks)


async def measure_burst(fn: AnalyzeFn, queries: list[str], concurrency: int, rounds: int | None = None) -> LatencyStats:
    """Run enough rounds that every workload query is issued at least once."""
    if rounds is None:
        rounds = max(12, -(-len(queries) // concurrency))
    stats = LatencyStats(f"burst@{concurrency}")
    for r in range(rounds):
        await _burst_once(fn, queries, concurrency, stats, offset=r * concurrency)
    return stats


def run_suite(fn: AnalyzeFn | None = None, concurrencies: tuple[int, ...] = (32, 64, 128)) -> dict[str, LatencyStats]:
    fn = fn or _default_fn()
    workload = build_perf_workload()
    warmup(fn, workload[:60])

    results: dict[str, LatencyStats] = {}
    solo = measure_solo(fn, workload)
    results["solo"] = solo

    for c in concurrencies:
        stats = asyncio.run(measure_burst(fn, workload, c))
        results[f"burst@{c}"] = stats
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--categories", action="store_true", help="per-branch breakdown")
    args = ap.parse_args()

    fn = _default_fn()
    workload = build_perf_workload()
    print(f"workload: {len(workload)} queries, corpus: {len(build_corpus())} cases\n")
    warmup(fn, workload[:60])

    print("latency (ms)")
    print(measure_solo(fn, workload).row())
    for c in args.concurrency:
        print(asyncio.run(measure_burst(fn, workload, c)).row())

    if args.categories:
        print("\nper-branch (solo, ms)")
        for stats in measure_by_category(fn):
            print(stats.row())


if __name__ == "__main__":
    main()
