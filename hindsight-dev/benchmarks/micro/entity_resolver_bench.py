"""Microbenchmark for entity resolver and intrabatch deduplication (PERF-R17 / PERF-R18 / P0-17).

Measures wall time, CPU time, and peak memory (tracemalloc) comparing:
- ``baseline_quadratic``: Previous O(N^2) pairwise comparisons without candidate caching;
- ``length_pruned_quadratic``: O(N^2) with length pre-pruning;
- ``prefix_filtering (prod)``: Production Prefix Filtering Principle (All-Pairs Set Similarity Join).

Usage:
    ./scripts/benchmarks/run-entity-resolver-bench.sh
    ./scripts/benchmarks/run-entity-resolver-bench.sh --repeats 10
"""

import argparse
import gc
import json
import os
import random
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

from hindsight_api.engine.entity_resolver import (
    _find_intrabatch_similar_pairs,
    _SimilarNamePair,
    _trigram_set,
)
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass(frozen=True)
class Workload:
    """A batch of entity names reflecting different real-world retain/dedup scenarios."""

    name: str
    description: str
    entity_names: list[str]
    threshold: float

    @property
    def total_entities(self) -> int:
        return len(self.entity_names)


@dataclass
class VariantResult:
    """One variant measured against one workload."""

    workload: str
    variant: str
    wall_ms: float  # best of --repeats, milliseconds
    cpu_ms: float  # process CPU (all threads) over that same best run
    peak_kib: float  # tracemalloc peak of a separate single run
    total_entities: int
    pairs_found: int
    matches_baseline: bool


# --- Baseline and Variant implementations ---


def _v_prefix_filtering(names: Sequence[str], threshold: float) -> list[_SimilarNamePair]:
    """Production implementation using Prefix Filtering."""
    return _find_intrabatch_similar_pairs(list(names), threshold)


def _v_baseline_quadratic(names: Sequence[str], threshold: float) -> list[_SimilarNamePair]:
    """Previous baseline: O(N^2) double-loop with len(ta & tb) and uncached trigrams."""
    trigrams = [_trigram_set(n) for n in names]
    pairs: list[_SimilarNamePair] = []
    n = len(names)
    for i in range(n):
        ti = trigrams[i]
        for j in range(i + 1, n):
            # len(ta & tb) allocation churn
            inter = len(ti & trigrams[j])
            union = len(ti) + len(trigrams[j]) - inter
            if union and (inter / union) >= threshold:
                pairs.append(_SimilarNamePair(name_a=names[i], name_b=names[j]))
    return pairs


def _v_length_pruned_quadratic(names: Sequence[str], threshold: float) -> list[_SimilarNamePair]:
    """Intermediate variant: O(N^2) with length pre-pruning."""
    trigrams = [_trigram_set(n) for n in names]
    trigrams_with_len = [(t, len(t)) for t in trigrams]
    pairs: list[_SimilarNamePair] = []
    n = len(names)
    for i in range(n):
        ti, len_i = trigrams_with_len[i]
        if not len_i:
            continue
        for j in range(i + 1, n):
            tj, len_j = trigrams_with_len[j]
            if not len_j:
                continue
            if len_i < len_j:
                if len_i / len_j < threshold:
                    continue
            elif len_j / len_i < threshold:
                continue
            inter = len(ti & tj)
            union = len_i + len_j - inter
            if union and (inter / union) >= threshold:
                pairs.append(_SimilarNamePair(name_a=names[i], name_b=names[j]))
    return pairs


def build_workloads(seed: int = 1234) -> list[Workload]:
    rng = random.Random(seed)

    base_entities = [
        "Barack Obama",
        "Apple Inc.",
        "New York City",
        "Dr. John Watson",
        "OpenAI GPT-4",
        "OpenAI GPT 4",
        "Microsoft Corporation",
        "Google LLC",
        "Amazon Web Services",
        "Meta Platforms Inc",
        "Nvidia RTX GPU",
        "Elon Musk",
        "Tesla Motors",
        "SpaceX Falcon 9",
        "DeepMind Technologies",
        "Anthropic Claude 3.5",
        "San Francisco",
        "State of California",
        "Los Angeles County",
        "Seattle WA",
        "Chicago Illinois",
        "Python Software Foundation",
        "Rust Language Foundation",
        "TypeScript",
        "PostgreSQL pg_trgm",
        "Linux Kernel Development",
        "Wren 🕯️",
        "Wren 🗯️",
        "Aster 🔑",
        "aster 0",
        "ke-aster",
        "Merrivale",
        "Merryvale",
        "Corvin",
        "Corvyn",
        "Astrid",
        "José García",
        "Jose Garcia",
        "北京",
        "北京市",
        "Jean-Luc",
        "Jean Luc",
    ]

    def make_batch(target_count: int) -> list[str]:
        res = []
        while len(res) < target_count:
            chosen = rng.choice(base_entities)
            # occasionally introduce slight noise/suffix
            r = rng.random()
            if r < 0.2:
                suffix = rng.choice([" Inc", " LLC", " 2", " Corp", " - New"])
                res.append(chosen + suffix)
            else:
                res.append(chosen)
        return res[:target_count]

    return [
        Workload(
            name="micro_batch_20",
            description="Minimal retain batch (20 entities, 190 comparisons)",
            entity_names=make_batch(20),
            threshold=0.5,
        ),
        Workload(
            name="small_batch_50",
            description="Typical standard retain batch (50 entities, 1,225 comparisons)",
            entity_names=make_batch(50),
            threshold=0.5,
        ),
        Workload(
            name="cap_batch_250",
            description="Previous system cap size (250 entities, 31,125 comparisons)",
            entity_names=make_batch(250),
            threshold=0.5,
        ),
        Workload(
            name="large_doc_500",
            description="Large document import (500 entities, 124,750 comparisons)",
            entity_names=make_batch(500),
            threshold=0.5,
        ),
        Workload(
            name="bulk_import_1000",
            description="Bulk knowledge import batch (1000 entities, 499,500 comparisons)",
            entity_names=make_batch(1000),
            threshold=0.5,
        ),
    ]


def build_variants() -> dict[str, Callable[[Sequence[str], float], list[_SimilarNamePair]]]:
    return {
        "prefix_filtering (prod)": _v_prefix_filtering,
        "length_pruned_quadratic": _v_length_pruned_quadratic,
        "baseline_quadratic": _v_baseline_quadratic,
    }


@dataclass(frozen=True)
class Timing:
    wall_ms: float
    cpu_ms: float
    pairs: list[_SimilarNamePair]


def _measure(
    fn: Callable[[Sequence[str], float], list[_SimilarNamePair]],
    names: Sequence[str],
    threshold: float,
    repeats: int,
) -> Timing:
    fn(names, threshold)  # warm up
    best_wall = float("inf")
    best_cpu = 0.0
    res: list[_SimilarNamePair] = []
    for _ in range(repeats):
        gc.collect()
        t0, c0 = time.perf_counter(), time.process_time()
        res = fn(names, threshold)
        wall = time.perf_counter() - t0
        cpu = time.process_time() - c0
        if wall < best_wall:
            best_wall, best_cpu = wall, cpu
    return Timing(wall_ms=best_wall * 1000, cpu_ms=best_cpu * 1000, pairs=res)


def _measure_peak_kib(
    fn: Callable[[Sequence[str], float], list[_SimilarNamePair]],
    names: Sequence[str],
    threshold: float,
) -> float:
    gc.collect()
    tracemalloc.start()
    try:
        fn(names, threshold)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 1024


def run(workloads: Sequence[Workload], repeats: int) -> list[VariantResult]:
    variants = build_variants()
    results: list[VariantResult] = []

    for wl in workloads:
        baseline_pairs: set[tuple[str, str]] | None = None
        for name, fn in variants.items():
            timing = _measure(fn, wl.entity_names, wl.threshold, repeats)
            peak_kib = _measure_peak_kib(fn, wl.entity_names, wl.threshold)
            pair_set = {(p.name_a, p.name_b) if p.name_a < p.name_b else (p.name_b, p.name_a) for p in timing.pairs}

            if baseline_pairs is None:
                baseline_pairs = pair_set

            results.append(
                VariantResult(
                    workload=wl.name,
                    variant=name,
                    wall_ms=timing.wall_ms,
                    cpu_ms=timing.cpu_ms,
                    peak_kib=peak_kib,
                    total_entities=wl.total_entities,
                    pairs_found=len(pair_set),
                    matches_baseline=(pair_set == baseline_pairs),
                )
            )
    return results


def _render(workloads: Sequence[Workload], results: list[VariantResult]) -> None:
    by_wl: dict[str, list[VariantResult]] = {}
    for r in results:
        by_wl.setdefault(r.workload, []).append(r)

    for wl in workloads:
        rows = by_wl.get(wl.name, [])
        if not rows:
            continue
        baseline = next((r for r in rows if "baseline" in r.variant), rows[0])

        table = Table(
            title=f"[bold]{wl.name}[/bold] — {wl.description} (N={wl.total_entities})",
            title_justify="left",
        )
        table.add_column("variant", style="cyan")
        table.add_column("wall ms", justify="right")
        table.add_column("speedup", justify="right", style="green")
        table.add_column("cpu ms", justify="right")
        table.add_column("peak KiB", justify="right")
        table.add_column("mem Δ", justify="right")
        table.add_column("pairs", justify="right")
        table.add_column("status", justify="center")

        for r in rows:
            speedup = baseline.wall_ms / r.wall_ms if r.wall_ms > 0 else float("inf")
            mem_diff = r.peak_kib - baseline.peak_kib
            if r.variant == baseline.variant:
                mem_str = "baseline"
            elif mem_diff > 0:
                mem_str = f"+{mem_diff:,.1f} KiB"
            else:
                mem_str = f"{mem_diff:,.1f} KiB"

            table.add_row(
                r.variant,
                f"{r.wall_ms:.3f}",
                f"{speedup:.2f}x" if r.variant != baseline.variant else "—",
                f"{r.cpu_ms:.3f}",
                f"{r.peak_kib:,.1f}",
                mem_str,
                f"{r.pairs_found}",
                "[green]exact[/green]" if r.matches_baseline else "[red]MISMATCH[/red]",
            )
        console.print(table)
        console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=5, help="timed repeats per variant (best-of); default 5")
    parser.add_argument("--seed", type=int, default=1234, help="seed for entity generation; default 1234")
    parser.add_argument("--json", dest="json_path", help="also write raw results to this path")
    args = parser.parse_args()

    workloads = build_workloads(args.seed)

    console.print(
        f"[dim]Running Entity Resolver & In-batch Dedup microbenchmarks | cpu_count={os.cpu_count()} | repeats={args.repeats}[/dim]\n"
    )
    results = run(workloads, repeats=args.repeats)
    _render(workloads, results)

    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        console.print(f"[dim]wrote {args.json_path}[/dim]")


if __name__ == "__main__":
    main()
