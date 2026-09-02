"""Microbenchmark for candidate entity string similarity matching on retain entity resolution paths.

Entity resolution compares extracted entity names against historical bank candidates:
* ``entity_resolver._tokens_match`` — checks word-level compatibility for multi-word names;
* ``entity_resolver._resolve_from_candidates`` — scores each candidate's name similarity (0~0.5 points).

The baseline implementation used pure-Python ``difflib.SequenceMatcher``:
    SequenceMatcher(None, entity_text_lower, canonical_lower).ratio()
For batches with hundreds to thousands of candidates, pure-Python dynamic programming consumed
several milliseconds of synchronous CPU on the event-loop thread, risking health probe timeouts.

The variants measured are:
``prod``
    The production ``rapidfuzz.distance.GestaltPatternMatching.normalized_similarity`` call (C++ SIMD).
``baseline_sequencematcher``
    The previous baseline: ``difflib.SequenceMatcher(None, a, b).ratio()``.

Usage (from the repo root):
    ./scripts/benchmarks/run-entity-matcher-bench.sh
    ./scripts/benchmarks/run-entity-matcher-bench.sh --repeats 10 --json out.json
    ./scripts/benchmarks/run-entity-matcher-bench.sh --workload large_batch_1000
"""

import argparse
import gc
import json
import math
import os
import random
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any

from rapidfuzz.distance.Indel import (
    normalized_similarity as _similarity_ratio,
)
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass(frozen=True)
class Workload:
    """A batch of entity pairs shaped like actual production entity resolution candidate scoring."""

    name: str
    description: str
    pairs: list[tuple[str, str]]

    @property
    def total_pairs(self) -> int:
        return len(self.pairs)


@dataclass
class VariantResult:
    """One variant measured against one workload."""

    workload: str
    variant: str
    wall_ms: float  # best of --repeats, milliseconds
    cpu_ms: float  # process CPU (all threads) over that same best run
    peak_kib: float  # tracemalloc peak of a separate single run
    total_pairs: int
    pairs_per_sec: float
    matches_baseline: bool


# --- Variant implementations ---


def _v_prod(pairs: Sequence[tuple[str, str]]) -> list[float]:
    return [_similarity_ratio(a, b) for a, b in pairs]


def _v_baseline_sequencematcher(pairs: Sequence[tuple[str, str]]) -> list[float]:
    return [SequenceMatcher(None, a, b).ratio() for a, b in pairs]


VARIANTS: dict[str, tuple[str, Callable[[Sequence[tuple[str, str]]], list[float]]]] = {
    "prod": ("Production (rapidfuzz C++ Indel/LCS)", _v_prod),
    "baseline_sequencematcher": ("Baseline (difflib.SequenceMatcher)", _v_baseline_sequencematcher),
}


# --- Synthetic & realistic workload generation ---


def _build_candidate_pairs(num_pairs: int, seed: int = 42) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    base_names = [
        "Dr. Johnathan Smith",
        "Jane Doe-Smith",
        "Google Cloud Platform",
        "Apple Computer Inc",
        "Amazon Web Services (AWS)",
        "PostgreSQL Relational Database",
        "Kubernetes Container Orchestrator",
        "San Francisco Bay Area",
        "University of California, Berkeley",
        "Michael Bloomberg",
        "Alexander the Great",
        "Artificial General Intelligence",
        "Vectorize AI Engine",
        "DeepMind Technologies",
        "Microsoft Azure Cloud",
    ]

    pairs: list[tuple[str, str]] = []
    for i in range(num_pairs):
        base = base_names[i % len(base_names)].lower()
        mode = rng.randint(0, 4)
        if mode == 0:
            # Exact match
            other = base
        elif mode == 1:
            # Single typo / character swap
            idx = rng.randint(0, len(base) - 1)
            other = base[:idx] + rng.choice("abcdefghijklmnopqrstuvwxyz") + base[idx + 1 :]
        elif mode == 2:
            # Abbreviation or truncated form
            words = base.split()
            other = " ".join(words[: max(1, len(words) - 1)]) if len(words) > 1 else base[: max(1, len(base) - 2)]
        elif mode == 3:
            # Decoration or prefix / suffix addition
            other = f"{base} corp" if rng.random() > 0.5 else f"the {base}"
        else:
            # Slightly related other name
            other = rng.choice(base_names).lower()
        pairs.append((base, other))
    return pairs


def make_workloads() -> dict[str, Workload]:
    return {
        "typical_batch_50": Workload(
            name="typical_batch_50",
            description="Typical Retain batch: 50 candidate pairs scored against DB",
            pairs=_build_candidate_pairs(50, seed=101),
        ),
        "medium_batch_200": Workload(
            name="medium_batch_200",
            description="Medium Retain batch: 200 candidate pairs (capped entity resolution max)",
            pairs=_build_candidate_pairs(200, seed=202),
        ),
        "large_batch_1000": Workload(
            name="large_batch_1000",
            description="Large Retain batch: 1000 candidate pairs across wide entity mentions",
            pairs=_build_candidate_pairs(1000, seed=303),
        ),
        "stress_batch_5000": Workload(
            name="stress_batch_5000",
            description="Stress scenario: 5000 candidate pairs simulating heavy entity resolution",
            pairs=_build_candidate_pairs(5000, seed=404),
        ),
        "extreme_limit_50000": Workload(
            name="extreme_limit_50000",
            description="True upper bound: 50,000 candidate pairs (250 new entities x 200 max candidates)",
            pairs=_build_candidate_pairs(50000, seed=505),
        ),
        "mega_scale_1000000": Workload(
            name="mega_scale_1000000",
            description="Mega scale stress: 1,000,000 candidate pairs (large-scale batch / full scan)",
            pairs=_build_candidate_pairs(1000000, seed=606),
        ),
    }


# --- Measurement harness ---


def _measure_time(fn: Callable[[], Any], repeats: int) -> tuple[float, float, Any]:
    best_wall = float("inf")
    best_cpu = float("inf")
    last_res = None
    for _ in range(repeats):
        gc.collect()
        t0_wall = time.perf_counter()
        t0_cpu = time.process_time()
        res = fn()
        t1_wall = time.perf_counter()
        t1_cpu = time.process_time()
        wall = (t1_wall - t0_wall) * 1000.0
        cpu = (t1_cpu - t0_cpu) * 1000.0
        if wall < best_wall:
            best_wall = wall
            best_cpu = cpu
            last_res = res
    return best_wall, best_cpu, last_res


def _measure_memory_kib(fn: Callable[[], Any]) -> float:
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak_bytes / 1024.0


def benchmark_one(
    workload: Workload, variant_key: str, repeats: int, baseline_results: list[float] | None
) -> VariantResult:
    _, fn = VARIANTS[variant_key]
    wall_ms, cpu_ms, output = _measure_time(lambda: fn(workload.pairs), repeats)
    peak_kib = _measure_memory_kib(lambda: fn(workload.pairs))

    # Assert decision & score equivalence (>=0.6 merge threshold agreement and score consistency)
    matches = True
    if baseline_results is not None:
        for a, b in zip(output, baseline_results):
            # 1. Decision agreement on merge threshold (0.6)
            if (a >= 0.6) != (b >= 0.6):
                matches = False
                break
            # 2. Score closeness on relevant candidates (>=0.5)
            if max(a, b) >= 0.5 and not math.isclose(a, b, abs_tol=0.01):
                matches = False
                break

    pairs_sec = (workload.total_pairs / (wall_ms / 1000.0)) if wall_ms > 0 else 0.0

    return VariantResult(
        workload=workload.name,
        variant=variant_key,
        wall_ms=wall_ms,
        cpu_ms=cpu_ms,
        peak_kib=peak_kib,
        total_pairs=workload.total_pairs,
        pairs_per_sec=pairs_sec,
        matches_baseline=matches,
    )


# --- CLI and reporting ---


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of iterations per measurement; records best wall time (default: %(default)s).",
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="all",
        help="Specific workload to run, or 'all' (default: %(default)s).",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        metavar="PATH",
        help="Write raw benchmark results to this JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workloads = make_workloads()
    selected: list[Workload]
    if args.workload == "all":
        selected = list(workloads.values())
    elif args.workload in workloads:
        selected = [workloads[args.workload]]
    else:
        console.print(f"[bold red]Unknown workload:[/bold red] {args.workload}. Available: {list(workloads.keys())}")
        raise SystemExit(2)

    console.rule("[bold cyan]Hindsight Entity Candidate String Matching Microbenchmark[/bold cyan]")
    console.print(
        "Comparing [bold green]prod (rapidfuzz C++)[/bold green] vs [bold yellow]difflib.SequenceMatcher[/bold yellow]"
    )
    console.print(f"Repeats: {args.repeats} per measurement | Workloads: {len(selected)}")
    console.print()

    all_results: list[VariantResult] = []

    for wl in selected:
        console.print(f"[bold blue]Workload:[/bold blue] {wl.name} ({wl.description})")
        # 1. Run baseline first to get reference scores
        _, base_fn = VARIANTS["baseline_sequencematcher"]
        base_res = benchmark_one(wl, "baseline_sequencematcher", args.repeats, None)
        baseline_outputs = base_fn(wl.pairs)

        # 2. Run prod variant and compare
        prod_res = benchmark_one(wl, "prod", args.repeats, baseline_outputs)

        all_results.extend([prod_res, base_res])

        # Print comparison table
        table = Table(title=f"Results for {wl.name} ({wl.total_pairs} pairs)", header_style="bold magenta")
        table.add_column("Variant", style="dim", no_wrap=True)
        table.add_column("Wall Time", justify="right")
        table.add_column("CPU Time", justify="right")
        table.add_column("Speedup", justify="right")
        table.add_column("Throughput", justify="right")
        table.add_column("Peak Memory", justify="right")
        table.add_column("Memory Cut", justify="right")
        table.add_column("Exact Match", justify="center")

        speedup_ratio = base_res.wall_ms / prod_res.wall_ms if prod_res.wall_ms > 0 else 1.0
        mem_diff_kib = base_res.peak_kib - prod_res.peak_kib
        mem_reduction_pct = (mem_diff_kib / base_res.peak_kib * 100.0) if base_res.peak_kib > 0 else 0.0

        # Prod row
        table.add_row(
            "[bold green]prod (rapidfuzz)[/bold green]",
            f"[bold]{prod_res.wall_ms:.3f} ms[/bold]",
            f"{prod_res.cpu_ms:.3f} ms",
            f"[bold green]{speedup_ratio:.2f}x[/bold green]",
            f"{prod_res.pairs_per_sec / 1000.0:.1f}k pairs/s",
            f"{prod_res.peak_kib:.1f} KiB",
            f"-{mem_reduction_pct:.1f}%" if mem_reduction_pct > 0 else f"{mem_reduction_pct:.1f}%",
            "[green]YES (100%)[/green]" if prod_res.matches_baseline else "[red]NO[/red]",
        )
        # Baseline row
        table.add_row(
            "baseline_sequencematcher",
            f"{base_res.wall_ms:.3f} ms",
            f"{base_res.cpu_ms:.3f} ms",
            "1.00x",
            f"{base_res.pairs_per_sec / 1000.0:.1f}k pairs/s",
            f"{base_res.peak_kib:.1f} KiB",
            "baseline",
            "[green]YES (ref)[/green]",
        )
        console.print(table)
        console.print()

    if args.json:
        out_path = os.path.abspath(args.json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        console.print(f"[bold green]Wrote JSON results to:[/bold green] {out_path}")


if __name__ == "__main__":
    main()
