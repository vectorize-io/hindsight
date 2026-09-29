"""Microbenchmark for the in-batch entity name dedup on the retain path.

``EntityResolver`` folds same-batch surface variants of a new entity name together before
inserting them, by finding every pair of new names whose pg_trgm similarity clears the merge
cutoff. Measures wall time, CPU time and peak memory (tracemalloc) for three ways of finding
those pairs:

- ``baseline_quadratic``: the O(N^2) double loop this replaced — also the conformance reference;
- ``length_pruned_quadratic``: the same loop with a set-size pre-filter, the obvious cheaper fix;
- ``prefix_filtering (prod)``: what ``_find_intrabatch_similar_pairs`` does today.

Every variant must return the same pairs; the ``status`` column says whether it did. Workloads
are built from *distinct* names because the caller passes ``rep_by_lower.values()`` — one entry
per distinct new name in the batch — and include the shape that defeats prefix filtering, so a
regression on it is visible here rather than only in review.

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
    matches_baseline: bool  # same pairs as baseline_quadratic, which is the reference for every variant


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


# Surname / given-name / place fragments recombined into distinct names. The production caller
# passes `rep_by_lower.values()` — one entry per *distinct* new name in the batch — so a workload
# built by sampling a small pool with replacement measures a batch shape that cannot occur.
_GIVEN = "James Mary John Patricia Robert Jennifer Michael Linda William Elizabeth David Barbara Richard Susan Joseph Jessica Thomas Sarah Charles Karen Christopher Nancy Daniel Lisa Matthew Betty Anthony Margaret Mark Sandra Donald Ashley Steven Kimberly Paul Emily Andrew Donna Joshua Michelle".split()
_FAMILY = "Smith Johnson Williams Brown Jones Garcia Miller Davis Rodriguez Martinez Hernandez Lopez Gonzalez Wilson Anderson Thomas Taylor Moore Jackson Martin Lee Perez Thompson White Harris Sanchez Clark Ramirez Lewis Robinson Walker Young Allen King Wright Scott Torres Nguyen Hill Flores".split()
_ORG_SUFFIX = "Systems Labs Holdings Group Ventures Partners Industries Technologies Solutions Networks Dynamics Analytics Robotics Biosciences Capital Foundation Institute Media Logistics Energy".split()
_PLACE = "Springfield Riverton Fairview Kingston Ashland Georgetown Clinton Salem Madison Franklin Greenville Bristol Newport Oxford Milton Dover Arlington Burlington Manchester Hudson".split()
_PLACE_KIND = ["County", "City", "Township", "District", "Metro Area"]


def _distinct_entity_names(count: int, rng: random.Random) -> list[str]:
    """``count`` distinct people / orgs / places, with the surface variants a retain batch carries.

    Roughly a sixth of the names are a near-miss of one already emitted — an accent dropped, a
    suffix added, a transposed letter — which is the material the in-batch dedup exists to catch.
    """
    names: list[str] = []
    seen: set[str] = set()

    def emit(name: str) -> None:
        if name.lower() not in seen:
            seen.add(name.lower())
            names.append(name)

    while len(names) < count:
        roll = rng.random()
        if roll < 0.5:
            emit(f"{rng.choice(_GIVEN)} {rng.choice(_FAMILY)}")
        elif roll < 0.8:
            emit(f"{rng.choice(_FAMILY)} {rng.choice(_ORG_SUFFIX)}")
        else:
            emit(f"{rng.choice(_PLACE)} {rng.choice(_PLACE_KIND)}")
        if names and rng.random() < 0.2:
            source = rng.choice(names)
            variant = rng.choice(
                [
                    source.replace("a", "á", 1),
                    f"{source} Inc",
                    f"{source.lower()}",
                    source.replace("e", "", 1),
                ]
            )
            emit(variant)
    return names[:count]


def build_workloads(seed: int = 1234) -> list[Workload]:
    rng = random.Random(seed)

    return [
        Workload(
            name="micro_batch_20",
            description="Minimal retain batch (20 distinct names, 190 comparisons)",
            entity_names=_distinct_entity_names(20, rng),
            threshold=0.5,
        ),
        Workload(
            name="small_batch_50",
            description="Typical retain batch (50 distinct names, 1,225 comparisons)",
            entity_names=_distinct_entity_names(50, rng),
            threshold=0.5,
        ),
        Workload(
            name="cap_batch_250",
            description="At the _INTRABATCH_MAX_NAMES cap (250 distinct names, 31,125 comparisons)",
            entity_names=_distinct_entity_names(250, rng),
            threshold=0.5,
        ),
        Workload(
            name="cap_batch_250_low_cutoff",
            description="At the cap with the merge cutoff lowered to 0.2 (more names survive the filter)",
            entity_names=_distinct_entity_names(250, rng),
            threshold=0.2,
        ),
        Workload(
            name="adversarial_250_alike",
            description="Worst case: 250 mutually similar names, prefix filter prunes nothing",
            entity_names=[f"Acme Corporation Subsidiary {i:04d}" for i in range(250)],
            threshold=0.5,
        ),
        Workload(
            name="above_cap_500",
            description="Above the current cap (500 distinct names, 124,750 comparisons)",
            entity_names=_distinct_entity_names(500, rng),
            threshold=0.5,
        ),
        Workload(
            name="above_cap_500_alike",
            description="Above the cap, worst case: 500 mutually similar names",
            entity_names=[f"Acme Corporation Subsidiary {i:04d}" for i in range(500)],
            threshold=0.5,
        ),
        Workload(
            name="bulk_import_1000",
            description="Bulk import (1000 distinct names, 499,500 comparisons)",
            entity_names=_distinct_entity_names(1000, rng),
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


def _normalised_pairs(pairs: Sequence[_SimilarNamePair]) -> set[tuple[str, str]]:
    """Pairs as an order-insensitive set, so variants are compared on membership alone."""
    return {(p.name_a, p.name_b) if p.name_a < p.name_b else (p.name_b, p.name_a) for p in pairs}


def run(workloads: Sequence[Workload], repeats: int) -> list[VariantResult]:
    variants = build_variants()
    results: list[VariantResult] = []

    for wl in workloads:
        # Conformance is measured against the quadratic loop specifically, not against whichever
        # variant happened to run first — taking the first would have compared the optimisation
        # to itself and reported "exact" however wrong it was.
        baseline_pairs = _normalised_pairs(_v_baseline_quadratic(wl.entity_names, wl.threshold))
        for name, fn in variants.items():
            timing = _measure(fn, wl.entity_names, wl.threshold, repeats)
            peak_kib = _measure_peak_kib(fn, wl.entity_names, wl.threshold)
            pair_set = _normalised_pairs(timing.pairs)

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
