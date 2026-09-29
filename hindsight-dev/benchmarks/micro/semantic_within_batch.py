"""Microbenchmark for ``compute_semantic_links_within_batch`` (retain phase 2).

Timing and memory are measured in *separate* passes on purpose: ``tracemalloc`` taxes every
allocation, and it taxes the allocation-heavy baseline hardest, so timing a traced run
understates the speedup by up to 4x.

Wall time is the best of N runs, and the before/after rounds are *interleaved* rather than run
back to back. On a thermally loaded laptop the same measurement drifts by more than 2x over a
few minutes, so running all of "before" and then all of "after" attributes that drift to the
code. Interleaving cancels it. Treat the speedup column as a ratio with roughly +/-20% of play
in it; the memory column is exact and repeats to the byte.
"""

import argparse
import gc
import time
import tracemalloc
from array import array
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from hindsight_api.engine.retain.link_utils import compute_semantic_links_within_batch
from rich.console import Console
from rich.table import Table

console = Console()

# Frozen snapshot of the implementation before #3977 -- float64 throughout, full argsort per
# row. Kept verbatim so the table has a real "before" column; it is not imported from the
# engine on purpose, since the point is to compare against code that no longer exists.
_BASELINE_BLOCK_ROWS = 256


def _baseline(unit_ids: list[str], embeddings: Sequence, top_k: int = 50, *, threshold: float) -> list[tuple]:
    if len(unit_ids) < 2:
        return []
    links = []
    matrix = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(matrix, axis=1)
    valid = np.isfinite(matrix).all(axis=1) & np.isfinite(norms) & (norms > 0)
    normalized = np.zeros_like(matrix)
    normalized[valid] = matrix[valid] / norms[valid, np.newaxis]
    for start in range(0, len(unit_ids), _BASELINE_BLOCK_ROWS):
        stop = min(start + _BASELINE_BLOCK_ROWS, len(unit_ids))
        block = normalized[start:stop] @ normalized.T
        block[:, ~valid] = -np.inf
        for i in range(start, stop):
            if not valid[i]:
                continue
            similarities = block[i - start]
            similarities[i] = -np.inf
            above = np.where(similarities >= threshold)[0]
            if len(above) > 0:
                for other in above[np.argsort(-similarities[above])][:top_k]:
                    score = float(min(1.0, max(0.0, similarities[other])))
                    links.append((unit_ids[i], unit_ids[other], "semantic", score, None))
    return links


@dataclass(frozen=True)
class Workload:
    """One synthetic similarity regime to measure the pass against."""

    threshold: float
    description: str


# "clustered" is the one that resembles a real retain batch: topical clusters, so a realistic
# fraction of pairs clears the default 0.7 threshold. "sparse" deliberately produces no links at
# all, which isolates the matrix product from the selection and materialisation work.
WORKLOADS = {
    "sparse": Workload(0.7, "uncorrelated facts, near-zero links -- isolates the matrix product"),
    "clustered": Workload(0.7, "topical clusters at the default threshold -- the realistic retain batch"),
    "dense": Workload(0.3, "everything similar -- stresses candidate extraction and top-k"),
}


@dataclass(frozen=True)
class Batch:
    unit_ids: list[str]
    embeddings: list[array]


def _make_batch(n: int, kind: str, dim: int, seed: int = 0) -> Batch:
    """Embeddings as ``array("f")`` -- the ``PackedEmbedding`` the retain pipeline carries."""
    rng = np.random.default_rng(seed)
    if kind == "sparse":
        matrix = rng.normal(size=(n, dim))
    elif kind == "clustered":
        centers = rng.normal(size=(max(2, n // 25), dim))
        matrix = centers[rng.integers(0, len(centers), n)] + rng.normal(scale=0.55, size=(n, dim))
    else:
        matrix = rng.random(size=(n, dim))
    return Batch(
        unit_ids=[f"u{i}" for i in range(n)],
        embeddings=[array("f", row.tolist()) for row in matrix.astype(np.float32)],
    )


@dataclass(frozen=True)
class Round:
    elapsed_ms: float
    links: int


@dataclass(frozen=True)
class Timings:
    before_ms: float
    after_ms: float
    links: int

    @property
    def speedup(self) -> float:
        return self.before_ms / self.after_ms


def _one_round(fn: Callable, batch: Batch, threshold: float, top_k: int) -> Round:
    gc.collect()
    started = time.perf_counter()
    links = fn(batch.unit_ids, batch.embeddings, top_k=top_k, threshold=threshold)
    elapsed = (time.perf_counter() - started) * 1000
    count = len(links)
    del links
    return Round(elapsed_ms=elapsed, links=count)


def _interleaved_best(
    before: Callable, after: Callable, batch: Batch, threshold: float, top_k: int, repeats: int
) -> Timings:
    """Alternate the two implementations so CPU frequency drift hits both equally."""
    for fn in (before, after):
        fn(batch.unit_ids, batch.embeddings, top_k=top_k, threshold=threshold)  # warm BLAS and the allocator
    best_before = best_after = float("inf")
    links = 0
    for _ in range(repeats):
        best_before = min(best_before, _one_round(before, batch, threshold, top_k).elapsed_ms)
        round_after = _one_round(after, batch, threshold, top_k)
        best_after = min(best_after, round_after.elapsed_ms)
        links = round_after.links
    return Timings(before_ms=best_before, after_ms=best_after, links=links)


def _peak_mb(fn: Callable, batch: Batch, threshold: float, top_k: int) -> float:
    gc.collect()
    tracemalloc.start()
    links = fn(batch.unit_ids, batch.embeddings, top_k=top_k, threshold=threshold)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del links
    return peak / (1024 * 1024)


def run_benchmark(
    n_values: Sequence[int],
    dim: int,
    top_k: int,
    repeats: int,
    workloads: Sequence[str],
) -> None:
    for kind in workloads:
        workload = WORKLOADS[kind]
        table = Table(title=f"{kind} (threshold={workload.threshold}) -- {workload.description}", title_justify="left")
        table.add_column("N", justify="right", style="cyan")
        table.add_column("Links", justify="right")
        table.add_column("Before (ms)", justify="right")
        table.add_column("After (ms)", justify="right", style="green")
        table.add_column("Speedup", justify="right", style="bold green")
        table.add_column("Before (MB)", justify="right")
        table.add_column("After (MB)", justify="right", style="yellow")
        table.add_column("Peak RAM", justify="right", style="bold yellow")

        for n in n_values:
            batch = _make_batch(n, kind, dim)
            timings = _interleaved_best(
                _baseline, compute_semantic_links_within_batch, batch, workload.threshold, top_k, repeats
            )
            before_mb = _peak_mb(_baseline, batch, workload.threshold, top_k)
            after_mb = _peak_mb(compute_semantic_links_within_batch, batch, workload.threshold, top_k)
            table.add_row(
                str(n),
                f"{timings.links:,}",
                f"{timings.before_ms:.2f}",
                f"{timings.after_ms:.2f}",
                f"{timings.speedup:.2f}x",
                f"{before_mb:.2f}",
                f"{after_mb:.2f}",
                f"-{(1 - after_mb / before_mb) * 100:.0f}%",
            )
        console.print(table)
        console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 500, 1700, 5000])
    parser.add_argument("--dim", type=int, default=1536, help="embedding dimensions (default: 1536)")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workloads", nargs="+", choices=sorted(WORKLOADS), default=sorted(WORKLOADS))
    args = parser.parse_args()
    run_benchmark(args.sizes, args.dim, args.top_k, args.repeats, args.workloads)


if __name__ == "__main__":
    main()
