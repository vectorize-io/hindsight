"""Microbenchmark for pgvector float vector serialization on retain and import paths.

Retain and import paths convert embeddings to pgvector vector literals:
* ``memories.pg.writes.insert_facts_batch`` — serializes 50~500 facts per batch into pgvector literals;
* ``retain.link_utils.compute_semantic_links_within_batch`` / temp table generation — prepares records;
* ``memories.pg.writes.update_memory_unit_embedding`` — updates individual memory embeddings.

The baseline implementation used a Python generator:
    "[" + ",".join(repr(float(value)) for value in embedding) + "]"
For 500 facts (768,000 floats), this allocated 768,000 PyFloat objects + 768,000 PyUnicode
strings, consuming ~200ms of pure CPU time and generating ~15MB of allocation churn.

The variants measured are:
``prod``
    The production ``embedding_to_pgvector`` call (zero-copy orjson SIMD Ryu formatting with fallback).
``listcomp_str``
    List comprehension with ``str()``: ``"[" + ",".join([str(v) for v in emb]) + "]"``
``baseline_generator``
    The previous generator baseline: ``"[" + ",".join(repr(float(v)) for v in emb) + "]"``

Usage (from the repo root):
    ./scripts/benchmarks/run-vector-serialization-bench.sh
    ./scripts/benchmarks/run-vector-serialization-bench.sh --repeats 10 --json out.json
    ./scripts/benchmarks/run-vector-serialization-bench.sh --workload batch_200_openai_1536
"""

import argparse
import gc
import json
import math
import os
import random
import time
import tracemalloc
from array import array
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from hindsight_api.engine.retain.types import embedding_to_pgvector
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass(frozen=True)
class Workload:
    """A batch of embedding vectors shaped like actual production call sites."""

    name: str
    call_site: str
    batch_size: int
    dim: int
    data_type: str  # "packed" (array('f')), "list" (list[float]), "numpy" (np.ndarray)
    vectors: list[Any]

    @property
    def total_floats(self) -> int:
        return self.batch_size * self.dim


@dataclass
class VariantResult:
    """One variant measured against one workload."""

    workload: str
    variant: str
    wall_ms: float  # best of --repeats, milliseconds
    cpu_ms: float  # process CPU (all threads) over that same best run
    peak_kib: float  # tracemalloc peak of a separate single run
    total_floats: int
    m_floats_per_sec: float
    matches_baseline: bool


# --- Variant implementations ---


def _v_prod(vectors: Sequence[Any]) -> list[str]:
    return [embedding_to_pgvector(emb) for emb in vectors]


def _v_baseline_generator(vectors: Sequence[Any]) -> list[str]:
    res = []
    for emb in vectors:
        if isinstance(emb, str):
            res.append(emb)
        else:
            res.append("[" + ",".join(repr(float(value)) for value in emb) + "]")
    return res


def _v_listcomp_str(vectors: Sequence[Any]) -> list[str]:
    res = []
    for emb in vectors:
        if isinstance(emb, str):
            res.append(emb)
        else:
            res.append("[" + ",".join([str(value) for value in emb]) + "]")
    return res


def build_workloads(seed: int) -> list[Workload]:
    rng = random.Random(seed)

    def make_vector(dim: int) -> list[float]:
        return [rng.uniform(-1.0, 1.0) for _ in range(dim)]

    def make_packed_batch(count: int, dim: int) -> list[array]:
        return [array("f", make_vector(dim)) for _ in range(count)]

    def make_list_batch(count: int, dim: int) -> list[list[float]]:
        return [make_vector(dim) for _ in range(count)]

    return [
        Workload(
            name="single_bge_384",
            call_site="single MiniLM/BGE fact update (384d)",
            batch_size=1,
            dim=384,
            data_type="packed",
            vectors=make_packed_batch(1, 384),
        ),
        Workload(
            name="single_openai_1536",
            call_site="single OpenAI fact insert (1536d)",
            batch_size=1,
            dim=1536,
            data_type="packed",
            vectors=make_packed_batch(1, 1536),
        ),
        Workload(
            name="batch_20_gemini_768",
            call_site="Gemini-001 conversational retain sub-batch (768d, 20 facts)",
            batch_size=20,
            dim=768,
            data_type="packed",
            vectors=make_packed_batch(20, 768),
        ),
        Workload(
            name="batch_200_openai_1536",
            call_site="typical OpenAI retain batch (1536d, 200 facts)",
            batch_size=200,
            dim=1536,
            data_type="packed",
            vectors=make_packed_batch(200, 1536),
        ),
        Workload(
            name="batch_500_large_doc_1536",
            call_site="large document retain batch (1536d, 500 facts)",
            batch_size=500,
            dim=1536,
            data_type="packed",
            vectors=make_packed_batch(500, 1536),
        ),
        Workload(
            name="batch_200_raw_list_1536",
            call_site="raw API list[float] before packing (1536d, 200 facts)",
            batch_size=200,
            dim=1536,
            data_type="list",
            vectors=make_list_batch(200, 1536),
        ),
    ]


def build_variants() -> dict[str, Callable[[Sequence[Any]], list[str]]]:
    return {
        "baseline_generator": _v_baseline_generator,
        "listcomp_str": _v_listcomp_str,
        "prod": _v_prod,
    }


@dataclass(frozen=True)
class Timing:
    wall_ms: float
    cpu_ms: float
    results: list[str]


def _measure(fn: Callable[[Sequence[Any]], list[str]], vectors: Sequence[Any], repeats: int) -> Timing:
    fn(vectors[:1])  # warmup
    best_wall = float("inf")
    best_cpu = 0.0
    res = []
    for _ in range(repeats):
        gc.collect()
        t0, c0 = time.perf_counter(), time.process_time()
        res = fn(vectors)
        wall = time.perf_counter() - t0
        cpu = time.process_time() - c0
        if wall < best_wall:
            best_wall, best_cpu = wall, cpu
    return Timing(wall_ms=best_wall * 1000, cpu_ms=best_cpu * 1000, results=res)


def _measure_peak_kib(fn: Callable[[Sequence[Any]], list[str]], vectors: Sequence[Any]) -> float:
    gc.collect()
    tracemalloc.start()
    try:
        fn(vectors)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 1024


def _check_conformance_output(baseline_str: str, candidate_str: str) -> bool:
    if not (candidate_str.startswith("[") and candidate_str.endswith("]")):
        return False
    b_parts = baseline_str[1:-1].split(",") if len(baseline_str) > 2 else []
    c_parts = candidate_str[1:-1].split(",") if len(candidate_str) > 2 else []
    if len(b_parts) != len(c_parts):
        return False
    for b_item, c_item in zip(b_parts, c_parts):
        b_val = float(b_item)
        c_val = float(c_item)
        if math.isnan(b_val):
            if not math.isnan(c_val):
                return False
        elif math.isinf(b_val):
            if not (math.isinf(c_val) and (b_val > 0) == (c_val > 0)):
                return False
        else:
            if abs(b_val - c_val) > 1e-6 and not math.isclose(b_val, c_val, rel_tol=1e-5, abs_tol=1e-6):
                return False
    return True


_CONFORMANCE_CASES: dict[str, Any] = {
    "standard_floats": array("f", [0.1, -0.25, 3.5, 0.0, -1.0, 1e-7, 123.456]),
    "subnormal_small": array("f", [1e-15, -1e-20, 1e-35]),
    "large_floats": array("f", [1e5, -2.5e6, 3.4e38]),
    "negative_zero": array("f", [-0.0, 0.0]),
    "non_finite": array("f", [float("nan"), float("inf"), -float("inf"), 1.0]),
    "empty_vector": array("f", []),
    "already_string": "[0.123,-0.456,7.89]",
    "raw_float_list": [0.1, -0.25, 3.5, 0.0, -1.0, 1e-7, 123.456],
    "raw_float_tuple": (0.1, -0.25, 3.5, 0.0, -1.0, 1e-7, 123.456),
    "numpy_array_f32": np.array([0.1, -0.25, 3.5, 0.0, -1.0, 1e-7, 123.456], dtype=np.float32),
}


def _conformance() -> None:
    variants = build_variants()
    table = Table(title="[bold]conformance[/bold] — vector serialization vs baseline", title_justify="left")
    table.add_column("test case")
    for name in variants:
        table.add_column(name, justify="right")

    for case, inp in _CONFORMANCE_CASES.items():
        cells: list[str] = []
        baseline_res = _v_baseline_generator([inp])[0]
        for name, fn in variants.items():
            try:
                got = fn([inp])[0]
            except Exception as err:
                cells.append(f"[red]{type(err).__name__}[/red]")
                continue
            is_valid = _check_conformance_output(baseline_res, got)
            if is_valid:
                cells.append("[green]ok[/green]")
            else:
                cells.append("[red]MISMATCH[/red]")
        table.add_row(case, *cells)

    console.print(table)
    console.print()


def run(workloads: Sequence[Workload], repeats: int) -> list[VariantResult]:
    variants = build_variants()
    results: list[VariantResult] = []
    for wl in workloads:
        baseline_res: list[str] | None = None
        for name, fn in variants.items():
            try:
                timing = _measure(fn, wl.vectors, repeats)
                peak_kib = _measure_peak_kib(fn, wl.vectors)
            except Exception as err:
                console.print(f"[red]{wl.name}/{name}: {type(err).__name__}: {err}[/red]")
                continue

            if baseline_res is None:
                baseline_res = timing.results

            matches = True
            if len(timing.results) != len(baseline_res):
                matches = False
            else:
                for b_str, c_str in zip(baseline_res, timing.results):
                    if not _check_conformance_output(b_str, c_str):
                        matches = False
                        break

            m_floats_s = (wl.total_floats / timing.wall_ms / 1000) if timing.wall_ms > 0 else 0.0
            results.append(
                VariantResult(
                    workload=wl.name,
                    variant=name,
                    wall_ms=timing.wall_ms,
                    cpu_ms=timing.cpu_ms,
                    peak_kib=peak_kib,
                    total_floats=wl.total_floats,
                    m_floats_per_sec=m_floats_s,
                    matches_baseline=matches,
                )
            )
    return results


def _render(workloads: Sequence[Workload], results: Sequence[VariantResult]) -> None:
    by_workload: dict[str, list[VariantResult]] = {}
    for r in results:
        by_workload.setdefault(r.workload, []).append(r)

    for wl in workloads:
        rows = by_workload.get(wl.name, [])
        if not rows:
            continue
        baseline = rows[0]
        table = Table(
            title=(
                f"[bold]{wl.name}[/bold] — {wl.call_site}  "
                f"({wl.batch_size} fact(s), {wl.dim}d, {wl.total_floats:,} floats, {wl.data_type})"
            ),
            title_justify="left",
        )
        table.add_column("variant")
        table.add_column("wall ms", justify="right")
        table.add_column("speedup", justify="right")
        table.add_column("cpu ms", justify="right")
        table.add_column("cpu/wall", justify="right")
        table.add_column("peak KiB", justify="right")
        table.add_column("Mfloat/s", justify="right")
        table.add_column("conformance", justify="right")

        for r in rows:
            speedup = baseline.wall_ms / r.wall_ms if r.wall_ms > 0 else float("inf")
            table.add_row(
                r.variant,
                f"{r.wall_ms:.3f}",
                "—" if r.variant == baseline.variant else f"{speedup:.2f}x",
                f"{r.cpu_ms:.3f}",
                f"{(r.cpu_ms / r.wall_ms):.2f}" if r.wall_ms > 0 else "—",
                f"{r.peak_kib:,.0f}",
                f"{r.m_floats_per_sec:.2f}",
                "[green]ok[/green]" if r.matches_baseline else "[red]MISMATCH[/red]",
            )
        console.print(table)
        console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=5, help="timed repeats per variant (best-of); default 5")
    parser.add_argument("--seed", type=int, default=1234, help="seed for vector generation; default 1234")
    parser.add_argument(
        "--workload",
        action="append",
        help="run only this workload (repeatable); default all",
    )
    parser.add_argument("--json", dest="json_path", help="also write raw results to this path")
    parser.add_argument(
        "--no-conformance",
        action="store_true",
        help="skip the conformance check on adversarial inputs",
    )
    args = parser.parse_args()

    workloads = build_workloads(args.seed)
    if args.workload:
        wanted = set(args.workload)
        unknown = wanted - {w.name for w in workloads}
        if unknown:
            parser.error(f"unknown workload(s): {', '.join(sorted(unknown))}")
        workloads = [w for w in workloads if w.name in wanted]

    console.print(
        f"[dim]Running pgvector serialization microbenchmarks | cpu_count={os.cpu_count()} | repeats={args.repeats}[/dim]\n"
    )
    if not args.no_conformance:
        _conformance()
    results = run(workloads, repeats=args.repeats)
    _render(workloads, results)

    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        console.print(f"[dim]wrote {args.json_path}[/dim]")


if __name__ == "__main__":
    main()
