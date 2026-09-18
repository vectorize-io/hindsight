"""Microbenchmark for Fact Extraction prompt and schema generation on the retain path.

Retain splits input text into chunks and processes them in parallel. Prior to this optimization,
`_build_extraction_prompt_and_schema` was invoked dynamically on each chunk inside
`build_chunk_prompt_parts`, triggering repetitive `create_model()` class generation,
JSON schema AST traversals, and prompt string template formatting.

This benchmark measures wall time, process CPU time, and peak memory (tracemalloc)
for two variants across multiple document chunk sizes and bank configuration scenarios:

- `unhoisted_baseline`: Rebuilding prompt and dynamic Pydantic schema on every chunk;
- `hoisted_prod`: Compiling prompt and schema once per retain item/batch and reusing.

Usage:
    ./scripts/benchmarks/run-fact-extraction-bench.sh
    ./scripts/benchmarks/run-fact-extraction-bench.sh --repeats 10
    ./scripts/benchmarks/run-fact-extraction-bench.sh --json /tmp/fact_extraction_bench.json
"""

import argparse
import gc
import json
import os
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from types import SimpleNamespace

from hindsight_api.engine.retain.fact_extraction import (
    ExtractionPrompt,
    _build_extraction_prompt_and_schema,
    build_chunk_prompt_parts,
)
from hindsight_api.engine.structured_output import strict_json_schema
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass(frozen=True)
class Workload:
    """A benchmark scenario reflecting document size and bank configuration."""

    name: str
    description: str
    total_chunks: int
    config: SimpleNamespace


@dataclass
class VariantResult:
    """One variant measured against one workload."""

    workload: str
    variant: str
    wall_ms: float  # best of --repeats, milliseconds
    cpu_ms: float  # process CPU (all threads) over that same best run
    peak_kib: float  # tracemalloc peak of a separate single run
    total_chunks: int
    matches_baseline: bool  # prompt and schema exactly match baseline


def _make_config(supports_pattern: bool = False, entity_labels: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        retain_extraction_mode="concise",
        retain_extract_causal_links=True,
        retain_custom_instructions=None,
        retain_mission=None,
        entity_labels=entity_labels,
        entities_allow_free_form=True,
        llm_output_language=None,
        llm_supports_string_pattern=supports_pattern,
    )


def build_workloads() -> list[Workload]:
    sample_labels = {
        "attributes": [
            {"key": "sentiment", "type": "value", "optional": False, "values": [{"value": "pos"}, {"value": "neg"}]},
            {"key": "category", "type": "multi-values", "values": [{"value": "work"}, {"value": "personal"}]},
        ]
    }

    cfg_default = _make_config(supports_pattern=False, entity_labels=None)
    cfg_pattern = _make_config(supports_pattern=True, entity_labels=None)
    cfg_labels = _make_config(supports_pattern=False, entity_labels=sample_labels)
    cfg_both = _make_config(supports_pattern=True, entity_labels=sample_labels)

    scales = [
        ("small_doc", "Small document", 5),
        ("typical_doc", "Typical document", 20),
        ("large_doc", "Large document", 50),
    ]

    configs = [
        ("default", "static schema (0x create_model)", cfg_default),
        ("pattern", "string pattern (2x create_model)", cfg_pattern),
        ("labels", "entity labels (3x create_model)", cfg_labels),
        ("both", "pattern + labels (5x create_model)", cfg_both),
    ]

    workloads: list[Workload] = []
    for scale_prefix, scale_desc, chunks in scales:
        for cfg_suffix, cfg_desc, cfg in configs:
            workloads.append(
                Workload(
                    name=f"{scale_prefix}_{chunks}_chunks_{cfg_suffix}",
                    description=f"{scale_desc} ({chunks} chunks, {cfg_desc})",
                    total_chunks=chunks,
                    config=cfg,
                )
            )
    return workloads


_DUMMY_CHUNK = "Alice met Bob at the Paris office on Monday morning to discuss the quarterly engineering roadmap."


@dataclass(frozen=True)
class ExtractionBenchOutput:
    prompt: str
    schema_dict: dict


def _run_unhoisted_baseline(config: SimpleNamespace, total_chunks: int) -> ExtractionBenchOutput:
    """Previous baseline: build prompt & schema on every chunk."""
    last_prompt = ""
    last_schema = None
    for i in range(total_chunks):
        parts = build_chunk_prompt_parts(
            config,
            chunk=_DUMMY_CHUNK,
            chunk_index=i,
            total_chunks=total_chunks,
        )
        last_prompt = parts.system_prompt
        last_schema = parts.response_schema
    assert last_schema is not None
    return ExtractionBenchOutput(prompt=last_prompt, schema_dict=strict_json_schema(last_schema))


def _run_hoisted_prod(config: SimpleNamespace, total_chunks: int) -> ExtractionBenchOutput:
    """Production implementation: compile prompt & schema once outside chunk loop."""
    extraction_prompt: ExtractionPrompt = _build_extraction_prompt_and_schema(config)
    last_prompt = ""
    last_schema = None
    for i in range(total_chunks):
        parts = build_chunk_prompt_parts(
            config,
            chunk=_DUMMY_CHUNK,
            chunk_index=i,
            total_chunks=total_chunks,
            extraction_prompt=extraction_prompt,
        )
        last_prompt = parts.system_prompt
        last_schema = parts.response_schema
    assert last_schema is not None
    return ExtractionBenchOutput(prompt=last_prompt, schema_dict=strict_json_schema(last_schema))


@dataclass(frozen=True)
class _Timing:
    wall_ms: float
    cpu_ms: float
    out: ExtractionBenchOutput


def _measure(
    fn: Callable[[SimpleNamespace, int], ExtractionBenchOutput],
    config: SimpleNamespace,
    total_chunks: int,
    repeats: int,
) -> _Timing:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    best_wall = float("inf")
    best_cpu = 0.0
    last_out: ExtractionBenchOutput | None = None

    for _ in range(repeats):
        gc.collect()
        t0_wall = time.perf_counter()
        t0_cpu = time.process_time()
        out = fn(config, total_chunks)
        t1_cpu = time.process_time()
        t1_wall = time.perf_counter()

        wall_ms = (t1_wall - t0_wall) * 1000.0
        cpu_ms = (t1_cpu - t0_cpu) * 1000.0
        if wall_ms < best_wall:
            best_wall = wall_ms
            best_cpu = cpu_ms
            last_out = out

    assert last_out is not None
    return _Timing(wall_ms=best_wall, cpu_ms=best_cpu, out=last_out)


def _measure_peak_kib(
    fn: Callable[[SimpleNamespace, int], ExtractionBenchOutput],
    config: SimpleNamespace,
    total_chunks: int,
) -> float:
    gc.collect()
    tracemalloc.start()
    try:
        fn(config, total_chunks)
        _current, peak = tracemalloc.get_traced_memory()
        return peak / 1024.0
    finally:
        tracemalloc.stop()


def run(workloads: Sequence[Workload], repeats: int) -> list[VariantResult]:
    variants = {
        "unhoisted_baseline": _run_unhoisted_baseline,
        "hoisted_prod": _run_hoisted_prod,
    }
    results: list[VariantResult] = []

    for wl in workloads:
        baseline_timing = _measure(variants["unhoisted_baseline"], wl.config, wl.total_chunks, repeats)
        baseline_peak_kib = _measure_peak_kib(variants["unhoisted_baseline"], wl.config, wl.total_chunks)
        base_prompt = baseline_timing.out.prompt
        base_schema_dict = baseline_timing.out.schema_dict

        results.append(
            VariantResult(
                workload=wl.name,
                variant="unhoisted_baseline",
                wall_ms=baseline_timing.wall_ms,
                cpu_ms=baseline_timing.cpu_ms,
                peak_kib=baseline_peak_kib,
                total_chunks=wl.total_chunks,
                matches_baseline=True,
            )
        )

        prod_timing = _measure(variants["hoisted_prod"], wl.config, wl.total_chunks, repeats)
        prod_peak_kib = _measure_peak_kib(variants["hoisted_prod"], wl.config, wl.total_chunks)
        prod_prompt = prod_timing.out.prompt
        prod_schema_dict = prod_timing.out.schema_dict

        matches = (prod_prompt == base_prompt) and (prod_schema_dict == base_schema_dict)

        results.append(
            VariantResult(
                workload=wl.name,
                variant="hoisted_prod",
                wall_ms=prod_timing.wall_ms,
                cpu_ms=prod_timing.cpu_ms,
                peak_kib=prod_peak_kib,
                total_chunks=wl.total_chunks,
                matches_baseline=matches,
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
            title=f"[bold]{wl.name}[/bold] — {wl.description} (N={wl.total_chunks} chunks)",
            title_justify="left",
        )
        table.add_column("variant", style="cyan")
        table.add_column("wall ms", justify="right")
        table.add_column("speedup", justify="right", style="green")
        table.add_column("cpu ms", justify="right")
        table.add_column("peak KiB", justify="right")
        table.add_column("mem Δ", justify="right")
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
                "[green]exact[/green]" if r.matches_baseline else "[red]MISMATCH[/red]",
            )
        console.print(table)
        console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=5, help="timed repeats per variant (best-of); default 5")
    parser.add_argument("--json", dest="json_path", help="also write raw results to this path")
    args = parser.parse_args()

    workloads = build_workloads()

    console.print(
        f"[dim]Running Fact Extraction Prompt & Schema Hoisting microbenchmarks | cpu_count={os.cpu_count()} | repeats={args.repeats}[/dim]\n"
    )
    results = run(workloads, repeats=args.repeats)
    _render(workloads, results)

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        console.print(f"[dim]Wrote raw results to {args.json_path}[/dim]")


if __name__ == "__main__":
    main()
