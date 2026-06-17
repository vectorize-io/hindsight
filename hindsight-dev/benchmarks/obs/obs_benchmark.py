"""Observation-quality benchmark.

Measures how many *duplicate observations* consolidation produces. For each
document in the dataset it ingests the content into a fresh bank, runs
consolidation, then reuses the observation-dedup tool (hindsight_dev.obs_dedup)
to score:

- exact duplicates: observations with byte-identical (normalised) text in a scope
- near duplicates: cosine-similarity clusters at configurable thresholds

The headline metric is the duplication rate (redundant observations / total).
Lower is better. Add more documents under ``datasets/`` to grow coverage as new
regressions are found.

Run with::

    ./scripts/benchmarks/run-obs.sh
    # or
    cd hindsight-dev && uv run python -m benchmarks.obs.obs_benchmark
"""

import argparse
import asyncio
import json
import os
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from hindsight_api.config import DEFAULT_EMBEDDINGS_LOCAL_MODEL, _get_raw_config
from hindsight_api.engine.consolidation.consolidator import run_consolidation_job
from hindsight_api.engine.memory_engine import MemoryEngine, fq_table
from hindsight_api.engine.task_backend import SyncTaskBackend
from hindsight_api.models import RequestContext
from rich.console import Console
from rich.table import Table

from hindsight_dev.obs_dedup.dedup import cluster_pairs, embed_observations, find_similar_pairs
from hindsight_dev.obs_dedup.models import Observation

console = Console()

DATASETS_DIR = Path(__file__).parent / "datasets"
NEAR_THRESHOLDS = (0.97, 0.92)
RETAIN_MISSION = (
    "Extract and keep durable facts: stated goals, decisions, preferences, constraints, "
    "and recurring plans. Ignore greetings and one-off small talk."
)


@dataclass
class _ThresholdMetric:
    threshold: float
    duplicate_clusters: int
    redundant_observations: int
    duplication_rate: float


@dataclass
class _DocResult:
    name: str
    facts: int
    facts_consolidated: int
    facts_covered: int  # facts referenced by >=1 observation's source_memory_ids
    facts_skipped: int  # consolidated but in no observation (no durable knowledge)
    observations: int
    avg_sources_per_obs: float
    exact_duplicate_groups: int
    exact_redundant: int
    near: list[_ThresholdMetric] = field(default_factory=list)


def _normalize(text: str) -> str:
    return " ".join((text or "").split()).strip().lower()


def _score_observations(observations: list[Observation], *, model_name: str, force_cpu: bool) -> _DocResult:
    """Compute exact + near-duplicate metrics for one bank's observations."""
    # Exact duplicates: identical normalised text within the same tag scope.
    groups: dict[tuple[tuple[str, ...], str], list[str]] = defaultdict(list)
    for obs in observations:
        groups[(obs.tags, _normalize(obs.text))].append(obs.id)
    exact_groups = {k: v for k, v in groups.items() if len(v) > 1}
    exact_redundant = sum(len(v) - 1 for v in exact_groups.values())

    result = _DocResult(
        name="",
        facts=0,
        facts_consolidated=0,
        facts_covered=0,
        facts_skipped=0,
        observations=len(observations),
        avg_sources_per_obs=0.0,
        exact_duplicate_groups=len(exact_groups),
        exact_redundant=exact_redundant,
    )

    # Near duplicates: embed once, cluster at each threshold.
    if len(observations) >= 2:
        matrix = embed_observations(observations, model_name=model_name, force_cpu=force_cpu)
        for threshold in NEAR_THRESHOLDS:
            pairs = find_similar_pairs(matrix, threshold=threshold)
            clusters = cluster_pairs(observations, pairs, min_cluster_size=2)
            redundant = sum(c.redundant_count for c in clusters)
            result.near.append(
                _ThresholdMetric(
                    threshold=threshold,
                    duplicate_clusters=len(clusters),
                    redundant_observations=redundant,
                    duplication_rate=redundant / len(observations),
                )
            )
    return result


async def _run_document(
    memory: MemoryEngine, dataset: Path, *, model_name: str, force_cpu: bool, fraction: float, wipe: bool
) -> _DocResult:
    bank_id = f"obs-bench-{uuid.uuid4().hex[:8]}"
    content = dataset.read_text(encoding="utf-8").rstrip("\n")
    if fraction < 1.0:
        # Run only the first `fraction` of the document — lets us scale up incrementally
        # (1/4 -> 1/2 -> full) instead of always paying the full ~1h drain.
        content = content[: int(len(content) * fraction)]
    ctx = RequestContext()
    console.print(f"   bank={bank_id} chars={len(content)}")

    await memory.get_bank_profile(bank_id=bank_id, request_context=ctx)
    # Confine consolidation to our explicit drain loop below so it's the single, well-defined
    # measurement point: disable retain's auto-consolidation for THIS bank so retain doesn't
    # also fire a consolidation pass during ingestion. Serial correctness is already guaranteed
    # by SyncTaskBackend (see main()); this just keeps the drain loop the sole consolidator,
    # it is not what prevents the race.
    await memory._config_resolver.update_bank_config(bank_id, {"enable_auto_consolidation": False}, ctx)
    try:
        await memory.retain_async(
            bank_id=bank_id,
            content=content,
            context="conversation between an assistant and the user",
            request_context=ctx,
        )
        # Consolidation is run explicitly (the in-process engine has no background worker
        # draining the queue). run_consolidation_job honors consolidation_max_memories_per_round
        # (default 100) and returns after one round, so loop until the document is fully
        # consolidated — exactly what the production worker does by calling it repeatedly.
        pool = await memory._get_pool()
        for _ in range(500):  # safety bound; 500 * round-size >> any single document
            await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=ctx)
            async with pool.acquire() as conn:
                pending = await conn.fetchval(
                    f"SELECT COUNT(*) FROM {fq_table('memory_units')} WHERE bank_id=$1 "
                    f"AND fact_type IN ('experience','world') "
                    f"AND consolidated_at IS NULL AND consolidation_failed_at IS NULL",
                    bank_id,
                )
            if pending == 0:
                break

        async with pool.acquire() as conn:
            facts = await conn.fetchval(
                f"SELECT COUNT(*) FROM {fq_table('memory_units')} "
                f"WHERE bank_id=$1 AND fact_type IN ('experience','world')",
                bank_id,
            )
            facts_consolidated = await conn.fetchval(
                f"SELECT COUNT(*) FROM {fq_table('memory_units')} "
                f"WHERE bank_id=$1 AND fact_type IN ('experience','world') AND consolidated_at IS NOT NULL",
                bank_id,
            )
            # Facts referenced by >=1 observation's source_memory_ids = "covered"; the rest of
            # the consolidated facts were skipped (no durable knowledge). This shows whether
            # few observations means heavy merging (high coverage) or discarding (low coverage).
            facts_covered = await conn.fetchval(
                f"""
                SELECT COUNT(*) FROM {fq_table("memory_units")} f
                WHERE f.bank_id=$1 AND f.fact_type IN ('experience','world')
                  AND f.id IN (
                    SELECT unnest(source_memory_ids) FROM {fq_table("memory_units")}
                    WHERE bank_id=$1 AND fact_type='observation'
                  )
                """,
                bank_id,
            )
            rows = await conn.fetch(
                f"SELECT id, text, tags, coalesce(array_length(source_memory_ids,1),0) AS n_src "
                f"FROM {fq_table('memory_units')} WHERE bank_id=$1 AND fact_type='observation' ORDER BY created_at",
                bank_id,
            )
        observations = [Observation(id=str(r["id"]), text=r["text"], tags=tuple(r["tags"] or [])) for r in rows]
        # _score_observations() embeds via the obs_dedup tool, which calls asyncio.run()
        # internally — illegal inside this running loop. Run it in a worker thread, which
        # also keeps the CPU-bound embedding off the event loop.
        result = await asyncio.to_thread(_score_observations, observations, model_name=model_name, force_cpu=force_cpu)
        result.name = dataset.name
        result.facts = facts
        result.facts_consolidated = facts_consolidated
        result.facts_covered = facts_covered
        result.facts_skipped = facts_consolidated - facts_covered
        result.avg_sources_per_obs = round(sum(r["n_src"] for r in rows) / len(rows), 1) if rows else 0.0
        return result
    finally:
        # Persist by default so the bank can be inspected after the run; --wipe-bank deletes.
        if wipe:
            await memory.delete_bank(bank_id, request_context=ctx)
        else:
            console.print(f"   [dim]kept bank {bank_id} for inspection (pass --wipe-bank to delete)[/dim]")


def _display(results: list[_DocResult]) -> None:
    table = Table(title="Observation Duplication Benchmark")
    table.add_column("Document", style="cyan")
    table.add_column("Facts", justify="right")
    table.add_column("Covered", justify="right")
    table.add_column("Skipped", justify="right")
    table.add_column("Obs", justify="right")
    table.add_column("src/obs", justify="right")
    table.add_column("Exact dup", justify="right")
    for threshold in NEAR_THRESHOLDS:
        table.add_column(f"Dup rate @{threshold}", justify="right", style="yellow")

    for r in results:
        cov = f"{r.facts_covered} ({r.facts_covered / r.facts_consolidated:.0%})" if r.facts_consolidated else "0"
        row = [
            r.name,
            str(r.facts),
            cov,
            str(r.facts_skipped),
            str(r.observations),
            str(r.avg_sources_per_obs),
            str(r.exact_redundant),
        ]
        by_threshold = {m.threshold: m for m in r.near}
        for threshold in NEAR_THRESHOLDS:
            m = by_threshold.get(threshold)
            row.append(f"{m.duplication_rate:.0%} ({m.redundant_observations})" if m else "—")
        table.add_row(*row)
    console.print("\n")
    console.print(table)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Observation duplication benchmark.")
    parser.add_argument(
        "--wipe-bank", action="store_true", help="Delete each bank after measuring (default: keep for inspection)."
    )
    parser.add_argument(
        "--fraction", type=float, default=1.0, help="Run only the first fraction (0-1] of each document (default: 1.0)."
    )
    parser.add_argument("--dataset", default=None, help="Run only the dataset whose filename contains this substring.")
    parser.add_argument(
        "--output",
        default=None,
        help="Also write the result JSON to this path (used by CI to publish to the perf dashboard).",
    )
    args = parser.parse_args()
    if not 0.0 < args.fraction <= 1.0:
        parser.error("--fraction must be in (0, 1]")

    console.print("\n[bold cyan]Observation Duplication Benchmark[/bold cyan]")
    console.print("=" * 80)

    datasets = sorted(DATASETS_DIR.glob("*.txt"))
    if args.dataset:
        datasets = [d for d in datasets if args.dataset in d.name]
    if not datasets:
        console.print(f"[red]No dataset .txt files found in {DATASETS_DIR}[/red]")
        return

    model_name = os.getenv("HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL", DEFAULT_EMBEDDINGS_LOCAL_MODEL)
    force_cpu = os.getenv("HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU", "").lower() in ("1", "true", "yes")

    # Consolidation must be enabled for observations to be produced.
    config = _get_raw_config()
    config.enable_observations = True
    config.retain_mission = RETAIN_MISSION
    # Auto-consolidation is disabled per-bank in _run_document so our drain loop is the
    # sole consolidator (see the comment there).

    db_url = os.getenv("HINDSIGHT_API_DATABASE_URL", "pg0")
    console.print(
        f"\nDatasets: {len(datasets)} | LLM: {os.getenv('HINDSIGHT_API_LLM_MODEL', 'not set')} "
        f"| fraction: {args.fraction} | persist: {not args.wipe_bank} | db: {db_url}"
    )

    memory = MemoryEngine(
        db_url=db_url,
        memory_llm_provider=os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq"),
        memory_llm_api_key=os.getenv("HINDSIGHT_API_LLM_API_KEY"),
        memory_llm_model=os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-120b"),
        memory_llm_base_url=os.getenv("HINDSIGHT_API_LLM_BASE_URL") or None,
        # SyncTaskBackend runs every submitted task INLINE/serially instead of queuing it
        # for a background worker poller — and that is what prevents consolidation
        # corruption here. The default BrokerTaskBackend queues consolidation ops (both
        # retain's auto-submit AND the consolidator's own round-limit re-submit,
        # consolidator.py ~L835) that a worker poller runs CONCURRENTLY with our drain loop;
        # two consolidators mutating the same rows leave facts with both consolidated_at AND
        # consolidation_failed_at. Inline execution serializes them all, so the drain loop is
        # the sole consolidator.
        task_backend=SyncTaskBackend(),
    )
    await memory.initialize()

    results: list[_DocResult] = []
    try:
        for dataset in datasets:
            console.print(f"\n[cyan]→ {dataset.name}[/cyan]")
            result = await _run_document(
                memory, dataset, model_name=model_name, force_cpu=force_cpu, fraction=args.fraction, wipe=args.wipe_bank
            )
            results.append(result)
            console.print(
                f"  facts={result.facts} consolidated={result.facts_consolidated} "
                f"covered={result.facts_covered} skipped={result.facts_skipped} "
                f"observations={result.observations} exact_redundant={result.exact_redundant}"
            )
    finally:
        pool = await memory._get_pool()
        await pool.close()

    _display(results)

    total_obs = sum(r.observations for r in results)
    total_exact = sum(r.exact_redundant for r in results)
    by_threshold = {
        str(threshold): sum(m.redundant_observations for r in results for m in r.near if m.threshold == threshold)
        for threshold in NEAR_THRESHOLDS
    }
    # Headline metric for the trend dashboard: near-duplicate rate at the tightest
    # threshold (lower is better). NEAR_THRESHOLDS[0] is the strictest (0.97).
    headline_threshold = NEAR_THRESHOLDS[0]
    overall_duplication_rate = (by_threshold[str(headline_threshold)] / total_obs) if total_obs else 0.0
    aggregate = {
        "total_observations": total_obs,
        "total_exact_redundant": total_exact,
        "overall_duplication_rate": overall_duplication_rate,
        "overall_exact_rate": (total_exact / total_obs) if total_obs else 0.0,
        "by_threshold": {str(t): {"redundant_observations": by_threshold[str(t)]} for t in NEAR_THRESHOLDS},
    }

    # Top-level headline fields mirror locomo's `overall_accuracy` so the publish
    # script and dashboard manifest can read a single number per run.
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_duplication_rate": overall_duplication_rate,
        "duplication_threshold": headline_threshold,
        "total_observations": total_obs,
        "config": {
            "llm_provider": os.getenv("HINDSIGHT_API_LLM_PROVIDER"),
            "llm_model": os.getenv("HINDSIGHT_API_LLM_MODEL"),
            "embedding_model": model_name,
            "near_thresholds": list(NEAR_THRESHOLDS),
            "fraction": args.fraction,
        },
        "documents": [asdict(r) for r in results],
        "aggregate": aggregate,
    }
    serialized = json.dumps(payload, indent=2, default=str)

    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"obs_benchmark_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    output_file.write_text(serialized)
    if args.output:
        Path(args.output).write_text(serialized)
    console.print(f"\n[green]✓[/green] Results saved to: {output_file}\n")


if __name__ == "__main__":
    asyncio.run(main())
