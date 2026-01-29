"""
Consolidation performance benchmark.

Measures consolidation throughput (op/sec) and identifies bottlenecks by:
1. Ingesting a batch of diverse memories
2. Running consolidation manually with detailed timing
3. Analyzing timing breakdown to identify bottlenecks
4. Reporting op/sec and time spent in each component
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hindsight_api.config import get_config
from hindsight_api.engine.consolidation.consolidator import run_consolidation_job
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.models import RequestContext
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


# Sample diverse memories to trigger different consolidation patterns
SAMPLE_MEMORIES = [
    # Similar memories (should merge)
    "Alice loves coffee and drinks it every morning.",
    "Alice prefers coffee over tea for her morning beverage.",
    "Alice switched to decaf coffee recently.",
    # Different person (should NOT merge with Alice)
    "Bob works at Google as a software engineer.",
    "Bob has been at Google for 5 years.",
    # Technical facts
    "Python is a programming language used for data science.",
    "Python supports object-oriented and functional programming.",
    # Product info
    "The new iPhone 15 was released in September 2023.",
    "The iPhone 15 features USB-C charging instead of Lightning.",
    # Contradictions (should merge with conflict resolution)
    "The meeting is scheduled for Tuesday at 2pm.",
    "The meeting was moved to Wednesday at 3pm.",
    # Entity-rich content
    "Sarah Smith works at Microsoft in Seattle.",
    "Sarah graduated from Stanford University in 2015.",
    # Temporal information
    "The project started on January 15, 2024.",
    "The project deadline is March 30, 2024.",
    # Preferences
    "User prefers dark mode in applications.",
    "User uses keyboard shortcuts extensively.",
    # World knowledge
    "Paris is the capital of France.",
    "The Eiffel Tower is located in Paris.",
    # Multiple entities
    "John and Mary went to the Italian restaurant on Main Street.",
    "The Italian restaurant on Main Street has excellent pizza.",
]


async def create_test_memories(memory_engine: MemoryEngine, bank_id: str, num_memories: int = 100) -> None:
    """
    Create test memories by repeating and varying the sample memories.

    Args:
        memory_engine: MemoryEngine instance
        bank_id: Bank ID to ingest into
        num_memories: Number of memories to create
    """
    console.print(f"\n[cyan]Creating {num_memories} test memories...[/cyan]")

    # Generate memories by cycling through samples
    memories = []
    for i in range(num_memories):
        base_memory = SAMPLE_MEMORIES[i % len(SAMPLE_MEMORIES)]
        # Add variation to avoid exact duplicates
        memory = f"{base_memory} (context: test {i + 1})"
        memories.append(
            {
                "content": memory,
                "context": f"Test memory {i + 1}",
            }
        )

    # Batch ingest
    console.print("[yellow]Ingesting memories in batch...[/yellow]")
    start_time = time.time()
    await memory_engine.retain_batch_async(
        bank_id=bank_id,
        contents=memories,
        request_context=RequestContext(),
    )
    ingest_time = time.time() - start_time
    console.print(
        f"[green]✓[/green] Ingested {num_memories} memories in {ingest_time:.2f}s ({num_memories / ingest_time:.2f} mem/sec)"
    )


async def run_consolidation_benchmark(
    memory_engine: MemoryEngine,
    bank_id: str,
    enable_detailed_logs: bool = True,
) -> dict[str, Any]:
    """
    Run consolidation and measure performance.

    Args:
        memory_engine: MemoryEngine instance
        bank_id: Bank ID to consolidate
        enable_detailed_logs: Enable detailed consolidation logs

    Returns:
        Performance metrics dict
    """
    console.print("\n[cyan]Running consolidation benchmark...[/cyan]")

    # Set log level to INFO to see consolidation logs
    if enable_detailed_logs:
        # Configure logging for consolidation
        consolidation_logger = logging.getLogger("hindsight_api.engine.consolidation.consolidator")
        consolidation_logger.setLevel(logging.INFO)

        # Add console handler if not present
        if not consolidation_logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            consolidation_logger.addHandler(handler)

        console.print("[yellow]Detailed logging enabled for consolidation[/yellow]")

    # Run consolidation and measure time
    start_time = time.time()
    result = await run_consolidation_job(
        memory_engine=memory_engine,
        bank_id=bank_id,
        request_context=RequestContext(),
    )
    total_time = time.time() - start_time

    # Calculate op/sec
    memories_processed = result.get("memories_processed", 0)
    ops_per_sec = memories_processed / total_time if total_time > 0 else 0

    console.print("\n[green]✓[/green] Consolidation complete!")
    console.print(f"  Total time: {total_time:.2f}s")
    console.print(f"  Memories processed: {memories_processed}")
    console.print(f"  Throughput: {ops_per_sec:.2f} op/sec")
    console.print(f"  Avg time per memory: {total_time / memories_processed:.3f}s" if memories_processed > 0 else "")

    return {
        "total_time": total_time,
        "memories_processed": memories_processed,
        "ops_per_sec": ops_per_sec,
        "consolidation_result": result,
    }


async def analyze_timing_breakdown(bank_id: str) -> None:
    """
    Analyze the timing breakdown from consolidation logs.

    NOTE: This relies on the performance logging in ConsolidationPerfLog.
    The logs will show timing breakdowns for: recall, llm, embedding, db_write
    """
    console.print("\n[cyan]Timing Breakdown Analysis:[/cyan]")
    console.print("Check the logs above for detailed timing breakdown:")
    console.print("  - recall: Time spent finding related observations")
    console.print("  - llm: Time spent in LLM calls for consolidation decisions")
    console.print("  - embedding: Time spent generating embeddings")
    console.print("  - db_write: Time spent writing to database")


async def get_bank_stats(memory_engine: MemoryEngine, bank_id: str) -> dict[str, Any]:
    """Get memory statistics for the bank."""
    pool = await memory_engine._get_pool()
    from hindsight_api.engine.memory_engine import fq_table

    async with pool.acquire() as conn:
        # Count memories by fact type
        stats = await conn.fetch(
            f"""
            SELECT fact_type, COUNT(*) as count
            FROM {fq_table("memory_units")}
            WHERE bank_id = $1
            GROUP BY fact_type
            """,
            bank_id,
        )

        return {row["fact_type"]: row["count"] for row in stats}


def display_results_table(metrics: dict[str, Any], stats_before: dict, stats_after: dict) -> None:
    """Display benchmark results in a formatted table."""
    table = Table(title="Consolidation Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Time", f"{metrics['total_time']:.2f}s")
    table.add_row("Memories Processed", str(metrics["memories_processed"]))
    table.add_row("Throughput", f"{metrics['ops_per_sec']:.2f} op/sec")
    table.add_row(
        "Avg Time/Memory",
        f"{metrics['total_time'] / metrics['memories_processed']:.3f}s" if metrics["memories_processed"] > 0 else "N/A",
    )

    result = metrics["consolidation_result"]
    table.add_row("", "")  # Separator
    table.add_row("Observations Created", str(result.get("observations_created", 0)))
    table.add_row("Observations Updated", str(result.get("observations_updated", 0)))
    table.add_row("Observations Merged", str(result.get("observations_merged", 0)))
    table.add_row("Skipped (No Durable Knowledge)", str(result.get("skipped", 0)))

    table.add_row("", "")  # Separator
    table.add_row("Memories Before", str(stats_before.get("experience", 0) + stats_before.get("world", 0)))
    table.add_row("Observations After", str(stats_after.get("observation", 0)))

    console.print("\n")
    console.print(table)


async def main():
    """Run the consolidation benchmark."""
    console.print("\n[bold cyan]Consolidation Performance Benchmark[/bold cyan]")
    console.print("=" * 80)

    # Configuration
    num_memories = int(os.getenv("NUM_MEMORIES", "100"))
    bank_id = f"consolidation-bench-{uuid.uuid4().hex[:8]}"

    console.print("\n[cyan]Configuration:[/cyan]")
    console.print(f"  Bank ID: {bank_id}")
    console.print(f"  Number of memories: {num_memories}")
    console.print(f"  LLM Provider: {os.getenv('HINDSIGHT_API_LLM_PROVIDER', 'not set')}")
    console.print(f"  LLM Model: {os.getenv('HINDSIGHT_API_LLM_MODEL', 'not set')}")

    # Check if consolidation is enabled
    config = get_config()
    if not config.enable_observations:
        console.print("\n[red]ERROR: Consolidation is disabled (enable_observations=False)[/red]")
        console.print("Set HINDSIGHT_API_ENABLE_OBSERVATIONS=true to enable consolidation")
        return

    # Initialize memory engine
    console.print("\n[1] Initializing memory engine...")
    memory = MemoryEngine(
        db_url=os.getenv("HINDSIGHT_API_DATABASE_URL", "pg0"),
        memory_llm_provider=os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq"),
        memory_llm_api_key=os.getenv("HINDSIGHT_API_LLM_API_KEY"),
        memory_llm_model=os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-120b"),
        memory_llm_base_url=os.getenv("HINDSIGHT_API_LLM_BASE_URL") or None,
    )
    await memory.initialize()
    console.print("[green]✓[/green] Memory engine initialized")

    try:
        # Create bank
        console.print("\n[2] Creating test bank...")
        await memory.get_bank_profile(bank_id=bank_id, request_context=RequestContext())
        console.print(f"[green]✓[/green] Created bank: {bank_id}")

        # Get initial stats
        stats_before = await get_bank_stats(memory, bank_id)

        # Create test memories
        console.print("\n[3] Creating test memories...")
        await create_test_memories(memory, bank_id, num_memories)

        # Run consolidation benchmark
        console.print("\n[4] Running consolidation benchmark...")
        metrics = await run_consolidation_benchmark(memory, bank_id, enable_detailed_logs=True)

        # Get final stats
        stats_after = await get_bank_stats(memory, bank_id)

        # Analyze timing
        console.print("\n[5] Analyzing performance...")
        await analyze_timing_breakdown(bank_id)

        # Display results
        console.print("\n[6] Results:")
        display_results_table(metrics, stats_before, stats_after)

        # Save results to file
        output_dir = Path("benchmarks/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"consolidation_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "num_memories": num_memories,
                "bank_id": bank_id,
                "llm_provider": os.getenv("HINDSIGHT_API_LLM_PROVIDER"),
                "llm_model": os.getenv("HINDSIGHT_API_LLM_MODEL"),
            },
            "metrics": metrics,
            "stats_before": stats_before,
            "stats_after": stats_after,
        }

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        console.print(f"\n[green]✓[/green] Results saved to: {output_file}")

    finally:
        # Cleanup
        console.print("\n[7] Cleaning up...")
        await memory.delete_bank(bank_id, request_context=RequestContext())
        console.print(f"[green]✓[/green] Deleted bank: {bank_id}")

        # Close memory engine connections
        pool = await memory._get_pool()
        await pool.close()
        console.print("[green]✓[/green] Memory engine connections closed")

    console.print("\n[bold green]✓ Benchmark Complete![/bold green]\n")


if __name__ == "__main__":
    asyncio.run(main())
