"""
Retain operation performance benchmark.

Measures retain operation performance by:
1. Loading a document from a file or directory
2. Sending it to the retain endpoint via HTTP (batched for directories)
3. Measuring time taken and token usage
4. Reporting performance metrics

Usage:
    # Single file
    uv run python hindsight-dev/benchmarks/perf/retain_perf.py --document <file_path> [options]

    # Directory (batches all files)
    uv run python hindsight-dev/benchmarks/perf/retain_perf.py --document <dir_path> [options]
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table

console = Console()


def _create_memory_engine():
    """Create a MemoryEngine from environment variables."""
    from hindsight_api import MemoryEngine

    return MemoryEngine(
        db_url=os.getenv("HINDSIGHT_API_DATABASE_URL", "pg0"),
        memory_llm_provider=os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq"),
        memory_llm_api_key=os.getenv("HINDSIGHT_API_LLM_API_KEY"),
        memory_llm_model=os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-20b"),
        memory_llm_base_url=os.getenv("HINDSIGHT_API_LLM_BASE_URL") or None,
    )


async def retain_via_memory_engine(
    bank_id: str,
    items: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    """
    Send retain request directly to MemoryEngine (in-memory, no HTTP).

    Args:
        bank_id: Bank ID to retain into
        items: List of items to retain

    Returns:
        Tuple of (duration_seconds, response_data)
    """
    from hindsight_api.models import RequestContext

    memory = _create_memory_engine()
    await memory.initialize()

    # Measure time
    start_time = time.time()

    try:
        # Call retain_batch_async directly
        result, usage = await memory.retain_batch_async(
            bank_id=bank_id,
            contents=items,
            request_context=RequestContext(),
            return_usage=True,
        )

        duration = time.time() - start_time

        # Format response to match HTTP response structure
        response_data = {
            "success": True,
            "bank_id": bank_id,
            "items_count": len(items),
            "async": False,
            "usage": usage.model_dump() if usage else None,
        }

        return duration, response_data
    finally:
        # Close memory engine connections
        pool = await memory._get_pool()
        await pool.close()


def _mock_fact_response(messages, scope):
    """Generate fact extraction responses with heavily overlapping entities.

    Uses a tiny entity pool (5 names) so every concurrent retain touches the
    same rows in the entities / unit_entities / memory_links tables, maximising
    the chance of deadlocks from row-lock ordering conflicts.
    """
    import hashlib
    import random as _rng

    # Deterministic seed from message content so results are repeatable
    content = str(messages)
    seed = int(hashlib.md5(content.encode()).hexdigest()[:8], 16)
    _rng.seed(seed)

    # Deliberately tiny pools → very high overlap across concurrent retains
    names = ["Alice", "Bob", "Carol", "Dave", "Eve"]
    places = ["New York", "London"]

    num_facts = _rng.randint(5, 12)
    facts = []
    for _ in range(num_facts):
        who1, who2 = _rng.sample(names, 2)
        place = _rng.choice(places)
        facts.append(
            {
                "what": f"{who1} met {who2} in {place}",
                "when": "2024-01-15",
                "where": place,
                "who": f"{who1}, {who2}",
                "why": "N/A",
                "fact_kind": "conversation",
                "fact_type": "world",
                "entities": [{"text": who1}, {"text": who2}, {"text": place}],
            }
        )
    return {"facts": facts}


async def retain_via_memory_engine_async(
    bank_id: str,
    items: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    """
    Submit retain via submit_async_retain and let the WorkerPoller process it.

    This reproduces the real async flow: documents are split into sub-batches,
    each becomes a separate worker task, and the worker processes them concurrently.
    """
    from hindsight_api.models import RequestContext
    from hindsight_api.worker.poller import WorkerPoller

    memory = _create_memory_engine()
    await memory.initialize()

    # Configure mock LLM to return realistic facts with entities (after init)
    for llm_config in [memory._llm_config, memory._retain_llm_config]:
        if hasattr(llm_config, "set_response_callback"):
            llm_config.set_response_callback(_mock_fact_response)
            console.print("    [cyan]Mock LLM configured with entity-rich fact responses[/cyan]")

    # Start a WorkerPoller so tasks get picked up
    poller = WorkerPoller(
        backend=memory._backend,
        worker_id="bench-worker",
        executor=memory.execute_task,
        poll_interval_ms=100,
        max_slots=50,
    )
    poller_task = asyncio.create_task(poller.run())

    start_time = time.time()

    try:
        # Submit async retain (splits into sub-batches as worker tasks)
        result = await memory.submit_async_retain(
            bank_id=bank_id,
            contents=items,
            request_context=RequestContext(),
        )
        operation_id = result["operation_id"]
        console.print(f"    Submitted operation {operation_id} ({result['items_count']} items)")

        # Poll for completion
        while True:
            status = await memory.get_operation_status(
                bank_id=bank_id,
                operation_id=operation_id,
                request_context=RequestContext(),
            )
            op_status = status.get("status")
            if op_status in ("completed", "failed"):
                if op_status == "failed":
                    console.print(f"    [red]Operation FAILED: {status.get('error_message')}[/red]")
                    # Print child statuses if available
                    for child in status.get("child_operations", []):
                        if child.get("status") == "failed":
                            console.print(f"      Child {child['operation_id']}: {child.get('error_message', '')}")
                break
            await asyncio.sleep(0.5)

        duration = time.time() - start_time

        response_data = {
            "success": op_status == "completed",
            "bank_id": bank_id,
            "items_count": result["items_count"],
            "async": True,
            "usage": None,
        }

        return duration, response_data
    finally:
        await poller.shutdown_graceful(timeout=5)
        poller_task.cancel()
        await memory.close()


async def stress_test_deadlocks(
    concurrency: int = 20,
    num_documents: int = 50,
    max_retain_concurrent: int | None = None,
) -> dict[str, Any]:
    """
    Fire many concurrent retains into the same bank with overlapping entities
    to reproduce deadlocks on entity/link tables.

    Each document gets a unique short text, but the mock LLM always returns
    facts referencing the same small set of entities — maximising row-lock
    contention on the entities and unit_entities tables.
    """
    import traceback
    from collections import Counter

    from hindsight_api.models import RequestContext

    # Override semaphore limit if requested (before engine init reads config)
    if max_retain_concurrent is not None:
        os.environ["HINDSIGHT_API_RETAIN_MAX_CONCURRENT"] = str(max_retain_concurrent)

    # Enable logging so deadlock retry warnings are visible
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("hindsight_api.engine.db_utils").setLevel(logging.DEBUG)

    # Force mock provider so LLM calls are instant — we're testing DB contention
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    memory = _create_memory_engine()
    await memory.initialize()

    # Wire up the mock callback so LLM calls return entity-rich facts instantly
    for llm_config in [memory._llm_config, memory._retain_llm_config]:
        if hasattr(llm_config, "set_response_callback"):
            llm_config.set_response_callback(_mock_fact_response)

    bank_id = f"stress-deadlock-{int(time.time())}"
    pool = await memory._get_pool()

    # Ensure the bank exists
    # Ensure bank exists before firing concurrent retains
    from hindsight_api.engine.retain.fact_storage import ensure_bank_exists

    async with pool.acquire() as conn:
        await ensure_bank_exists(conn, bank_id)

    console.print("\n[bold]Stress test config:[/bold]")
    console.print(f"  Bank:          {bank_id}")
    console.print(f"  Documents:     {num_documents}")
    console.print(f"  Concurrency:   {concurrency}")
    console.print(f"  DB semaphore:  {max_retain_concurrent or 'default'}")
    console.print()

    # Generate synthetic documents — large enough to produce many chunks.
    # Default chunk_size is 3000 chars, so 100k content ≈ 33 chunks per doc.
    # Each chunk triggers the mock LLM which returns 5-12 facts with overlapping
    # entities, maximising row-lock contention across concurrent transactions.
    content_size = int(os.getenv("STRESS_CONTENT_SIZE", "100000"))
    console.print(f"  Content/doc:   ~{content_size:,} chars (~{content_size // 3000} chunks)")

    # Pre-populate the bank with seed documents so that the bank already has
    # units with embeddings. Subsequent concurrent retains will create semantic
    # and temporal links to these existing units — and to each other's new units
    # — triggering INSERT ON CONFLICT share-lock deadlocks on memory_links.
    seed_count = int(os.getenv("STRESS_SEED_DOCS", "5"))
    if seed_count > 0:
        console.print(f"\n[cyan]Seeding bank with {seed_count} documents (serial)...[/cyan]")
        for i in range(seed_count):
            await memory.retain_batch_async(
                bank_id=bank_id,
                contents=[
                    {
                        "content": f"Seed document {i}: Alice discussed machine learning with Bob in New York. "
                        f"Carol and Dave reviewed the quarterly results in London. "
                        f"Eve presented blockchain research findings to Frank in Berlin."
                    }
                ],
                request_context=RequestContext(),
            )
        console.print(f"  [green]Seeded {seed_count} documents[/green]")

    documents = []
    for i in range(num_documents):
        # Build a large document from repeated paragraphs with slight variation
        paragraphs = []
        while len("\n\n".join(paragraphs)) < content_size:
            j = len(paragraphs)
            paragraphs.append(
                f"Section {j} of document {i}: Alice and Bob met Carol in New York to discuss "
                f"the progress on project Alpha. Dave joined from London via video call. "
                f"Eve presented the quarterly results while Frank took notes. "
                f"The team agreed to reconvene next week in San Francisco. "
                f"Key topics included machine learning infrastructure, deployment pipelines, "
                f"and the upcoming product launch scheduled for Q2."
            )
        content = "\n\n".join(paragraphs)[:content_size]
        documents.append([{"content": content}])

    # Track outcomes per task
    results: list[dict] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def _retain_one(doc_idx: int, items: list[dict]) -> dict:
        async with semaphore:
            t0 = time.time()
            try:
                await memory.retain_batch_async(
                    bank_id=bank_id,
                    contents=items,
                    request_context=RequestContext(),
                )
                return {"idx": doc_idx, "status": "ok", "duration": time.time() - t0}
            except Exception as e:
                return {
                    "idx": doc_idx,
                    "status": "error",
                    "error": type(e).__name__,
                    "message": str(e)[:200],
                    "traceback": traceback.format_exc(),
                    "duration": time.time() - t0,
                }

    console.print("[cyan]Firing concurrent retains...[/cyan]")
    start = time.time()

    tasks = [asyncio.create_task(_retain_one(i, docs)) for i, docs in enumerate(documents)]
    results = await asyncio.gather(*tasks)

    total_time = time.time() - start

    # Summarise
    status_counts = Counter(r["status"] for r in results)
    error_types = Counter(r.get("error", "") for r in results if r["status"] == "error")
    durations = [r["duration"] for r in results]
    durations.sort()

    console.print(f"\n[bold]Results ({total_time:.2f}s total):[/bold]")
    table = Table(title="Stress Test Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total documents", str(num_documents))
    table.add_row("Concurrency", str(concurrency))
    table.add_row("OK", str(status_counts.get("ok", 0)))
    table.add_row("Errors", str(status_counts.get("error", 0)))
    table.add_row("Wall time", f"{total_time:.2f}s")
    table.add_row("p50 latency", f"{durations[len(durations) // 2]:.3f}s")
    table.add_row("p95 latency", f"{durations[int(len(durations) * 0.95)]:.3f}s")
    table.add_row("p99 latency", f"{durations[int(len(durations) * 0.99)]:.3f}s")
    table.add_row("Max latency", f"{durations[-1]:.3f}s")
    console.print(table)

    if error_types:
        console.print("\n[bold red]Error breakdown:[/bold red]")
        for err_type, count in error_types.most_common():
            console.print(f"  {err_type}: {count}")
            # Show first traceback for each error type
            for r in results:
                if r.get("error") == err_type:
                    console.print(f"    [dim]{r.get('traceback', '(no traceback)')}[/dim]")
                    break

    # Check for deadlock retries in logs
    deadlock_count = sum(1 for r in results if r.get("error") == "DeadlockDetectedError")
    if deadlock_count:
        console.print(f"\n[bold red]Deadlocks that exhausted retries: {deadlock_count}[/bold red]")
    elif status_counts.get("error", 0) == 0:
        console.print(
            "\n[bold green]No errors — deadlocks may still have occurred but were retried successfully.[/bold green]"
        )
        console.print("[dim]Check logs above for 'Deadlock detected' warnings from retry_with_backoff.[/dim]")

    await pool.close()
    return {"ok": status_counts.get("ok", 0), "errors": status_counts.get("error", 0), "wall_time": total_time}


async def retain_via_http(
    base_url: str,
    bank_id: str,
    items: list[dict[str, Any]],
    timeout: float = 300.0,
) -> tuple[float, dict[str, Any]]:
    """
    Send retain request via HTTP and measure performance.

    Args:
        base_url: API base URL (e.g., http://localhost:8000)
        bank_id: Bank ID to retain into
        items: List of items to retain (each with 'content' and optional 'context', 'metadata')
        timeout: Request timeout in seconds

    Returns:
        Tuple of (duration_seconds, response_data)
    """
    url = f"{base_url}/v1/default/banks/{bank_id}/memories"

    payload = {"items": items}

    headers = {"Content-Type": "application/json"}

    # Measure time
    start_time = time.time()

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

    duration = time.time() - start_time

    return duration, result


def load_documents(path: str) -> tuple[list[dict[str, Any]], int]:
    """
    Load document(s) from file or directory.

    For directories: loads all .json, .txt, and .md files
    For JSON files with 'content' field: extracts content
    For other files: reads entire file as content

    Returns:
        Tuple of (items_list, total_content_length)
        items_list: List of dicts with 'content' and optional 'metadata'/'context'
        total_content_length: Total character count across all documents
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    items = []
    total_length = 0

    if file_path.is_file():
        # Single file
        content, metadata = _load_single_file(file_path)
        total_length = len(content)
        item = {"content": content}
        if metadata:
            item["metadata"] = metadata
        items.append(item)
    else:
        # Directory - load all supported files
        supported_extensions = {".json", ".txt", ".md"}
        files = [f for f in file_path.rglob("*") if f.is_file() and f.suffix in supported_extensions]

        if not files:
            raise ValueError(f"No supported files (.json, .txt, .md) found in directory: {path}")

        console.print(f"Found {len(files)} files in directory")

        for file in sorted(files):
            try:
                content, metadata = _load_single_file(file)
                total_length += len(content)
                item = {"content": content}
                if metadata:
                    item["metadata"] = metadata
                # Add filename as context for batch processing
                item["context"] = f"Source: {file.name}"
                items.append(item)
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to load {file.name}: {e}[/yellow]")
                continue

    return items, total_length


def _load_single_file(file_path: Path) -> tuple[str, dict[str, Any] | None]:
    """
    Load a single file and extract content.

    Returns:
        Tuple of (content, metadata)
    """
    if file_path.suffix == ".json":
        # Try to parse as JSON and extract 'content' field
        try:
            data = json.loads(file_path.read_text())
            if isinstance(data, dict) and "content" in data:
                # Extract metadata if present
                metadata = data.get("metadata", {})
                # Add doc_id to metadata if present
                if "doc_id" in data:
                    metadata["doc_id"] = data["doc_id"]
                return data["content"], metadata if metadata else None
            else:
                # Fallback: use entire JSON as string
                return file_path.read_text(), None
        except json.JSONDecodeError:
            # Not valid JSON, read as text
            return file_path.read_text(), None
    else:
        # Read as plain text
        return file_path.read_text(), None


def display_results(
    duration: float,
    usage: dict[str, int] | None,
    content_length: int,
    bank_id: str,
    num_documents: int,
) -> None:
    """Display benchmark results in a formatted table."""
    table = Table(title="Retain Performance Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Bank ID", bank_id)
    table.add_row("Documents", f"{num_documents:,}")
    table.add_row("Total Content Length", f"{content_length:,} chars")
    if num_documents > 1:
        table.add_row("Avg Content/Doc", f"{content_length / num_documents:,.0f} chars")
    table.add_row("", "")  # Separator
    table.add_row("Duration", f"{duration:.3f}s")
    table.add_row("Throughput", f"{content_length / duration:,.0f} chars/sec")
    if num_documents > 1:
        table.add_row("Docs/Second", f"{num_documents / duration:.2f}")

    if usage:
        table.add_row("", "")  # Separator
        table.add_row("Input Tokens", f"{usage.get('input_tokens', 0):,}")
        table.add_row("Output Tokens", f"{usage.get('output_tokens', 0):,}")
        table.add_row("Total Tokens", f"{usage.get('total_tokens', 0):,}")
        table.add_row("Tokens/Second", f"{usage.get('total_tokens', 0) / duration:,.1f}")
        if num_documents > 1:
            table.add_row("Avg Tokens/Doc", f"{usage.get('total_tokens', 0) / num_documents:,.0f}")
    else:
        table.add_row("", "")  # Separator
        table.add_row("Token Usage", "Not available (async mode or error)")

    console.print("\n")
    console.print(table)


def save_results(
    output_path: Path,
    duration: float,
    usage: dict[str, int] | None,
    content_length: int,
    bank_id: str,
    document_path: str,
    num_documents: int,
) -> None:
    """Save results to JSON file."""
    results = {
        "bank_id": bank_id,
        "document_path": document_path,
        "num_documents": num_documents,
        "content_length": content_length,
        "avg_content_per_doc": content_length / num_documents if num_documents > 0 else 0,
        "duration_seconds": duration,
        "chars_per_second": content_length / duration,
        "docs_per_second": num_documents / duration if num_documents > 0 else 0,
        "usage": usage,
    }

    if usage:
        results["tokens_per_second"] = usage.get("total_tokens", 0) / duration
        results["avg_tokens_per_doc"] = usage.get("total_tokens", 0) / num_documents if num_documents > 0 else 0

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]✓[/green] Results saved to {output_path}")


async def main():
    """Run the retain performance benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark retain operation performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark with a single document file
  uv run python hindsight-dev/benchmarks/perf/retain_perf.py \\
      --document ./test_data/large_doc.txt \\
      --bank-id perf-test-001

  # Benchmark with a directory (batches all files)
  uv run python hindsight-dev/benchmarks/perf/retain_perf.py \\
      --document ~/Documents/my-docs/ \\
      --bank-id perf-test-batch \\
      --output results/batch_perf.json

  # With custom API URL and save results
  uv run python hindsight-dev/benchmarks/perf/retain_perf.py \\
      --document ./test_data/ \\
      --bank-id perf-test-001 \\
      --api-url http://localhost:8000 \\
      --output results/retain_perf_001.json
        """,
    )

    parser.add_argument(
        "--document",
        required=False,
        help="Path to document file or directory (for directories, batches all .json/.txt/.md files)",
    )
    parser.add_argument(
        "--bank-id",
        default="perf-test",
        help="Bank ID to use (default: perf-test)",
    )
    parser.add_argument(
        "--context",
        help="Optional context for the retain operation (only used for single file mode)",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save results JSON (optional)",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Use in-memory MemoryEngine instead of HTTP (bypasses API server, useful for isolating performance)",
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use async retain (submit_async_retain + worker poller). Only works with --in-memory.",
    )
    parser.add_argument(
        "--max-retain-concurrent",
        type=int,
        default=None,
        help="Override HINDSIGHT_API_RETAIN_MAX_CONCURRENT for this run (default: from config)",
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Run deadlock stress test: fire many concurrent retains with overlapping entities into the same bank",
    )
    parser.add_argument(
        "--stress-concurrency",
        type=int,
        default=20,
        help="Max concurrent retains for stress test (default: 20)",
    )
    parser.add_argument(
        "--stress-documents",
        type=int,
        default=50,
        help="Number of documents to retain in stress test (default: 50)",
    )

    args = parser.parse_args()

    # Stress test mode — standalone, doesn't need --document
    if args.stress:
        console.print("\n[bold cyan]Retain Deadlock Stress Test[/bold cyan]")
        console.print("=" * 80)
        await stress_test_deadlocks(
            concurrency=args.stress_concurrency,
            num_documents=args.stress_documents,
            max_retain_concurrent=args.max_retain_concurrent,
        )
        return

    if not args.document:
        parser.error("--document is required (unless using --stress)")

    console.print("\n[bold cyan]Retain Performance Benchmark[/bold cyan]")
    console.print("=" * 80)

    # Check mode
    if args.in_memory:
        console.print("\n[cyan]Mode: IN-MEMORY (direct MemoryEngine, no HTTP)[/cyan]")
    else:
        console.print(f"\n[cyan]Mode: HTTP (via {args.api_url})[/cyan]")

    # Check if server is running (skip for in-memory mode)
    if not args.in_memory:
        console.print(f"\n[1] Checking API server at {args.api_url}...")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{args.api_url}/health", timeout=5.0)
                response.raise_for_status()
            console.print("    [green]✓[/green] API server is running")
        except Exception as e:
            console.print(f"    [red]✗[/red] API server is not accessible: {e}")
            console.print("\n[yellow]Please ensure the API server is running:[/yellow]")
            console.print("  ./scripts/dev/start-api.sh")
            sys.exit(1)

    # Override retain_max_concurrent if specified
    if args.max_retain_concurrent is not None:
        os.environ["HINDSIGHT_API_RETAIN_MAX_CONCURRENT"] = str(args.max_retain_concurrent)
        console.print(f"    [cyan]Retain max concurrent: {args.max_retain_concurrent}[/cyan]")

    # Load document(s)
    doc_path = Path(args.document)
    if doc_path.is_dir():
        console.print(f"\n[2] Loading documents from directory {args.document}...")
    else:
        console.print(f"\n[2] Loading document from {args.document}...")

    try:
        items, total_content_length = load_documents(args.document)
        num_docs = len(items)

        # Add context to single file if provided
        if num_docs == 1 and args.context:
            items[0]["context"] = args.context

        console.print(
            f"    [green]✓[/green] Loaded {num_docs:,} document{'s' if num_docs > 1 else ''} ({total_content_length:,} characters)"
        )
        if num_docs > 1:
            console.print(
                f"    [cyan]Average content per document: {total_content_length / num_docs:,.0f} chars[/cyan]"
            )
    except Exception as e:
        console.print(f"    [red]✗[/red] Failed to load documents: {e}")
        sys.exit(1)

    # Run benchmark
    console.print(f"\n[3] {'Processing' if args.in_memory else 'Sending retain request to'} bank '{args.bank_id}'...")
    console.print(f"    [cyan]Retaining {num_docs:,} document{'s' if num_docs > 1 else ''} in batch...[/cyan]")
    try:
        if args.in_memory and args.use_async:
            # In-memory async mode: submit_async_retain + worker poller
            duration, result = await retain_via_memory_engine_async(
                bank_id=args.bank_id,
                items=items,
            )
        elif args.in_memory:
            # In-memory sync mode: call MemoryEngine directly
            duration, result = await retain_via_memory_engine(
                bank_id=args.bank_id,
                items=items,
            )
        else:
            # HTTP mode: call API endpoint
            duration, result = await retain_via_http(
                base_url=args.api_url,
                bank_id=args.bank_id,
                items=items,
                timeout=args.timeout,
            )
        console.print(f"    [green]✓[/green] Retain completed in {duration:.3f}s")

        # Extract usage
        usage = result.get("usage")

    except httpx.HTTPStatusError as e:
        console.print(f"    [red]✗[/red] HTTP error: {e.response.status_code}")
        console.print(f"    Response: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        console.print(f"    [red]✗[/red] Request failed: {e}")
        sys.exit(1)

    # Display results
    console.print("\n[4] Results:")
    display_results(
        duration=duration,
        usage=usage,
        content_length=total_content_length,
        bank_id=args.bank_id,
        num_documents=num_docs,
    )

    # Save results if requested
    if args.output:
        console.print("\n[5] Saving results...")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_results(
            output_path=args.output,
            duration=duration,
            usage=usage,
            content_length=total_content_length,
            bank_id=args.bank_id,
            document_path=args.document,
            num_documents=num_docs,
        )

    console.print("\n[bold green]✓ Benchmark Complete![/bold green]\n")


if __name__ == "__main__":
    asyncio.run(main())
