"""Memory benchmark for the CPU-bound front half of retain (issue #3756).

Retain's memory used to scale with the size of the document rather than with a bounded
working set: a 45 MB body allocated ~385 MB just to *measure* its token count, and ~200 MB
more to chunk it, all before a single fact or embedding existed.

This exercises exactly that front half — token sizing, chunking, and sub-batch splitting —
with no database, no LLM and no embeddings, so the numbers are attributable to the code
under test and nothing else::

    uv run --directory hindsight-dev python -m benchmarks.perf.retain_memory
    uv run --directory hindsight-dev python -m benchmarks.perf.retain_memory --mb 8

**``allocated`` is the metric to read, and it comes from ``tracemalloc``, not RSS.**
Peak RSS cannot attribute cost to a step: the allocator maps arenas on first touch and
reuses them silently afterwards, so whichever step runs first is charged for growth the
later ones get for free. Measured by RSS, #3756 reported a generator that holds one chunk
at a time as costing +80 MB, and reported a genuinely unbounded buffer as costing nothing.
``tracemalloc`` counts the Python bytes a step allocates, in any order, on every run.

``time`` is inflated — ``tracemalloc`` traces every allocation, roughly tripling the wall
clock — so read it only against other rows in the same run, never as real retain latency.

``retained`` is RSS the process did not give back — a fragmented heap becomes a permanent
baseline increase for the worker, so it is worth watching even though it is noisier.

Compare a row against the same row before a change. There is no total: the two chunking
rows are two implementations of one step, and the document string itself is shared.
"""

import argparse
import gc
import os
import subprocess
import time
import tracemalloc
from dataclasses import dataclass

# The measured document: prose with sentence boundaries the recursive splitter can cut on,
# so chunking does real work rather than falling into a degenerate fixed-width path.
_FILLER = "The quick brown fox jumps over the lazy dog near the river bank in 2023. "

# Issue #3756 reported a 47,280,119-character document at chunk_size=1500.
DEFAULT_DOC_MB = 45
DEFAULT_CHUNK_SIZE = 1500

# The token budget the in-process retain path splits an oversized item against
# (``HINDSIGHT_API_RETAIN_BATCH_TOKENS``).
_SUB_BATCH_TOKENS = 10_000


def _current_rss_mb() -> float:
    """Current RSS. Read from /proc on Linux, else shell out to ps (macOS)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout
    return int(out.strip()) / 1024


@dataclass
class Measurement:
    """One measured step: what it allocated, and what the process did not give back."""

    label: str
    seconds: float
    allocated_mb: float
    retained_mb: float
    detail: str


def measure(label: str, fn, detail_fn=None) -> Measurement:
    """Run ``fn`` and report its peak Python allocation and its retained RSS.

    ``fn``'s return value is dropped before the retained reading is taken, so "retained"
    means heap the allocator kept, not the result still being held.
    """
    gc.collect()
    base_rss = _current_rss_mb()
    tracemalloc.start()
    started = time.time()
    result = fn()
    elapsed = time.time() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    detail = detail_fn(result) if detail_fn else ""
    del result
    gc.collect()
    return Measurement(
        label=label,
        seconds=elapsed,
        allocated_mb=peak / 1024 / 1024,
        retained_mb=_current_rss_mb() - base_rss,
        detail=detail,
    )


def build_document(target_chars: int) -> str:
    repeats = target_chars // len(_FILLER) + 1
    return (_FILLER * repeats)[:target_chars]


def run(doc_mb: int, chunk_size: int) -> list[Measurement]:
    from hindsight_api.engine.memory_engine import _split_contents_into_sub_batches
    from hindsight_api.engine.retain import fact_extraction
    from hindsight_api.engine.token_encoding import count_tokens_windowed

    doc = build_document(doc_mb * 1024 * 1024)
    print(f"document: {len(doc):,} chars ({len(doc) / 1024 / 1024:.1f} MB), chunk_size={chunk_size}")

    # Load tiktoken's encoding table before measuring. It is ~80 MB, paid once per process
    # and shared with recall, so it is a worker's startup cost rather than a retain's.
    count_tokens_windowed(_FILLER)
    print(f"resident before measurement: {_current_rss_mb():.1f} MB\n")

    return [
        measure(
            "sizing check (windowed)",
            lambda: count_tokens_windowed(doc),
            lambda n: f"{n:,} tokens",
        ),
        # What retain does: consume chunks one at a time, never holding the list. Both the
        # sub-batch splitter and the streaming orchestrator iterate `iter_chunks`.
        measure(
            "chunking, streamed (iter_chunks)",
            lambda: sum(1 for _ in fact_extraction.iter_chunks(doc, chunk_size, structured_chunk_size=None)),
            lambda n: f"{n:,} chunks",
        ),
        # The materialising form, for comparison: `chunk_text` still has callers that need
        # random access, and it is what the whole pre-#3756 path used. Its cost is the chunk
        # list itself — about one copy of the document — and is irreducible for that caller.
        measure(
            "chunking, materialised (chunk_text)",
            lambda: fact_extraction.chunk_text(doc, chunk_size, structured_chunk_size=None),
            lambda chunks: f"{len(chunks):,} chunks",
        ),
        # Also about one copy of the document: the returned slices ARE the document, cut up
        # for the caller to process sequentially.
        measure(
            "split into sub-batches",
            lambda: _split_contents_into_sub_batches(
                [{"content": doc}],
                _SUB_BATCH_TOKENS,
                chunk_size=chunk_size,
                structured_chunk_size=None,
            ),
            lambda split: f"{len(split.sub_batches):,} sub-batches",
        ),
    ]


def report(measurements: list[Measurement]) -> None:
    print(f"{'step':<38} {'time':>8} {'allocated':>13} {'retained':>12}   detail")
    print("-" * 100)
    for m in measurements:
        print(f"{m.label:<38} {m.seconds:7.1f}s {m.allocated_mb:11.1f} MB {m.retained_mb:+10.1f} MB   {m.detail}")
    print("-" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mb", type=int, default=DEFAULT_DOC_MB, help="document size in MB")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="retain chunk size in chars")
    args = parser.parse_args()
    report(run(args.mb, args.chunk_size))


if __name__ == "__main__":
    main()
