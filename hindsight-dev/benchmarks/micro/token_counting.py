"""Microbenchmark for the token counting on the recall path.

Recall calls ``count_tokens``-shaped code once per candidate, in a serial Python
loop, at four call sites:

* ``_truncate_query_to_token_limit`` — once per recall, on the query;
* the chunk token-budget loop in ``memory_engine`` — once per candidate chunk;
* ``select_facts_within_budget`` — once per ranked fact;
* ``select_source_facts_within_budget`` — once per source fact;

plus ``cross_encoder._truncate_to_tokens``, which encodes every reranker
document. All of them go through the same encoder, so the question this
benchmark answers is narrow: for a batch of N texts, what does counting cost in
wall time, CPU time and peak allocation, and how much of that is avoidable?

The variants measured are the production call and the cheaper spellings of the
same count:

``prod``
    ``token_encoding.count_tokens`` exactly as recall calls it — quicktok's
    count-only API under the configured encoding.
``encode``
    ``Encoding.encode(disallowed_special=())`` — the production call without the
    ``_SafeEncoding`` wrapper, isolating per-call Python overhead.
``encode_ordinary``
    ``Encoding.encode_ordinary`` — same counts for content that mentions a
    special-token literal, without the special-token machinery.
``encode_to_numpy``
    ``Encoding.encode_to_numpy`` — tiktoken has no count-only API, and this is the
    closest it comes: the ids land in a ``uint32`` array instead of a Python list,
    so counting them costs no per-token ``PyLong``. This is the variant a
    count-only candidate has to beat to be worth a dependency.
``ordinary_batch_1`` / ``ordinary_batch_N``
    ``Encoding.encode_ordinary_batch`` — one Rust call for the whole list. The
    batch releases the GIL and fans out over rayon, so ``cpu/wall > 1`` here is
    the parallelism, and peak memory is the cost of holding every token list at
    once instead of one at a time.
``chars_div_4``
    ``len(text) // 4``. Not a proposal — the floor, so the numbers above have
    something to be compared against.

``quicktok_count`` / ``quicktok_encode`` / ``quicktok_count_batch``
    The spellings of the production tokenizer's own API. ``quicktok_count`` is
    what ``prod`` resolves to; the other two show what the truncate call sites
    (which need the ids) and a batched counter would cost instead.

The tiktoken variants are kept as the baseline this repo migrated *from*, so the
comparison stays reproducible — tiktoken is no longer a Hindsight dependency, so
install it explicitly to include them.

Every variant is checked against ``prod``'s total count; a variant that counts
differently is reported as MISMATCH rather than as a speedup.

Usage (from the repo root):

    ./scripts/benchmarks/run-token-count-bench.sh
    ./scripts/benchmarks/run-token-count-bench.sh --repeats 10 --json out.json
    ./scripts/benchmarks/run-token-count-bench.sh --workload facts_200

    # include the tiktoken baseline (no longer a project dependency):
    cd hindsight-dev && uv run --with tiktoken token-count-bench
"""

import argparse
import gc
import json
import os
import pathlib
import random
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import quicktok
from hindsight_api.config import DEFAULT_TOKENIZER_ENCODING
from hindsight_api.engine.token_encoding import count_tokens
from rich.console import Console
from rich.table import Table

console = Console()

# A small closed vocabulary keeps the char/token ratio in the range real memory text
# sits at (~4 chars/token) without shipping a corpus file.
_WORDS = (
    "memory recall bank observation consolidation embedding retrieval fact entity "
    "postgres pgvector migration reranker chunk document token budget disposition "
    "the user asked about deployment latency during the incident review yesterday "
    "resolved merged reverted scheduled increased dropped measured configured"
).split()


@dataclass(frozen=True)
class Workload:
    """A batch of texts shaped like one recall call site."""

    name: str
    call_site: str
    texts: list[str]

    @property
    def total_chars(self) -> int:
        return sum(len(t) for t in self.texts)


@dataclass
class VariantResult:
    """One variant measured against one workload."""

    workload: str
    variant: str
    wall_ms: float  # best of --repeats, milliseconds
    cpu_ms: float  # process CPU (all threads) over that same best run
    peak_kib: float  # tracemalloc peak of a separate single run
    tokens: int
    matches_baseline: bool


def _make_text(rng: random.Random, approx_tokens: int) -> str:
    # ~1.3 tokens per word for this vocabulary; overshoot slightly and let the
    # measured count be whatever it is — the absolute size is what matters, not
    # hitting the target exactly.
    words = [rng.choice(_WORDS) for _ in range(max(1, int(approx_tokens / 1.3)))]
    return " ".join(words)


def _corpus_texts(path: str, rng: random.Random, count: int, approx_tokens: int) -> list[str]:
    """``count`` slices of a real corpus, each roughly ``approx_tokens`` long.

    The synthetic generator draws from a tiny vocabulary, which flatters every BPE
    implementation — the merges it exercises stay in cache. Point ``--corpus`` at a
    file of real memory text to check whether a measured speedup survives it.
    """
    body = pathlib.Path(path).read_text(errors="replace")
    width = max(1, approx_tokens * 4)  # ~4 chars/token
    if len(body) <= width:
        return [body] * count
    return [body[start : start + width] for start in (rng.randrange(len(body) - width) for _ in range(count))]


def build_workloads(seed: int, corpus: str | None = None) -> list[Workload]:
    """Batches sized like what recall actually hands the encoder."""
    rng = random.Random(seed)

    def texts(count: int, approx_tokens: int) -> list[str]:
        if corpus:
            return _corpus_texts(corpus, rng, count, approx_tokens)
        return [_make_text(rng, approx_tokens) for _ in range(count)]

    return [
        Workload(
            name="query_1",
            call_site="_truncate_query_to_token_limit",
            texts=texts(1, 30),
        ),
        Workload(
            name="facts_200",
            call_site="select_facts_within_budget",
            texts=texts(200, 45),
        ),
        Workload(
            name="source_facts_500",
            call_site="select_source_facts_within_budget",
            texts=texts(500, 30),
        ),
        Workload(
            name="chunks_50",
            call_site="chunk token-budget loop",
            texts=texts(50, 500),
        ),
        Workload(
            name="rerank_docs_100",
            call_site="cross_encoder._truncate_to_tokens",
            texts=texts(100, 250),
        ),
        Workload(
            name="one_large_doc",
            call_site="retain count_tokens (memory reference)",
            texts=texts(1, 100_000),
        ),
    ]


# --- variants ---------------------------------------------------------------
# Each returns the total token count for the batch, so results are comparable.


# The encoding production actually uses. ``--encoding`` re-runs the same comparison
# on another vocabulary: a 200k vocabulary is a different amount of work per byte,
# so the ranking here does not transfer between them for free.
PROD_ENCODING = DEFAULT_TOKENIZER_ENCODING


def _tiktoken_encoding(name: str):
    """The tiktoken encoding for ``name``, or ``None`` when tiktoken is absent.

    tiktoken is what this module measures *against*, not with — it was dropped as a
    Hindsight dependency once these numbers justified the swap. The variants stay so
    the decision can be re-checked, but the package is never required: install it
    explicitly (``uv run --with tiktoken token-count-bench``) to include them.
    """
    try:
        import tiktoken
    except ImportError:
        return None
    return tiktoken.get_encoding(name)


def _v_prod(texts: Sequence[str], name: str) -> int:
    # The real production call, wrapper and all, when measuring the configured
    # encoding; the same shape on quicktok directly when measuring another.
    if name == PROD_ENCODING:
        return sum(count_tokens(t) for t in texts)
    return sum(quicktok.get_encoding(name).count(t) for t in texts)


def _v_quicktok_encode(texts: Sequence[str], name: str) -> int:
    # The truncate call sites need the ids, not just the count.
    enc = quicktok.get_encoding(name)
    return sum(len(enc.encode(t, disallowed_special=())) for t in texts)


def _v_quicktok_count_batch(texts: Sequence[str], name: str) -> int:
    return int(quicktok.count_batch(quicktok.get_encoding(name), list(texts)).sum())


def _v_chars_div_4(texts: Sequence[str], name: str) -> int:
    return sum(len(t) // 4 for t in texts)


def _v_tiktoken_encode(texts: Sequence[str], name: str) -> int:
    enc = _tiktoken_encoding(name)
    return sum(len(enc.encode(t, disallowed_special=())) for t in texts)


def _v_tiktoken_encode_ordinary(texts: Sequence[str], name: str) -> int:
    enc = _tiktoken_encoding(name)
    return sum(len(enc.encode_ordinary(t)) for t in texts)


def _v_tiktoken_encode_to_numpy(texts: Sequence[str], name: str) -> int:
    # tiktoken's nearest thing to a count-only API: the ids come back as a uint32
    # array, so no Python list of PyLongs is built just to be measured and dropped.
    enc = _tiktoken_encoding(name)
    return sum(len(enc.encode_to_numpy(t, disallowed_special=())) for t in texts)


def _v_tiktoken_ordinary_batch(texts: Sequence[str], name: str, num_threads: int) -> int:
    enc = _tiktoken_encoding(name)
    return sum(len(t) for t in enc.encode_ordinary_batch(list(texts), num_threads=num_threads))


def build_variants(encoding: str, threads: int) -> dict[str, Callable[[Sequence[str]], int]]:
    def bind(fn, **extra):
        return lambda texts: fn(texts, encoding, **extra)

    variants: dict[str, Callable[[Sequence[str]], int]] = {
        "prod": bind(_v_prod),
        "quicktok_encode": bind(_v_quicktok_encode),
        "quicktok_count_batch": bind(_v_quicktok_count_batch),
        "chars_div_4": bind(_v_chars_div_4),
    }

    if _tiktoken_encoding(encoding) is not None:
        variants["tiktoken_encode"] = bind(_v_tiktoken_encode)
        variants["tiktoken_encode_ordinary"] = bind(_v_tiktoken_encode_ordinary)
        variants["tiktoken_encode_to_numpy"] = bind(_v_tiktoken_encode_to_numpy)
        variants["tiktoken_batch_1"] = bind(_v_tiktoken_ordinary_batch, num_threads=1)
        variants[f"tiktoken_batch_{threads}"] = bind(_v_tiktoken_ordinary_batch, num_threads=threads)
    return variants


@dataclass(frozen=True)
class Timing:
    """One variant's best timed run over a workload."""

    wall_ms: float
    cpu_ms: float  # CPU of that same run, summed across threads
    tokens: int


def _measure(fn: Callable[[Sequence[str]], int], texts: Sequence[str], repeats: int) -> Timing:
    """Best-of-``repeats`` wall time, with the CPU time of that same run.

    Best-of rather than mean: the thing being measured is the encoder's cost, and
    the scheduler only ever adds to it.
    """
    fn(texts)  # warm up caches (encoder load, regex compile, branch prediction)
    best_wall = float("inf")
    best_cpu = 0.0
    tokens = 0
    for _ in range(repeats):
        gc.collect()
        t0, c0 = time.perf_counter(), time.process_time()
        tokens = fn(texts)
        wall = time.perf_counter() - t0
        cpu = time.process_time() - c0
        if wall < best_wall:
            best_wall, best_cpu = wall, cpu
    return Timing(wall_ms=best_wall * 1000, cpu_ms=best_cpu * 1000, tokens=tokens)


def _measure_peak_kib(fn: Callable[[Sequence[str]], int], texts: Sequence[str]) -> float:
    """Peak *traced* allocation of a single run.

    tracemalloc, not RSS: RSS is order-dependent because the allocator reuses
    arenas, which makes a bounded implementation look unbounded (and the reverse)
    depending on what ran before it.
    """
    gc.collect()
    tracemalloc.start()
    try:
        fn(texts)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 1024


def run(workloads: Sequence[Workload], encoding: str, threads: int, repeats: int) -> list[VariantResult]:
    variants = build_variants(encoding, threads)
    results: list[VariantResult] = []
    for wl in workloads:
        baseline_tokens: int | None = None
        for name, fn in variants.items():
            try:
                timing = _measure(fn, wl.texts, repeats)
                peak_kib = _measure_peak_kib(fn, wl.texts)
            except Exception as err:
                # One variant that raises must not take the whole run with it —
                # raising *is* a result for a candidate encoder (issue #1883).
                console.print(f"[red]{wl.name}/{name}: {type(err).__name__}: {err}[/red]")
                continue
            if baseline_tokens is None:
                baseline_tokens = timing.tokens
            results.append(
                VariantResult(
                    workload=wl.name,
                    variant=name,
                    wall_ms=timing.wall_ms,
                    cpu_ms=timing.cpu_ms,
                    peak_kib=peak_kib,
                    tokens=timing.tokens,
                    # chars_div_4 is an estimator, not a candidate — never flagged.
                    matches_baseline=(timing.tokens == baseline_tokens or name == "chars_div_4"),
                )
            )
    return results


# Inputs that have historically broken token counting here. A candidate encoder is
# only interesting if it agrees with the production count on every one of them —
# a faster wrong number is not a speedup.
_CONFORMANCE_CASES: dict[str, str] = {
    "ascii": "the user asked about deployment latency during the incident review",
    "special_literal": "the model emits <|endoftext|> and <|fim_prefix|> markers",  # issue #1883
    "unicode": "🧠 memory ✅ done — naïve café 東京 مرحبا",
    "empty": "",
    "whitespace": "   \n\t  ",
    "code": "def f(x: int) -> str:\n    return f'{x!r}'  # ok\n",
}


def _conformance(encoding: str, threads: int) -> None:
    """Check every variant against the production count on adversarial inputs."""
    variants = build_variants(encoding, threads)
    table = Table(title=f"[bold]conformance[/bold] — {encoding} token count vs prod", title_justify="left")
    table.add_column("case")
    for name in variants:
        table.add_column(name, justify="right")

    for case, text in _CONFORMANCE_CASES.items():
        cells: list[str] = []
        expected: int | None = None
        for name, fn in variants.items():
            try:
                got = fn([text])
            except Exception as err:  # a raise is the failure mode #1883 was about
                cells.append(f"[red]{type(err).__name__}[/red]")
                continue
            if expected is None:
                expected = got
                cells.append(str(got))
            elif name == "chars_div_4":
                cells.append(f"[dim]{got}[/dim]")  # estimator, never expected to match
            else:
                cells.append(str(got) if got == expected else f"[red]{got}≠{expected}[/red]")
        table.add_row(case, *cells)

    console.print(table)
    console.print()


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
                f"({len(wl.texts)} text(s), {baseline.tokens:,} tokens, {wl.total_chars:,} chars)"
            ),
            title_justify="left",
        )
        table.add_column("variant")
        table.add_column("wall ms", justify="right")
        table.add_column("speedup", justify="right")
        table.add_column("cpu ms", justify="right")
        table.add_column("cpu/wall", justify="right")
        table.add_column("peak KiB", justify="right")
        table.add_column("Mtok/s", justify="right")
        table.add_column("counts", justify="right")

        for r in rows:
            speedup = baseline.wall_ms / r.wall_ms if r.wall_ms > 0 else float("inf")
            mtok_s = (r.tokens / r.wall_ms / 1000) if r.wall_ms > 0 else 0.0
            table.add_row(
                r.variant,
                f"{r.wall_ms:.3f}",
                "—" if r.variant == baseline.variant else f"{speedup:.2f}x",
                f"{r.cpu_ms:.3f}",
                f"{(r.cpu_ms / r.wall_ms):.2f}" if r.wall_ms > 0 else "—",
                f"{r.peak_kib:,.0f}",
                f"{mtok_s:.2f}",
                "ok" if r.matches_baseline else "[red]MISMATCH[/red]",
            )
        console.print(table)
        console.print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repeats", type=int, default=5, help="timed repeats per variant (best-of); default 5")
    parser.add_argument(
        "--threads",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="num_threads for the parallel batch variant; default min(8, cpu_count)",
    )
    parser.add_argument("--seed", type=int, default=1234, help="text-generation seed; default 1234")
    parser.add_argument(
        "--encoding",
        default=PROD_ENCODING,
        help=(
            f"encoding to measure; default {PROD_ENCODING} (what production uses). "
            "o200k_base is the current OpenAI vocabulary — measuring it says nothing about switching "
            "production to it, which would change every token count and therefore every budget."
        ),
    )
    parser.add_argument(
        "--corpus",
        help="slice the workloads out of this text file instead of generating them (real text is harder)",
    )
    parser.add_argument(
        "--workload",
        action="append",
        help="run only this workload (repeatable); default all",
    )
    parser.add_argument("--json", dest="json_path", help="also write raw results to this path")
    parser.add_argument(
        "--no-conformance",
        action="store_true",
        help="skip the count-agreement check on adversarial inputs",
    )
    args = parser.parse_args()

    workloads = build_workloads(args.seed, args.corpus)
    if args.workload:
        wanted = set(args.workload)
        unknown = wanted - {w.name for w in workloads}
        if unknown:
            parser.error(f"unknown workload(s): {', '.join(sorted(unknown))}")
        workloads = [w for w in workloads if w.name in wanted]

    # Encoder load is a one-off (and may hit the network on a cold tiktoken cache).
    # Time it separately so it never lands inside a variant's numbers.
    t0 = time.perf_counter()
    quicktok.get_encoding(args.encoding)
    console.print(
        f"[dim]{args.encoding} load: {(time.perf_counter() - t0) * 1000:.1f} ms  "
        f"(one-off, lru_cached) | cpu_count={os.cpu_count()} | repeats={args.repeats}[/dim]\n"
    )

    if not args.no_conformance:
        _conformance(args.encoding, args.threads)

    results = run(workloads, encoding=args.encoding, threads=args.threads, repeats=args.repeats)
    _render(workloads, results)

    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        console.print(f"[dim]wrote {args.json_path}[/dim]")


if __name__ == "__main__":
    main()
