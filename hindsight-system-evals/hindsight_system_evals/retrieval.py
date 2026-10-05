"""Score `recall` over a FROZEN bank of realistic memories.

This eval measures **retrieval**, not the pipeline that fed it. So the memories are
built once and committed as a bank archive, and the eval restores that archive
instead of re-ingesting the corpus:

    build (on purpose, costs ~500 model calls)     eval (every run, ZERO model calls)
    ---------------------------------------        ----------------------------------
    retain 500 BEIR abstracts, real extraction     import the archive
    export_bank -> fixtures/beir-scifact-bank.zip  recall each query, score vs. qrels

Re-ingesting per run would mean paying for extraction to measure something
extraction is not under test for, and it would move the corpus underneath every
comparison — a model that extracted different claims this afternoon changes the
score without anything in retrieval changing. The frozen archive makes the
memories a constant, which is the only way an A/B on a retrieval setting means
anything.

**Embeddings deliberately do not travel in the archive** — `aexport_bank` leaves
them out and the importer regenerates them with its own model. That is not a
limitation here, it is the feature that makes the embedder itself one of the
settings this eval can A/B.

What is a build-time decision (baked into the zip) and what is a test-time one:

| baked in at build | free at test time |
|---|---|
| extraction model and mode | embedder, reranker |
| which claims each abstract became | `budget`, `max_tokens` |
| observations / consolidation off | fusion and scoring settings |

The scorer collapses memory units back onto their source document, because a qrel
labels a document and `recall` returns facts — several per document — so a raw
top-10 can be one abstract ten times. And `recall` takes no `k`: it budgets by
`max_tokens`, so the list is widened with tokens and cut at k in the scorer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from hindsight_client import Hindsight

from hindsight_system_evals.beir import BeirQuery, BeirSlice, load_beir_slice
from hindsight_system_evals.retrieval_metrics import QueryScore, RunScore, aggregate, score_query

#: Waits until a bank has no pending work — the suite's `settled` fixture.
SettleFn = Callable[[str], Awaitable[None]]

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

#: Overridable so a rebuilt archive can be measured before it replaces the committed one.
BANK_FIXTURE = Path(os.getenv("HINDSIGHT_EVAL_BEIR_BANK") or _FIXTURES / "beir-scifact-bank.zip")

#: What model wrote these memories, and from how many documents. Unknowable from
#: the zip, and the single most important thing about it — a score over claims a
#: weak model extracted is a different measurement from the same retrieval over a
#: strong one's. Written beside the archive at build time.
BANK_MANIFEST = Path(str(BANK_FIXTURE).removesuffix(".zip") + ".json")

#: Wide enough that the top 100 *documents* exist to be ranked. `recall` has no
#: `k`; it budgets by tokens, so every cutoff above the mean number of documents a
#: response reaches measures the BUDGET rather than the ranking. At the 4096
#: default a handful of long facts fill it and even @20 is a budget reading.
#: `RunScore` prints `mean documents returned` precisely so this can be checked
#: rather than assumed, and the eval asserts on it.
RECALL_MAX_TOKENS = int(os.getenv("HINDSIGHT_EVAL_RECALL_MAX_TOKENS", "131072"))

#: `high` searches widest. A floor set against `mid` would move the day someone
#: tuned a default, so the budget this eval measures is named, not inherited.
RECALL_BUDGET = os.getenv("HINDSIGHT_EVAL_RECALL_BUDGET", "high")

#: Queries run at once. Each is one recall and they are independent; this only
#: keeps a local reranker from thrashing.
QUERY_CONCURRENCY = int(os.getenv("HINDSIGHT_EVAL_RECALL_CONCURRENCY", "4"))

# --------------------------------------------------------------------------- #
# Build: the expensive half, run on purpose
# --------------------------------------------------------------------------- #

#: Documents per retain request. Each is one model call server-side, so a single
#: 500-item batch would hold one HTTP request open for many minutes.
BUILD_BATCH = 50

#: The extraction the committed archive was built with. `concise` is also the
#: server default, but naming it keeps the archive's provenance explicit — and
#: `chunks` builds the one-unit-per-document arm for comparison.
BUILD_EXTRACTION_MODE = os.getenv("HINDSIGHT_EVAL_RETAIN_EXTRACTION_MODE", "concise")


@dataclass(frozen=True)
class Manifest:
    """Provenance of a built archive."""

    built_at: str
    llm_provider: str
    llm_model: str
    extraction_mode: str
    documents: int
    queries: int
    facts: int

    def describe(self) -> str:
        return (
            f"{self.documents} documents -> {self.facts} memories, extraction={self.extraction_mode} "
            f"by {self.llm_provider}/{self.llm_model}, built {self.built_at}"
        )


def read_manifest() -> Manifest | None:
    """The archive's provenance, or None when it was built before manifests existed."""
    if not BANK_MANIFEST.exists():
        return None
    return Manifest(**json.loads(BANK_MANIFEST.read_text(encoding="utf-8")))


async def build_bank(client: Hindsight, sliced: BeirSlice, drain: SettleFn) -> bytes:
    """Retain every document with real extraction and export the bank as an archive.

    The corpus id becomes `document_id`, which is what makes scoring possible at
    all: every fact a document produced carries it back on the recall result, so a
    unit can be attributed to the document the qrels label. The archive preserves
    it, which the eval asserts rather than assumes.

    Observations and consolidation are off. Both synthesise units with no single
    source document — an observation merges facts from many — so they would put
    rows in the ranking that no qrel can label, and a retrieval score cannot tell
    a correct observation from a miss.
    """
    bank_id = f"beir-build-{uuid.uuid4().hex[:8]}"
    await client.aupdate_bank_config(
        bank_id,
        retain_extraction_mode=BUILD_EXTRACTION_MODE,
        enable_observations=False,
        enable_auto_consolidation=False,
    )
    for start in range(0, len(sliced.documents), BUILD_BATCH):
        batch = sliced.documents[start : start + BUILD_BATCH]
        await client.aretain_batch(
            bank_id=bank_id,
            items=[{"content": doc.content, "document_id": doc.id} for doc in batch],
            retain_async=True,
        )
        # Drained per batch rather than once at the end: a 500-document build is
        # ~500 model calls, and a provider error 400 documents in should fail here
        # with the batch that caused it, not after another eight minutes of work.
        await drain(bank_id)

    # `limit=1` for the count only: the manifest wants `total`, not the units.
    facts = (await client.alist_memories(bank_id, limit=1)).total
    archive = await client.aexport_bank(bank_id, include_history=False, timeout=1800)
    _write_manifest(sliced, facts)
    return archive


def _write_manifest(sliced: BeirSlice, facts: int) -> None:
    manifest = Manifest(
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
        llm_provider=os.getenv("HINDSIGHT_EVAL_BUILD_LLM_PROVIDER", "unknown"),
        llm_model=os.getenv("HINDSIGHT_EVAL_BUILD_LLM_MODEL", "unknown"),
        extraction_mode=BUILD_EXTRACTION_MODE,
        documents=len(sliced.documents),
        queries=len(sliced.queries),
        facts=facts,
    )
    BANK_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    BANK_MANIFEST.write_text(json.dumps(manifest.__dict__, indent=1) + "\n", encoding="utf-8")


async def _drain(client: Hindsight, bank_id: str, timeout: float = 7200) -> None:
    """Wait until nothing is queued, then fail on anything that ended failed.

    Not `wait_until_settled`: that gives up on the first operation carrying an
    error, while a build runs hundreds of model calls and a transient provider
    error is what the worker's retries are for. Only an operation that exhausted
    them is a broken build. Same reasoning as `refresh_cost._drain`.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        busy = [
            op
            for status in ("pending", "processing")
            for op in (await client.operations.list_operations(bank_id, status=status, limit=100)).operations
        ]
        if not busy:
            break
        await asyncio.sleep(5)
    else:
        raise TimeoutError(f"bank {bank_id} still busy after {timeout}s")
    failed = (await client.operations.list_operations(bank_id, status="failed", limit=100)).operations
    if failed:
        raise RuntimeError(f"bank {bank_id}: {', '.join(f'{op.task_type}: {op.error_message}' for op in failed)}")


# --------------------------------------------------------------------------- #
# Eval: the free half, run every time
# --------------------------------------------------------------------------- #


async def restore_bank(client: Hindsight, host_bank: str, settled: SettleFn) -> str:
    """Restore the frozen archive into a fresh bank and wait for the re-embedding.

    The import operation is recorded against an existing bank while the target
    must NOT exist, so there are two banks and two settles. The re-embedding is
    the slow part and it is local, not billed.
    """
    if not BANK_FIXTURE.exists():
        raise RuntimeError(
            f"missing {BANK_FIXTURE} — build it with: uv run python -m hindsight_system_evals.retrieval build"
        )
    await client.aupdate_bank_config(host_bank, enable_observations=False, enable_auto_consolidation=False)
    target = f"{host_bank}-beir"
    await client.aimport_bank(host_bank, BANK_FIXTURE.read_bytes(), target_bank_id=target)
    await settled(host_bank)
    await settled(target)
    # Nothing may call the model after import: the archive is the corpus, and a
    # consolidation pass firing mid-run would add unlabelable units to the ranking.
    await client.aupdate_bank_config(target, enable_observations=False, enable_auto_consolidation=False)
    await _await_full_restore(client, target)
    return target


#: How long to keep waiting after the bank looks settled.
_RESTORE_SETTLE_GRACE_SECONDS = 900.0

#: Consecutive identical readings before the restore is believed finished.
_STABLE_READINGS = 3

#: Seconds between readings.
_STABLE_INTERVAL_SECONDS = 4.0


async def _await_full_restore(client: Hindsight, bank_id: str) -> None:
    """Block until the restore has stopped changing what `recall` can see.

    Two waits, because neither alone is enough and both were learned the hard way.

    `wait_until_settled` goes quiet when no operation is pending or processing,
    but the restore is still writing after that. So first the memory count has to
    reach the number the manifest records for the archive.

    That is still not sufficient: the rows land BEFORE their embeddings do, and a
    memory without a vector is invisible to the semantic arm. A bank holding all
    1,812 memories therefore still returns a growing number of documents for a
    while, and scoring during that window reported nDCG@10 0.874 and 137 documents
    where a finished bank gives 0.887 and 160 — a 1.5% "regression" that is purely
    an artefact of reading too early. On a floor-asserting gate that is a random
    red build, which is worse than no gate.

    So the second wait measures the thing the metrics actually depend on: run the
    real query and wait for the document count to stop moving. Behavioural, uses
    only the public API, and cannot drift from what the scorer sees because it is
    the same call.
    """
    manifest = read_manifest()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _RESTORE_SETTLE_GRACE_SECONDS

    if manifest and manifest.facts:
        total = 0
        while loop.time() < deadline:
            total = (await client.alist_memories(bank_id, limit=1)).total
            if total >= manifest.facts:
                break
            await asyncio.sleep(_STABLE_INTERVAL_SECONDS)
        else:
            raise AssertionError(
                f"bank {bank_id} settled with {total} of the archive's {manifest.facts} memories after "
                f"{_RESTORE_SETTLE_GRACE_SECONDS:.0f}s — scoring a partial restore reads as a ranking regression"
            )

    probe = load_beir_slice().queries[0].text
    seen: list[int] = []
    while loop.time() < deadline:
        response = await client.arecall(
            bank_id=bank_id, query=probe, max_tokens=RECALL_MAX_TOKENS, budget=RECALL_BUDGET
        )
        seen.append(len({r.document_id for r in response.results if r.document_id}))
        if len(seen) >= _STABLE_READINGS and len(set(seen[-_STABLE_READINGS:])) == 1:
            return
        await asyncio.sleep(_STABLE_INTERVAL_SECONDS)
    raise AssertionError(
        f"bank {bank_id} never stopped changing: the probe query returned {seen[-6:]} distinct documents "
        f"over the last readings, so embeddings were still landing after {_RESTORE_SETTLE_GRACE_SECONDS:.0f}s"
    )


async def score(client: Hindsight, bank_id: str, sliced: BeirSlice) -> RunScore:
    """Recall each query once and score the document ranking it produced."""
    semaphore = asyncio.Semaphore(QUERY_CONCURRENCY)

    async def one(query: BeirQuery) -> QueryScore:
        async with semaphore:
            response = await client.arecall(
                bank_id=bank_id,
                query=query.text,
                max_tokens=RECALL_MAX_TOKENS,
                budget=RECALL_BUDGET,
            )
        # A result with no document_id cannot be scored either way. There should be
        # none — every unit here came from a retained document — so it is dropped
        # rather than counted as a miss, and the eval asserts the count instead of
        # letting it be silently absorbed into a lower score.
        ranked = [result.document_id for result in response.results if result.document_id]
        return score_query(query.id, ranked, query.relevant)

    scores = await asyncio.gather(*(one(query) for query in sliced.queries))
    return aggregate(list(scores))


def run_label() -> str:
    """A short name for the retrieval configuration this run measured.

    The reranker is a SERVER-level setting — it is not in `_CONFIGURABLE_FIELDS`,
    so no bank can switch it — which means comparing two rerankers means two
    server processes and therefore two runs. A label per run is how the two
    results are told apart afterwards, and guessing it from a filename is how
    someone eventually publishes the wrong arm.
    """
    provider = os.getenv("HINDSIGHT_EVAL_SET_RERANKER_PROVIDER", "local")
    if provider != "typesafe":
        return f"{provider} reranker"
    model = os.getenv("HINDSIGHT_EVAL_SET_RERANKER_TYPESAFE_MODEL", "jev-latest")
    prune = os.getenv("HINDSIGHT_EVAL_SET_RERANKER_TYPESAFE_PRUNE_CANDIDATES", "false").lower() == "true"
    return f"typesafe {model} ({'drop' if prune else 'keep'} irrelevant)"


def prunes_candidates() -> bool:
    """Whether this run's reranker drops candidates it judges irrelevant.

    Changes what the metrics mean, so it cannot be left implicit: a pruning run
    returns a couple of documents instead of ~160, and every cutoff above that is
    then bounded by the prune decision rather than by the token budget or the
    ranking. Recall@k stays honest (the gold document is either in the kept set or
    it is not); Recall@100 stops being comparable with a non-pruning arm.
    """
    return os.getenv("HINDSIGHT_EVAL_SET_RERANKER_TYPESAFE_PRUNE_CANDIDATES", "false").lower() == "true"


def run_notes(manifest: Manifest | None) -> dict[str, str]:
    """Everything needed to reproduce a run's number, as flat strings.

    Collected from the environment rather than from the server: these are the
    `HINDSIGHT_EVAL_SET_*` values this process chose, which is what a reader needs
    to re-run it. Deliberately a plain string map — it is provenance for a JSON
    file and a chart caption, not structured data anything branches on.
    """
    notes = {
        "recall_budget": RECALL_BUDGET,
        "recall_max_tokens": str(RECALL_MAX_TOKENS),
        "reranker_provider": os.getenv("HINDSIGHT_EVAL_SET_RERANKER_PROVIDER", "local (default)"),
        "embeddings_provider": os.getenv("HINDSIGHT_EVAL_SET_EMBEDDINGS_PROVIDER", "local (default)"),
    }
    for suffix in ("RERANKER_TYPESAFE_MODEL", "RERANKER_TYPESAFE_PRUNE_CANDIDATES", "RERANKER_LOCAL_MODEL"):
        if value := os.getenv(f"HINDSIGHT_EVAL_SET_{suffix}"):
            notes[suffix.lower()] = value
    if manifest:
        notes["memories"] = f"{manifest.facts} from {manifest.documents} documents"
        notes["extracted_by"] = f"{manifest.llm_provider}/{manifest.llm_model} ({manifest.extraction_mode})"
        notes["archive_built_at"] = manifest.built_at
    return notes


def worst_queries(run: RunScore, *, count: int = 5) -> str:
    """The queries that cost the most, with which half of the pipeline to blame.

    A macro-average that dropped says nothing about what to fix. These lines do:
    a run where every loser reports `retrieval` is a different bug from one where
    they all report `ranking`.
    """
    losers = sorted(run.per_query, key=lambda s: (s.ndcg[10], s.mrr))[:count]
    return "\n".join(
        f"  {s.query_id}: nDCG@10={s.ndcg[10]:.3f} MRR={s.mrr:.3f} "
        f"({s.gold_found}/{s.gold_total} gold, {s.retrieved} documents returned) — {s.blame}"
        for s in losers
    )


# --------------------------------------------------------------------------- #
# `python -m hindsight_system_evals.retrieval build`
# --------------------------------------------------------------------------- #


async def _build(out: Path, api_url: str, api_key: str | None) -> None:
    """Build against a server the developer already configured with a real model.

    Deliberately not starting one here. The suite's own server points every model
    at the stub, which is the whole reason it needs no secrets — and extraction is
    the one step of this that must NOT be stubbed, since the claims the model
    writes are the corpus the floors are measured over. Rather than smuggle a
    second, real-LLM server into a package built to avoid exactly that, the
    builder asks for the URL of one. It runs by hand, rarely, and the archive it
    writes is what every test run reads.
    """
    sliced = load_beir_slice()
    print(f"building from {sliced.describe()} against {api_url}, extraction={BUILD_EXTRACTION_MODE}")
    client = Hindsight(base_url=api_url, api_key=api_key, timeout=1800)
    try:
        archive = await build_bank(client, sliced, lambda bank: _drain(client, bank))
    finally:
        await client.aclose()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(archive)
    manifest = read_manifest()
    print(f"wrote {out} ({len(archive) / 1e6:.1f} MB)" + (f" — {manifest.describe()}" if manifest else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="rebuild the frozen bank archive (real model, slow, costs money)")
    build.add_argument("--out", type=Path, default=BANK_FIXTURE)
    build.add_argument(
        "--api-url",
        required=True,
        help="a running hindsight-api configured with the extraction model to bake into the archive",
    )
    build.add_argument("--api-key", default=None)
    args = parser.parse_args()
    if args.command == "build":
        asyncio.run(_build(args.out, args.api_url, args.api_key))


if __name__ == "__main__":
    main()
