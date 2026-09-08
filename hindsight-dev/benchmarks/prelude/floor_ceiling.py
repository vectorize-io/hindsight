"""Floor/ceiling experiment: can query wording move retrieval on this corpus at all?

Reflect opens every run by walking its enabled retrieval layers (mental models ->
observations -> recall). Historically each layer was a forced LLM turn, so its
query was written having READ the layers above it. Planning all the queries up
front (#3865) makes them "cold" — and because the mental-model result drives the
deterministic short-circuit that decides whether the lower layers run at all,
cold queries can change the whole evidence set, not just one search.

Before measuring what a planner produces, measure whether the wording matters:

* **floor** — search each layer with the user's question verbatim. This is not a
  strawman: it is exactly what the prelude falls back to when planning fails.
* **ceiling** — search with the hand-written ``ideal_query`` from the corpus, the
  best a planner could plausibly do.

No LLM is involved in either arm, so both are deterministic and the gap is
attributable to wording alone. If floor is approximately ceiling, wording does
not move retrieval here and the planned prelude costs nothing — no budget
branch, no stochastic eval needed. A wide gap is the signal to go build the
stochastic tier and measure where a real planner lands between the two.

Run with::

    ./scripts/benchmarks/run-prelude-eval.sh
    # or
    cd hindsight-dev && uv run python -m benchmarks.prelude.floor_ceiling
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.reflect.tools import (
    tool_recall,
    tool_search_mental_models,
    tool_search_observations,
)
from hindsight_api.engine.retain import embedding_utils
from hindsight_api.engine.task_backend import SyncTaskBackend
from hindsight_api.models import RequestContext
from rich.console import Console
from rich.table import Table

from benchmarks.prelude.fixture import Corpus, Question, build_corpus

console = Console()


# How deep into the evidence the metric looks. A set-membership metric ("was the
# gold id returned at all") saturates at 1.0 as soon as the corpus is smaller
# than one recall page — which says nothing about whether the query was any good.
# Rank matters: the synthesis model reads the top of the evidence, and a token
# budget truncates the tail. So score the top K and the position of the first hit.
TOP_K = 3


@dataclass
class ArmResult:
    """What one query string retrieved for one question, scored by RANK."""

    arm: str
    query: str
    retrieved: list[str]
    gold_ranks: list[int]  # 1-based position of each gold id, -1 when absent
    gold_total: int
    n_retrieved: int
    short_circuit: bool

    @property
    def recall_at_k(self) -> float:
        """Fraction of gold ids inside the top K. Undefined (1.0) with no gold."""
        if self.gold_total == 0:
            return 1.0
        return sum(1 for r in self.gold_ranks if 1 <= r <= TOP_K) / self.gold_total

    @property
    def mrr(self) -> float:
        """Reciprocal rank of the FIRST gold hit — how far down the reader must go."""
        hits = [r for r in self.gold_ranks if r > 0]
        return 1.0 / min(hits) if hits else 0.0


@dataclass
class QuestionResult:
    id: str
    category: str
    question: str
    floor: ArmResult
    ceiling: ArmResult
    short_circuit_expected: bool


async def _run_layers(
    memory: MemoryEngine,
    corpus: Corpus,
    ctx: RequestContext,
    query: str,
) -> tuple[dict[str, list[str]], bool]:
    """Search all three layers with ONE query and return (ids per layer, short-circuited).

    Results stay SEPARATED BY LAYER because rank only means anything within a
    layer: the agent reads each layer's results as its own block, and a raw fact
    competes with other raw facts, never with the mental models listed above it.
    Ranking a flat concatenation would score every fact behind every mental model
    no matter how good the query was.

    Deliberately not the reflect agent: this measures the retrieval stack given a
    fixed query, with no model in the loop to add variance.
    """
    by_layer: dict[str, list[str]] = {"mental_models": [], "observations": [], "recall": []}

    # Mirror memory_engine's search_mental_models_fn closure: the tool takes a
    # connection, a query embedding, and the bank's last-write watermark that its
    # staleness check is resolved against.
    embeddings = await embedding_utils.generate_embeddings_batch(memory.embeddings, [query], input_type="query")
    freshness = await memory.get_bank_freshness(corpus.bank_id, request_context=ctx)
    raw_watermark = freshness.get("last_memory_write_at")
    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        mm = await tool_search_mental_models(
            memory,
            conn,
            corpus.bank_id,
            query,
            embeddings[0],
            last_memory_write_at=datetime.fromisoformat(raw_watermark) if raw_watermark else None,
        )
    models = mm.get("mental_models") or []
    by_layer["mental_models"] = [str(m["id"]) for m in models if "id" in m]

    # The production short-circuit, verbatim: every returned model explicitly
    # fresh and non-empty. Reproduced rather than imported because the eval must
    # keep measuring the *rule*, not whatever the agent's internals become.
    short_circuit = bool(models) and all(
        m.get("is_stale") is False and str(m.get("content") or "").strip() for m in models
    )
    if short_circuit:
        return by_layer, True

    obs = await tool_search_observations(memory, corpus.bank_id, query, ctx)
    by_layer["observations"] = [str(o["id"]) for o in (obs.get("observations") or []) if "id" in o]

    rec = await tool_recall(memory, corpus.bank_id, query, ctx)
    by_layer["recall"] = [str(m["id"]) for m in (rec.get("memories") or []) if "id" in m]

    return by_layer, False


async def _score(
    memory: MemoryEngine, corpus: Corpus, ctx: RequestContext, question: Question, arm: str, query: str
) -> ArmResult:
    by_layer, short_circuit = await _run_layers(memory, corpus, ctx, query)
    gold = corpus.gold_row_ids(question)

    def _rank(row_id: str) -> int:
        """1-based rank of ``row_id`` within whichever layer returned it."""
        for ids in by_layer.values():
            if row_id in ids:
                return ids.index(row_id) + 1
        return -1

    return ArmResult(
        arm=arm,
        query=query,
        retrieved=[i for ids in by_layer.values() for i in ids],
        gold_ranks=[_rank(g) for g in sorted(gold)],
        gold_total=len(gold),
        n_retrieved=sum(len(v) for v in by_layer.values()),
        short_circuit=short_circuit,
    )


def _display(results: list[QuestionResult]) -> None:
    table = Table(title=f"Floor (question verbatim) vs ceiling (ideal query) — recall@{TOP_K} / MRR")
    table.add_column("question")
    table.add_column("category")
    table.add_column("floor", justify="right")
    table.add_column("ceiling", justify="right")
    table.add_column("Δ r@k", justify="right")
    table.add_column("ranks f→c", justify="center")
    table.add_column("SC f/c/want", justify="center")

    for r in results:
        delta = r.ceiling.recall_at_k - r.floor.recall_at_k
        colour = "" if abs(delta) < 1e-9 else ("green" if delta > 0 else "red")
        delta_cell = f"[{colour}]{delta:+.2f}[/{colour}]" if colour else "0.00"
        sc = f"{int(r.floor.short_circuit)}/{int(r.ceiling.short_circuit)}/{int(r.short_circuit_expected)}"
        sc_bad = (
            r.floor.short_circuit != r.short_circuit_expected or r.ceiling.short_circuit != r.short_circuit_expected
        )
        if r.floor.gold_total == 0:
            # `absent` questions have nothing to rank; what matters is how much
            # unrelated evidence the query dragged in.
            floor_cell = f"{r.floor.n_retrieved} rows"
            ceiling_cell = f"{r.ceiling.n_retrieved} rows"
            ranks = "—"
            delta_cell = ""
        else:
            floor_cell = f"{r.floor.recall_at_k:.2f} / {r.floor.mrr:.2f}"
            ceiling_cell = f"{r.ceiling.recall_at_k:.2f} / {r.ceiling.mrr:.2f}"
            ranks = f"{r.floor.gold_ranks}→{r.ceiling.gold_ranks}"
        table.add_row(
            r.id,
            r.category,
            floor_cell,
            ceiling_cell,
            delta_cell,
            ranks,
            f"[red]{sc}[/red]" if sc_bad else sc,
        )
    console.print(table)

    scored = [r for r in results if r.floor.gold_total > 0]
    floor_k = sum(r.floor.recall_at_k for r in scored) / len(scored)
    ceiling_k = sum(r.ceiling.recall_at_k for r in scored) / len(scored)
    floor_mrr = sum(r.floor.mrr for r in scored) / len(scored)
    ceiling_mrr = sum(r.ceiling.mrr for r in scored) / len(scored)
    sc_ok = sum(1 for r in results if r.floor.short_circuit == r.short_circuit_expected)

    console.print(
        f"\n[bold]recall@{TOP_K}[/bold]  floor={floor_k:.3f}  ceiling={ceiling_k:.3f}  gap={ceiling_k - floor_k:+.3f}"
        f"\n[bold]MRR[/bold]       floor={floor_mrr:.3f}  ceiling={ceiling_mrr:.3f}  gap={ceiling_mrr - floor_mrr:+.3f}"
        f"\n[bold]short-circuit[/bold] matches intent on the floor arm: {sc_ok}/{len(results)}"
    )

    gap = max(ceiling_k - floor_k, ceiling_mrr - floor_mrr)
    if gap < 0.05:
        console.print(
            "\n[yellow]Wording barely moves retrieval on this corpus.[/yellow] Before concluding the "
            "planned prelude is safe, check that the near_miss rows actually have competing "
            "neighbours — a corpus smaller than one recall page cannot discriminate anything."
        )
    else:
        console.print(
            f"\n[green]Wording moves retrieval by up to {gap:+.3f}.[/green] Worth building the "
            "stochastic tier to see where a real planner lands between these two arms."
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write raw per-question results here")
    parser.add_argument("--bank-id", help="reuse an existing fixture bank instead of building one")
    args = parser.parse_args()

    # The corpus must be authored, not extracted — see fixture.py.
    os.environ.setdefault("HINDSIGHT_API_LLM_PROVIDER", "mock")
    _get_raw_config()

    memory = MemoryEngine(
        db_url=os.getenv("HINDSIGHT_API_DATABASE_URL", "pg0"),
        memory_llm_provider="mock",
        memory_llm_api_key="unused",
        memory_llm_model="mock",
        task_backend=SyncTaskBackend(),
    )
    await memory.initialize()

    ctx = RequestContext()
    try:
        corpus = await build_corpus(memory, bank_id=args.bank_id)
        console.print(f"bank={corpus.bank_id} rows={len(corpus.stored_ids)} questions={len(corpus.questions)}\n")

        results: list[QuestionResult] = []
        for q in corpus.questions:
            results.append(
                QuestionResult(
                    id=q.id,
                    category=q.category,
                    question=q.question,
                    floor=await _score(memory, corpus, ctx, q, "floor", q.question),
                    ceiling=await _score(memory, corpus, ctx, q, "ceiling", q.ideal_query),
                    short_circuit_expected=q.short_circuit,
                )
            )
    finally:
        pool = await memory._get_pool()
        await pool.close()

    _display(results)
    if args.json:
        args.json.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
        console.print(f"\nwrote {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
