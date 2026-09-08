"""Answer-level eval: run the real reflect and grade the prose it writes.

The retrieval eval (``floor_ceiling.py``) scores the evidence set. That is a
*necessary* condition — reflect is grounded, so an unretrieved fact cannot be
answered — but nowhere near a sufficient one. The gap is not hypothetical: for
the ``mm_stale`` question, retrieval can return the stale "Stripe" mental model
AND the Adyen facts, score recall@k = 1.0, and the answer can still say Stripe
because the model trusted the summary over the raw facts. The retrieval eval
calls that a pass. Only reading the answer catches it.

So this tier runs ``reflect_async`` end to end against a real LLM and grades the
text with an independent judge. Two scores per question, because they fail
differently:

* **correct** — the answer meets ``answer_criteria``. Missing it can just mean
  incomplete.
* **trap** — the answer asserts ``must_not_claim``, the specific wrong thing the
  question baits. This is a grounding failure, and it is the number that matters:
  a confidently wrong answer is worse than a hedged one.

Non-deterministic by construction, so every question runs N times and the output
is a rate, not a verdict. Do not put this in CI.

Run with::

    ./scripts/benchmarks/run-prelude-eval.sh --answers
    # or
    cd hindsight-dev && uv run python -m benchmarks.prelude.answer_eval --runs 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.task_backend import SyncTaskBackend
from hindsight_api.models import RequestContext
from rich.console import Console
from rich.table import Table

from benchmarks.prelude import judge
from benchmarks.prelude.fixture import Question, build_corpus, build_hard_corpus

console = Console()


@dataclass
class RunResult:
    """One reflect call, graded, with enough of its trace to attribute a failure."""

    answer: str
    correct: bool
    correct_reason: str
    hit_trap: bool
    trap_reason: str
    error: str = ""
    # What reflect actually did, so a wrong answer can be blamed on the right thing.
    queries: list[str] = field(default_factory=list)
    gold_retrieved: int = 0
    gold_total: int = 0

    @property
    def blame(self) -> str:
        """Why this run failed: the evidence never arrived, or it did and was misused.

        This is the question a score alone can never answer, and the two have
        opposite fixes — a retrieval miss is a query/ranking problem, a reasoning
        miss is a prompt/model problem.
        """
        if self.error:
            return "error"
        if self.correct and not self.hit_trap:
            return "ok"
        if self.gold_total and self.gold_retrieved < self.gold_total:
            return f"RETRIEVAL ({self.gold_retrieved}/{self.gold_total} gold reached the model)"
        return "REASONING (all gold was retrieved)"


@dataclass
class QuestionResult:
    id: str
    category: str
    runs: list[RunResult] = field(default_factory=list)

    @property
    def correct_rate(self) -> float:
        ok = [r for r in self.runs if not r.error]
        return sum(1 for r in ok if r.correct) / len(ok) if ok else 0.0

    @property
    def trap_rate(self) -> float:
        ok = [r for r in self.runs if not r.error]
        return sum(1 for r in ok if r.hit_trap) / len(ok) if ok else 0.0


def _trace_evidence(result: Any, gold: set[str]) -> tuple[list[str], int]:
    """The queries reflect issued, and how many gold rows its tools actually returned.

    Walks ``tool_trace`` rather than ``based_on``: based_on is what the model
    *declared it used*, which is downstream of the thing being diagnosed. The
    tool outputs are what it was handed.
    """
    queries: list[str] = []
    seen: set[str] = set()
    for call in result.tool_trace or []:
        query = (call.input or {}).get("query")
        if query:
            queries.append(f"{call.tool}({query})")
        payload = json.dumps(call.output or {}, default=str)
        seen.update(g for g in gold if g in payload)
    return queries, len(seen)


async def _one_run(
    memory: MemoryEngine, bank_id: str, ctx: RequestContext, q: Question, budget: str, gold: set[str]
) -> RunResult:
    try:
        result = await memory.reflect_async(bank_id=bank_id, query=q.question, budget=budget, request_context=ctx)
        answer = (result.text or "").strip()
        queries, gold_hits = _trace_evidence(result, gold)
    except Exception as exc:  # a failed reflect is a data point, not a crash
        return RunResult(answer="", correct=False, correct_reason="", hit_trap=False, trap_reason="", error=str(exc))

    correct = await judge.evaluate(answer, q.answer_criteria, context=f"Question asked: {q.question}")
    trap = (
        await judge.evaluate(
            answer,
            f"The answer asserts, as currently true, that {q.must_not_claim}.",
            context=f"Question asked: {q.question}",
        )
        if q.must_not_claim
        else None
    )
    return RunResult(
        answer=answer,
        correct=correct.meets_criteria,
        correct_reason=correct.reasoning,
        hit_trap=bool(trap and trap.meets_criteria),
        trap_reason=trap.reasoning if trap else "",
        queries=queries,
        gold_retrieved=gold_hits,
        gold_total=len(gold),
    )


def _display(results: list[QuestionResult], runs: int) -> None:
    table = Table(title=f"Reflect answer quality — {runs} run(s) per question")
    table.add_column("question")
    table.add_column("category")
    table.add_column("correct", justify="right")
    table.add_column("hit trap", justify="right")
    table.add_column("errors", justify="right")
    table.add_column("why it failed", justify="left")

    for r in results:
        errors = sum(1 for x in r.runs if x.error)
        trap_cell = "—" if not any(x.trap_reason for x in r.runs) else f"{r.trap_rate:.0%}"
        if r.trap_rate > 0:
            trap_cell = f"[red]{trap_cell}[/red]"
        blames = sorted({x.blame for x in r.runs if x.blame not in ("ok", "error")})
        table.add_row(
            r.id,
            r.category,
            f"{r.correct_rate:.0%}" if r.correct_rate == 1.0 else f"[yellow]{r.correct_rate:.0%}[/yellow]",
            trap_cell,
            f"[red]{errors}[/red]" if errors else "0",
            "; ".join(blames) or "—",
        )
    console.print(table)

    by_category: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)
    console.print("\n[bold]by category[/bold]")
    for category, rows in sorted(by_category.items()):
        rate = sum(x.correct_rate for x in rows) / len(rows)
        console.print(f"  {category:<20} correct={rate:.0%}")

    overall = sum(r.correct_rate for r in results) / len(results)
    trapped = [r for r in results if r.trap_rate > 0]
    console.print(f"\n[bold]overall correct[/bold] {overall:.1%}")
    if trapped:
        console.print(
            f"[red]{len(trapped)} question(s) asserted the wrong answer at least once:[/red] "
            + ", ".join(f"{r.id} ({r.trap_rate:.0%})" for r in trapped)
            + "\nThat is a grounding failure — the retrieval eval cannot see it."
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="reflect calls per question (default 3)")
    parser.add_argument(
        "--corpus",
        choices=("hard", "authored"),
        default="hard",
        help="hard: dense same-topic near-misses incl. reasoning categories (default). authored: the small corpus.",
    )
    parser.add_argument("--budget", default="low", help="reflect budget: low, medium or high")
    parser.add_argument("--json", type=Path, help="write raw per-run results (including answers) here")
    args = parser.parse_args()

    provider = os.getenv("HINDSIGHT_API_LLM_PROVIDER", "")
    if not provider or provider == "mock":
        raise SystemExit(
            "The answer eval needs a REAL reflect model: set HINDSIGHT_API_LLM_PROVIDER/_MODEL/_API_KEY. "
            "(The corpus is still seeded deterministically — the fixture scripts extraction itself.)"
        )

    if judge._MODEL == os.getenv("HINDSIGHT_API_LLM_MODEL") and judge._PROVIDER == provider:
        console.print(
            "[yellow]Warning: the judge is the same model as reflect.[/yellow] It will grade its own "
            "output and agree with itself. Set HINDSIGHT_TEST_JUDGE_MODEL to something else.\n"
        )

    memory = MemoryEngine(
        db_url=os.getenv("HINDSIGHT_API_DATABASE_URL", "pg0"),
        # Retain runs on mock so the corpus text stays authored; reflect runs on
        # the configured provider, which is what this tier measures.
        memory_llm_provider="mock",
        memory_llm_api_key="unused",
        memory_llm_model="mock",
        reflect_llm_provider=provider,
        reflect_llm_api_key=os.getenv("HINDSIGHT_API_LLM_API_KEY"),
        reflect_llm_model=os.getenv("HINDSIGHT_API_LLM_MODEL"),
        reflect_llm_base_url=os.getenv("HINDSIGHT_API_LLM_BASE_URL") or None,
        task_backend=SyncTaskBackend(),
    )
    await memory.initialize()

    ctx = RequestContext()
    try:
        corpus = await (build_hard_corpus(memory) if args.corpus == "hard" else build_corpus(memory))
        console.print(
            f"bank={corpus.bank_id} questions={len(corpus.questions)} "
            f"reflect={provider}/{os.getenv('HINDSIGHT_API_LLM_MODEL', '?')} "
            f"judge={judge._PROVIDER}/{judge._MODEL} runs={args.runs} budget={args.budget}\n"
        )

        results: list[QuestionResult] = []
        for q in corpus.questions:
            qr = QuestionResult(id=q.id, category=q.category)
            gold = corpus.gold_row_ids(q)
            for _ in range(args.runs):
                qr.runs.append(await _one_run(memory, corpus.bank_id, ctx, q, args.budget, gold))
            results.append(qr)
            console.print(f"  {q.id:<24} correct={qr.correct_rate:.0%} trap={qr.trap_rate:.0%}")
    finally:
        pool = await memory._get_pool()
        await pool.close()

    console.print()
    _display(results, args.runs)
    if args.json:
        args.json.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
        console.print(f"\nwrote {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
