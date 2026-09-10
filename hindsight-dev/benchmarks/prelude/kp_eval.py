"""Knowledge-page eval: does a page ACCUMULATE the right answer, one wave at a time?

``answer_eval.py`` asks reflect a question once, over a bank that is already
complete. That is not how Hindsight is used. A knowledge page is created with a
source query and then *converges* — data arrives over time and each delta refresh
edits the stored document.

That difference is the whole point of testing it separately:

* A wrong reflect answer is read once and gone. A wrong page is **written down**,
  and every later reflect reads it back as fact. Fabrication compounds instead of
  evaporating.
* A page must also do things a one-shot answer never has to: keep what is still
  true, supersede what is not, and not drift while nothing relevant is arriving.

The flow per question, mirroring how a page is really used:

    create page (source_query = the question)
      -> ingest wave 1  -> refresh -> page content
      -> ingest wave 2  -> refresh -> page content
      -> judge the FINAL content against the same criteria answer_eval uses

The corpus, criteria and traps are shared with ``answer_eval`` deliberately: same
questions, same grading, different production path, so any difference in score is
attributable to the path and not to the test.

Pages are pointed at raw facts rather than the default observation-only trigger.
Observations come from consolidation, which is LLM-written, and a corpus whose
text is generated rather than authored has no stable gold to label against — the
same reason ``fixture.py`` seeds authored rows instead of running consolidation.

Run with::

    ./scripts/benchmarks/run-prelude-eval.sh --pages --runs 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.task_backend import SyncTaskBackend
from hindsight_api.models import RequestContext
from rich.console import Console
from rich.table import Table

from benchmarks.prelude import hard_corpus, judge
from benchmarks.prelude.fixture import _require_mock, _retain_many

console = Console()

#: Pages read raw facts here, not observations — see the module docstring.
_PAGE_TRIGGER = {
    "mode": "delta",
    "fact_types": ["world", "experience"],
    "exclude_mental_models": True,
    "refresh_after_consolidation": False,
}

#: How many waves the corpus is split into. More than one is the point: a single
#: wave is just a slow full build, and never exercises a delta edit at all.
_WAVES = 2


@dataclass
class WaveSnapshot:
    """The page as it stood after one wave — the record of how it converged."""

    wave: int
    facts_ingested: int
    content: str = ""
    chars: int = 0
    error: str = ""


@dataclass
class PageRunResult:
    """One page, built across every wave, then graded on its final content."""

    question_id: str
    category: str
    waves: list[WaveSnapshot] = field(default_factory=list)
    final_content: str = ""
    correct: bool = False
    correct_reason: str = ""
    hit_trap: bool = False
    trap_reason: str = ""
    error: str = ""

    @property
    def grew_monotonically(self) -> bool:
        """Whether the page only ever gained content.

        Not a pass/fail on its own — a delta SHOULD shrink a page when a claim is
        superseded. It is reported because an unexplained collapse to near-empty
        is the signature of a destructive edit, which is invisible in a score that
        only grades the final text.
        """
        sizes = [w.chars for w in self.waves if not w.error]
        return all(b >= a for a, b in zip(sizes, sizes[1:]))


def _split_into_waves(facts: list[hard_corpus.HardFact], count: int) -> list[list[tuple[str, str]]]:
    """Split the corpus into ingest waves WITHOUT separating a subject's facts.

    A round-robin split put every "deployed to production" fact in wave 1 and
    every staging/canary fact for the same releases in wave 2. The second wave
    then read as a correction of the first -- later statements about the same
    subject, saying it went to staging and canary rather than production -- and
    the refresh superseded a correct claim by the book. The bug was the split.

    Facts sharing a ``subject`` therefore always travel together; subjects are
    dealt round-robin so the waves stay comparable in size.
    """
    by_subject: dict[str, list[tuple[str, str]]] = {}
    for index, fact in enumerate(facts):
        # A fact with no subject is its own subject: it cannot contradict anything.
        key = f"{fact.cluster}:{fact.subject}" if fact.subject else f"_solo:{index}"
        by_subject.setdefault(key, []).append((fact.id, fact.text))

    waves: list[list[tuple[str, str]]] = [[] for _ in range(count)]
    for position, key in enumerate(sorted(by_subject)):
        waves[position % count].extend(by_subject[key])
    return waves


async def _build_page(
    memory: MemoryEngine,
    bank_id: str,
    ctx: RequestContext,
    question: hard_corpus.HardQuestion,
    waves: list[list[tuple[str, str]]],
) -> PageRunResult:
    """Create one page for one question, then feed it the corpus wave by wave."""
    result = PageRunResult(question_id=question.id, category=question.category)

    page = await memory.create_knowledge_page(
        bank_id=bank_id,
        name=f"KP {question.id}",
        source_query=question.question,
        content="",
        trigger=_PAGE_TRIGGER,
        request_context=ctx,
    )
    if page is None:
        result.error = "create_knowledge_page returned None (name collision)"
        return result
    mental_model_id = page["mental_model_id"]

    for index, wave in enumerate(waves, start=1):
        snapshot = WaveSnapshot(wave=index, facts_ingested=len(wave))
        try:
            await _retain_many(memory, bank_id, ctx, wave, "world")
            # Refresh explicitly rather than waiting on the auto trigger: the
            # benchmark runs in-process with a SyncTaskBackend and no worker
            # polling a queue, so "wait for it to become up to date" would be
            # waiting for something that is never going to run.
            await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mental_model_id, request_context=ctx)
            model = await memory.get_mental_model(bank_id=bank_id, mental_model_id=mental_model_id, request_context=ctx)
            snapshot.content = (model or {}).get("content", "") or ""
            snapshot.chars = len(snapshot.content)
        except Exception as exc:  # a failed wave is a data point, not a crash
            snapshot.error = str(exc)
        result.waves.append(snapshot)

    result.final_content = result.waves[-1].content if result.waves else ""
    return result


async def _grade(result: PageRunResult, question: hard_corpus.HardQuestion) -> None:
    """Grade the accumulated page with the criteria answer_eval uses for reflect."""
    if not result.final_content.strip():
        result.error = result.error or "page ended empty"
        return
    verdict = await judge.evaluate(
        result.final_content,
        question.answer_criteria,
        context=f"This is a knowledge page built to answer: {question.question}",
    )
    result.correct = verdict.meets_criteria
    result.correct_reason = verdict.reasoning
    if question.must_not_claim:
        trap = await judge.evaluate(
            result.final_content,
            f"The page asserts, as currently true, that {question.must_not_claim}.",
            context=f"This is a knowledge page built to answer: {question.question}",
        )
        result.hit_trap = trap.meets_criteria
        result.trap_reason = trap.reasoning


def _display(results: list[PageRunResult]) -> None:
    table = Table(title=f"Knowledge-page convergence — {_WAVES} ingest waves per page")
    table.add_column("question")
    table.add_column("category")
    table.add_column("correct", justify="right")
    table.add_column("hit trap", justify="right")
    table.add_column("page size by wave", justify="left")
    table.add_column("note", justify="left")

    for r in results:
        sizes = " → ".join(str(w.chars) for w in r.waves) or "—"
        notes = []
        if r.error:
            notes.append(f"[red]{r.error[:40]}[/red]")
        if not r.grew_monotonically:
            notes.append("[yellow]shrank[/yellow]")
        table.add_row(
            r.question_id,
            r.category,
            "[green]yes[/green]" if r.correct else "[red]no[/red]",
            "[red]yes[/red]" if r.hit_trap else "no",
            sizes,
            "; ".join(notes) or "—",
        )
    console.print(table)

    scored = [r for r in results if not r.error]
    if scored:
        correct = sum(1 for r in scored if r.correct) / len(scored)
        trapped = [r for r in scored if r.hit_trap]
        console.print(f"\n[bold]pages correct[/bold] {correct:.1%} ({len(scored)} graded)")
        if trapped:
            console.print(
                f"[red]{len(trapped)} page(s) assert the wrong answer:[/red] "
                + ", ".join(r.question_id for r in trapped)
                + "\nA wrong page is worse than a wrong answer: it is stored, and every later "
                "reflect reads it back as fact."
            )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write raw per-page results, including every wave")
    parser.add_argument("--only", help="run a single question id")
    args = parser.parse_args()

    provider = os.getenv("HINDSIGHT_API_LLM_PROVIDER", "")
    if not provider or provider == "mock":
        raise SystemExit(
            "The page eval needs a REAL model for refresh: set HINDSIGHT_API_LLM_PROVIDER/_MODEL/_API_KEY."
        )

    memory = MemoryEngine(
        db_url=os.getenv("HINDSIGHT_API_DATABASE_URL", "pg0"),
        # Retain stays on mock so the corpus text is authored, not extracted;
        # reflect (which is what a refresh runs) uses the configured model.
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
    corpus = hard_corpus.build()
    facts, questions = corpus.facts, corpus.questions
    if args.only:
        questions = [q for q in questions if q.id == args.only]
        if not questions:
            raise SystemExit(f"No question with id {args.only!r}")

    waves = _split_into_waves(facts, _WAVES)

    console.print(
        f"pages={len(questions)} facts={sum(len(w) for w in waves)} waves={_WAVES} "
        f"refresh={provider}/{os.getenv('HINDSIGHT_API_LLM_MODEL', '?')} judge={judge._PROVIDER}/{judge._MODEL}\n"
    )

    results: list[PageRunResult] = []
    try:
        for question in questions:
            # One bank per page. Sharing a bank would let every page's refresh
            # reflect over every other question's corpus, which is not the
            # scenario and would make a failure impossible to attribute.
            bank_id = f"prelude-kp-{uuid.uuid4().hex[:8]}"
            await memory.get_bank_profile(bank_id=bank_id, request_context=ctx)
            await memory._config_resolver.update_bank_config(bank_id, {"enable_auto_consolidation": False}, ctx)
            _require_mock(memory)
            result = await _build_page(memory, bank_id, ctx, question, waves)
            await _grade(result, question)
            results.append(result)
            console.print(
                f"  {question.id:<22} correct={result.correct} trap={result.hit_trap} "
                f"sizes={[w.chars for w in result.waves]}"
            )
    finally:
        pool = await memory._get_pool()
        await pool.close()

    console.print()
    _display(results)
    if args.json:
        args.json.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
        console.print(f"\nwrote {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
