"""Reflect: the answer it gives from what the bank holds.

Each case has a ``category`` and its test id is ``<category>/<case id>``, so
``-k "precision/"`` runs one group and ``eval-table`` compares models per group:

* precision — the answer differs from its near-misses by one attribute: a version,
  an entity one character off, a number (``hq-release-prod``, ``hq-entity-port``,
  ``hq-numeric-outage``);
* update — what is true now, when the bank also holds what was true before
  (``hq-owner-latest``, ``sp-rotation``);
* aggregation — the answer needs several memories together: a count, a rule and its
  exception (``hq-count-billing``, ``sp-friday-deploy``);
* no_overreach — the bank does not say it, or says it only for one scope
  (``hq-absent-2024``, ``hq-scoped-2fa``);
* source_authority — the handbook and the chatter disagree, or only one answers
  (``sp-*``);
* pages — the bank holds a page about the question; the answer comes from it.

The benchmark at the bottom (``test_reflect_benchmark_coding_agent_decision_model``) is
not a case: it compares two reflect modes over frozen coding-agent banks and grades rates.

Corpus questions — Reflect must answer the corpus questions — and never assert the bait.

The whole corpus is in the bank at once, so this is the easy case for retrieval
and the hard one for grounding: every category's near-misses are present
together. The incident behind the minimum-acceptance question: asked for the
2024 engineering headcount in a bank that only covers 2025-2026, reflect
extrapolated backwards from the later growth and reported a specific number as
"reliably deduced". Nothing in the bank said it.

Same two assertions as the knowledge-page suite, in the same order: the trap
(the answer asserts the specific wrong thing) before correctness (the answer is
incomplete). A failure also says whether the gold evidence reached the model,
which decides where the fix goes.

Source authority — Reflect must rank its sources, not just retrieve them.

One bank holds the engineering handbook and the conversations about it. Recall
ranks both by similarity to the question, which cannot tell an approved policy
from someone guessing in a thread — so "how many approvals does a PR need"
retrieves the handbook sentence and "I think one is enough" equally well, and the
answer is whichever the model happened to read first.

Provenance is recorded in each document's metadata, which is the harder case:
metadata is stripped from reflect's tool results, so the ranking has to be
resolved server-side and only its verdict sent. A corpus marking provenance with
tags would pass even with that resolution broken.

Six questions, chosen so that no blunt rule passes all of them: two where the
handbook is right and the chatter is confidently wrong, one only the handbook
answers, one only the chatter answers, one where a dated decision in a meeting
explicitly supersedes the handbook, and one where the handbook's rule and an
approved exception to it are both needed. Filtering the conversations away fails
the fourth; trusting whatever ranks highest fails the first two.

Same two assertions and the same order as the other answer suites: the trap
before correctness. ``source_order`` is recorded alongside — which kind of source
reached the model first — because a wrong answer that never saw the handbook is a
retrieval problem and one that saw it and wrote the chatter's version is not.

Pages — A reflect that has a page about the question must answer from the page.

The other suites cannot see this: their pages are created with
``exclude_mental_models``, so ``has_mental_models`` is false and reflect is never
offered the page tools at all. This one builds the page first and then asks the
question with the page layer live, which is the only shape where the top of the
retrieval ladder is exercised.

It exists because that layer changed: ``search_mental_models`` used to hand back
five pages whole — 8.7-19k tokens, the largest single item in a reflect's floor
(#4533) — and now returns whole pages up to a token budget and a snippet of the
rest, with ``read_mental_models`` pulling the full text of the ones the model chooses. That trade is only safe if the model
still *reads* the page it needs, so the assertions are: the answer is right, it
does not assert the bait, and the evidence names the page.

Coding-agent benchmark — A coding agent's first-prompt reflect, on a real sde-bench bank, with and without the decision model.

The coding-agents plugin reflects once on a session's first prompt and injects the answer.
It does not send the developer's prompt as written: it wraps it in ~2k characters of
rendering rules (``buildReflectQuery``, copied below). And the bank it reads is nothing like
this package's corpus: the task's repo history, the one chat where the decision was made,
and 140 decoy developer conversations, plus pages the plugin wrote about the code.

Reflect with the decision model passed every other suite here and still failed this, measured on
sde-bench ``boltons-budget-001`` (2026-10-06): it injected "The bank holds no decision on the
Retrier" in both tasks while agent mode injected the decision, and it took 14s against 7-10s.
Three causes, none visible on short questions over an on-topic corpus:

* the whole wrapped prompt was the search query, which found nothing and took 4-7s per arm;
* the plugin's pages were fresh but about the code, and freshness alone hid the facts;
* the decision model's "is this enough" read an expected score of 1.43 as "partly" though
  "fully" was its likeliest answer, buying two more LLM turns.

So this suite runs both modes over several frozen banks of that shape, interleaved, and
grades them as rates: one bank and three repeats measured a single task's luck (the decision model
lost one task twice to a consolidated observation the agent happened to search around, and
agent mode lost different ones). With the decision model, reflect must get as many decisions right
as agent mode, less one, answer "no decision" no more often, and make fewer LLM calls (the cost) in
less time. Every bank carries a reflect mission ranking its sources (``MISSION``), the way a
team would tell reflect which records hold its decisions, so both modes are also held to
following it. The banks and each task's bug report and recorded policy are in
``fixtures/coding-agent-banks/``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from hindsight_client import Hindsight
from hindsight_client_api.models.bank_config_update import BankConfigUpdate

from hindsight_system_evals import (
    AnswerOutcome,
    Target,
    ask,
    build_page,
    evaluate,
    facts,
    questions,
    seed_bank,
    sources,
    split_into_waves,
    wait_until_settled,
)
from hindsight_system_evals.corpus import HardQuestion
from hindsight_system_evals.orchestration import CASES as FOLLOW_UP_CASES
from hindsight_system_evals.orchestration import FollowUpCase, server_with_scripted_decision_model
from hindsight_system_evals.pages import SettleFn
from hindsight_system_evals.pages import facts as page_facts
from hindsight_system_evals.report import RECORDED, EvalRecord, case_params

log = logging.getLogger(__name__)


# ── Corpus questions: precision, update, aggregation, no_overreach ───────────
async def _run(client: Hindsight, bank_id: str, settled: SettleFn, question: HardQuestion) -> AnswerOutcome:
    question_id = question.id
    corpus = facts()
    await seed_bank(client, bank_id, corpus, settled)
    outcome = await ask(client, bank_id, question, corpus)
    assert outcome.answer, f"{question_id}: reflect returned no text — bank {bank_id}"

    context = f"Question asked: {question.question}"
    verdict = await evaluate(outcome.answer, question.answer_criteria, context=context)
    outcome.correct, outcome.correct_reason = verdict.meets_criteria, verdict.reasoning
    if question.must_not_claim:
        trap = await evaluate(
            outcome.answer, f"The answer asserts, as currently true, that {question.must_not_claim}.", context=context
        )
        outcome.hit_trap, outcome.trap_reason = trap.meets_criteria, trap.reasoning

    RECORDED.append(
        EvalRecord(
            kind="reflect",
            question_id=question_id,
            category=question.category,
            correct=outcome.correct,
            hit_trap=outcome.hit_trap,
            bank_id=bank_id,
            reason=outcome.trap_reason if outcome.hit_trap else outcome.correct_reason,
        )
    )
    log.info("%s: correct=%s trap=%s blame=%s", question_id, outcome.correct, outcome.hit_trap, outcome.blame)
    return outcome


def _assert_sound(outcome: AnswerOutcome) -> None:
    evidence = f"blame: {outcome.blame}\nqueries: {outcome.queries}\nbank {outcome.bank_id}"
    assert not outcome.hit_trap, (
        f"{outcome.question_id}: the answer asserts the wrong thing — {outcome.trap_reason}\n"
        f"{evidence}\nanswer: {outcome.answer[:400]}"
    )
    assert outcome.correct, (
        f"{outcome.question_id}: {outcome.correct_reason}\n{evidence}\nanswer: {outcome.answer[:400]}"
    )


@pytest.mark.parametrize("question", case_params(questions(), {"hq-absent-2024"}))
async def test_reflect(client: Hindsight, bank_id: str, settled: SettleFn, question: HardQuestion) -> None:
    _assert_sound(await _run(client, bank_id, settled, question))


# ── Source authority: handbook vs chatter ────────────────────────────────────
async def _run_sources(client: Hindsight, bank_id: str, settled: SettleFn, question: HardQuestion) -> AnswerOutcome:
    question_id = question.id
    # The ``query`` strategy carries the ranking on the question itself; every
    # other strategy leaves it exactly as written.
    question = dataclasses.replace(question, question=sources.decorate_query(question.question))
    corpus = sources.facts()
    await sources.seed_bank(client, bank_id, corpus, settled)
    outcome = await ask(client, bank_id, question, corpus)
    assert outcome.answer, f"{question_id}: reflect returned no text — bank {bank_id}"

    context = f"Question asked: {question.question}"
    verdict = await evaluate(outcome.answer, question.answer_criteria, context=context)
    outcome.correct, outcome.correct_reason = verdict.meets_criteria, verdict.reasoning
    trap = await evaluate(
        outcome.answer, f"The answer asserts, as currently true, that {question.must_not_claim}.", context=context
    )
    outcome.hit_trap, outcome.trap_reason = trap.meets_criteria, trap.reasoning

    RECORDED.append(
        EvalRecord(
            kind="source_priority",
            question_id=question_id,
            category=question.category,
            correct=outcome.correct,
            hit_trap=outcome.hit_trap,
            bank_id=bank_id,
            reason=outcome.trap_reason if outcome.hit_trap else outcome.correct_reason,
        )
    )
    log.info(
        "%s: correct=%s trap=%s guide_first=%s sources=%s",
        question_id,
        outcome.correct,
        outcome.hit_trap,
        outcome.source_order[:1] == ["guide"],
        outcome.source_order,
    )
    return outcome


def _assert_sources_sound(outcome: AnswerOutcome) -> None:
    evidence = (
        f"blame: {outcome.blame}\nqueries: {outcome.queries}\n"
        f"sources in the order they reached the model: {outcome.source_order}\nbank {outcome.bank_id}"
    )
    assert not outcome.hit_trap, (
        f"{outcome.question_id}: the answer asserts the wrong thing — {outcome.trap_reason}\n"
        f"{evidence}\nanswer: {outcome.answer[:400]}"
    )
    assert outcome.correct, (
        f"{outcome.question_id}: {outcome.correct_reason}\n{evidence}\nanswer: {outcome.answer[:400]}"
    )


@pytest.mark.parametrize("question", case_params(sources.questions(), {"sp-datastore", "sp-approvals"}))
async def test_reflect_sources(client: Hindsight, bank_id: str, settled: SettleFn, question: HardQuestion) -> None:
    _assert_sources_sound(await _run_sources(client, bank_id, settled, question))


# ── Pages: reflect reads the page it was given ───────────────────────────────
#: Two questions whose answers a page states outright, so a reflect that reads the
#: page can answer and one that stops at the snippet cannot. Their own category here.
PAGE_BACKED = tuple(
    dataclasses.replace(q, category="pages") for q in questions() if q.id in ("hq-release-prod", "hq-count-billing")
)


async def _page_then_reflect(client: Hindsight, bank_id: str, settled: SettleFn, question: HardQuestion) -> None:
    """Build the page, ask its own question, and check the answer came from it.

    Deliberately NOT graded against the corpus gold: whether the page itself got
    the answer right is what test_01 grades, and a page that converged on the
    wrong number would fail this for a reason that has nothing to do with the
    retrieval layer. What matters here is that reflect used the page — it cited
    it, and what it said matches what the page says.
    """
    question_id = question.id
    outcome = await build_page(client, bank_id, question, split_into_waves(page_facts()), settled)
    assert outcome.final_content.strip(), f"{question_id}: the page is empty, so the reflect below proves nothing"

    # The page layer is live here: no exclusion, so reflect is offered
    # search_mental_models / read_mental_models over the page just built.
    response = await client.areflect(bank_id=bank_id, query=question.question, budget="low", include_tool_calls=True)
    answer = (response.text or "").strip()
    assert answer, f"{question_id}: reflect returned no text — bank {bank_id}"

    # From the tool trace, not ``based_on``: the citation lists are only populated
    # when the caller asks for them, while the trace says what actually ran — and
    # it reads the same on a server that returns pages whole and one that returns
    # snippets plus a read.
    tools_used = [call.tool for call in ((response.trace.tool_calls if response.trace else None) or [])]
    page_tools = [t for t in tools_used if "mental_model" in t]

    verdict = await evaluate(
        answer,
        "The answer states what the page states on this question — the same facts, figures and dates, "
        "with nothing asserted that the page does not say.",
        context=f"Question asked: {question.question}\n\nThe page the bank holds says:\n{outcome.final_content}",
    )

    RECORDED.append(
        EvalRecord(
            kind="reflect",
            question_id=f"{question_id}-from-page",
            category=question.category,
            correct=verdict.meets_criteria,
            # Not a trap: a trap is a stored falsehood the answer repeated, counted
            # on its own in the report. "Never went to the page layer" is a
            # different failure, and it lands in ``reason`` and in the assert below.
            hit_trap=False,
            bank_id=bank_id,
            page_id=outcome.page_id,
            reason=f"page tools used: {page_tools or 'none'} — {verdict.reasoning}",
        )
    )
    log.info("%s: grounded=%s page_tools=%s answer=%s", question_id, verdict.meets_criteria, page_tools, answer[:200])

    assert page_tools, (
        f"{question_id}: reflect answered without ever going to the page layer, though the bank holds a page "
        f"written for this exact question. Tools used: {tools_used}\nanswer: {answer[:400]}"
    )
    assert verdict.meets_criteria, f"{question_id}: {verdict.reasoning}\nanswer: {answer[:400]}"


@pytest.mark.parametrize("question", case_params(PAGE_BACKED, {q.id for q in PAGE_BACKED}))
async def test_reflect_from_page(client: Hindsight, bank_id: str, settled: SettleFn, question: HardQuestion) -> None:
    await _page_then_reflect(client, bank_id, settled, question)


# ── Benchmark: coding-agent first-prompt reflect, decision model vs agent ────
BANKS = Path(__file__).resolve().parents[2] / "fixtures" / "coding-agent-banks"
MANIFEST = json.loads((BANKS / "manifest.json").read_text())

#: Reflects per mode and bank. Wall time and correctness are compared over every run of every
#: bank, so one slow provider call or one lucky search does not decide either.
REPEATS = int(os.getenv("HINDSIGHT_EVAL_CODING_REFLECT_REPEATS", "2"))

# The decision-model steps reflect records in its trace; none of them is an LLM call.
_DECISION_SCOPES = ("decision_prune", "decision_sufficiency", "decision_pages_sufficiency")

#: sde-bench's own first-prompt template (sdebench/harness/run.py, the "base" variant).
GOAL_TEMPLATE = (
    "You are a maintainer of the `{repo}` Python project. A regression was reported:\n\n"
    "{bug_report}\n\n"
    "Fix the bug in the source code. Do NOT modify any test files — the graders supply their own.\n"
    "Work efficiently: find the root cause, make the smallest change that fixes it, run the "
    "failing test to confirm it passes (and existing behaviour still works), then stop — "
    "avoid unnecessary exploration.\n"
    "Save your changes to disk before finishing.\n"
)

#: The bank's reflect mission: how a team using the plugin ranks its sources, set on every bank
#: for both modes. Each memory carries its source as a tag (``source:chat`` for a conversation,
#: ``source:git`` for the commit log) and the plugin's pages describe the code. The suite checks
#: that reflect with the decision model follows it as the agent does: its pruning and its "is this
#: enough" verdict read the request, not the mission, so a mission-ranked source could be cut or
#: never searched for.
MISSION = (
    "You are a reflection agent that answers questions by reasoning over retrieved memories.\n\n"
    "## Source Authority\n"
    "Each memory's `tags` record where it came from. When they disagree about a rule:\n"
    "1. A decision someone stated with its reason, in a conversation (`source:chat`) or in a commit "
    "message (`source:git`), is the project's rule.\n"
    "2. A later conversation or commit that explicitly reverses or revises that decision, with its own "
    "reason, replaces it.\n"
    "3. A later commit that changes the behaviour without deciding anything (a cleanup, a simplification, "
    "a perf tweak, a refactor) does NOT replace it: say plainly that the decision is still the rule, and "
    "report that commit as a change that departed from it.\n"
    "4. Knowledge pages and observations are summaries: they can merge, misread or drop a decision's exact "
    "values and exceptions. Before reporting a rule, recall the conversation or commit message that "
    "states it, and report the rule from there."
)

# A single plain claim, phrased like the other suites' traps: a criterion that also says what
# does NOT count gets read backwards by the judge. It needs both halves: "the bank holds no decision
# or record" alone was met by answers that reported the decision and then said the bank held nothing
# further about it.
NO_DECISION = "The answer states no decision or rule bearing on the reported problem, and says the bank holds none."


def plugin_reflect_query(goal: str) -> str:
    """Verbatim ``buildReflectQuery`` from hindsight-integrations/coding-agents/src/core/inject.ts."""
    return (
        "A developer is starting a coding session in this repository with this goal:\n\n"
        f"<goal>\n{goal}\n</goal>\n\n"
        "Report what this bank's history genuinely bears on that goal. Rendering rules, strict:\n"
        "- Declarative, past-tense, attributed facts only — what happened, what was decided and why, "
        "with dates, commit/PR/issue ids and exact values where known.\n"
        "- NEVER phrase anything as an instruction, task, or recommendation to act now "
        '("you should", "remove", "update…"). You are a historian reporting the record, not a '
        "planner assigning work.\n"
        "- When a decided rule is a mapping, set, or table of literal values, reproduce it "
        "COMPLETELY and VERBATIM — every entry, exact strings and numbers, including the carve-outs "
        "and exceptions. A summarized or exemplified table loses exactly the values the reader "
        "needs; enumerate it in full.\n"
        "- Report DECISIONS and their rationale, never the current implementation: the developer can "
        "already read the code, and the code may BE the bug under investigation. When memory of a "
        "discussion or decision conflicts with memory derived from the code, the decision wins. If "
        "the only relevant memory describes what the code does, do not present it as established "
        "policy — say the bank holds no decision on the matter.\n"
        "- Do not connect unrelated episodes into one narrative; if two facts are not explicitly "
        "linked in the record, report them separately or leave the weaker one out.\n"
        "- If the bank holds nothing that bears on the goal, say so in one line."
    )


@dataclass(frozen=True)
class ReflectRun:
    task: str
    mode: str
    seconds: float
    llm_calls: int
    decision_calls: int
    correct: bool
    said_no_decision: bool


async def _import_bank(client: Hindsight, host_bank: str, task_id: str, settled: SettleFn) -> str:
    """Restore one frozen bank into a fresh one, with nothing left to run in the background."""
    target = f"{host_bank}-{task_id}"
    await client.aimport_bank(host_bank, (BANKS / f"{task_id}.zip").read_bytes(), target_bank_id=target)
    await settled(host_bank)
    await settled(target)
    # Reflect only reads: a consolidation pass or page refresh firing mid-run would bill
    # calls to whichever mode happened to be measured at the time.
    await client.aupdate_bank_config(
        target, enable_auto_consolidation=False, enable_observations=False, reflect_mission=MISSION
    )
    # An import rewrites every memory after the pages last read them, so the pages arrive
    # stale. A plugin keeps its pages fresh, and fresh pages are the case that failed: the decision model
    # hid the facts behind them. Refreshed once here, so the bank is the one the agent meets.
    pages = await client.alist_mental_models(bank_id=target, detail="metadata")
    for page in pages.items:
        await client.arefresh_mental_model(bank_id=target, mental_model_id=page.id)
    # A refresh can fail once and be retried by the worker (an empty answer from the model);
    # the staleness check below is the real gate, so a failed-then-retried attempt is let through.
    await wait_until_settled(client, target, allow_failed=True)
    refreshed = await client.alist_mental_models(bank_id=target, detail="metadata")
    stale = [page.name for page in refreshed.items if page.is_stale]
    assert not stale, f"{task_id}: pages still stale after their refresh: {stale}"
    return target


async def _reflect(client: Hindsight, bank: str, task_id: str, mode: str) -> ReflectRun:
    task = MANIFEST["tasks"][task_id]
    # The generated API takes any config field; the wrapper names only the common ones.
    await client.banks.update_bank_config(
        bank, BankConfigUpdate(updates={"reflect_use_decision_model": mode == "decision"})
    )
    goal = GOAL_TEMPLATE.format(repo=task["repo"], bug_report=task["bug_report"])
    start = time.monotonic()
    response = await client.areflect(
        bank_id=bank, query=plugin_reflect_query(goal), budget="low", include_tool_calls=True
    )
    seconds = time.monotonic() - start
    scopes = [call.scope for call in (response.trace.llm_calls if response.trace else None) or []]
    decision = sum(1 for scope in scopes if scope in _DECISION_SCOPES)
    answer = response.text or ""
    context = f"Bug report the developer is working on:\n{task['bug_report']}"
    correct = await evaluate(
        answer,
        "The answer reports the project's recorded decision on this matter, consistent with: "
        f"{task['policy']}, as the rule that stands. It may word it differently, and it may also report a "
        "later change that departed from that decision, as long as it does not present that change as the "
        "project's rule.",
        context=context,
    )
    no_decision = await evaluate(answer, NO_DECISION, context=context)
    log.info(
        "%s %s: %.1fs llm=%d decision=%d correct=%s",
        task_id,
        mode,
        seconds,
        len(scopes) - decision,
        decision,
        correct.meets_criteria,
    )
    return ReflectRun(
        task_id, mode, seconds, len(scopes) - decision, decision, correct.meets_criteria, no_decision.meets_criteria
    )


async def test_reflect_benchmark_coding_agent_decision_model(
    client: Hindsight, bank_id: str, settled: SettleFn
) -> None:
    await client.aupdate_bank_config(bank_id, enable_auto_consolidation=False)
    runs: list[ReflectRun] = []
    for task_id in MANIFEST["tasks"]:
        bank = await _import_bank(client, bank_id, task_id, settled)
        # Interleaved, so a provider that slows down part-way through hits both modes alike.
        for _ in range(REPEATS):
            for mode in ("agent", "decision"):
                runs.append(await _reflect(client, bank, task_id, mode))

    agent = [r for r in runs if r.mode == "agent"]
    decision = [r for r in runs if r.mode == "decision"]
    log.info(
        "correct: agent %d/%d, decision model %d/%d",
        sum(r.correct for r in agent),
        len(agent),
        sum(r.correct for r in decision),
        len(decision),
    )
    # The bug this suite exists for: the decision model answering "no decision" where agent mode found one.
    # Agent mode can say it too on the hardest bank (dedupe-history, where a later commit undid
    # the decision), so it is held to agent's count, not to zero.
    no_decision = {
        mode: [r.task for r in runs if r.mode == mode and r.said_no_decision] for mode in ("agent", "decision")
    }
    assert len(no_decision["decision"]) <= len(no_decision["agent"]), f"answered 'no decision': {no_decision}"
    assert sum(r.correct for r in decision) >= sum(r.correct for r in agent) - 1, (
        f"with the decision model, reflect reported the decision in {sum(r.correct for r in decision)}/{len(decision)} runs, agent in "
        f"{sum(r.correct for r in agent)}/{len(agent)}; missed: {[r.task for r in decision if not r.correct]}"
    )

    if not any(r.decision_calls for r in decision):
        pytest.skip(
            "no decision model is set up, so both arms ran agent turns and only the answers were graded; set "
            "HINDSIGHT_EVAL_SET_DECISION_MODEL_PROVIDER=typesafe and HINDSIGHT_EVAL_SET_DECISION_MODEL_API_KEY "
            "to compare cost and time"
        )
    assert statistics.median(r.llm_calls for r in decision) < statistics.median(r.llm_calls for r in agent)
    decision_s, agent_s = statistics.median(r.seconds for r in decision), statistics.median(r.seconds for r in agent)
    assert decision_s < agent_s, (
        f"with the decision model reflect took {decision_s:.1f}s (median), agent {agent_s:.1f}s"
    )


# Full runs only: each case starts a server of its own.
@pytest.mark.parametrize("case", case_params(FOLLOW_UP_CASES, minimum=()))
async def test_reflect_follows_decision_model(target: Target, tmp_path: Path, case: FollowUpCase) -> None:
    """Reflect runs the follow-up search the decision model asks for.

    The decision model here always answers "the evidence answers part of the question", so a
    decision-orchestrated reflect must write one more search after the first round. A model
    that answers that forced turn with ``done`` instead skips it, and the stub hears one
    verdict instead of two.
    """
    if target.is_remote:
        pytest.skip("starts its own server, with a scripted decision model")
    with server_with_scripted_decision_model(tmp_path / "server.log") as orchestrated:
        client = Hindsight(base_url=orchestrated.server.url)
        try:
            bank = f"syseval-{case.id}"
            await client.aretain_batch(bank_id=bank, items=[{"content": text} for text in case.facts])
            await wait_until_settled(client, bank)
            orchestrated.decision_model.reset()
            await client.areflect(bank_id=bank, query=case.question, budget="low")
            verdicts = len(orchestrated.decision_model.verdicts)
        finally:
            await client.aclose()
    followed = verdicts >= 2
    reason = "" if followed else f"the decision model asked for a follow-up search; reflect ran {verdicts - 1}"
    RECORDED.append(
        EvalRecord(
            kind="reflect",
            question_id=case.id,
            category=case.category,
            correct=followed,
            hit_trap=False,
            bank_id=bank,
            reason=reason,
        )
    )
    assert followed, f"{case.id}: {reason}"
