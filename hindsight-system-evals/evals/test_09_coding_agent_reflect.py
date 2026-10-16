"""A coding agent's first-prompt reflect, on a real sde-bench bank, with and without the decision model.

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

from hindsight_system_evals import wait_until_settled
from hindsight_system_evals.judge import evaluate
from hindsight_system_evals.pages import SettleFn

log = logging.getLogger(__name__)

BANKS = Path(__file__).resolve().parents[1] / "fixtures" / "coding-agent-banks"
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


async def test_coding_agent_reflect_with_decision_model_matches_agent_and_costs_less(
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
