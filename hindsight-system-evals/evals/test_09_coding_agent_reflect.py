"""A coding agent's first-prompt reflect, on a real sde-bench bank, in both reflect modes.

The coding-agents plugin reflects once on a session's first prompt and injects the answer.
It does not send the developer's prompt as written: it wraps it in ~2k characters of
rendering rules (``buildReflectQuery``, copied below). And the bank it reads is nothing like
this package's corpus: the task's repo history, the one chat where the decision was made,
and 140 decoy developer conversations, plus pages the plugin wrote about the code.

Fast reflect passed every other suite here and still failed this, measured on sde-bench
``boltons-budget-001`` (2026-10-06): it injected "The bank holds no decision on the
Retrier" in both tasks while agent mode injected the decision, and it took 14s against 7-10s.
Three causes, none visible on short questions over an on-topic corpus:

* the whole wrapped prompt was the search query, which found nothing and took 4-7s per arm;
* the plugin's pages were fresh but about the code, and freshness alone hid the facts;
* the decision model's "is this enough" read an expected score of 1.43 as "partly" though
  "fully" was its likeliest answer, buying two more LLM turns.

So this suite runs both modes over several frozen banks of that shape, interleaved, and
grades them as rates: one bank and three repeats measured a single task's luck (fast mode
lost one task twice to a consolidated observation the agent happened to search around, and
agent mode lost different ones). Fast mode must get as many decisions right as agent mode,
less one, answer "no decision" no more often than agent mode, and be faster with fewer LLM calls. The banks and each
task's bug report and recorded policy are in ``fixtures/coding-agent-banks/``.
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

# The decision-model steps fast reflect records in its trace; none of them is an LLM call.
_DECISION_SCOPES = ("fast_prune", "fast_sufficiency", "fast_pages_sufficiency")

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

# A single plain claim, phrased like the other suites' traps: a criterion that also says what
# does NOT count gets read backwards by the judge.
NO_DECISION = "The answer asserts that the bank holds no decision or record bearing on the reported problem."


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
    await client.aupdate_bank_config(target, enable_auto_consolidation=False, enable_observations=False)
    # An import rewrites every memory after the pages last read them, so the pages arrive
    # stale. A plugin keeps its pages fresh, and fresh pages are the case that failed: fast mode
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
    await client.banks.update_bank_config(bank, BankConfigUpdate(updates={"reflect_mode": mode}))
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
        f"{task['policy']}. It may word it differently, but it must state that decision and must not "
        "contradict it.",
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


async def test_coding_agent_reflect_fast_matches_agent_and_is_faster(
    client: Hindsight, bank_id: str, settled: SettleFn
) -> None:
    await client.aupdate_bank_config(bank_id, enable_auto_consolidation=False)
    runs: list[ReflectRun] = []
    for task_id in MANIFEST["tasks"]:
        bank = await _import_bank(client, bank_id, task_id, settled)
        # Interleaved, so a provider that slows down part-way through hits both modes alike.
        for _ in range(REPEATS):
            for mode in ("agent", "fast"):
                runs.append(await _reflect(client, bank, task_id, mode))

    agent = [r for r in runs if r.mode == "agent"]
    fast = [r for r in runs if r.mode == "fast"]
    log.info(
        "correct: agent %d/%d, fast %d/%d",
        sum(r.correct for r in agent),
        len(agent),
        sum(r.correct for r in fast),
        len(fast),
    )
    # The bug this suite exists for: fast mode answering "no decision" where agent mode found one.
    # Agent mode can say it too on the hardest bank (dedupe-history, where a later commit undid
    # the decision), so fast is held to agent's count, not to zero.
    no_decision = {mode: [r.task for r in runs if r.mode == mode and r.said_no_decision] for mode in ("agent", "fast")}
    assert len(no_decision["fast"]) <= len(no_decision["agent"]), f"answered 'no decision': {no_decision}"
    assert sum(r.correct for r in fast) >= sum(r.correct for r in agent) - 1, (
        f"fast reported the decision in {sum(r.correct for r in fast)}/{len(fast)} runs, agent in "
        f"{sum(r.correct for r in agent)}/{len(agent)}; missed: {[r.task for r in fast if not r.correct]}"
    )

    if not any(r.decision_calls for r in fast):
        pytest.skip(
            "fast mode ran without a decision model, so only its answers were graded; set "
            "HINDSIGHT_EVAL_SET_RERANKER_TYPESAFE_API_KEY to compare its speed"
        )
    fast_s, agent_s = statistics.median(r.seconds for r in fast), statistics.median(r.seconds for r in agent)
    assert fast_s < agent_s, f"fast reflect took {fast_s:.1f}s (median), agent {agent_s:.1f}s"
    assert statistics.median(r.llm_calls for r in fast) < statistics.median(r.llm_calls for r in agent)
