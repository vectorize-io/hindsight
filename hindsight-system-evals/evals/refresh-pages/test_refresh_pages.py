"""Refresh pages: a knowledge page must stay right as the bank changes under it.

Each case has a ``category`` and its test id is ``<category>/<case id>``, so
``-k "update/"`` runs one group and ``eval-table`` compares models per group:

* precision — a page built across waves keeps the attribute the question asks
  about apart from its near-misses (``hq-release-prod``, ``hq-entity-port``,
  ``hq-numeric-outage``);
* update — a later wave changes what the page says, and the page follows it
  (``hq-owner-latest``, ``du-oncall-departed``, ``du-renamed-service``,
  ``du-backup-correction``);
* aggregation — a list or count grows or loses a member (``hq-count-billing``,
  ``du-enterprise-plan``, ``du-launch-risk``);
* no_overreach — the bank does not say it, or says it only for one scope
  (``hq-absent-2024``, ``hq-scoped-2fa``);
* stability — a page that should keep what it has does: a wave with nothing for it,
  and no collapse across waves (``du-quiet-wave``, ``page-never-collapses``);
* consistency — a changed value is changed everywhere the page restates it
  (``pump-maintenance-en``, ``pump-maintenance-zh``).

The benchmarks at the bottom (``test_refresh_pages_benchmark_cost*``) are not cases:
they record what a refresh spends, on a frozen bank.

Convergence — A knowledge page must converge on the right answer as data arrives.

This is the eval that motivated the package. A page is created with a source
query and then accumulates: each ingest triggers a delta refresh that edits what
is already stored. Two failures found that way, both of which a one-shot reflect
eval scores as passes because reflect sees the whole bank at once:

* a page that said "Release 0.9.3 was deployed to production on 18 March 2026"
  came back one wave later saying "No release was deployed to production on 18
  March 2026" — the fact was in the earlier wave and outside the delta window;
* a page counting 3 customers, then handed 4 more, reported 4 and dropped the
  first three.

Both traced to the same cause: the delta step treated the reflect synthesis as
authoritative, when that synthesis is written from the new batch alone. Its
totals count only the batch and its absences describe only the batch.

Two assertions per case, and the second matters more. ``correct`` can fail
because an answer is incomplete. ``hit_trap`` means the page asserts the specific
wrong thing — a stored falsehood every later reflect will read back as true.

Delta updates — A page already written must take the update the next wave brings.

The cases and why each is hard are in ``hindsight_system_evals.delta_updates``.
The short version: a delta refresh that drops an update loses it for good, since
the next refresh starts after it, and nothing but the page shows it happened.
The decision-model gate in front of the delta makes that a per-memory yes/no,
so these are the updates a yes/no on one memory finds hardest to see.

Run it with the gate on and off to compare them. Both keep the decision model, so the
only difference is whether a delta refresh asks it which memories to read:

    HINDSIGHT_EVAL_SET_DECISION_MODEL_PROVIDER=typesafe \\
    HINDSIGHT_EVAL_SET_DECISION_MODEL_API_KEY=... uv run pytest evals/refresh-pages/test_11_delta_updates.py --full

    HINDSIGHT_EVAL_SET_MENTAL_MODEL_DELTA_USE_DECISION_MODEL=false \\
    HINDSIGHT_EVAL_SET_DECISION_MODEL_PROVIDER=typesafe \\
    HINDSIGHT_EVAL_SET_DECISION_MODEL_API_KEY=... uv run pytest evals/refresh-pages/test_11_delta_updates.py --full

Consistency — A refresh that changes a value must change it everywhere the page states it.

#5228. A delta refresh edits a page block by block, and every block it does not
name stays exactly as it was. When a new fact changed a value, the model rewrote
the paragraph the fact was about and left the other blocks that still said the
old thing — so the page contradicted itself, and the refresh reported success:

* an information-gaps line still said "only the 1,000-operating-hour replacement
  interval" next to a main section that now said 800 hours;
* a source note still said "no evidence of a maintenance-owner change was found"
  right after the owner was changed.

Nothing failed to apply: the model never emitted an operation for those blocks.

The fixture is built so the first page restates the values. The page is asked to
open with a summary and end with sources and gaps, and wave one carries an audit
note ("V1 is the only document, no owner change recorded"). A first version
without them produced short pages with nothing to go stale, and passed 16/16 on
gemini-2.5-flash-lite while the bug was live. With them the same model kept a
stale line in 10 of 16 runs, nearly always the opening summary. The outcome
mostly follows the wave-one page, so ``HINDSIGHT_EVAL_CONSISTENCY_REPEATS`` runs
each case on fresh pages to get a rate.

The trap is the stale restatement; ``correct`` also requires that the gaps the
new notice did NOT fill are still recorded, because "rewrite every block that
mentions the pump" would pass the trap and erase them.

Refresh cost — How many tokens a knowledge-page refresh spends, and on what.

A baseline, not a gate. Every other suite here grades what a refresh wrote; this
one records what writing it cost — each LLM call's input, cached and output
tokens, and the full prompt it sent — so a change to how reflect budgets its
tools can be judged against numbers instead of a feeling.

Behind it: refreshes on real project banks sent 30-90k-token synthesis prompts,
because a tool's ``max_tokens`` counts fact text only and the JSON around each
fact is three to four times larger (#4566). That one fact explains a 500 on the
30s call deadline (#4568), a context guard ending the search before ``recall``
ran (#4563), a small-window model answering "I don't have information" (#4561),
and a background loop billing a metered key all night (#4532).

The bank is a frozen archive (see ``refresh_cost``), so two runs refresh the same
facts, observations and pages. The model is not frozen, so numbers still move
between runs: compare medians over repeats, not single runs.

Two refreshes per page:

* ``full`` — the page rebuilt from the whole bank: the reflect loop at its widest.
* ``delta`` — a small wave of new facts, then the page's own default refresh:
  what a page pays every time it keeps itself current.

Run ``--cost-output DIR`` to keep the report and every prompt for comparison.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import uuid
from pathlib import Path

import pytest
from hindsight_client import Hindsight

from hindsight_system_evals import (
    Target,
    build_page,
    evaluate,
    provider_environment,
    questions,
    split_into_waves,
    wait_until_settled,
)
from hindsight_system_evals.corpus import HardFact, HardQuestion
from hindsight_system_evals.delta_updates import CASES as DELTA_CASES
from hindsight_system_evals.delta_updates import MINIMUM_ACCEPTANCE as DELTA_MINIMUM
from hindsight_system_evals.delta_updates import DeltaUpdate
from hindsight_system_evals.delta_updates import build as build_delta
from hindsight_system_evals.orchestration import CASES as FOLLOW_UP_CASES
from hindsight_system_evals.orchestration import FollowUpCase, server_with_scripted_decision_model
from hindsight_system_evals.pages import PageOutcome, SettleFn, facts
from hindsight_system_evals.refresh_cost import (
    FIXTURE,
    PAGES,
    PRICES,
    CostReport,
    RefreshCost,
    llm_calls,
    measure_refresh,
    new_wave,
    page_trigger,
    summary_table,
)
from hindsight_system_evals.report import RECORDED, EvalRecord, case_params

log = logging.getLogger(__name__)

# ── Convergence: a page built across waves ───────────────────────────────────
#: The two questions that have actually regressed. Everything else runs with ``--full``.
CONVERGENCE_MINIMUM = ("hq-release-prod", "hq-count-billing")


async def _run(
    client: Hindsight, bank_id: str, settled: SettleFn, question: HardQuestion, *, record: bool = True
) -> PageOutcome:
    question_id = question.id
    outcome = await build_page(client, bank_id, question, split_into_waves(facts()), settled)

    assert outcome.final_content.strip(), (
        f"{question_id}: the page is empty after {len(outcome.waves)} waves — sizes {[w.chars for w in outcome.waves]}"
    )

    context = f"This is a knowledge page built to answer: {question.question}"
    verdict = await evaluate(outcome.final_content, question.answer_criteria, context=context)
    outcome.correct, outcome.correct_reason = verdict.meets_criteria, verdict.reasoning

    if question.must_not_claim:
        trap = await evaluate(
            outcome.final_content,
            f"The page asserts, as currently true, that {question.must_not_claim}.",
            context=context,
        )
        outcome.hit_trap, outcome.trap_reason = trap.meets_criteria, trap.reasoning

    # A check that re-runs a question for a different property (the collapse
    # check below) must not count as a second graded page: it would inflate the
    # total and weight that one question twice in the published correct rate.
    if record:
        RECORDED.append(
            EvalRecord(
                kind="knowledge_page",
                question_id=question_id,
                category=question.category,
                correct=outcome.correct,
                hit_trap=outcome.hit_trap,
                sizes=[w.chars for w in outcome.waves],
                bank_id=outcome.bank_id,
                page_id=outcome.page_id,
                reason=outcome.trap_reason if outcome.hit_trap else outcome.correct_reason,
            )
        )
    log.info(
        "%s: correct=%s trap=%s sizes=%s",
        question_id,
        outcome.correct,
        outcome.hit_trap,
        [w.chars for w in outcome.waves],
    )
    return outcome


def _assert_sound(outcome: PageOutcome) -> None:
    # The trap first: a page that asserts the wrong thing is worse than one that
    # is merely incomplete, and reporting them in that order makes a failure
    # readable without opening the log.
    assert not outcome.hit_trap, (
        f"{outcome.question_id}: the page asserts the wrong answer — {outcome.trap_reason}\n"
        f"bank {outcome.bank_id}, page {outcome.page_id}\n"
        f"page reads: {outcome.final_content[:400]}"
    )
    assert outcome.correct, (
        f"{outcome.question_id}: {outcome.correct_reason}\n"
        f"bank {outcome.bank_id}, page {outcome.page_id}\n"
        f"page sizes by wave: {[w.chars for w in outcome.waves]}\n"
        f"page reads: {outcome.final_content[:400]}"
    )


@pytest.mark.parametrize("question", case_params(questions(), CONVERGENCE_MINIMUM))
async def test_refresh_pages(client: Hindsight, bank_id: str, settled: SettleFn, question: HardQuestion) -> None:
    _assert_sound(await _run(client, bank_id, settled, question))


# One case, id'd like the others so ``-k "stability/"`` selects it.
@pytest.mark.parametrize("case", [pytest.param("page-never-collapses", id="stability/page-never-collapses")])
@pytest.mark.full
async def test_refresh_pages_never_collapse(client: Hindsight, bank_id: str, settled: SettleFn, case: str) -> None:
    """A wave that shrinks a page to almost nothing is a destructive edit.

    Separate from correctness because a page can shrink legitimately — a
    superseded claim should go — but a collapse to a fraction of its former size
    is the signature of a replace that should have been a merge, and a final-text
    score cannot see it. Not recorded by ``_run``: the same question is graded on
    its answer by ``test_refresh_pages``, and counting it twice would weight it twice.
    """
    question = next(q for q in questions() if q.id == "hq-count-billing")
    outcome = await _run(client, bank_id, settled, question, record=False)
    sizes = [w.chars for w in outcome.waves]
    held = sizes[-1] >= sizes[0] * 0.5
    RECORDED.append(
        EvalRecord(
            kind="knowledge_page",
            question_id=case,
            category="stability",
            correct=held,
            hit_trap=False,
            sizes=sizes,
            bank_id=outcome.bank_id,
            page_id=outcome.page_id,
            reason=f"sizes by wave: {sizes}",
        )
    )
    assert held, (
        f"the page collapsed across waves ({sizes}) — bank {outcome.bank_id}\npage reads: {outcome.final_content[:400]}"
    )


# ── Delta updates: an established page, then a wave that changes it ──────────
async def _run_delta(client: Hindsight, bank_id: str, settled: SettleFn, case: DeltaUpdate) -> None:
    case_id = case.id
    question = case.question
    try:
        outcome = await build_delta(client, bank_id, case, settled)
    finally:
        # Also when a refresh failed and raised: that is the run whose replies matter.
        await _dump_llm_calls(client, bank_id, case_id)
    context = f"This is a knowledge page built to answer: {question.question}"
    where = f"bank {outcome.bank_id}, page {outcome.page_id}"

    # Without the old value on the wave-1 page the update has nothing to replace,
    # and a pass would mean nothing. That is the full rebuild's failure, not the
    # delta's, so it is reported as such and not recorded as a grade.
    base = await evaluate(outcome.waves[0].content, case.base_criteria, context=context)
    assert base.meets_criteria, (
        f"{case_id}: the wave-1 page never said what the update changes — {base.reasoning}\n"
        f"{where}\npage reads: {outcome.waves[0].content[:400]}"
    )

    verdict = await evaluate(outcome.final_content, question.answer_criteria, context=context)
    outcome.correct, outcome.correct_reason = verdict.meets_criteria, verdict.reasoning
    if question.must_not_claim:
        trap = await evaluate(
            outcome.final_content,
            f"The page asserts, as currently true, that {question.must_not_claim}.",
            context=context,
        )
        outcome.hit_trap, outcome.trap_reason = trap.meets_criteria, trap.reasoning
    if case.listed and outcome.correct:
        listing = _listing_problem(case, outcome.waves[0].content, outcome.final_content)
        if listing:
            outcome.correct, outcome.correct_reason = False, listing

    RECORDED.append(
        EvalRecord(
            kind="delta_update",
            question_id=case_id,
            category=question.category,
            correct=outcome.correct,
            hit_trap=outcome.hit_trap,
            sizes=[w.chars for w in outcome.waves],
            bank_id=outcome.bank_id,
            page_id=outcome.page_id,
            reason=outcome.trap_reason if outcome.hit_trap else outcome.correct_reason,
        )
    )
    log.info(
        "%s: correct=%s trap=%s sizes=%s", case_id, outcome.correct, outcome.hit_trap, [w.chars for w in outcome.waves]
    )

    assert not outcome.hit_trap, (
        f"{case_id}: the page still asserts what the update replaced — {outcome.trap_reason}\n"
        f"{where}\npage reads: {outcome.final_content[:400]}"
    )
    assert outcome.correct, f"{case_id}: {outcome.correct_reason}\n{where}\npage reads: {outcome.final_content[:400]}"


def _count(item: str, text: str) -> int:
    """Whole-token occurrences, so ``3.1.0`` is not found inside ``3.11.0``."""
    return len(re.findall(rf"(?<![\w.]){re.escape(item)}(?![\w]|\.\d)", text))


def _listing_problem(case: DeltaUpdate, before: str, after: str) -> str:
    """What the delta did wrong to the page's listing of ``case.listed``, or "".

    Only the delta's own work is graded: an item the wave-1 page never carried is
    the full rebuild's omission, unless the update is what brings it. So: nothing
    the wave-1 page listed may vanish, every item the update brings must land, and
    at most one item may appear twice (a "latest" line may restate the newest).
    """
    base = "\n".join(case.base)
    lost = [i for i in case.listed if _count(i, before) and not _count(i, after)]
    not_added = [i for i in case.listed if not _count(i, base) and not _count(i, after)]
    twice = [i for i in case.listed if _count(i, after) > 1]
    if lost:
        return f"the delta dropped {len(lost)} item(s) the page already listed: {lost[:8]}"
    if not_added:
        return f"the delta never added {len(not_added)} new item(s): {not_added[:8]}"
    if len(twice) > 1:
        return f"the page lists {len(twice)} of {len(case.listed)} items more than once: {twice[:8]}"
    return ""


async def _dump_llm_calls(client: Hindsight, bank_id: str, case_id: str) -> None:
    """With ``HINDSIGHT_EVAL_DELTA_DUMP=DIR``, write the bank's LLM calls in the cost-report
    shape, so a failing reply can be replayed against the model prompt-for-prompt."""
    out = os.getenv("HINDSIGHT_EVAL_DELTA_DUMP")
    if not out:
        return
    calls = await llm_calls(client, bank_id, "2000-01-01T00:00:00Z", "2100-01-01T00:00:00Z")
    path = Path(out) / f"{case_id}-{bank_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"refreshes": [{"page": case_id, "mode": "delta", "calls": [c.model_dump() for c in calls]}]})
    )


#: How many times each delta case runs. A failure that shows one reply in ten needs
#: more than one run to be seen; ``-k`` a case and raise this to measure its rate.
DELTA_REPEATS = int(os.getenv("HINDSIGHT_EVAL_DELTA_REPEATS", "1"))


@pytest.mark.parametrize("repeat", range(DELTA_REPEATS))
@pytest.mark.parametrize("case", case_params(DELTA_CASES, DELTA_MINIMUM))
async def test_refresh_pages_delta(
    client: Hindsight, bank_id: str, settled: SettleFn, case: DeltaUpdate, repeat: int
) -> None:
    await _run_delta(client, bank_id, settled, case)


# ── Consistency: a changed value changes everywhere ──────────────────────────
REPEATS = int(os.getenv("HINDSIGHT_EVAL_CONSISTENCY_REPEATS", "1"))

_EN_QUESTION = HardQuestion(
    id="pump-maintenance-en",
    category="consistency",
    question=(
        "Describe the Blue Heron pump's current maintenance requirements. Open with a short "
        "summary, then cover the filter replacement interval, who owns maintenance, the "
        "operating checks and dated changes, and end with a Sources section and an "
        "Information gaps section."
    ),
    ideal_query="Blue Heron pump maintenance filter interval owner",
    answer_criteria=(
        "The page gives 800 operating hours as the current filter replacement interval and "
        "Brian Lee as the current maintenance owner, AND it still records that the filter "
        "type / part number / cleaning guidance and the electrical isolation instructions are "
        "not documented."
    ),
    must_not_claim=(
        "the filter replacement interval is 1,000 operating hours (including a gap or summary "
        "line saying only the 1,000-hour interval is recorded), OR that Alice Morgan owns "
        "maintenance, OR that manual V1 is the only source, OR that no change of owner or "
        "manual update has been recorded. A statement clearly framed as dated history (for "
        "example 'V1, effective 2026-10-01, specified 1,000 hours') does NOT count"
    ),
)

_EN_WAVES = [
    [
        HardFact(
            "en-v1-interval",
            "Blue Heron pump maintenance manual V1, effective 2026-10-01: replace the filter "
            "every 1,000 operating hours.",
            "pump",
        ),
        HardFact(
            "en-v1-owner",
            "According to manual V1, Alice Morgan owns maintenance of the Blue Heron pump.",
            "pump",
        ),
        HardFact(
            "en-v1-check",
            "Manual V1 requires Blue Heron pump operators to check the pressure gauge before each shift.",
            "pump",
        ),
        HardFact(
            "en-v1-gaps",
            "For the Blue Heron pump filter, manual V1 gives only the 1,000-operating-hour "
            "replacement interval: no filter type, part number or cleaning guidance. It has no "
            "electrical isolation instructions either.",
            "pump",
        ),
        HardFact(
            "en-v1-audit",
            "Maintenance audit, 2026-10-02: manual V1 is the only maintenance document issued for "
            "the Blue Heron pump, and no change of maintenance owner has been recorded.",
            "pump",
        ),
    ],
    [
        HardFact(
            "en-v2-interval",
            "Blue Heron pump maintenance notice V2, effective 2026-10-05, supplements manual V1: "
            "the filter replacement interval changes from 1,000 to 800 operating hours.",
            "pump",
        ),
        HardFact(
            "en-v2-owner",
            "From 2026-10-05, under notice V2, Brian Lee replaces Alice Morgan as owner of Blue "
            "Heron pump maintenance.",
            "pump",
        ),
        HardFact(
            "en-v2-unchanged",
            "Notice V2 leaves the Blue Heron pump's pressure-gauge check unchanged and adds no "
            "filter part number and no electrical isolation procedure.",
            "pump",
        ),
    ],
]

# The reporter's case, kept in Chinese: the stale line there holds no old value at
# all ("no evidence of an owner change was found"), so a rule about finding other
# copies of the value that changed cannot reach it.
_ZH_QUESTION = HardQuestion(
    id="pump-maintenance-zh",
    category="consistency",
    question="蓝鹭泵目前的维护要求是什么？请先给出简短摘要，再说明滤芯更换周期、维护负责人、运行检查和有日期的变更，最后列出“信息来源”和“信息缺口”两个部分。",
    ideal_query="蓝鹭泵 维护 滤芯 负责人",
    answer_criteria=(
        "The page (in Chinese) gives 800 operating hours as the current filter replacement "
        "interval and 李工 (Li) as the current maintenance owner, AND it still records that "
        "the filter model / part number and the electrical isolation steps are not documented."
    ),
    must_not_claim=(
        "the filter interval is 1,000 hours, OR that 张工 (Zhang) owns maintenance, OR that the "
        "V1 manual is the only source, OR that no evidence of a manual update or maintenance-owner "
        "change was found. A statement clearly framed as dated history does NOT count"
    ),
)

_ZH_WAVES = [
    [
        HardFact(
            "zh-v1-interval", "《蓝鹭泵维护手册 V1》于 2026 年 10 月 1 日生效：滤芯每运行 1000 小时更换一次。", "pump"
        ),
        HardFact("zh-v1-owner", "根据《蓝鹭泵维护手册 V1》，蓝鹭泵的维护负责人为张工。", "pump"),
        HardFact("zh-v1-check", "《蓝鹭泵维护手册 V1》要求操作员每班开机前检查压力表。", "pump"),
        HardFact(
            "zh-v1-gaps",
            "关于蓝鹭泵滤芯，《蓝鹭泵维护手册 V1》只给出每运行 1000 小时更换的周期，未记录滤芯型号和零件编号，也未给出断电隔离步骤。",
            "pump",
        ),
        HardFact(
            "zh-v1-audit",
            "2026 年 10 月 2 日维护审查：《蓝鹭泵维护手册 V1》是蓝鹭泵唯一的维护文件，也未发现手册版本更新或维护负责人变更的记录。",
            "pump",
        ),
    ],
    [
        HardFact(
            "zh-v2-interval",
            "《维护变更通知 V2》于 2026 年 10 月 5 日生效，补充手册 V1：蓝鹭泵滤芯更换周期由每运行 1000 小时缩短为 800 小时。",
            "pump",
        ),
        HardFact(
            "zh-v2-owner",
            "自 2026 年 10 月 5 日起，根据《维护变更通知 V2》，蓝鹭泵维护负责人由张工变更为李工。",
            "pump",
        ),
        HardFact(
            "zh-v2-unchanged",
            "《维护变更通知 V2》确认压力表检查要求不变，且未新增滤芯零件编号或断电隔离步骤。",
            "pump",
        ),
    ],
]


#: The waves of each consistency case, by question id.
_CONSISTENCY_WAVES = {_EN_QUESTION.id: _EN_WAVES, _ZH_QUESTION.id: _ZH_WAVES}


async def _run_consistency(client: Hindsight, bank_id: str, settled: SettleFn, question: HardQuestion) -> None:
    waves = _CONSISTENCY_WAVES[question.id]
    case_id = question.id
    outcome = await build_page(client, bank_id, question, waves, settled)
    assert outcome.final_content.strip(), f"{case_id}: the page is empty — bank {bank_id}"

    context = f"This is a knowledge page built to answer: {question.question}"
    trap = await evaluate(
        outcome.final_content,
        f"The page asserts anywhere — a summary, a source note or an information-gap line "
        f"included — as currently true, that {question.must_not_claim}.",
        context=context,
    )
    verdict = await evaluate(outcome.final_content, question.answer_criteria, context=context)
    RECORDED.append(
        EvalRecord(
            kind="knowledge_page",
            question_id=case_id,
            category=question.category,
            correct=verdict.meets_criteria,
            hit_trap=trap.meets_criteria,
            sizes=[w.chars for w in outcome.waves],
            bank_id=bank_id,
            page_id=outcome.page_id,
            reason=trap.reasoning if trap.meets_criteria else verdict.reasoning,
        )
    )
    log.info("%s: correct=%s trap=%s", case_id, verdict.meets_criteria, trap.meets_criteria)

    # The first build is printed too: a stale line can only survive if the build
    # wrote one, so a pass on a page that never restated the value proves little.
    pages = "\n\n".join(f"--- after wave {w.wave} ---\n{w.content}" for w in outcome.waves)
    assert not trap.meets_criteria, f"{case_id}: the page kept a stale claim — {trap.reasoning}\n{pages}"
    assert verdict.meets_criteria, f"{case_id}: {verdict.reasoning}\n{pages}"


@pytest.mark.parametrize("repeat", range(REPEATS))
@pytest.mark.parametrize("question", case_params((_EN_QUESTION, _ZH_QUESTION), {_EN_QUESTION.id}))
async def test_refresh_pages_consistency(
    client: Hindsight, bank_id: str, settled: SettleFn, question: HardQuestion, repeat: int
) -> None:
    """EN: the old value is literally in the stale block. ZH, the reporter's case: the stale
    block denies the change happened."""
    await _run_consistency(client, bank_id, settled, question)


# ── Benchmark: what a refresh spends ─────────────────────────────────────────
#: One page per run in minimum acceptance: enough to see a regression in the
#: token count, cheap enough to run daily. ``--full`` measures all of them.
MINIMUM_PAGES = ("Known bugs and fixes",)


async def _import_fixture(client: Hindsight, host_bank: str, settled: SettleFn) -> str:
    """Restore the frozen bank into a fresh one and wait for the re-embedding."""
    if not FIXTURE.exists():
        pytest.fail(f"missing {FIXTURE} — build it with `uv run python -m hindsight_system_evals.refresh_cost build`")
    # The import operation is recorded against an existing bank; the target must
    # not exist yet.
    await client.aupdate_bank_config(host_bank, enable_auto_consolidation=False)
    target = f"{host_bank}-cost"
    await client.aimport_bank(host_bank, FIXTURE.read_bytes(), target_bank_id=target)
    await settled(host_bank)
    await settled(target)
    # A new wave is retained as written; extraction would add calls outside any
    # refresh, and consolidation would too.
    await client.aupdate_bank_config(
        target, retain_extraction_mode="chunks", enable_auto_consolidation=False, enable_observations=False
    )
    return target


async def _page_ids(client: Hindsight, bank: str) -> dict[str, str]:
    models = await client.alist_mental_models(bank_id=bank, detail="metadata")
    return {m.name: m.id for m in models.items}


async def _measure(client: Hindsight, bank: str, settled: SettleFn, pages: tuple[str, ...]) -> list[RefreshCost]:
    ids = await _page_ids(client, bank)
    missing = [p for p in pages if p not in ids]
    assert not missing, f"the fixture has no page named {missing}; it has {sorted(ids)}"

    results: list[RefreshCost] = []
    for page in pages:
        await client.aupdate_mental_model(bank_id=bank, mental_model_id=ids[page], trigger=page_trigger("full"))
        await settled(bank)
        results.append(await measure_refresh(client, bank, ids[page], page=page, mode="full"))

    # The delta pass after every full one, so each delta edits a page built from
    # the same bank, and the wave is the only thing new to all of them.
    await client.aretain_batch(bank_id=bank, items=new_wave())
    await settled(bank)
    for page in pages:
        await client.aupdate_mental_model(bank_id=bank, mental_model_id=ids[page], trigger=page_trigger("delta"))
        await settled(bank)
        results.append(await measure_refresh(client, bank, ids[page], page=page, mode="delta"))
    return results


def _report(request: pytest.FixtureRequest, results: list[RefreshCost]) -> None:
    model = provider_environment()["HINDSIGHT_API_LLM_MODEL"] if not request.config.getoption("--api-url") else "remote"
    price = PRICES.get(model)
    # Reflect builds no explicit per-step caches any more, so cached tokens are implicit hits.
    explicit = False
    table = summary_table(results, price, explicit)
    log.info("refresh cost baseline:\n%s", table)
    out = request.config.getoption("--cost-output")
    if not out:
        return
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    report = CostReport(
        timestamp=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        model=provider_environment()["HINDSIGHT_API_LLM_MODEL"]
        if not request.config.getoption("--api-url")
        else "remote",
        fixture=FIXTURE.name,
        explicit_cache=explicit,
        refreshes=results,
    )
    (directory / "refresh-cost.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (directory / "refresh-cost.txt").write_text(table + "\n", encoding="utf-8")
    # The synthesis prompt of every refresh, readable on its own: the thing to
    # open when a number moves.
    for r in results:
        if r.final_call is not None:
            name = f"{r.page.lower().replace(' ', '-')}-{r.mode}-final-prompt.json"
            (directory / name).write_text(
                json.dumps(r.final_call.prompt, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
            )
    log.info("refresh cost report written to %s", directory)


def _assert_measured(results: list[RefreshCost]) -> None:
    # The only hard claims: every refresh reached the model, and the trace saw
    # it. A refresh with no traced calls would record a baseline of zero.
    for r in results:
        # No synthesis call is not a failure: a refresh that answers through the
        # ``done`` tool writes its page from the tool loop alone.
        assert r.calls, f"{r.page} ({r.mode}): no LLM calls traced — is HINDSIGHT_API_LLM_TRACE_ENABLED off?"


async def test_refresh_pages_benchmark_cost(
    request: pytest.FixtureRequest, client: Hindsight, settled: SettleFn
) -> None:
    bank = await _import_fixture(client, f"syseval-{uuid.uuid4().hex[:10]}", settled)
    results = await _measure(client, bank, settled, MINIMUM_PAGES)
    _report(request, results)
    _assert_measured(results)


@pytest.mark.full
async def test_refresh_pages_benchmark_cost_all_pages(
    request: pytest.FixtureRequest, client: Hindsight, settled: SettleFn
) -> None:
    bank = await _import_fixture(client, f"syseval-{uuid.uuid4().hex[:10]}", settled)
    results = await _measure(client, bank, settled, tuple(PAGES))
    _report(request, results)
    _assert_measured(results)


# Full runs only: each case starts a server of its own.
@pytest.mark.parametrize("case", case_params(FOLLOW_UP_CASES, minimum=()))
async def test_refresh_pages_follows_decision_model(target: Target, tmp_path: Path, case: FollowUpCase) -> None:
    """A page refresh runs the follow-up search the decision model asks for.

    Same scripted decision model as the reflect case (``hindsight_system_evals/orchestration.py``):
    it always answers "the evidence answers part of the question", so the refresh must search a
    second time before it writes the page.
    """
    if target.is_remote:
        pytest.skip("starts its own server, with a scripted decision model")
    with server_with_scripted_decision_model(tmp_path / "server.log") as orchestrated:
        client = Hindsight(base_url=orchestrated.server.url)
        try:
            bank = f"syseval-{case.id}"
            await client.aretain_batch(bank_id=bank, items=[{"content": text} for text in case.facts])
            await wait_until_settled(client, bank)
            created = await client.knowledge_base.create_knowledge_page(
                bank,
                {
                    "name": f"eval {case.id}",
                    "source_query": case.question,
                    "trigger": {"mode": "full", "exclude_mental_models": True, "refresh_after_consolidation": False},
                },
            )
            await wait_until_settled(client, bank)
            orchestrated.decision_model.reset()
            await client.arefresh_mental_model(bank_id=bank, mental_model_id=created.mental_model_id)
            await wait_until_settled(client, bank)
            verdicts = len(orchestrated.decision_model.verdicts)
        finally:
            await client.aclose()
    followed = verdicts >= 2
    reason = "" if followed else f"the decision model asked for a follow-up search; the refresh ran {verdicts - 1}"
    RECORDED.append(
        EvalRecord(
            kind="knowledge_page",
            question_id=case.id,
            category=case.category,
            correct=followed,
            hit_trap=False,
            bank_id=bank,
            reason=reason,
            page_id=created.page_id,
        )
    )
    assert followed, f"{case.id}: {reason}"
