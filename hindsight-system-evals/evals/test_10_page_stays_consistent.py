"""A refresh that changes a value must change it everywhere the page states it.

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
"""

from __future__ import annotations

import logging
import os

import pytest
from hindsight_client import Hindsight

from hindsight_system_evals import build_page, evaluate
from hindsight_system_evals.corpus import HardFact, HardQuestion
from hindsight_system_evals.pages import SettleFn
from hindsight_system_evals.report import RECORDED, EvalRecord

log = logging.getLogger(__name__)

REPEATS = int(os.getenv("HINDSIGHT_EVAL_CONSISTENCY_REPEATS", "1"))

_EN_QUESTION = HardQuestion(
    id="pump-maintenance-en",
    category="stale_restatement",
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
    category="stale_restatement",
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


async def _run(
    client: Hindsight, bank_id: str, settled: SettleFn, question: HardQuestion, waves: list[list[HardFact]]
) -> None:
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
async def test_refresh_updates_every_restatement(
    client: Hindsight, bank_id: str, settled: SettleFn, repeat: int
) -> None:
    """The English case: the old value is literally in the stale block."""
    await _run(client, bank_id, settled, _EN_QUESTION, _EN_WAVES)


@pytest.mark.full
@pytest.mark.parametrize("repeat", range(REPEATS))
async def test_refresh_updates_absence_claims(client: Hindsight, bank_id: str, settled: SettleFn, repeat: int) -> None:
    """The reporter's Chinese case: the stale block denies the change happened."""
    await _run(client, bank_id, settled, _ZH_QUESTION, _ZH_WAVES)
