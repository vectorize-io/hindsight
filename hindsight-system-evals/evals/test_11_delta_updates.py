"""A page already written must take the update the next wave brings.

The cases and why each is hard are in ``hindsight_system_evals.delta_updates``.
The short version: a delta refresh that drops an update loses it for good, since
the next refresh starts after it, and nothing but the page shows it happened.
The decision-model gate in front of the delta makes that a per-memory yes/no,
so these are the updates a yes/no on one memory finds hardest to see.

Run it with the gate on and off to compare them. Both keep the decision model, so the
only difference is whether a delta refresh asks it which memories to read:

    HINDSIGHT_EVAL_SET_DECISION_MODEL_PROVIDER=typesafe \\
    HINDSIGHT_EVAL_SET_DECISION_MODEL_API_KEY=... uv run pytest evals/test_11_delta_updates.py --full

    HINDSIGHT_EVAL_SET_MENTAL_MODEL_DELTA_USE_DECISION_MODEL=false \\
    HINDSIGHT_EVAL_SET_DECISION_MODEL_PROVIDER=typesafe \\
    HINDSIGHT_EVAL_SET_DECISION_MODEL_API_KEY=... uv run pytest evals/test_11_delta_updates.py --full
"""

from __future__ import annotations

import logging

import pytest
from hindsight_client import Hindsight

from hindsight_system_evals import evaluate
from hindsight_system_evals.delta_updates import CASES, MINIMUM_ACCEPTANCE, DeltaUpdate, build
from hindsight_system_evals.pages import SettleFn
from hindsight_system_evals.report import RECORDED, EvalRecord

log = logging.getLogger(__name__)

_CASES = {case.question.id: case for case in CASES}


async def _run(client: Hindsight, bank_id: str, settled: SettleFn, case_id: str) -> None:
    case: DeltaUpdate = _CASES[case_id]
    question = case.question
    outcome = await build(client, bank_id, case, settled)
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


@pytest.mark.parametrize("case_id", MINIMUM_ACCEPTANCE)
async def test_delta_takes_the_update_minimum(client: Hindsight, bank_id: str, settled: SettleFn, case_id: str) -> None:
    """The update only the page connects to the topic."""
    await _run(client, bank_id, settled, case_id)


@pytest.mark.full
@pytest.mark.parametrize("case_id", [c for c in _CASES if c not in MINIMUM_ACCEPTANCE])
async def test_delta_takes_the_update_full(client: Hindsight, bank_id: str, settled: SettleFn, case_id: str) -> None:
    await _run(client, bank_id, settled, case_id)
