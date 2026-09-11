"""Reflect must answer the corpus questions — and never assert the bait.

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
"""

from __future__ import annotations

import logging

import pytest
from hindsight_client import Hindsight

from hindsight_system_evals import AnswerOutcome, ask, evaluate, facts, questions, seed_bank
from hindsight_system_evals.pages import SettleFn
from hindsight_system_evals.report import RECORDED, EvalRecord

log = logging.getLogger(__name__)

_QUESTIONS = {question.id: question for question in questions()}

#: The one that has actually regressed. Everything else runs with ``--full``.
MINIMUM_ACCEPTANCE = ("hq-absent-2024",)


async def _run(client: Hindsight, bank_id: str, settled: SettleFn, question_id: str) -> AnswerOutcome:
    question = _QUESTIONS[question_id]
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


@pytest.mark.parametrize("question_id", MINIMUM_ACCEPTANCE)
async def test_reflect_answers_minimum_acceptance(
    client: Hindsight, bank_id: str, settled: SettleFn, question_id: str
) -> None:
    _assert_sound(await _run(client, bank_id, settled, question_id))


@pytest.mark.full
@pytest.mark.parametrize("question_id", [q for q in _QUESTIONS if q not in MINIMUM_ACCEPTANCE])
async def test_reflect_answers_full(client: Hindsight, bank_id: str, settled: SettleFn, question_id: str) -> None:
    _assert_sound(await _run(client, bank_id, settled, question_id))
