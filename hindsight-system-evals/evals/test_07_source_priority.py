"""Reflect must rank its sources, not just retrieve them.

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
"""

from __future__ import annotations

import dataclasses
import logging

import pytest
from hindsight_client import Hindsight

from hindsight_system_evals import AnswerOutcome, ask, evaluate
from hindsight_system_evals.pages import SettleFn
from hindsight_system_evals.report import RECORDED, EvalRecord
from hindsight_system_evals.sources import decorate_query, facts, questions, seed_bank

log = logging.getLogger(__name__)

_QUESTIONS = {question.id: question for question in questions()}

#: The two where the chatter actively contradicts the handbook — the failure the
#: suite exists for. Everything else runs with ``--full``.
MINIMUM_ACCEPTANCE = ("sp-datastore", "sp-approvals")


async def _run(client: Hindsight, bank_id: str, settled: SettleFn, question_id: str) -> AnswerOutcome:
    question = _QUESTIONS[question_id]
    # The ``query`` strategy carries the ranking on the question itself; every
    # other strategy leaves it exactly as written.
    question = dataclasses.replace(question, question=decorate_query(question.question))
    corpus = facts()
    await seed_bank(client, bank_id, corpus, settled)
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


def _assert_sound(outcome: AnswerOutcome) -> None:
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


@pytest.mark.parametrize("question_id", MINIMUM_ACCEPTANCE)
async def test_source_priority_minimum_acceptance(
    client: Hindsight, bank_id: str, settled: SettleFn, question_id: str
) -> None:
    _assert_sound(await _run(client, bank_id, settled, question_id))


@pytest.mark.full
@pytest.mark.parametrize("question_id", [q for q in _QUESTIONS if q not in MINIMUM_ACCEPTANCE])
async def test_source_priority_full(client: Hindsight, bank_id: str, settled: SettleFn, question_id: str) -> None:
    _assert_sound(await _run(client, bank_id, settled, question_id))
