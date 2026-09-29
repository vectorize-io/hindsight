"""A knowledge page must converge on the right answer as data arrives.

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
"""

from __future__ import annotations

import logging

import pytest
from hindsight_client import Hindsight

from hindsight_system_evals import build_page, evaluate, questions, split_into_waves
from hindsight_system_evals.pages import PageOutcome, SettleFn, facts
from hindsight_system_evals.report import RECORDED, EvalRecord

log = logging.getLogger(__name__)

_QUESTIONS = {question.id: question for question in questions()}

#: The minimum-acceptance set: one question per failure mode that has actually
#: regressed, kept small so CI stays inside a sane budget. Everything else is
#: marked ``full``. These two are here because they are the two that broke.
MINIMUM_ACCEPTANCE = ("hq-release-prod", "hq-count-billing")


async def _run(
    client: Hindsight, bank_id: str, settled: SettleFn, question_id: str, *, record: bool = True
) -> PageOutcome:
    question = _QUESTIONS[question_id]
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


@pytest.mark.parametrize("question_id", MINIMUM_ACCEPTANCE)
async def test_page_converges_minimum_acceptance(
    client: Hindsight, bank_id: str, settled: SettleFn, question_id: str
) -> None:
    """The two cases that have actually regressed. This is the CI gate."""
    outcome = await _run(client, bank_id, settled, question_id)
    _assert_sound(outcome)


@pytest.mark.full
@pytest.mark.parametrize("question_id", [q for q in _QUESTIONS if q not in MINIMUM_ACCEPTANCE])
async def test_page_converges_full(client: Hindsight, bank_id: str, settled: SettleFn, question_id: str) -> None:
    """The rest of the categories: supersession, entity confusion, scoped truth,
    dense absence, numeric precision. Run with ``--full``."""
    outcome = await _run(client, bank_id, settled, question_id)
    _assert_sound(outcome)


@pytest.mark.full
async def test_a_page_never_silently_collapses(client: Hindsight, bank_id: str, settled: SettleFn) -> None:
    """A wave that shrinks a page to almost nothing is a destructive edit.

    Separate from correctness because a page can shrink legitimately — a
    superseded claim should go — but a collapse to a fraction of its former size
    is the signature of a replace that should have been a merge, and a final-text
    score cannot see it.
    """
    outcome = await _run(client, bank_id, settled, "hq-count-billing", record=False)
    sizes = [w.chars for w in outcome.waves]
    assert sizes[-1] >= sizes[0] * 0.5, (
        f"the page collapsed across waves ({sizes}) — bank {outcome.bank_id}\npage reads: {outcome.final_content[:400]}"
    )
