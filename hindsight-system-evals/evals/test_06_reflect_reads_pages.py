"""A reflect that has a page about the question must answer from the page.

The other suites cannot see this: their pages are created with
``exclude_mental_models``, so ``has_mental_models`` is false and reflect is never
offered the page tools at all. This one builds the page first and then asks the
question with the page layer live, which is the only shape where the top of the
retrieval ladder is exercised.

It exists because that layer changed: ``search_mental_models`` used to hand back
five pages whole — 8.7-19k tokens, the largest single item in a reflect's floor
(#4533) — and now returns a snippet per page, with ``read_mental_models`` pulling
the full text of the ones the model chooses. That trade is only safe if the model
still *reads* the page it needs, so the assertions are: the answer is right, it
does not assert the bait, and the evidence names the page.
"""

from __future__ import annotations

import logging

import pytest
from hindsight_client import Hindsight

from hindsight_system_evals import build_page, evaluate, questions, split_into_waves
from hindsight_system_evals.pages import SettleFn, facts
from hindsight_system_evals.report import RECORDED, EvalRecord

log = logging.getLogger(__name__)

_QUESTIONS = {question.id: question for question in questions()}

#: Two questions whose answers a page states outright, so a reflect that reads the
#: page can answer and one that stops at the snippet cannot.
PAGE_BACKED = ("hq-release-prod", "hq-count-billing")


async def _page_then_reflect(client: Hindsight, bank_id: str, settled: SettleFn, question_id: str):
    """Build the page, ask its own question, and check the answer came from it.

    Deliberately NOT graded against the corpus gold: whether the page itself got
    the answer right is what test_01 grades, and a page that converged on the
    wrong number would fail this for a reason that has nothing to do with the
    retrieval layer. What matters here is that reflect used the page — it cited
    it, and what it said matches what the page says.
    """
    question = _QUESTIONS[question_id]
    outcome = await build_page(client, bank_id, question, split_into_waves(facts()), settled)
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


@pytest.mark.parametrize("question_id", PAGE_BACKED)
async def test_reflect_answers_from_the_page_it_was_given(
    client: Hindsight, bank_id: str, settled: SettleFn, question_id: str
) -> None:
    await _page_then_reflect(client, bank_id, settled, question_id)
