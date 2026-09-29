"""An over-long answer is shortened to the budget the caller asked for.

`max_tokens` on reflect is a target for the *visible answer*, not a provider cap.
That distinction is the whole feature: an agent embedding the answer in its own
context has a budget, and an answer that overruns it does not fail loudly — it
silently pushes something else out of the window, and what gets pushed out is
whatever was least recently added.

So when the answer overruns, reflect makes a second call that rewrites it to fit
and returns the rewrite. Two things follow that are worth pinning: the rewrite
only happens when it is needed, and an answer already inside the budget is
returned untouched rather than paraphrased for no reason.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

QUERY = "What about Alice?"
SHORT_ANSWER = "Alice lives in Berlin."
LONG_ANSWER = " ".join(f"Sentence {i} about Alice living in Berlin and playing the cello." for i in range(80))
TRIMMED = "Alice lives in Berlin and plays the cello."


@pytest.fixture
async def bank_with_facts(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)
    return bank_id


async def test_an_over_long_answer_is_rewritten_to_fit(client, llm, bank_with_facts):
    reflect_loop(llm, answer=LONG_ANSWER, query="Alice Berlin")
    llm.on_step("reflect_trim").returns_text(TRIMMED)

    response = await client.areflect(bank_id=bank_with_facts, query=QUERY, max_tokens=40)

    assert response.text == TRIMMED


async def test_the_rewrite_is_told_the_budget_and_the_text(client, llm, bank_with_facts):
    """It is a real call to a real model, so it has to carry what the model needs
    to do the job — the target, and the whole text being cut down."""
    reflect_loop(llm, answer=LONG_ANSWER, query="Alice Berlin")
    llm.on_step("reflect_trim").returns_text(TRIMMED)

    await client.areflect(bank_id=bank_with_facts, query=QUERY, max_tokens=40)

    prompt = llm.prompts_for("reflect_trim")[0]
    assert "Target budget: 40 tokens" in prompt
    assert "Sentence 79" in prompt, "the rewrite was shown a truncated copy of the answer it is trimming"


async def test_an_answer_already_inside_the_budget_is_left_alone(client, llm, bank_with_facts):
    """No rewrite call at all — declaring no rule for the trim step *is* the
    assertion, since an unscripted call fails the test.

    Paraphrasing an answer that already fits would spend a model call to make it
    worse, and reflect already costs several.
    """
    reflect_loop(llm, answer=SHORT_ANSWER, query="Alice Berlin")

    response = await client.areflect(bank_id=bank_with_facts, query=QUERY, max_tokens=4096)

    assert response.text == SHORT_ANSWER
    assert llm.prompts_for("reflect_trim") == []


async def test_no_budget_means_no_rewrite_however_long_the_answer(client, llm, bank_with_facts):
    """Without `max_tokens` there is no target to overrun, so a long answer is
    returned whole. Same assertion by absence: a trim call here would be
    unscripted."""
    reflect_loop(llm, answer=LONG_ANSWER, query="Alice Berlin")

    response = await client.areflect(bank_id=bank_with_facts, query=QUERY)

    assert response.text == LONG_ANSWER
    assert llm.prompts_for("reflect_trim") == []
