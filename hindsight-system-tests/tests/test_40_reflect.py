"""Reflect answers a question by searching, then reasoning over what it found.

Recall returns memories; reflect returns an answer. The difference is the loop:
the server drives an agentic conversation that must search before it speaks, and
the answer it produces is supposed to rest on what those searches returned.

Two properties matter more than the wording of any answer. The loop has to
actually run — a reflect that skips straight to speaking is a language model
guessing, which is exactly the failure memory is meant to prevent. And the
answer has to come back attached to its evidence, so a caller can check it.

The stub supplies the answer text, so nothing here judges *quality* — that needs
a real model and belongs in the `hs_llm_core` judge tests. What is asserted is
the mechanism around it.
"""

from __future__ import annotations

import pytest
from hindsight_client_api.exceptions import ApiException

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

QUERY = "Where does Alice live?"
ANSWER = "Alice lives in Berlin, where she renewed her lease."


@pytest.fixture
async def bank_with_facts(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice renewed her Berlin lease", who="Alice", entities=["Alice", "Berlin"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin. Alice renewed her Berlin lease.")
    await settled(bank_id)
    return bank_id


async def test_reflect_returns_the_answer_the_loop_produced(client, llm, bank_with_facts):
    reflect_loop(llm, answer=ANSWER)

    response = await client.areflect(bank_id=bank_with_facts, query=QUERY)

    assert response.text == ANSWER


async def test_the_search_ladder_is_climbed_before_answering(client, llm, bank_with_facts):
    """The loop must not be skippable — asserted by leaving it unscripted.

    Only the *answering* turn is declared here. If the server could reach an
    answer without searching first, the reflect would succeed and no search turn
    would ever reach the stub. Instead the unscripted search turn arrives, which
    is the proof, so this is the one test in the suite whose subject is an
    unmatched call.
    """
    llm.on_step("reflect_answer").returns_text(ANSWER)

    with pytest.raises(ApiException):
        await client.areflect(bank_id=bank_with_facts, query=QUERY)

    searches = [call for call in llm.unmatched if call.tools]
    assert searches, "reflect answered without taking a single search turn"

    # Consumed deliberately: the suite-wide guard exists to catch calls a test
    # forgot to script, and this one is the point of the test.
    llm.unmatched.clear()


async def test_the_answer_can_carry_the_memories_behind_it(client, llm, bank_with_facts):
    """`based_on` is what makes an answer checkable. Without it a caller has a
    paragraph and no way to tell whether it came from the bank or from the
    model."""
    reflect_loop(llm, answer=ANSWER)

    response = await client.areflect(bank_id=bank_with_facts, query=QUERY, include_facts=True)

    assert response.text == ANSWER
    assert response.based_on, "an answer with no evidence cannot be audited"


async def test_evidence_is_omitted_unless_asked_for(client, llm, bank_with_facts):
    """It costs tokens to return, so the default answer is the answer alone."""
    reflect_loop(llm, answer=ANSWER)

    response = await client.areflect(bank_id=bank_with_facts, query=QUERY)

    assert response.based_on is None


async def test_reflect_reports_what_the_answer_cost(client, llm, bank_with_facts):
    """Reflect runs several model turns, so its cost is neither obvious nor
    small. A caller budgeting tokens needs it reported, not inferred."""
    reflect_loop(llm, answer=ANSWER)

    response = await client.areflect(bank_id=bank_with_facts, query=QUERY)

    assert response.usage.total_tokens > 0
    assert response.usage.input_tokens > 0
