"""Reflect that may say nothing: a memory watcher asks at every step, and most steps it must stay silent.

A watcher beside a coding agent asks reflect, at each thing the agent does, whether memory holds
something the agent must hear, naming its own searches (``search_queries``) and allowing no answer
(``abstain``). The two meet the decision-orchestrated path end to end: the caller's searches replace
the query rewrite, the decision model's pick ("which memory, or none") replaces the sufficiency
check, and an empty answer comes back as ``abstained`` instead of a failure.

The decision model is server-level, so the server under test is ``typesafe_server``; the stub
answers the pick's "none" with ``stubs.rerank.pick_none``.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from collections.abc import AsyncIterator

import pytest

from hindsight_system_tests import wait_until_settled
from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.steps import anchor_for

pytestmark = pytest.mark.asyncio

QUESTION = "Does my change to parse_quantity go against a recorded decision?"
SEARCHES = ["parse_quantity memory units", "parse_quantity disk units"]
NOTE = "Memory suffixes are binary: K is 1024, not 1000."


@pytest.fixture
async def decision_bank(typesafe_client, client, llm) -> AsyncIterator[str]:
    """A bank on the TypeSafe server holding one decision about parse_quantity.

    Asks for ``client`` only to have the main server running: it consolidates this bank.
    """
    bank = f"systest-{uuid.uuid4().hex[:12]}"
    llm.on_step("extract_facts").returns(
        extracted(
            fact("parse_quantity reads memory suffixes as binary, K is 1024", entities=["parse_quantity"]),
            fact("parse_quantity reads disk suffixes as decimal, M is 1000000", entities=["parse_quantity"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    await typesafe_client.aretain(bank_id=bank, content="We decided parse_quantity memory units are binary, disk decimal.")
    await wait_until_settled(typesafe_client, bank)
    yield bank
    with contextlib.suppress(Exception):
        await typesafe_client.banks.delete_bank(bank)


def _llm_turns(llm, step: str) -> int:
    anchor = anchor_for(step)
    return sum(1 for call in llm.calls if anchor in call.all_text)


async def test_nothing_to_say_costs_no_llm_call(typesafe_client, llm, stubs, decision_bank):
    """The decision model sees no memory the answer depends on: empty, abstained, no LLM at all."""
    stubs.rerank.cut_level = 5
    stubs.rerank.pick_none = 0.97
    llm.calls.clear()

    response = await typesafe_client.areflect(
        bank_id=decision_bank, query=QUESTION, abstain=True, search_queries=SEARCHES, include_tool_calls=True
    )

    assert response.abstained and response.text == ""
    searched = {call.input["query"] for call in response.trace.tool_calls}
    assert searched == set(SEARCHES), "the caller's searches ran as given"
    assert _llm_turns(llm, "reflect_round_query") == 0, "no rewrite: the caller named its searches"
    assert _llm_turns(llm, "reflect") == 0 and _llm_turns(llm, "reflect_abstain_answer") == 0, "no answer call"


async def test_something_to_say_is_written_from_what_was_picked(typesafe_client, llm, stubs, decision_bank):
    stubs.rerank.cut_level = 5
    stubs.rerank.pick_none = 0.1
    llm.on_step("reflect_abstain_answer").returns_text(json.dumps({"applies": True, "answer": NOTE}))

    response = await typesafe_client.areflect(
        bank_id=decision_bank,
        query=QUESTION,
        context="I changed parse_quantity to read K as 1000.",
        abstain=True,
        search_queries=SEARCHES,
        include_facts=True,
    )

    assert not response.abstained and response.text == NOTE
    assert response.based_on and response.based_on.memories, "the note carries the memory it came from"
    (answer_call,) = [call for call in llm.calls if anchor_for("reflect_abstain_answer") in call.all_text]
    assert "I changed parse_quantity to read K as 1000." in answer_call.all_text, "the context reaches the answer"


async def test_the_llm_finding_nothing_applies_is_silence_not_text(typesafe_client, llm, stubs, decision_bank):
    """Unsure, the decision model leaves it to the LLM; its "does not apply" comes back empty, not as text."""
    stubs.rerank.cut_level = 5
    stubs.rerank.pick_none = 0.6
    llm.on_step("reflect_abstain_answer").returns_text(json.dumps({"applies": False, "answer": ""}))

    response = await typesafe_client.areflect(
        bank_id=decision_bank, query=QUESTION, abstain=True, search_queries=SEARCHES
    )

    assert response.abstained and response.text == ""


async def test_without_abstain_the_same_request_answers(typesafe_client, llm, stubs, decision_bank):
    """The pick is only asked for an abstain request: an ordinary reflect is unchanged."""
    stubs.rerank.cut_level = 5
    stubs.rerank.pick_none = 0.97
    llm.on_step("reflect", tool="done").returns_tool_call("done", answer=NOTE)

    response = await typesafe_client.areflect(bank_id=decision_bank, query=QUESTION, search_queries=SEARCHES)

    assert not response.abstained and response.text == NOTE
