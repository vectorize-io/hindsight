"""Fast reflect: one rewritten query, every layer at once, and a decision model that ends the loop.

A bank set to ``reflect_mode: fast`` skips the agent's per-layer search turns. One LLM
call rewrites the request into a search query, every retrieval layer runs with it at
once, and a decision model (TypeSafe's System One API) prunes what came back and
judges whether it already answers the request. When it does, the answer is the only
other LLM call.

Two capabilities meet here — a bank config field and the TypeSafe provider — so the
server under test is `typesafe_server`, the second process whose TypeSafe settings
point at the stub. Its cut level doubles as the sufficiency verdict: the stub answers
every ``score`` question with it, and fast reflect reads level 2 or more as "the
evidence fully answers the question".
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator

import pytest

from hindsight_system_tests import wait_until_settled
from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.steps import anchor_for

pytestmark = pytest.mark.asyncio

QUERY = "Where does Alice live?"
SEARCH = "Alice home city"
ANSWER = "Alice lives in Berlin."
# The stub's top cut level: it keeps every candidate, and as a sufficiency verdict it
# reads as "fully answers".
ENOUGH = 5


@pytest.fixture
async def fast_bank(typesafe_client, client, llm) -> AsyncIterator[str]:
    """A fast-mode bank on the TypeSafe server, with two facts about Alice.

    Asks for ``client`` only to have the main server running: the TypeSafe server's
    worker is off (see ``typesafe_server``), so the main one consolidates this bank.
    """
    bank = f"systest-{uuid.uuid4().hex[:12]}"
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice renewed her Berlin lease", who="Alice", entities=["Alice", "Berlin"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    await typesafe_client.aretain(bank_id=bank, content="Alice moved to Berlin. Alice renewed her Berlin lease.")
    await wait_until_settled(typesafe_client, bank)
    await typesafe_client.banks.update_bank_config(bank, {"updates": {"reflect_mode": "fast"}})
    yield bank
    with contextlib.suppress(Exception):
        await typesafe_client.banks.delete_bank(bank)


def _reflect_turns(llm) -> int:
    """How many calls the reflect loop itself made (the query rewrite is not one)."""
    anchor = anchor_for("reflect")
    return sum(1 for call in llm.calls if anchor in call.all_text)


async def test_enough_evidence_answers_after_one_parallel_retrieval(typesafe_client, llm, stubs, fast_bank):
    """No agent search turns: the rewritten query drives every layer, then the answer."""
    stubs.rerank.cut_level = ENOUGH
    llm.on_step("reflect_fast_query").returns_text(SEARCH)
    llm.on_step("reflect", tool="done").returns_tool_call("done", answer=ANSWER)
    llm.calls.clear()

    response = await typesafe_client.areflect(
        bank_id=fast_bank, query=QUERY, include_tool_calls=True, include_facts=True
    )

    assert response.text == ANSWER
    assert response.based_on, "the answer must still carry its evidence"
    searches = [call for call in response.trace.tool_calls if call.tool != "done"]
    assert {call.tool for call in searches} == {"search_observations", "recall"}
    assert {call.input["query"] for call in searches} == {SEARCH}, "every layer searches with the rewritten query"
    assert _reflect_turns(llm) == 1, "the answer is the only reflect turn"


async def test_a_request_with_nothing_to_search_returns_empty_at_once(typesafe_client, llm, stubs, fast_bank):
    """An acknowledgement ("yes, go ahead") costs one small call, not a retrieval."""
    stubs.rerank.cut_level = ENOUGH
    llm.on_step("reflect_fast_query").returns_text("NONE")
    llm.calls.clear()

    response = await typesafe_client.areflect(bank_id=fast_bank, query="yes, go ahead", include_tool_calls=True)

    assert response.text == ""
    assert not (response.trace.tool_calls if response.trace else None), "nothing was searched"
    assert _reflect_turns(llm) == 0
