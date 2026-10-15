"""Decision-orchestrated reflect: one rewritten query, every layer at once, and a decision model
that ends the loop.

On a server with a decision model set up, reflect skips the agent's per-layer search turns. One LLM
call rewrites the request into a search query, every retrieval layer runs with it at
once, and a decision model (TypeSafe's System One API) prunes what came back and
judges whether it already answers the request. When it does, the answer is the only
other LLM call.

The decision model is server-level, so the server under test is `typesafe_server`, the
second process whose decision model points at the stub; a bank opts out with
``reflect_use_decision_model``. Its cut level doubles as the sufficiency verdict: the stub answers
every ``score`` question with it, and reflect reads level 2 or more as "the
evidence fully answers the question".
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator

import pytest

from hindsight_system_tests import wait_until_settled
from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.reflect import reflect_loop
from hindsight_system_tests.steps import anchor_for

pytestmark = pytest.mark.asyncio

QUERY = "Where does Alice live?"
SEARCH = "Alice home city"
ANSWER = "Alice lives in Berlin."
# The stub's top cut level: it keeps every candidate, and as a sufficiency verdict it
# reads as "fully answers".
ENOUGH = 5


@pytest.fixture
async def decision_bank(typesafe_client, client, llm) -> AsyncIterator[str]:
    """A bank on the TypeSafe server, with two facts about Alice.

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
    yield bank
    with contextlib.suppress(Exception):
        await typesafe_client.banks.delete_bank(bank)


def _reflect_turns(llm) -> int:
    """How many calls the reflect loop itself made (the query rewrite is not one)."""
    anchor = anchor_for("reflect")
    return sum(1 for call in llm.calls if anchor in call.all_text)


async def test_enough_evidence_answers_after_one_parallel_retrieval(typesafe_client, llm, stubs, decision_bank):
    """No agent search turns: the rewritten query drives every layer, then the answer."""
    stubs.rerank.cut_level = ENOUGH
    llm.on_step("reflect_round_query").returns_text(SEARCH)
    llm.on_step("reflect", tool="done").returns_tool_call("done", answer=ANSWER)
    llm.calls.clear()

    response = await typesafe_client.areflect(
        bank_id=decision_bank, query=QUERY, include_tool_calls=True, include_facts=True
    )

    assert response.text == ANSWER
    assert response.based_on, "the answer must still carry its evidence"
    searches = [call for call in response.trace.tool_calls if call.tool != "done"]
    assert {call.tool for call in searches} == {"search_observations", "recall"}
    assert {call.input["query"] for call in searches} == {SEARCH}, "every layer searches with the rewritten query"
    assert _reflect_turns(llm) == 1, "the answer is the only reflect turn"


async def test_a_request_with_nothing_to_search_returns_empty_at_once(typesafe_client, llm, stubs, decision_bank):
    """An acknowledgement ("yes, go ahead") costs one small call, not a retrieval."""
    stubs.rerank.cut_level = ENOUGH
    llm.on_step("reflect_round_query").returns_text("NONE")
    llm.calls.clear()

    response = await typesafe_client.areflect(bank_id=decision_bank, query="yes, go ahead", include_tool_calls=True)

    assert response.text == ""
    assert not (response.trace.tool_calls if response.trace else None), "nothing was searched"
    assert _reflect_turns(llm) == 0


async def test_a_bank_that_opts_out_runs_agent_turns(typesafe_client, llm, stubs, decision_bank):
    """``reflect_use_decision_model: false`` keeps the agent's per-layer searches on that bank."""
    await typesafe_client.banks.update_bank_config(decision_bank, {"updates": {"reflect_use_decision_model": False}})
    stubs.rerank.cut_level = ENOUGH
    reflect_loop(llm, answer=ANSWER, query=SEARCH)
    llm.calls.clear()

    response = await typesafe_client.areflect(bank_id=decision_bank, query=QUERY, include_tool_calls=True)

    assert response.text == ANSWER
    assert not _rewrote_the_query(llm), "no query rewrite"
    assert _reflect_turns(llm) > 1, "the agent wrote its own searches"


def _rewrote_the_query(llm) -> bool:
    return any(anchor_for("reflect_round_query") in call.all_text for call in llm.calls)


async def test_a_knowledge_page_refresh_follows_its_own_switch(typesafe_client, llm, stubs, decision_bank):
    """``mental_model_refresh_use_decision_model``, not reflect's switch, decides a refresh's path.

    Run as dry runs: they execute in the API process, the one with the decision model (this
    server has no worker), and leave the page as it is between the two.
    """
    stubs.rerank.cut_level = ENOUGH
    llm.on_step("reflect_round_query").returns_text(SEARCH)
    reflect_loop(llm, answer=ANSWER, query=SEARCH)
    created = await typesafe_client.mental_models.create_mental_model(
        decision_bank, {"name": "Home", "source_query": QUERY}
    )
    await wait_until_settled(typesafe_client, decision_bank)

    await typesafe_client.banks.update_bank_config(decision_bank, {"updates": {"reflect_use_decision_model": False}})
    llm.calls.clear()
    await typesafe_client.mental_models.dry_run_refresh_mental_model(decision_bank, created.mental_model_id)
    assert _rewrote_the_query(llm), "reflect's switch turned the refresh's decision model off"

    await typesafe_client.banks.update_bank_config(
        decision_bank, {"updates": {"mental_model_refresh_use_decision_model": False}}
    )
    llm.calls.clear()
    await typesafe_client.mental_models.dry_run_refresh_mental_model(decision_bank, created.mental_model_id)
    assert not _rewrote_the_query(llm), "the refresh kept the decision model after the bank turned it off"
