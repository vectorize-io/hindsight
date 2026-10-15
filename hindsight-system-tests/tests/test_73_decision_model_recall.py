"""Recall on a server with a decision model set up.

With a decision model, recall stops scoring candidates with the reranker. It asks the
model of each candidate whether showing it would help answer the question, orders the
pool by that, and — on a question that asks for a value — asks which of the candidates
that disagree states the current one, dropping the rest.

Each step is unit-tested on its own. What only a story shows is that the verdicts
survive the pipeline: that the model's order is what the client receives, with no boost
reordering it afterwards, and that a dropped candidate is actually gone from the
response rather than ranked last.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator

import pytest
from hindsight_client_api.exceptions import ApiException

from hindsight_system_tests import wait_until_settled
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

BERLIN = "Alice moved to Berlin in 2021"
CELLO = "Alice plays the cello professionally"
STALE = "The launch deadline for the mobile app is April 1"
CURRENT = "The launch deadline for the mobile app is April 5"
# What the model says of each memory on the deadline question.
DEADLINE_VERDICTS = {CURRENT: 0.9, STALE: 0.8, CELLO: 0.5, BERLIN: 0.3}


@pytest.fixture
async def bank(decision_model_client, hindsight_server, llm) -> AsyncIterator[str]:
    """Four facts in a bank, consolidated. ``hindsight_server`` is asked for by name: the
    decision-model server runs no worker, so the main server's has to settle the bank."""
    bank = f"systest-{uuid.uuid4().hex[:12]}"
    llm.on_step("extract_facts").returns(
        extracted(
            fact(BERLIN, who="Alice", entities=["Alice", "Berlin"]),
            fact(CELLO, who="Alice", entities=["Alice", "cello"]),
            fact(STALE, who="Alice", entities=["Alice", "mobile app"]),
            fact(CURRENT, who="Alice", entities=["Alice", "mobile app"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    await decision_model_client.aretain(
        bank_id=bank,
        content="Alice moved to Berlin, plays the cello, and the app launch moved from April 1 to April 5.",
    )
    await wait_until_settled(decision_model_client, bank)
    yield bank
    with contextlib.suppress(Exception):
        await decision_model_client.banks.delete_bank(bank)


def _texts(response) -> list[str]:
    """Each result's fact, without the " | Involving: ..." suffix recall adds."""
    return [result.text.split(" | ")[0] for result in response.results]


async def test_the_models_order_reaches_the_client(decision_model_client, bank, stubs):
    """The model's order is what comes back even where it goes against relevance, and
    nothing is dropped on a question that asks for no value."""
    stubs.rerank.verdicts = {CELLO: 0.9, STALE: 0.7, CURRENT: 0.5, BERLIN: 0.3}

    response = await decision_model_client.arecall(bank_id=bank, query="Where did Alice move to?")

    assert _texts(response) == [CELLO, STALE, CURRENT, BERLIN]


async def test_a_superseded_value_is_dropped(decision_model_client, bank, stubs):
    """The model says April 5 is the current deadline, so April 1 is gone — not ranked
    last, where a caller reading everything would still see it."""
    stubs.rerank.verdicts = DEADLINE_VERDICTS
    stubs.rerank.current_value = "April 5"

    response = await decision_model_client.arecall(
        bank_id=bank, query="When is the launch deadline for the mobile app?"
    )

    assert _texts(response) == [CURRENT, CELLO, BERLIN]


async def test_a_hedged_verdict_keeps_both_values(decision_model_client, bank, stubs):
    """Without a confident verdict nothing is dropped: returning both is where recall
    was before, and dropping the wrong one would delete the answer."""
    stubs.rerank.verdicts = DEADLINE_VERDICTS
    response = await decision_model_client.arecall(
        bank_id=bank, query="When is the launch deadline for the mobile app?"
    )

    assert _texts(response) == [CURRENT, STALE, CELLO, BERLIN]


async def test_a_rank_position_is_not_published_or_floored(decision_model_client, bank):
    """The model's scores are positions in one pool, so a reranker floor is refused."""
    response = await decision_model_client.arecall(bank_id=bank, query="Where did Alice move to?")
    assert all(result.scores is None or result.scores.reranker is None for result in response.results)

    with pytest.raises(ApiException) as exc:
        await decision_model_client.arecall(
            bank_id=bank, query="Where did Alice move to?", min_scores={"reranker": 0.5}
        )
    assert exc.value.status == 400
