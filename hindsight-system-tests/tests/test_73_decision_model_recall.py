"""Recall on a server with a decision model set up.

With a decision model, recall stops scoring candidates with the reranker. It asks the
model of each candidate whether showing it would help answer the question, telling it the
date the question is asked, and orders the pool by that.

Each step is unit-tested on its own. What only a story shows is that the verdicts survive
the pipeline: that the model's order is what the client receives, with no boost
reordering it afterwards, and that the recall's own question date is what the model is
told, not the server's clock.
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


async def test_the_model_is_told_the_question_date(decision_model_client, bank, stubs):
    """A question asked "as of" a date is judged as of that date, so the model can weigh
    how recent each memory is from where the question stands."""
    await decision_model_client.arecall(
        bank_id=bank, query="When is the launch deadline?", query_timestamp="2024-05-01T12:00:00"
    )

    assert stubs.rerank.states
    assert set(stubs.rerank.states) == {"Current date: 2024-05-01\n\nQuestion: When is the launch deadline?"}


async def test_a_rank_position_is_not_published_or_floored(decision_model_client, bank):
    """The model's scores are positions in one pool, so a reranker floor is refused."""
    response = await decision_model_client.arecall(bank_id=bank, query="Where did Alice move to?")
    assert all(result.scores is None or result.scores.reranker is None for result in response.results)

    with pytest.raises(ApiException) as exc:
        await decision_model_client.arecall(
            bank_id=bank, query="Where did Alice move to?", min_scores={"reranker": 0.5}
        )
    assert exc.value.status == 400
