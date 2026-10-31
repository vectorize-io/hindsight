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


TEA = "Marta's favourite tea is genmaicha"
TEA_AGAIN = "Marta's favorite tea is genmaicha"
SENCHA = "Marta drinks sencha in the afternoon"
UNRELATED = "Marta set up the staging cluster on Kubernetes"


@pytest.fixture
async def tea_bank(decision_model_prune_client, hindsight_server, llm) -> AsyncIterator[str]:
    """A fact, a reworded repeat of it, a second fact and an unrelated one."""
    bank = f"systest-{uuid.uuid4().hex[:12]}"
    llm.on_step("extract_facts").returns(
        extracted(
            fact(TEA, who="Marta", entities=["Marta", "genmaicha"]),
            fact(TEA_AGAIN, who="Marta", entities=["Marta", "genmaicha"]),
            fact(SENCHA, who="Marta", entities=["Marta", "sencha"]),
            fact(UNRELATED, who="Marta", entities=["Marta", "Kubernetes"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    await decision_model_prune_client.aretain(bank_id=bank, content="Marta's teas, and the staging cluster.")
    await wait_until_settled(decision_model_prune_client, bank)
    yield bank
    with contextlib.suppress(Exception):
        await decision_model_prune_client.banks.delete_bank(bank)


async def test_pruning_drops_what_is_no_help(decision_model_prune_client, tea_bank, stubs):
    """With HINDSIGHT_API_RECALL_DECISION_MODEL_PRUNE on, the client does not get the memory
    judged no help. Repeats are not pruned: both copies of the tea fact come back."""
    stubs.rerank.verdicts = {TEA: 0.9, TEA_AGAIN: 0.85, SENCHA: 0.8, UNRELATED: 0.05}

    response = await decision_model_prune_client.arecall(bank_id=tea_bank, query="What tea does Marta like?")

    assert _texts(response) == [TEA, TEA_AGAIN, SENCHA]


async def test_without_pruning_every_candidate_comes_back(decision_model_client, tea_bank, stubs):
    """The same bank and verdicts on a server without the flag: the unrelated memory is
    only ordered last, not dropped."""
    stubs.rerank.verdicts = {TEA: 0.9, TEA_AGAIN: 0.85, SENCHA: 0.8, UNRELATED: 0.05}

    response = await decision_model_client.arecall(bank_id=tea_bank, query="What tea does Marta like?")

    assert _texts(response) == [TEA, TEA_AGAIN, SENCHA, UNRELATED]


async def test_a_request_turns_prune_and_dedup_on(decision_model_client, tea_bank, stubs):
    """On a server with neither flag, the request asks for both: the repeat of the tea fact
    and the memory judged no help are left out; a different fact about Marta stays."""
    stubs.rerank.verdicts = {TEA: 0.9, TEA_AGAIN: 0.85, SENCHA: 0.8, UNRELATED: 0.05}
    stubs.rerank.repeats = {(TEA, TEA_AGAIN)}
    query = "What tea does Marta like?"

    dedup_only = await decision_model_client.arecall(bank_id=tea_bank, query=query, rerank={"dedup": True})
    both = await decision_model_client.arecall(
        bank_id=tea_bank, query=query, rerank={"prune": True, "dedup": True}
    )

    assert _texts(dedup_only) == [TEA, SENCHA, UNRELATED]
    assert _texts(both) == [TEA, SENCHA]


async def test_a_request_turns_the_servers_prune_off(decision_model_prune_client, tea_bank, stubs):
    """The server default is only a default: a request saying rerank.prune=false gets everything."""
    stubs.rerank.verdicts = {TEA: 0.9, TEA_AGAIN: 0.85, SENCHA: 0.8, UNRELATED: 0.05}

    response = await decision_model_prune_client.arecall(
        bank_id=tea_bank, query="What tea does Marta like?", rerank={"prune": False}
    )

    assert _texts(response) == [TEA, TEA_AGAIN, SENCHA, UNRELATED]


async def test_the_bank_sets_dedup_for_requests_that_omit_it(decision_model_client, tea_bank, stubs):
    """A bank with recall_decision_model_dedup on dedups every recall that does not say
    otherwise, and a request can still turn it off."""
    stubs.rerank.verdicts = {TEA: 0.9, TEA_AGAIN: 0.85, SENCHA: 0.8, UNRELATED: 0.05}
    stubs.rerank.repeats = {(TEA, TEA_AGAIN)}
    await decision_model_client.aupdate_bank_config(tea_bank, recall_decision_model_dedup=True)
    query = "What tea does Marta like?"

    omitted = await decision_model_client.arecall(bank_id=tea_bank, query=query)
    off = await decision_model_client.arecall(bank_id=tea_bank, query=query, rerank={"dedup": False})

    assert _texts(omitted) == [TEA, SENCHA, UNRELATED]
    assert _texts(off) == [TEA, TEA_AGAIN, SENCHA, UNRELATED]
