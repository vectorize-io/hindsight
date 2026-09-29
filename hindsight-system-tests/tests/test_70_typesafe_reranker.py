"""Recall through the TypeSafe reranker, which ranks the pool and can cut it.

Every other reranker Hindsight speaks to scores one (query, document) pair at a
time and hands back a number per pair. TypeSafe does neither: it ranks the whole
pool by making the candidates the options of a single question, and — with
`prune_candidates` on — asks a second question for where relevance ends, so recall
returns the relevant memories and nothing else.

That makes two things worth a story rather than a unit test. The ordering has to
survive the trip from a Choice's probability distribution, through the provider's
rank positions, through the recency and proof-count boosts the pipeline applies
on top, to the results a client sees. And the cut has to actually shrink what
recall returns — a pruning reranker whose verdict the pipeline ignored would look
identical in every unit test of the provider itself.

The server under test is `typesafe_server`, a second process configured for the
provider: the reranker is server-level, so no bank can opt into it.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator

import pytest

from hindsight_system_tests import wait_until_settled
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

QUERY = "Where does Alice live?"

# How many levels the provider's cut question offers. Mirrored rather than
# imported: this suite is blackbox and must not reach into the server package.
CUT_LEVEL_COUNT = 6
WIDEST_CUT = CUT_LEVEL_COUNT - 1

BERLIN = "Alice moved to Berlin in 2021 | Involving: Alice"
CELLO = "Alice plays the cello professionally | Involving: Alice"
HIKING = "Alice went hiking in the Alps last summer | Involving: Alice"


@pytest.fixture
async def typesafe_bank(typesafe_client, llm) -> AsyncIterator[str]:
    """Three facts in a bank on the TypeSafe server, cleaned up afterwards."""
    bank = f"systest-{uuid.uuid4().hex[:12]}"
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin in 2021", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice plays the cello professionally", who="Alice", entities=["Alice", "cello"]),
            fact("Alice went hiking in the Alps last summer", who="Alice", entities=["Alice", "Alps"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    await typesafe_client.aretain(
        bank_id=bank,
        content="Alice moved to Berlin in 2021, plays the cello, and hiked the Alps last summer.",
    )
    await wait_until_settled(typesafe_client, bank)
    yield bank
    with contextlib.suppress(Exception):
        await typesafe_client.banks.delete_bank(bank)


async def test_the_ranking_reaches_the_client(typesafe_client, typesafe_bank, stubs):
    """The Choice orders the pool, and that order is what recall returns.

    The stub ranks lexically against the query, so Berlin beats the other two, and
    the cut is opened wide enough to keep them all — this test is about the order,
    not the cut.
    """
    stubs.rerank.cut_level = WIDEST_CUT  # "all of the listed candidates"

    response = await typesafe_client.arecall(bank_id=typesafe_bank, query=QUERY)

    texts = [result.text for result in response.results]
    assert texts[0] == BERLIN, "the lexically closest memory must rank first"
    assert sorted(texts) == sorted([BERLIN, CELLO, HIKING]), "and nothing is dropped at the widest cut"


async def test_the_cut_shrinks_what_recall_returns(typesafe_client, typesafe_bank, stubs):
    """Level 0 — "only the first candidate is relevant" — and recall returns one.

    The other two memories exist, were retrieved, and were ranked; the reranker's
    verdict is the only reason they are not in the response. A pipeline that
    ignored the verdict would return all three here.
    """
    stubs.rerank.cut_level = 0

    response = await typesafe_client.arecall(bank_id=typesafe_bank, query=QUERY)

    assert [result.text for result in response.results] == [BERLIN]


async def test_a_deeper_cut_keeps_the_top_of_the_same_ranking(typesafe_client, typesafe_bank, stubs):
    """Level 1 is "the first two", and they are the first two of the full ranking.

    The full order is read back rather than assumed: which memory places second is
    the stub's business, and pinning it here would assert the fixture instead of
    the behaviour. What must hold is that the cut takes a prefix — it removes from
    the bottom of the same order, and never reshuffles.
    """
    stubs.rerank.cut_level = WIDEST_CUT
    everything = await typesafe_client.arecall(bank_id=typesafe_bank, query=QUERY)
    full_order = [result.text for result in everything.results]

    stubs.rerank.cut_level = 1
    response = await typesafe_client.arecall(bank_id=typesafe_bank, query=QUERY)

    assert [result.text for result in response.results] == full_order[:2]
