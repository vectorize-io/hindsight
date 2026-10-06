"""Recall through the TypeSafe reranker, which ranks the pool and can cut it.

Every other reranker Hindsight speaks to scores one (query, document) pair at a
time and hands back a number per pair. TypeSafe does neither: it ranks the whole
pool by making the candidates the options of a single question, and — with
`prune_candidates` on — asks a second question for where relevance ends, so recall
returns the relevant memories and nothing else.

That makes three things worth a story rather than a unit test. The ordering has to
survive the trip from a Choice's probability distribution, through the provider's
rank positions, to the results a client sees — as a decision model its order is
final, so the recency and proof-count boosts a cross-encoder gets are not applied.
The cut has to actually shrink what recall returns — a pruning reranker whose
verdict the pipeline ignored would look identical in every unit test of the
provider itself. And a bank's ranking rules, written in plain words in its config,
have to reach the model and change what the client sees.

The server under test is `typesafe_server`, a second process configured for the
provider: the reranker is server-level, so no bank can opt into it.
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


async def test_a_rank_position_is_not_published_or_floored(typesafe_client, typesafe_bank, stubs):
    """TypeSafe's per-memory number is a rank position, not a relevance score (#4901).

    The top memory would read 1.0 on every recall however weak it is, so recall
    publishes no reranker score, and a reranker floor — which would only keep a
    fixed share of the pool — is refused rather than silently applied.
    """
    stubs.rerank.cut_level = WIDEST_CUT

    response = await typesafe_client.arecall(bank_id=typesafe_bank, query=QUERY)
    assert len(response.results) == 3
    assert [result.scores.reranker for result in response.results] == [None, None, None]

    with pytest.raises(ApiException) as exc:
        await typesafe_client.arecall(bank_id=typesafe_bank, query=QUERY, min_scores={"reranker": 0.5})
    assert exc.value.status == 400
    assert "min_scores.reranker" in str(exc.value.body)


async def test_a_banks_ranking_rule_changes_the_order(typesafe_client, typesafe_bank, stubs):
    """A rule in the bank's config reaches the model and reorders recall.

    Lexically Berlin answers the question best. The bank then says to prefer
    memories about the cello, and the stub model follows rules of that shape, so
    the cello memory must come first — on the order alone, with no boost moving
    anything back. Without the rule the same recall leads with Berlin, so the
    reorder is the rule's doing.
    """
    stubs.rerank.cut_level = WIDEST_CUT
    before = await typesafe_client.arecall(bank_id=typesafe_bank, query=QUERY)
    assert before.results[0].text == BERLIN
    assert "Today's date is" in stubs.rerank.rank_instructions[-1]

    await typesafe_client.banks.update_bank_config(
        typesafe_bank, {"updates": {"reranker_instructions": "- prefer memories mentioning cello"}}
    )
    after = await typesafe_client.arecall(bank_id=typesafe_bank, query=QUERY)

    # The rule lifts the cello memory over everything; the rest keep their order.
    before_order = [result.text for result in before.results]
    assert [result.text for result in after.results] == [CELLO] + [text for text in before_order if text != CELLO]
    assert "- prefer memories mentioning cello" in stubs.rerank.rank_instructions[-1]
