"""`min_scores` floors two different things, and only one of them is a filter.

This parameter is the most misreadable in the recall API, so the story exists
mostly to write the real contract down.

`reranker` and `final` are applied *after* ranking, to the results themselves.
They do what the name suggests: raise the floor, get fewer results.

`semantic` and `keyword` are applied *inside* their retrieval arm, before
fusion. A fact excluded from the semantic arm is still returned if the keyword
arm found it, so a floor on one arm alone removes nothing at all — the other arm
supplies it right back. `min_scores={"semantic": 0.9}` reads like "only very
similar memories" and is, on its own, a no-op.

That is defensible: dropping a fact because one of several strategies scored it
poorly would defeat the point of searching several ways. But it is not what the
parameter name suggests, and nothing fails when a caller gets it wrong, so both
behaviours are pinned here — the no-op included.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

QUERY = "Where does Alice live?"

BERLIN = "Alice moved to Berlin in 2021 | Involving: Alice"
CELLO = "Alice plays the cello professionally | Involving: Alice"

# Measured against this fixture, and stable because the stub's embedder and
# reranker are pure functions of the text:
#
#   fact    semantic   keyword   reranker   final
#   Berlin    0.4695      0.30     0.4427   0.4870
#   cello     0.4146      0.30     0.3333   0.3667
#
# The thresholds below are chosen to sit between those numbers.


@pytest.fixture
async def scored_bank(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin in 2021", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice plays the cello professionally", who="Alice", entities=["Alice", "cello"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin in 2021 and works as a cellist.")
    await settled(bank_id)
    return bank_id


async def test_a_post_rerank_floor_filters_results(client, scored_bank):
    """`final` above the second fact's score, below the first's."""
    response = await client.arecall(bank_id=scored_bank, query=QUERY, min_scores={"final": 0.40})

    assert [r.text for r in response.results] == [BERLIN]


async def test_a_reranker_floor_filters_results(client, scored_bank):
    response = await client.arecall(bank_id=scored_bank, query=QUERY, min_scores={"reranker": 0.36})

    assert [r.text for r in response.results] == [BERLIN]


async def test_a_semantic_floor_alone_removes_nothing(client, scored_bank):
    """The footgun, asserted so it cannot change silently.

    0.50 is above *both* facts' semantic scores, so the semantic arm returns
    nothing — and both facts still come back, because the keyword arm found them
    independently. This is per-arm gating, not result filtering.
    """
    response = await client.arecall(bank_id=scored_bank, query=QUERY, min_scores={"semantic": 0.50})

    assert sorted(r.text for r in response.results) == sorted([BERLIN, CELLO])


async def test_a_keyword_floor_alone_removes_nothing(client, scored_bank):
    """The mirror image: the semantic arm now supplies what the keyword arm dropped."""
    response = await client.arecall(bank_id=scored_bank, query=QUERY, min_scores={"keyword": 0.35})

    assert sorted(r.text for r in response.results) == sorted([BERLIN, CELLO])


async def test_a_fact_disappears_only_when_every_arm_excludes_it(client, scored_bank):
    """Both arms floored. The cello fact is under both cutoffs (semantic 0.41 <
    0.45, keyword 0.30 < 0.35) so no arm returns it; Berlin clears the semantic
    one and survives. This is the only way retrieval-level floors remove a fact.
    """
    response = await client.arecall(bank_id=scored_bank, query=QUERY, min_scores={"semantic": 0.45, "keyword": 0.35})

    assert [r.text for r in response.results] == [BERLIN]


async def test_flooring_every_arm_can_empty_the_answer(client, scored_bank):
    """Above every score in both arms, nothing is retrieved at all — so unlike
    the token budget, there is no "never answer with nothing" floor here. The
    caller asked for a quality bar and gets an honest empty result."""
    response = await client.arecall(bank_id=scored_bank, query=QUERY, min_scores={"semantic": 0.50, "keyword": 0.35})

    assert response.results == []


async def test_an_unknown_stage_is_rejected_rather_than_ignored(client, scored_bank):
    """A misspelled stage silently applying no floor is the worst outcome — the
    caller believes they filtered. It raises instead, and names the valid keys."""
    with pytest.raises(ValueError, match="Unknown min_scores keys"):
        await client.arecall(bank_id=scored_bank, query=QUERY, min_scores={"rerank": 0.5})
