"""`budget` buys search depth, and it is a different dial from `max_tokens`.

The docs sell these as two independent dimensions: `budget` decides how hard the
pipeline looks, `max_tokens` decides how much of what it found comes back. That
independence is easy to lose — it would be quite natural for a widened budget to
also widen the answer, or for a tightened token cap to quietly stop the search
early — and neither mistake shows up as an error.

The depth itself is asserted through the trace, because on any corpus small
enough to be a readable test every level finds everything. Counting results
would prove nothing; the number the pipeline was configured with is the honest
observable.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

QUERY = "Where does Alice live?"


@pytest.fixture
async def populated_bank(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin in 2021", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice plays the cello professionally", who="Alice", entities=["Alice", "cello"]),
            fact("Alice grew up in Lisbon", who="Alice", entities=["Alice", "Lisbon"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin in 2021, plays cello, grew up in Lisbon.")
    await settled(bank_id)
    return bank_id


async def test_each_budget_level_configures_a_deeper_search(client, populated_bank):
    depths = {}
    for level in ("low", "mid", "high"):
        response = await client.arecall(bank_id=populated_bank, query=QUERY, budget=level, trace=True)
        depths[level] = response.trace["query"]["budget"]

    assert depths["low"] < depths["mid"] < depths["high"], f"each level must widen the search, got {depths}"


async def test_budget_does_not_change_what_a_small_bank_returns(client, populated_bank):
    """Depth and answer size are separate dials.

    Every fact here is reachable at the shallowest setting, so all three levels
    must return the same three facts in the same order. A level that returns
    *fewer* is truncating rather than searching less deeply — the two are only
    distinguishable on a corpus like this one, where depth is not the binding
    constraint.
    """
    answers = {}
    for level in ("low", "mid", "high"):
        response = await client.arecall(bank_id=populated_bank, query=QUERY, budget=level)
        answers[level] = [r.text for r in response.results]

    assert answers["low"] == answers["mid"] == answers["high"]
    assert len(answers["low"]) == 3


async def test_a_tight_token_cap_does_not_shrink_the_search(client, populated_bank):
    """The other half of the independence claim: capping the *answer* must not
    cap the *search*. If `max_tokens` leaked into the pipeline's depth, the
    budget reported in the trace would move with it."""
    wide = await client.arecall(bank_id=populated_bank, query=QUERY, budget="mid", max_tokens=4096, trace=True)
    narrow = await client.arecall(bank_id=populated_bank, query=QUERY, budget="mid", max_tokens=12, trace=True)

    assert narrow.trace["query"]["budget"] == wide.trace["query"]["budget"]
    assert len(narrow.results) < len(wide.results), "the tight cap must still shorten the answer"
