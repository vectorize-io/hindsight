"""`max_tokens` spends the budget in rank order, and one long fact does not
take the rest of the answer down with it.

This is the shape of issue #3688. The obvious implementation walks the ranked
facts and `break`s at the first one that does not fit — which silently drops
every shorter fact behind it, and in the pathological case returns nothing at
all for a query that matched plenty. The correct behaviour is to skip only the
oversized fact and keep spending, plus a floor so a run that matched something
never answers with nothing.

The middle assertion below therefore looks wrong at a glance and is the whole
point: at a tight budget the *second*-ranked fact is what comes back, because
the top-ranked one does not fit. Anyone refactoring toward "the best result
always survives" reintroduces the bug.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

QUERY = "Where does Alice live?"

# Ranked first by the pipeline, and the longer of the two once rendered with its
# `When:`/`Involving:` clauses — which is what makes the skip observable.
LONG_TOP_FACT = "Alice moved to Berlin in 2021 | When: 2021 | Involving: Alice"
SHORT_SECOND_FACT = "Alice plays the cello professionally | Involving: Alice"


@pytest.fixture
async def bank_with_two_facts(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin in 2021", when="2021", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice plays the cello professionally", who="Alice", entities=["Alice", "cello"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin in 2021 and works as a cellist.")
    await settled(bank_id)
    return bank_id


async def test_a_generous_budget_returns_everything_in_rank_order(client, bank_with_two_facts):
    response = await client.arecall(bank_id=bank_with_two_facts, query=QUERY, max_tokens=4096)

    assert [r.text for r in response.results] == [LONG_TOP_FACT, SHORT_SECOND_FACT]


async def test_an_oversized_top_fact_skips_only_itself(client, bank_with_two_facts):
    """The regression guard. A budget that fits the second fact but not the first.

    A `break`-based implementation returns nothing here; a `continue`-based one
    returns the fact that fits. The counter-intuitive result is the correct one.
    """
    response = await client.arecall(bank_id=bank_with_two_facts, query=QUERY, max_tokens=12)

    assert [r.text for r in response.results] == [SHORT_SECOND_FACT]


async def test_zero_tokens_asks_for_no_facts_at_all(client, bank_with_two_facts):
    """`max_tokens=0` is the documented way to recall chunks without facts, so it
    selects nothing — and the "never answer with nothing" floor deliberately does
    not apply, or the parameter would be impossible to express."""
    response = await client.arecall(bank_id=bank_with_two_facts, query=QUERY, max_tokens=0)

    assert response.results == []
