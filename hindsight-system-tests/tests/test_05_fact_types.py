"""What the world knows versus what the agent did — and the rename in between.

Hindsight separates objective facts from the agent's own actions, and lets a
caller recall one without the other. The catch is that the two halves of the API
do not agree on the vocabulary: extraction classifies a fact as ``world`` or
``assistant``, and recall returns ``world`` or ``experience``. Nothing in either
schema hints that ``assistant`` and ``experience`` are the same thing.

That translation is exactly the kind of detail a refactor drops on one side. It
is asserted end to end here — retained as ``assistant``, read back as
``experience``, filtered as ``experience`` — so the mapping cannot quietly stop
being applied.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

QUERY = "What do we know about Alice?"

WORLD_FACT = "Alice moved to Berlin in 2021 | Involving: Alice"
AGENT_FACT = "The agent booked Alice a flight to Munich | Involving: Alice"


@pytest.fixture
async def bank_with_both_kinds(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin in 2021", who="Alice", entities=["Alice", "Berlin"]),
            fact(
                "The agent booked Alice a flight to Munich",
                fact_type="assistant",
                who="Alice",
                entities=["Alice", "Munich"],
            ),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin. The agent booked her a flight to Munich.")
    await settled(bank_id)
    return bank_id


async def test_an_assistant_fact_is_read_back_as_experience(client, bank_with_both_kinds):
    """The rename, stated plainly. `assistant` goes in; `experience` comes out."""
    response = await client.arecall(bank_id=bank_with_both_kinds, query=QUERY)

    types_by_text = {r.text: r.type for r in response.results}
    assert types_by_text[WORLD_FACT] == "world"
    assert types_by_text[AGENT_FACT] == "experience"


async def test_recalling_one_type_excludes_the_other(client, bank_with_both_kinds):
    """Both directions, because a filter that silently matches everything passes
    a one-sided test."""
    experiences = await client.arecall(bank_id=bank_with_both_kinds, query=QUERY, types=["experience"])
    assert [r.text for r in experiences.results] == [AGENT_FACT]

    world = await client.arecall(bank_id=bank_with_both_kinds, query=QUERY, types=["world"])
    assert [r.text for r in world.results] == [WORLD_FACT]


async def test_asking_for_both_types_returns_both(client, bank_with_both_kinds):
    response = await client.arecall(bank_id=bank_with_both_kinds, query=QUERY, types=["world", "experience"])

    assert sorted(r.text for r in response.results) == sorted([WORLD_FACT, AGENT_FACT])
