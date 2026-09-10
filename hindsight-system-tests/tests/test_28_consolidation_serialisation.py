"""One bank consolidates one run at a time.

Consolidation reads facts, writes observations, and *deletes* the ones it
supersedes. Two runs over the same bank at once therefore race on the thing they
are both editing: each reads the observation layer, each decides independently
what to merge and retire, and the second to write undoes half of what the first
concluded. The visible symptoms are the ones story 22 and 23 guard against —
duplicate observations for one claim, a retirement that reappears — arriving via
a route those stories cannot reach.

So the server serialises per bank, and a second request joins the run in flight
rather than starting a rival. That is what makes it safe for consolidation to be
triggered from several directions — a retain, a scheduled sweep, an impatient
caller — without any of them coordinating.
"""

from __future__ import annotations

import asyncio

import pytest

from hindsight_system_tests.payloads import extracted, fact, observes

pytestmark = pytest.mark.asyncio

OBSERVATION = "Alice is settled in Berlin"


@pytest.fixture
async def consolidated_bank(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice renewed her Berlin lease", who="Alice", entities=["Alice", "Berlin"]),
        )
    )
    llm.on_step("consolidate").answers_with(observes(OBSERVATION))
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin. Alice renewed her Berlin lease.")
    await settled(bank_id)
    return bank_id


async def _observations(client, bank: str) -> list[str]:
    memories = await client.memory.list_memories(bank, limit=100)
    return sorted(m.text for m in memories.items if m.fact_type == "observation")


async def test_a_second_request_joins_the_run_already_in_flight(client, consolidated_bank):
    """Not merely "both succeed" — the *same* operation. Two ids would mean two
    passes over one observation layer, which is the race."""
    first = await client.banks.trigger_consolidation(consolidated_bank)
    second = await client.banks.trigger_consolidation(consolidated_bank)

    assert second.deduplicated is True
    assert second.operation_id == first.operation_id


async def test_a_burst_of_requests_produces_one_pass(client, consolidated_bank, settled):
    """Five callers, no coordination between them — the shape of a scheduled
    sweep landing on top of a retain that just finished."""
    before = (await client.operations.list_operations(consolidated_bank, type="consolidation", limit=100)).total

    await asyncio.gather(*(client.banks.trigger_consolidation(consolidated_bank) for _ in range(5)))
    await settled(consolidated_bank)

    after = (await client.operations.list_operations(consolidated_bank, type="consolidation", limit=100)).total
    assert after - before <= 1, f"a burst of 5 requests produced {after - before} consolidation runs"


async def test_the_observation_layer_survives_the_burst(client, consolidated_bank, settled):
    """The reason serialisation matters. Concurrent passes would each decide what
    to create and retire from their own read, and the loser's decisions would
    reappear as duplicates."""
    await asyncio.gather(*(client.banks.trigger_consolidation(consolidated_bank) for _ in range(5)))
    await settled(consolidated_bank)

    assert await _observations(client, consolidated_bank) == [OBSERVATION]


async def test_the_facts_survive_the_burst(client, consolidated_bank, settled):
    """And the layer underneath is untouched, however many passes were asked for."""
    await asyncio.gather(*(client.banks.trigger_consolidation(consolidated_bank) for _ in range(5)))
    await settled(consolidated_bank)

    memories = await client.memory.list_memories(consolidated_bank, limit=100)
    raw = sorted(m.text for m in memories.items if m.fact_type != "observation")
    assert raw == sorted(
        ["Alice moved to Berlin | Involving: Alice", "Alice renewed her Berlin lease | Involving: Alice"]
    )
