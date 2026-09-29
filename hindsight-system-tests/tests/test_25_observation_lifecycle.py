"""Removing what an observation was built from, and resetting the layer entirely.

An observation is only as good as the evidence under it. Two operations attack
that from opposite ends: deleting the document its facts came from, and clearing
the observation layer wholesale to rebuild it.

The second is the safer of the two and still worth pinning: `clear_observations`
must remove the synthesis and leave the raw facts completely intact, because
rebuilding the layer is only possible if the material it was derived from is
still there. A "clear" that took the facts with it would be a one-way door
dressed up as a maintenance operation.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import extracted, fact, observes

pytestmark = pytest.mark.asyncio

OBSERVATION = "Alice is settled in Berlin"
BERLIN = "Alice moved to Berlin | Involving: Alice"
LEASE = "Alice renewed her Berlin lease | Involving: Alice"


async def _split(client, bank: str) -> tuple[list[str], list[str]]:
    memories = await client.memory.list_memories(bank, limit=100)
    observations = sorted(m.text for m in memories.items if m.fact_type == "observation")
    raw = sorted(m.text for m in memories.items if m.fact_type != "observation")
    return observations, raw


@pytest.fixture
async def observed_bank(client, llm, bank_id, settled) -> str:
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


async def test_the_starting_point(client, observed_bank):
    observations, raw = await _split(client, observed_bank)

    assert observations == [OBSERVATION]
    assert raw == sorted([BERLIN, LEASE])


async def test_clearing_observations_leaves_every_fact_behind(client, observed_bank, settled):
    """The layer is derived, so it must be safe to throw away and rebuild. That
    is only true if clearing it cannot reach the facts underneath."""
    result = await client.banks.clear_observations(observed_bank)
    await settled(observed_bank)

    assert result.success is True
    assert result.deleted_count == 1

    observations, raw = await _split(client, observed_bank)
    assert observations == []
    assert raw == sorted([BERLIN, LEASE]), "clearing the derived layer must not touch the source facts"


async def test_a_cleared_observation_stops_answering_recalls(client, observed_bank, settled):
    await client.banks.clear_observations(observed_bank)
    await settled(observed_bank)

    response = await client.arecall(bank_id=observed_bank, query="Where is Alice?", types=["observation"])
    assert response.results == []

    # And the facts still answer, so the bank is diminished rather than broken.
    response = await client.arecall(bank_id=observed_bank, query="Where is Alice?", types=["world"])
    assert sorted(r.text for r in response.results) == sorted([BERLIN, LEASE])


async def test_deleting_the_source_document_takes_the_observation_with_it(client, llm, bank_id, settled):
    """The other direction. An observation whose entire evidence base has been
    deleted is a claim with nothing behind it — it cannot be justified, and it
    must not keep answering as though it could.
    """
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").answers_with(observes(OBSERVATION))

    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.", document_id="d1")
    await settled(bank_id)
    observations, _ = await _split(client, bank_id)
    assert observations == [OBSERVATION]

    await client.documents.delete_document(bank_id, "d1")
    await settled(bank_id)

    response = await client.arecall(bank_id=bank_id, query="Where is Alice?", types=["observation"])
    assert [r.text for r in response.results] == [], (
        "an observation outlived every fact it was drawn from and is still being recalled"
    )
