"""Consolidation writes an observation over the facts — it does not consume them.

An observation is a synthesis: "Alice is settled in Berlin", drawn from several
facts that each say something narrower. The facts it was drawn from are its
evidence and must survive it. That sounds obvious and is exactly what has gone
wrong before — a consolidation that treats its inputs as spent leaves the bank
holding a summary with nothing underneath, and the loss is unrecoverable because
the source text has already been reduced.

The evidence link matters as much as the survival: an observation whose
`source_fact_ids` do not resolve is an unfalsifiable claim. Nobody can ask why it
is believed.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import extracted, fact, observes

pytestmark = pytest.mark.asyncio

BERLIN = "Alice moved to Berlin | Involving: Alice"
STAYED = "Alice renewed her Berlin lease | Involving: Alice"
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


async def test_the_observation_is_written(client, consolidated_bank):
    memories = await client.memory.list_memories(consolidated_bank, limit=100)
    observations = [m for m in memories.items if m.fact_type == "observation"]

    assert [o.text for o in observations] == [OBSERVATION]


async def test_the_facts_behind_it_survive(client, consolidated_bank):
    """The regression that matters. A summary is not a replacement for its
    evidence, and losing the evidence cannot be undone."""
    memories = await client.memory.list_memories(consolidated_bank, limit=100)
    raw = sorted(m.text for m in memories.items if m.fact_type != "observation")

    assert raw == sorted([BERLIN, STAYED])


async def test_the_observation_cites_evidence_that_resolves(client, consolidated_bank):
    """Every cited id must name a fact that is actually there. An observation
    whose evidence dangles cannot be justified to anyone asking why."""
    memories = await client.memory.list_memories(consolidated_bank, limit=100)
    by_id = {m.id: m for m in memories.items}
    observation = next(m for m in memories.items if m.fact_type == "observation")

    assert observation.source_memory_ids, "an observation with no evidence is an unfalsifiable claim"
    for source_id in observation.source_memory_ids:
        assert source_id in by_id
        assert by_id[source_id].fact_type != "observation"


async def test_an_observation_is_recallable_on_its_own(client, consolidated_bank):
    """Observations are a distinct fact type, so a caller can ask for the
    synthesis without the raw material underneath it."""
    response = await client.arecall(bank_id=consolidated_bank, query="Where is Alice?", types=["observation"])

    assert [r.text for r in response.results] == [OBSERVATION]
    assert [r.type for r in response.results] == ["observation"]


async def test_raw_recall_is_unaffected_by_the_observation(client, consolidated_bank):
    """Asking for facts still returns facts. Consolidation adds a layer; it does
    not replace the one below."""
    response = await client.arecall(bank_id=consolidated_bank, query="Where is Alice?", types=["world"])

    assert sorted(r.text for r in response.results) == sorted([BERLIN, STAYED])
