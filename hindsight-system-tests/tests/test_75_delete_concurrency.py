"""Concurrent document cleanup removes shared observations without a partial batch."""

import asyncio

import pytest
from hindsight_system_tests.payloads import Consolidation, ObservationUpdate, extracted, fact, fact_ids_in, observes

pytestmark = pytest.mark.asyncio


async def _shared_observation(client, llm, bank_id, settled) -> None:
    """Retain `morning` and `evening`, both sources of one observation."""
    llm.on_step("extract_facts").returns(extracted(fact("The morning weather was mild")))
    llm.on_step("consolidate").answers_with(observes("The weather was mild"))
    await client.aretain(bank_id=bank_id, content="Morning weather record", document_id="morning")
    await settled(bank_id)
    memories = await client.memory.list_memories(bank_id, limit=100)
    observation_id = next(m.id for m in memories.items if m.fact_type == "observation")

    llm.reset()
    llm.on_step("extract_facts").returns(extracted(fact("The evening weather was mild")))

    def merge(request) -> Consolidation:
        return Consolidation(
            updates=[
                ObservationUpdate(
                    text="The weather stayed mild all day",
                    observation_id=observation_id,
                    source_fact_ids=fact_ids_in(request.all_text),
                )
            ]
        )

    llm.on_step("consolidate").answers_with(merge)
    await client.aretain(bank_id=bank_id, content="Evening weather record", document_id="evening")
    await settled(bank_id)
    memories = await client.memory.list_memories(bank_id, limit=100)
    sources = [m for m in memories.items if m.fact_type == "world"]
    observations = [m for m in memories.items if m.fact_type == "observation"]
    assert {m.document_id for m in sources} == {"morning", "evening"}
    assert len(observations) == 1
    assert set(observations[0].source_memory_ids) == {m.id for m in sources}


async def test_concurrent_deletes_of_observation_sources(client, llm, bank_id, settled):
    await _shared_observation(client, llm, bank_id, settled)

    results = await asyncio.gather(
        client.documents.delete_document(bank_id, "morning"),
        client.documents.delete_document(bank_id, "evening"),
        return_exceptions=True,
    )
    assert not [r for r in results if isinstance(r, Exception)], results
    await settled(bank_id)
    assert not (await client.memory.list_memories(bank_id, limit=100)).items


async def test_reingest_races_delete_of_another_observation_source(client, llm, bank_id, settled):
    await _shared_observation(client, llm, bank_id, settled)

    llm.reset()
    llm.on_step("extract_facts").returns(extracted(fact("The evening weather turned cold")))
    llm.on_step("consolidate").answers_with(observes("The evening turned cold"))
    results = await asyncio.gather(
        client.documents.delete_document(bank_id, "morning"),
        client.aretain(bank_id=bank_id, content="Evening weather, corrected", document_id="evening"),
        return_exceptions=True,
    )
    assert not [r for r in results if isinstance(r, Exception)], results
    await settled(bank_id)
    memories = (await client.memory.list_memories(bank_id, limit=100)).items
    assert sorted((m.fact_type, m.document_id, m.text) for m in memories if m.fact_type == "world") == [
        ("world", "evening", "The evening weather turned cold")
    ]
    observations = [m for m in memories if m.fact_type == "observation"]
    assert [m.text for m in observations] == ["The evening turned cold"]
