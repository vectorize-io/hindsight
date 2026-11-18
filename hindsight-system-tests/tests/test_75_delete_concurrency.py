"""Concurrent document cleanup and tag updates over a shared observation all complete."""

import asyncio

import pytest
from hindsight_client_api.models.update_document_request import UpdateDocumentRequest
from hindsight_system_tests.payloads import (
    Consolidation,
    Observation,
    ObservationUpdate,
    batch_fact_id,
    batch_fact_ids,
    extracted,
    fact,
    fact_ids_in,
    observes,
)

pytestmark = pytest.mark.asyncio

WEATHER = "log:weather"
MORNING = "log:morning"
EVENING = "log:evening"


async def _shared_observation(client, llm, bank_id, settled, tags: list[str] | None = None) -> str:
    """Retain `morning` and `evening`, both sources of one observation, and return its id."""
    llm.on_step("extract_facts").returns(extracted(fact("The morning weather was mild")))
    llm.on_step("consolidate").answers_with(observes("The weather was mild"))
    await client.aretain(bank_id=bank_id, content="Morning weather record", document_id="morning", tags=tags)
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
    await client.aretain(bank_id=bank_id, content="Evening weather record", document_id="evening", tags=tags)
    await settled(bank_id)
    memories = await client.memory.list_memories(bank_id, limit=100)
    sources = [m for m in memories.items if m.fact_type == "world"]
    observations = [m for m in memories.items if m.fact_type == "observation"]
    assert {m.document_id for m in sources} == {"morning", "evening"}
    assert len(observations) == 1
    assert set(observations[0].source_memory_ids) == {m.id for m in sources}
    return observations[0].id


def _one_part_of_the_day(request) -> Consolidation:
    """Once the documents' tags part, each fact is consolidated alone, in its own scope."""
    part = "morning" if batch_fact_id(request, "morning") else "evening"
    return Consolidation(creates=[Observation(text=f"The {part} was mild", source_fact_ids=batch_fact_ids(request))])


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


async def test_concurrent_tag_updates_of_observation_sources(client, llm, bank_id, settled):
    """Both documents share one tag, so one scope and one observation. Each tag update moves
    its document to a scope of its own, which deletes that observation and requeues the other
    document's fact. Both must complete, and each fact is rebuilt into its own observation.
    Catches the tag updates deadlocking over the shared observation (#6653)."""
    shared = await _shared_observation(client, llm, bank_id, settled, tags=[WEATHER])

    llm.reset()
    llm.on_step("consolidate").answers_with(_one_part_of_the_day)
    results = await asyncio.gather(
        client.documents.update_document(bank_id, "morning", UpdateDocumentRequest(tags=[MORNING])),
        client.documents.update_document(bank_id, "evening", UpdateDocumentRequest(tags=[EVENING])),
        return_exceptions=True,
    )
    assert not [r for r in results if isinstance(r, Exception)], results
    await settled(bank_id)
    memories = (await client.memory.list_memories(bank_id, limit=100)).items
    facts = {m.document_id: m.id for m in memories if m.fact_type == "world"}
    assert sorted((m.document_id, m.text, m.tags) for m in memories if m.fact_type == "world") == [
        ("evening", "The evening weather was mild", [EVENING]),
        ("morning", "The morning weather was mild", [MORNING]),
    ]
    observations = [m for m in memories if m.fact_type == "observation"]
    assert sorted((m.text, m.tags, m.source_memory_ids) for m in observations) == [
        ("The evening was mild", [EVENING], [facts["evening"]]),
        ("The morning was mild", [MORNING], [facts["morning"]]),
    ]
    assert shared not in {m.id for m in observations}


async def test_tag_update_races_delete_of_another_observation_source(client, llm, bank_id, settled):
    """The delete's sweep and the tag update both remove the shared observation and requeue
    the surviving fact. Both must complete, and that fact is rebuilt under its new tag.
    Catches the tag update deadlocking with the delete over the shared observation (#6653)."""
    shared = await _shared_observation(client, llm, bank_id, settled, tags=[WEATHER])

    llm.reset()
    llm.on_step("consolidate").answers_with(_one_part_of_the_day)
    results = await asyncio.gather(
        client.documents.delete_document(bank_id, "morning"),
        client.documents.update_document(bank_id, "evening", UpdateDocumentRequest(tags=[EVENING])),
        return_exceptions=True,
    )
    assert not [r for r in results if isinstance(r, Exception)], results
    await settled(bank_id)
    memories = (await client.memory.list_memories(bank_id, limit=100)).items
    facts = [m for m in memories if m.fact_type == "world"]
    assert [(m.document_id, m.text, m.tags) for m in facts] == [("evening", "The evening weather was mild", [EVENING])]
    observations = [m for m in memories if m.fact_type == "observation"]
    assert [(m.text, m.tags, m.source_memory_ids) for m in observations] == [
        ("The evening was mild", [EVENING], [facts[0].id])
    ]
    assert shared not in {m.id for m in observations}
