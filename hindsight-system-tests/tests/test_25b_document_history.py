"""A fact edit that rebuilds an observation leaves a trace of both halves (#5638).

Editing a fact — even only its date — deletes every observation built on it, and
consolidation rebuilds them from scratch. The deleted rows are gone for good; before
the document history, nothing in the system could say an observation had ever
existed, so a rebuild that merged two unrelated events could not be found afterwards.

The story pins the composition: retain → consolidate → edit → re-consolidate, read
back through one endpoint. The id the edit removed must be the id the first
consolidation created, and the rebuild must show up as a new one.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import extracted, fact, observes

pytestmark = pytest.mark.asyncio

OBSERVATION = "Alice is settled in Berlin"


async def test_an_edit_and_its_rebuild_are_both_on_the_document_timeline(client, llm, bank_id, settled):
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").answers_with(observes(OBSERVATION))

    # Async: the history is rebuilt from operations, and a synchronous retain has none.
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.", document_id="call-1", retain_async=True)
    await settled(bank_id)

    memories = await client.memory.list_memories(bank_id, limit=100)
    [first_obs] = [m.id for m in memories.items if m.fact_type == "observation"]
    [source] = [m.id for m in memories.items if m.fact_type != "observation"]

    # A date-only correction: the text is unchanged, the observation is still dropped.
    await client.memory.update_memory(bank_id, source, {"occurred_start": "2026-03-01T00:00:00Z"})
    await settled(bank_id)

    memories = await client.memory.list_memories(bank_id, limit=100)
    [rebuilt_obs] = [m.id for m in memories.items if m.fact_type == "observation"]
    assert rebuilt_obs != first_obs, "the edit should have rebuilt the observation under a new id"

    history = await client.documents.get_document_history(bank_id, "call-1")
    kinds = [e.kind for e in history.items]
    assert kinds == ["consolidation", "fact_edited", "consolidation", "retain"], kinds

    rebuild, edit, first_consolidation, retain = history.items
    assert edit.operation_id == rebuild.operation_id, "the edit is recorded on the consolidation it queued"
    assert retain.facts_added == 1
    assert first_consolidation.observations_created.ids == [first_obs]
    assert edit.fact_id == source
    assert edit.observations_removed.ids == [first_obs], "the deleted observation must still be named somewhere"
    assert rebuild.observations_created.ids == [rebuilt_obs]
