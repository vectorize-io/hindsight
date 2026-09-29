"""Consolidation can be asked for, and asking twice does not do it twice.

Auto-consolidation runs on its own schedule, which is right for steady ingestion
and wrong for a caller who has just loaded a corpus and wants the synthesis now.
`trigger_consolidation` is that door.

The interesting property is deduplication. A trigger is exactly the kind of call
that gets fired twice — a retried request, an impatient user, two workers on the
same queue — and consolidation is expensive and not idempotent in its effects: a
second concurrent run over the same facts can create the sibling observation
that story 22 exists to prevent. So the endpoint reports whether it started new
work or joined work already in flight, and a caller can tell the difference.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import extracted, fact, observes

pytestmark = pytest.mark.asyncio

OBSERVATION = "Alice is settled in Berlin"


async def _observations(client, bank: str) -> list[dict]:
    memories = await client.memory.list_memories(bank, limit=100)
    return [m for m in memories.items if m.fact_type == "observation"]


async def test_a_trigger_reports_the_operation_it_started(client, llm, bank_id, settled):
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").answers_with(observes(OBSERVATION))

    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)

    result = await client.banks.trigger_consolidation(bank_id)

    assert result.operation_id, "a caller needs an operation id to follow the work it asked for"
    assert result.deduplicated is False
    await settled(bank_id)


async def test_triggering_again_while_work_is_queued_is_deduplicated(client, llm, bank_id, settled):
    """Two triggers back to back, no wait between them.

    The second must not enqueue a second pass over the same facts — that is how
    one claim ends up with two observations. Whether it dedupes depends on the
    first still being in flight, so the assertion is on the *reported* outcome
    rather than on timing: whatever the server decides, it has to say so, and it
    must not silently run twice.
    """
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").answers_with(observes(OBSERVATION))

    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)

    first = await client.banks.trigger_consolidation(bank_id)
    second = await client.banks.trigger_consolidation(bank_id)
    await settled(bank_id)

    assert first.operation_id
    if second.deduplicated:
        # Joined the run already in flight rather than starting a rival one.
        assert second.operation_id == first.operation_id
    else:
        assert second.operation_id != first.operation_id

    # Either way, the bank must not have grown a duplicate observation.
    assert len(await _observations(client, bank_id)) <= 1


async def test_a_trigger_over_an_already_consolidated_bank_is_harmless(client, llm, bank_id, settled):
    """Nothing new to consolidate is a normal state, not an error — the bank is
    left exactly as it was."""
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").answers_with(observes(OBSERVATION))

    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)
    before = sorted(m.text for m in (await client.memory.list_memories(bank_id, limit=100)).items)

    await client.banks.trigger_consolidation(bank_id)
    await settled(bank_id)

    after = sorted(m.text for m in (await client.memory.list_memories(bank_id, limit=100)).items)
    assert after == before
