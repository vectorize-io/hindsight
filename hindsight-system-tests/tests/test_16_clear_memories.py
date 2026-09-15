"""Clearing a bank's memories says how many it erased (#4307).

The delete is the only participant that knows the exact figure: a client that
differences two listings around it races any retain landing in between. So the
count must come back on the response, scoped to the `type` filter when one is
given, and be `0` (not absent) when there was nothing to clear.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio


async def _fact_types(client, bank: str) -> list[str]:
    memories = await client.memory.list_memories(bank, limit=100)
    return sorted(item.fact_type for item in memories.items)


async def test_a_clear_reports_what_it_erased_per_type(client, llm, bank_id, settled):
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin", who="Alice", entities=["Alice"]),
            fact("Alice plays cello", who="Alice", entities=["Alice"]),
            fact("I recommended a cello teacher to Alice", fact_type="assistant", entities=["Alice"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin and plays cello; I found her a teacher.")
    await settled(bank_id)
    assert await _fact_types(client, bank_id) == ["experience", "world", "world"]

    typed = await client.memory.clear_bank_memories(bank_id, type="world")
    assert typed.deleted_count == 2
    assert await _fact_types(client, bank_id) == ["experience"]

    rest = await client.memory.clear_bank_memories(bank_id)
    assert rest.deleted_count == 1
    assert await _fact_types(client, bank_id) == []

    empty = await client.memory.clear_bank_memories(bank_id)
    assert empty.deleted_count == 0
