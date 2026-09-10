"""Appending to a document adds to it without re-litigating what is already there.

`update_mode="append"` is the conversation path: each turn extends the same
document rather than replacing it. The distinction from `replace` is the whole
point — an append must *not* run the removal diff, because everything already in
the document is still true, it simply is not in the new fragment.

Getting that backwards is quiet and expensive: appending with replace semantics
deletes the entire history on every turn, and the bank ends up remembering only
the most recent message.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

DOCUMENT_ID = "session-alice"

FIRST = "Alice moved to Berlin."
SECOND = "Alice plays cello."

BERLIN = "Alice moved to Berlin | Involving: Alice"
CELLO = "Alice plays cello | Involving: Alice"


async def _fact_texts(client, bank: str) -> list[str]:
    memories = await client.memory.list_memories(bank, limit=100)
    return sorted(item.text for item in memories.items if item.state == "valid")


@pytest.fixture
async def appended(client, llm, bank_id, settled):
    llm.on_step("extract_facts", contains="Berlin").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("extract_facts", contains="cello").returns(
        extracted(fact("Alice plays cello", who="Alice", entities=["Alice", "cello"]))
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content=FIRST, document_id=DOCUMENT_ID)
    await settled(bank_id)
    await client.aretain(bank_id=bank_id, content=SECOND, document_id=DOCUMENT_ID, update_mode="append")
    await settled(bank_id)


async def test_the_earlier_turn_survives_the_later_one(client, bank_id, appended):
    """The failure this guards is total: with replace semantics only the cello
    fact would remain, and every earlier turn of the conversation would be gone."""
    assert await _fact_texts(client, bank_id) == sorted([BERLIN, CELLO])


async def test_the_document_accumulates_the_text(client, bank_id, appended):
    """One document holding both turns, joined — not two documents, and not the
    second turn overwriting the first."""
    document = await client.documents.get_document(bank_id, DOCUMENT_ID)

    assert document.original_text == f"{FIRST}\n{SECOND}"
    assert document.memory_unit_count == 2


async def test_appending_the_same_turn_twice_does_not_duplicate_the_fact(client, bank_id, appended, settled):
    """Re-sending a turn is normal — a retried request, a replayed buffer — and
    must not leave the bank claiming the same thing twice."""
    await client.aretain(bank_id=bank_id, content=SECOND, document_id=DOCUMENT_ID, update_mode="append")
    await settled(bank_id)

    assert await _fact_texts(client, bank_id) == sorted([BERLIN, CELLO])
