"""No bank can see, count, or damage another bank's memories.

Bank isolation is the security invariant the whole product rests on: a bank is
one user's or one agent's brain, and a leak is not a ranking bug, it is showing
someone else's data to the wrong person.

The dangerous case is not two obviously different banks — it is two banks that
collide on a caller-supplied key. `document_id` is unique only *within* a bank,
so `profile`, `session-1` and `notes` exist in every bank at once, and a
statement that filters on one without a `bank_id` predicate reaches all of them.
That is #3429, and it is invisible to any test that uses one bank.

So every test here runs two banks that deliberately share ids, and checks the
four verbs separately — read, count, update, delete — because a missing
predicate is per-statement, and getting three of them right proves nothing about
the fourth.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

SHARED_DOCUMENT_ID = "profile"

ALICE_FACT = "Alice moved to Berlin | Involving: Alice"
BOB_FACT = "Bob moved to Lisbon | Involving: Bob"


@pytest.fixture
async def colliding_banks(client, llm, settled) -> AsyncIterator[tuple[str, str]]:
    """Two banks holding different content under the *same* document id."""
    llm.on_step("extract_facts", contains="Berlin").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("extract_facts", contains="Lisbon").returns(
        extracted(fact("Bob moved to Lisbon", who="Bob", entities=["Bob", "Lisbon"]))
    )
    llm.on_step("consolidate").returns(consolidation())

    first = f"systest-{uuid.uuid4().hex[:12]}"
    second = f"systest-{uuid.uuid4().hex[:12]}"
    try:
        await client.aretain(bank_id=first, content="Alice moved to Berlin.", document_id=SHARED_DOCUMENT_ID)
        await client.aretain(bank_id=second, content="Bob moved to Lisbon.", document_id=SHARED_DOCUMENT_ID)
        await settled(first)
        await settled(second)
        yield first, second
    finally:
        for bank in (first, second):
            await client.banks.delete_bank(bank)


async def _fact_texts(client, bank: str) -> list[str]:
    memories = await client.memory.list_memories(bank, limit=100)
    return sorted(m.text for m in memories.items if m.state == "valid")


async def test_reading_a_shared_document_id_returns_only_this_banks_copy(client, colliding_banks):
    first, second = colliding_banks

    assert (await client.documents.get_document(first, SHARED_DOCUMENT_ID)).original_text == "Alice moved to Berlin."
    assert (await client.documents.get_document(second, SHARED_DOCUMENT_ID)).original_text == "Bob moved to Lisbon."


async def test_counting_does_not_include_the_neighbours_rows(client, colliding_banks):
    """A leak shows up in a count long before anyone notices it in a list, and a
    count that over-reports is the same missing predicate."""
    first, second = colliding_banks

    for bank in (first, second):
        listing = await client.documents.list_documents(bank)
        assert listing.total == 1

        memories = await client.memory.list_memories(bank, limit=100)
        assert memories.total == 1

        stats = await client.banks.get_agent_stats(bank)
        assert stats.total_documents == 1
        assert stats.total_nodes == 1


async def test_recall_never_crosses_the_boundary(client, colliding_banks):
    """The user-visible leak. Both banks talk about someone moving to a European
    city, so a query that matches one matches the other — if it could reach it.
    """
    first, second = colliding_banks

    from_first = await client.arecall(bank_id=first, query="Who moved where?")
    assert [r.text for r in from_first.results] == [ALICE_FACT]

    from_second = await client.arecall(bank_id=second, query="Who moved where?")
    assert [r.text for r in from_second.results] == [BOB_FACT]


async def test_updating_a_shared_document_id_leaves_the_neighbour_untouched(client, llm, colliding_banks, settled):
    """`update` is its own statement and needs its own predicate. Re-retaining
    over the shared id in one bank must not rewrite the other's copy."""
    first, second = colliding_banks

    llm.on_step("extract_facts", contains="Munich").returns(
        extracted(fact("Alice moved to Munich", who="Alice", entities=["Alice", "Munich"]))
    )
    await client.aretain(
        bank_id=first, content="Alice moved to Munich.", document_id=SHARED_DOCUMENT_ID, update_mode="replace"
    )
    await settled(first)

    assert await _fact_texts(client, first) == ["Alice moved to Munich | Involving: Alice"]
    assert await _fact_texts(client, second) == [BOB_FACT]
    assert (await client.documents.get_document(second, SHARED_DOCUMENT_ID)).original_text == "Bob moved to Lisbon."


async def test_deleting_a_shared_document_id_leaves_the_neighbour_untouched(client, colliding_banks, settled):
    """The unrecoverable one."""
    first, second = colliding_banks

    await client.documents.delete_document(first, SHARED_DOCUMENT_ID)
    await settled(first)

    assert await _fact_texts(client, first) == []
    assert await _fact_texts(client, second) == [BOB_FACT]
    assert (await client.documents.get_document(second, SHARED_DOCUMENT_ID)).original_text == "Bob moved to Lisbon."


async def test_deleting_a_whole_bank_leaves_its_neighbour_intact(client, colliding_banks, settled):
    first, second = colliding_banks

    await client.banks.delete_bank(first)

    assert await _fact_texts(client, second) == [BOB_FACT]
