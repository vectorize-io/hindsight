"""Deleting a document takes its facts with it — and nobody else's.

Two properties, and the second is the one that has actually gone wrong.

`document_id` is caller-supplied and unique only *within* a bank, so the same id
legitimately exists in every bank at once. A delete that filters on
`document_id` alone therefore reaches into every bank that happens to use that
id — the shape of #3429. It is invisible in a single-bank test, silent when it
happens, and unrecoverable afterwards.

So the second test here runs two banks that deliberately share an id, deletes
from one, and requires the other to be untouched. That collision is the normal
case in production, not an edge case: `session-1`, `profile`, `notes` are what
callers actually name things.
"""

from __future__ import annotations

import uuid

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

SHARED_ID = "profile"

BERLIN = "Alice moved to Berlin | Involving: Alice"
CELLO = "Alice plays cello | Involving: Alice"
LISBON = "Bob moved to Lisbon | Involving: Bob"


async def _fact_texts(client, bank: str) -> list[str]:
    memories = await client.memory.list_memories(bank, limit=100)
    return sorted(item.text for item in memories.items if item.state == "valid")


@pytest.fixture
async def two_documents(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts", contains="Berlin").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("extract_facts", contains="cello").returns(
        extracted(fact("Alice plays cello", who="Alice", entities=["Alice", "cello"]))
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.", document_id="d-move")
    await client.aretain(bank_id=bank_id, content="Alice plays cello.", document_id="d-music")
    await settled(bank_id)
    return bank_id


async def test_deleting_a_document_removes_only_its_own_facts(client, two_documents, settled):
    await client.documents.delete_document(two_documents, "d-move")
    await settled(two_documents)

    assert await _fact_texts(client, two_documents) == [CELLO]

    listing = await client.documents.list_documents(two_documents)
    assert [d.id for d in listing.items] == ["d-music"]


async def test_a_deleted_fact_stops_answering_recalls(client, two_documents, settled):
    """Removal from the read model, not just the listing — a fact that survives
    in the search index is still shaping answers."""
    await client.documents.delete_document(two_documents, "d-move")
    await settled(two_documents)

    response = await client.arecall(bank_id=two_documents, query="Where does Alice live?")

    assert BERLIN not in [r.text for r in response.results]


async def test_deleting_from_one_bank_leaves_the_same_id_alone_in_another(client, llm, settled):
    """The cross-bank guard (#3429).

    Two banks, the same `document_id`, different content. Deleting from the first
    must leave the second whole. A delete missing its `bank_id` predicate passes
    every other test in this file and silently empties the neighbour.
    """
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
        await client.aretain(bank_id=first, content="Alice moved to Berlin.", document_id=SHARED_ID)
        await client.aretain(bank_id=second, content="Bob moved to Lisbon.", document_id=SHARED_ID)
        await settled(first)
        await settled(second)

        assert await _fact_texts(client, first) == [BERLIN]
        assert await _fact_texts(client, second) == [LISBON]

        await client.documents.delete_document(first, SHARED_ID)
        await settled(first)

        assert await _fact_texts(client, first) == []
        assert await _fact_texts(client, second) == [LISBON], "the neighbouring bank's document was collateral damage"
    finally:
        for bank in (first, second):
            await client.banks.delete_bank(bank)
