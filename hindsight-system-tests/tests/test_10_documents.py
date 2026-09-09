"""A retain is a document, and the document stays addressable afterwards.

The facts are what recall returns, but the document is what a caller *manages* —
it is the handle for updating, reprocessing and deleting. So the identity a
retain establishes has to survive: the id you supplied, the text you sent, the
chunks it was split into, and the count of what was extracted from it.

`document_id` is caller-supplied and only unique *within* a bank, which is why
every statement touching it needs a bank predicate — the class of bug behind
#3429. Story 13 covers that directly; this one establishes what a well-formed
document looks like so the later stories have something to contradict.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

CONTENT = "Alice moved to Berlin and plays cello."
DOCUMENT_ID = "profile-alice"


@pytest.fixture
async def bank_with_document(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice plays cello", who="Alice", entities=["Alice", "cello"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content=CONTENT, document_id=DOCUMENT_ID)
    await settled(bank_id)
    return bank_id


async def test_the_document_keeps_the_id_and_text_it_was_given(client, bank_with_document):
    document = await client.documents.get_document(bank_with_document, DOCUMENT_ID)

    assert document.id == DOCUMENT_ID
    assert document.bank_id == bank_with_document
    # The original text is kept verbatim — this is what a reprocess re-extracts
    # from, so a document that stored a transformed or truncated copy would
    # silently change what a later re-extraction can find.
    assert document.original_text == CONTENT
    assert document.memory_unit_count == 2
    assert document.nodes_by_fact_type == {"world": 2, "experience": 0, "observation": 0}


async def test_the_document_appears_in_the_bank_listing(client, bank_with_document):
    """Typed rows, like the single fetch.

    The two used to disagree — `get_document` returned a model and
    `list_documents` returned dicts, so `items[0].id` raised and a field renamed
    server-side was a compile error on one path and a runtime KeyError on the
    other (#4218). Attribute access here is the assertion.
    """
    listing = await client.documents.list_documents(bank_with_document)

    assert listing.total == 1
    assert [d.id for d in listing.items] == [DOCUMENT_ID]
    assert listing.items[0].memory_unit_count == 2


async def test_the_chunk_is_addressable_by_its_composite_id(client, bank_with_document):
    """Chunk ids are `{bank_id}_{document_id}_{index}`. The bank is baked into the
    id, which is what makes a chunk id globally unique — and therefore safe to
    query without a separate bank predicate, unlike `document_id` itself."""
    chunks = await client.documents.list_document_chunks(bank_with_document, DOCUMENT_ID)

    assert chunks.total == 1
    chunk = chunks.items[0]
    assert chunk.chunk_id == f"{bank_with_document}_{DOCUMENT_ID}_0"
    assert chunk.chunk_index == 0
    assert chunk.chunk_text == CONTENT


async def test_every_fact_points_back_at_the_document_it_came_from(client, bank_with_document):
    """Provenance. A fact that cannot name its document cannot be re-extracted,
    invalidated with it, or deleted with it."""
    memories = await client.memory.list_memories(bank_with_document, limit=100)

    assert memories.total == 2
    for item in memories.items:
        assert item.document_id == DOCUMENT_ID
        assert item.chunk_id == f"{bank_with_document}_{DOCUMENT_ID}_0"
