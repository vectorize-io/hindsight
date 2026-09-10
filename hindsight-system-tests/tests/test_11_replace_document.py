"""Re-retaining a document replaces what it said — including the parts it stopped saying.

`update_mode="replace"` is the update path: same `document_id`, new content, and
the bank should end up describing the new content and nothing else. The half
that is easy to get wrong is removal. Adding and changing facts is obvious work;
noticing that a fact present in the *old* text is absent from the new one, and
retiring it, is a diff the pipeline has to do deliberately.

When it goes wrong the bank does not look broken — it looks *fuller*. Stale
facts keep answering recalls, and nothing in the response says they were
superseded. The last test here is the pure-deletion case: content that removes a
statement and adds nothing, which is the shape that exercises removal on its own.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

DOCUMENT_ID = "profile-alice"

ORIGINAL = "Alice moved to Berlin and plays cello."
REVISED = "Alice moved to Lisbon and plays cello."
TRIMMED = "Alice plays cello."

BERLIN = "Alice moved to Berlin | Involving: Alice"
LISBON = "Alice moved to Lisbon | Involving: Alice"
CELLO = "Alice plays cello | Involving: Alice"


async def _fact_texts(client, bank: str) -> list[str]:
    memories = await client.memory.list_memories(bank, limit=100)
    return sorted(item.text for item in memories.items if item.state == "valid")


@pytest.fixture
async def retained(client, llm, bank_id, settled):
    """Declare all three revisions up front, then retain the first."""
    llm.on_step("extract_facts", contains="Berlin").returns(
        extracted(
            fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice plays cello", who="Alice", entities=["Alice", "cello"]),
        )
    )
    llm.on_step("extract_facts", contains="Lisbon").returns(
        extracted(
            fact("Alice moved to Lisbon", who="Alice", entities=["Alice", "Lisbon"]),
            fact("Alice plays cello", who="Alice", entities=["Alice", "cello"]),
        )
    )
    # The trimmed revision: cello only, no move at all. Its content mentions no
    # city, so this substring reaches it and neither of the two above.
    llm.on_step("extract_facts", contains="Alice plays cello.").returns(
        extracted(fact("Alice plays cello", who="Alice", entities=["Alice", "cello"]))
    )
    llm.on_step("consolidate").returns(consolidation())

    async def _retain(content: str, **kwargs) -> None:
        await client.aretain(bank_id=bank_id, content=content, document_id=DOCUMENT_ID, **kwargs)
        await settled(bank_id)

    await _retain(ORIGINAL)
    return _retain


async def test_the_starting_point(client, bank_id, retained):
    assert await _fact_texts(client, bank_id) == sorted([BERLIN, CELLO])


async def test_a_replaced_fact_is_gone_and_its_successor_is_present(client, bank_id, retained):
    await retained(REVISED, update_mode="replace")

    assert await _fact_texts(client, bank_id) == sorted([LISBON, CELLO])


async def test_the_document_text_is_replaced_too(client, bank_id, retained):
    """The stored original text is what a reprocess re-reads, so it has to move
    with the facts. A document left holding the old text would silently undo the
    update the next time it was re-extracted."""
    await retained(REVISED, update_mode="replace")

    document = await client.documents.get_document(bank_id, DOCUMENT_ID)
    assert document.original_text == REVISED


async def test_a_fact_the_new_text_stopped_saying_is_removed(client, bank_id, retained):
    """Pure deletion: the revision drops the move entirely and adds nothing.

    Nothing new arrives to overwrite anything, so removal is the only mechanism
    that can produce the right answer. If the diff only ever adds, the bank keeps
    claiming Alice moved to Berlin forever.
    """
    await retained(TRIMMED, update_mode="replace")

    assert await _fact_texts(client, bank_id) == [CELLO]


async def test_replacing_does_not_multiply_the_document(client, bank_id, retained):
    """Same id, three revisions, one document — not three."""
    await retained(REVISED, update_mode="replace")
    await retained(TRIMMED, update_mode="replace")

    listing = await client.documents.list_documents(bank_id)
    assert listing.total == 1
    assert listing.items[0].id == DOCUMENT_ID
