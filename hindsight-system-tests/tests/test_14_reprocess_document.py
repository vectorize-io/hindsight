"""Reprocessing a document actually re-extracts it.

Reprocess exists for the case where the *pipeline* changed and the content did
not — a better prompt, a different model, a fixed bug. Its whole value is
re-reading stored text and replacing what was derived from it.

Which makes its failure mode uniquely hard to see: a reprocess that quietly does
nothing returns success, completes its operation, and leaves a bank that looks
exactly as it should. Nobody finds out until they notice the improvement never
arrived.

The test therefore changes what the extraction step answers *between* the
original retain and the reprocess. Same stored text, different result — so the
only way the new facts can appear is if the document was genuinely re-read. A
no-op reprocess leaves the old facts and fails here.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

DOCUMENT_ID = "profile-alice"
CONTENT = "Alice moved to Berlin and plays cello."

ORIGINAL_FACT = "Alice moved to Berlin | Involving: Alice"
IMPROVED_FACT = "Alice relocated to Berlin, Germany | Involving: Alice"


async def _fact_texts(client, bank: str) -> list[str]:
    memories = await client.memory.list_memories(bank, limit=100)
    return sorted(item.text for item in memories.items if item.state == "valid")


async def test_reprocessing_picks_up_a_changed_extraction(client, llm, bank_id, settled):
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content=CONTENT, document_id=DOCUMENT_ID)
    await settled(bank_id)
    assert await _fact_texts(client, bank_id) == [ORIGINAL_FACT]

    # The pipeline "improves" while the document sits unchanged. Resetting the
    # rulebook is what makes this a real test: the new rule is the *only* rule,
    # so old facts cannot be reproduced by accident.
    llm.reset()
    llm.on_step("extract_facts").returns(
        extracted(
            fact(
                "Alice relocated to Berlin, Germany",
                where="Berlin",
                who="Alice",
                entities=["Alice", "Berlin", "Germany"],
            )
        )
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.documents.reprocess_document(bank_id, DOCUMENT_ID)
    await settled(bank_id)

    assert await _fact_texts(client, bank_id) == [IMPROVED_FACT], (
        "the document was not re-extracted — a reprocess that reports success and changes nothing"
    )


async def test_reprocessing_does_not_change_the_stored_text(client, llm, bank_id, settled):
    """Reprocess re-reads; it does not rewrite. The document is the input, and an
    input that drifts on every re-extraction is not reproducible."""
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content=CONTENT, document_id=DOCUMENT_ID)
    await settled(bank_id)

    await client.documents.reprocess_document(bank_id, DOCUMENT_ID)
    await settled(bank_id)

    document = await client.documents.get_document(bank_id, DOCUMENT_ID)
    assert document.original_text == CONTENT
    assert document.id == DOCUMENT_ID


async def test_reprocessing_reports_an_operation_to_wait_on(client, llm, bank_id, settled):
    """The work is asynchronous, so the response has to hand back something a
    caller can follow — otherwise "success" only means "accepted"."""
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content=CONTENT, document_id=DOCUMENT_ID)
    await settled(bank_id)

    result = await client.documents.reprocess_document(bank_id, DOCUMENT_ID)

    assert result.success is True
    assert result.operation_id
    assert result.items_count == 1
    await settled(bank_id)
