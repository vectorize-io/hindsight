"""A provider outage fails the write; it never half-applies it.

Retain leans on two third parties per chunk — the model that extracts facts and
the backend that embeds them — and either can be down, rate-limited or out of
quota. A failed retain is fine: the caller sees it and tries again. What must not
happen is a failed retain that still changed the bank. An update that dies mid-way
and leaves the document with its old memories deleted and its new ones never
written looks, from outside, like a document that says nothing at all — and
nothing reports it, because the operation that did it is marked failed (#5589).

Each story here updates a document across two chunks: one that yields a fact and
so reaches the embeddings backend, one that yields nothing and never does. That
split is the shape of the bug: during an outage the empty chunk still succeeds,
and its success was enough to commit the document.
"""

from __future__ import annotations

import asyncio

import pytest
from hindsight_client_api.exceptions import ServiceException

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

DOCUMENT_ID = "profile-alice"
CHUNK_SIZE = 200

ORIGINAL = "Alice moved to Berlin and plays cello."
# Two paragraphs, together over the chunk size: the chunker cuts between them.
FILLER = "The afternoon drizzle kept on all day long, the kettle sang in the next room, and the radio murmured about traffic on roads nobody in the house would be driving on before the weekend."
REVISED = f"Alice moved to Lisbon and plays cello.\n\n{FILLER}"

BERLIN = "Alice moved to Berlin | Involving: Alice"
LISBON = "Alice moved to Lisbon | Involving: Alice"
CELLO = "Alice plays cello | Involving: Alice"


async def _fact_texts(client, bank: str) -> list[str]:
    memories = await client.memory.list_memories(bank, limit=100)
    return sorted(item.text for item in memories.items if item.state == "valid")


@pytest.fixture
async def chunked_bank(client, llm, bank_id):
    await client.acreate_bank(bank_id=bank_id, name="Alice")
    await client.banks.update_bank_config(bank_id, {"updates": {"retain_chunk_size": CHUNK_SIZE}})
    llm.on_step("extract_facts", contains="drizzle").returns(extracted())
    llm.on_step("consolidate").returns(consolidation())
    return bank_id


def _extracts_the_revision(llm) -> None:
    llm.on_step("extract_facts", contains="Lisbon").returns(
        extracted(
            fact("Alice moved to Lisbon", who="Alice", entities=["Alice", "Lisbon"]),
            fact("Alice plays cello", who="Alice", entities=["Alice", "cello"]),
        )
    )


@pytest.fixture
async def original(client, llm, chunked_bank, settled):
    """The document as it stood before the outage."""
    llm.on_step("extract_facts", contains="Berlin").returns(
        extracted(
            fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice plays cello", who="Alice", entities=["Alice", "cello"]),
        )
    )
    await client.aretain(bank_id=chunked_bank, content=ORIGINAL, document_id=DOCUMENT_ID)
    await settled(chunked_bank)
    assert await _fact_texts(client, chunked_bank) == sorted([BERLIN, CELLO])
    return chunked_bank


async def _assert_still_the_original(client, bank: str) -> None:
    document = await client.documents.get_document(bank, DOCUMENT_ID)
    assert document.original_text == ORIGINAL, "the failed update overwrote the document's text"
    assert document.memory_unit_count == 2, "the failed update deleted the document's memories"
    assert await _fact_texts(client, bank) == sorted([BERLIN, CELLO])


async def _assert_the_revision_landed(client, bank: str) -> None:
    chunks = await client.documents.list_document_chunks(bank, DOCUMENT_ID)
    assert chunks.total == 2, "the revision was not split in two, so this story covered nothing"
    assert await _fact_texts(client, bank) == sorted([LISBON, CELLO])


async def test_an_embeddings_outage_during_an_update_keeps_the_previous_version(
    client, llm, stubs, original, settled
):
    _extracts_the_revision(llm)
    stubs.embeddings.fail_with_status = 503

    with pytest.raises(ServiceException):
        await client.aretain(bank_id=original, content=REVISED, document_id=DOCUMENT_ID)

    await _assert_still_the_original(client, original)

    # The backend comes back, and the same update now goes through whole.
    stubs.embeddings.fail_with_status = None
    await client.aretain(bank_id=original, content=REVISED, document_id=DOCUMENT_ID)
    await settled(original)
    await _assert_the_revision_landed(client, original)


async def test_a_model_outage_during_an_update_keeps_the_previous_version(client, llm, original, settled):
    llm.on_step("extract_facts", contains="Lisbon").fails_with_status(503)

    with pytest.raises(ServiceException):
        await client.aretain(bank_id=original, content=REVISED, document_id=DOCUMENT_ID)

    await _assert_still_the_original(client, original)

    llm.reset()
    _extracts_the_revision(llm)
    llm.on_step("extract_facts", contains="drizzle").returns(extracted())
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=original, content=REVISED, document_id=DOCUMENT_ID)
    await settled(original)
    await _assert_the_revision_landed(client, original)


async def test_a_first_retain_during_an_outage_leaves_no_document_behind(client, llm, stubs, chunked_bank, settled):
    """Nothing to preserve here, so the failure must leave nothing at all — and the
    retry after it must produce one copy of each fact, not one per attempt."""
    _extracts_the_revision(llm)
    stubs.embeddings.fail_with_status = 429

    with pytest.raises(ServiceException):
        await client.aretain(bank_id=chunked_bank, content=REVISED, document_id=DOCUMENT_ID)

    documents = await client.documents.list_documents(chunked_bank)
    assert documents.items == [], "a failed first retain left a document behind"
    assert await _fact_texts(client, chunked_bank) == []

    stubs.embeddings.fail_with_status = None
    await client.aretain(bank_id=chunked_bank, content=REVISED, document_id=DOCUMENT_ID)
    await settled(chunked_bank)
    await _assert_the_revision_landed(client, chunked_bank)


async def _failed_operation(client, bank: str):
    for _ in range(60):
        listing = await client.operations.list_operations(bank, status="failed", limit=100)
        if listing.operations:
            return listing.operations[0]
        await asyncio.sleep(1)
    raise AssertionError("the retain never reported its failure")


async def test_a_failed_async_update_keeps_the_previous_version_until_its_retry(client, llm, stubs, original):
    """The background path, where nobody is waiting on the response: the operation
    fails, the document stays as it was, and retrying that same operation once the
    backend is back applies the update whole."""
    _extracts_the_revision(llm)
    stubs.embeddings.fail_with_status = 503

    await client.aretain(bank_id=original, content=REVISED, document_id=DOCUMENT_ID, retain_async=True)
    failed = await _failed_operation(client, original)

    await _assert_still_the_original(client, original)

    stubs.embeddings.fail_with_status = None
    await client.operations.retry_operation(original, failed.id)

    # Polled on the outcome: `settled()` refuses a bank with a failed operation on
    # record, and this one keeps its failure record after the retry fixes it.
    for _ in range(60):
        if await _fact_texts(client, original) == sorted([LISBON, CELLO]):
            break
        await asyncio.sleep(1)
    await _assert_the_revision_landed(client, original)


async def _deferred_operation(client, bank: str):
    for _ in range(60):
        listing = await client.operations.list_operations(bank, status="pending", limit=100)
        deferred = [op for op in listing.operations if op.next_retry_at is not None]
        if deferred:
            return deferred[0]
        await asyncio.sleep(1)
    raise AssertionError("the retain was never deferred to the provider's quota reset")


async def test_a_retain_deferred_to_a_quota_reset_runs_now_on_retry(client, llm, chunked_bank, settled):
    """A provider out of quota says when it resets, and the worker parks the retain
    until then. Providers overstate that window — days out for a quota that is back
    in hours (#5684) — so retrying a parked operation runs it now, not at the reset."""
    llm.on_step("extract_facts", contains="Lisbon").fails_with_status(429, retry_after_seconds=3 * 86400)

    await client.aretain(bank_id=chunked_bank, content=REVISED, document_id=DOCUMENT_ID, retain_async=True)
    deferred = await _deferred_operation(client, chunked_bank)
    assert await _fact_texts(client, chunked_bank) == []

    llm.reset()
    _extracts_the_revision(llm)
    llm.on_step("extract_facts", contains="drizzle").returns(extracted())
    llm.on_step("consolidate").returns(consolidation())
    await client.operations.retry_operation(chunked_bank, deferred.id)

    await settled(chunked_bank)
    await _assert_the_revision_landed(client, chunked_bank)
