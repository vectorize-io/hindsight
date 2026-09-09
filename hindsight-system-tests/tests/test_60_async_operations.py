"""An async retain hands back a receipt, and replaying the receipt is free.

`retain_async=True` accepts the work and returns immediately, which moves the
whole burden of "did it land?" onto the operation record. So the record has to
be complete enough to answer that on its own — a status that reaches a terminal
state, and a result that says what was actually stored.

The idempotency half is where the money is. A caller retries when a request
times out, when a network blips, when a buffered turn is replayed hours later —
and it cannot tell a lost request from a slow one. Without a caller-supplied
`operation_id` the safe retry does not exist: retrying risks duplicating the
memory, and not retrying risks losing it. With one, the second submission joins
the first instead of doing the work twice.
"""

from __future__ import annotations

import uuid

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

CONTENT = "Alice moved to Berlin."
BERLIN = "Alice moved to Berlin | Involving: Alice"


@pytest.fixture(autouse=True)
def _extraction(llm):
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())


async def test_an_async_retain_returns_an_operation_to_follow(client, bank_id, settled):
    """Accepted, not done. The receipt is the only handle the caller gets."""
    response = await client.aretain(bank_id=bank_id, content=CONTENT, retain_async=True)

    assert response.success is True
    assert response.var_async is True
    assert response.operation_id

    await settled(bank_id)
    status = await client.operations.get_operation_status(bank_id, response.operation_id)
    assert status.status == "completed"
    assert status.completed_at is not None
    assert status.error_message is None


async def test_the_operation_says_what_it_actually_stored(client, bank_id, settled):
    """A status of "completed" alone cannot distinguish work done from work
    skipped. The result metadata is what makes the receipt auditable."""
    response = await client.aretain(bank_id=bank_id, content=CONTENT, retain_async=True)
    await settled(bank_id)

    status = await client.operations.get_operation_status(bank_id, response.operation_id)
    assert status.result_metadata["items_count"] == 1
    assert status.result_metadata["unit_ids_count"] == 1
    assert status.result_metadata["extraction_errors_count"] == 0


async def test_the_work_really_happened(client, bank_id, settled):
    """The receipt is not the point; the memory is."""
    await client.aretain(bank_id=bank_id, content=CONTENT, retain_async=True)
    await settled(bank_id)

    memories = await client.memory.list_memories(bank_id, limit=100)
    assert [m.text for m in memories.items] == [BERLIN]


async def test_replaying_an_operation_id_does_not_store_it_twice(client, bank_id, settled):
    """The retry a caller cannot avoid making.

    Same id, same content, submitted again after the first completed. One memory
    must exist afterwards — not two — or every timeout in production quietly
    duplicates a memory.
    """
    operation_id = str(uuid.uuid4())

    first = await client.aretain(bank_id=bank_id, content=CONTENT, retain_async=True, operation_id=operation_id)
    await settled(bank_id)
    second = await client.aretain(bank_id=bank_id, content=CONTENT, retain_async=True, operation_id=operation_id)
    await settled(bank_id)

    assert first.operation_id == second.operation_id == operation_id

    memories = await client.memory.list_memories(bank_id, limit=100)
    assert [m.text for m in memories.items] == [BERLIN]


async def test_two_different_ids_are_two_different_operations(client, bank_id, settled):
    """The other direction: deduplication must key on the id the caller chose,
    not on the content. Two deliberate retains of the same sentence are a real
    thing to want, and collapsing them would be its own silent data loss."""
    first = await client.aretain(bank_id=bank_id, content=CONTENT, retain_async=True, operation_id=str(uuid.uuid4()))
    second = await client.aretain(bank_id=bank_id, content=CONTENT, retain_async=True, operation_id=str(uuid.uuid4()))
    await settled(bank_id)

    assert first.operation_id != second.operation_id
    for operation_id in (first.operation_id, second.operation_id):
        status = await client.operations.get_operation_status(bank_id, operation_id)
        assert status.status == "completed"


async def test_operations_are_listed_for_the_bank(client, bank_id, settled):
    """Whoever is debugging a stuck ingestion needs to see the queue without
    having kept every receipt."""
    response = await client.aretain(bank_id=bank_id, content=CONTENT, retain_async=True)
    await settled(bank_id)

    listing = await client.operations.list_operations(bank_id, status="completed", limit=100)
    assert response.operation_id in [op.id for op in listing.operations]
