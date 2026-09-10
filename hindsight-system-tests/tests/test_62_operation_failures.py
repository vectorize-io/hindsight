"""A failed operation says what went wrong, and can be run again.

Async work fails — a provider outage, a model returning nonsense, a timeout. What
separates a recoverable system from a lossy one is what the record looks like
afterwards: a terminal status a caller can act on, the error kept rather than
swallowed, and a way to try again without re-submitting the original request and
risking a duplicate.

The batch story is the same argument one level up. A batch retain is a parent
operation over per-item children, and a caller who submitted fifty documents
needs to know *which* failed — a parent that reports only its own status turns
one bad item into an all-or-nothing retry.
"""

from __future__ import annotations

import asyncio

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

BERLIN = "Alice moved to Berlin | Involving: Alice"
CELLO = "Alice plays cello | Involving: Alice"


async def _fact_texts(client, bank: str) -> list[str]:
    memories = await client.memory.list_memories(bank, limit=100)
    return sorted(m.text for m in memories.items)


async def _first_failed(client, bank: str):
    """Wait for a failed operation to appear, and return it."""
    for _ in range(60):
        listing = await client.operations.list_operations(bank, status="failed", limit=100)
        if listing.operations:
            return listing.operations[0]
        await asyncio.sleep(1)
    raise AssertionError("no operation reported a failure")


async def _wait_for_facts(client, bank: str, expected: list[str]) -> list[str]:
    """Poll for an outcome rather than for an idle bank.

    `settled()` refuses to return while *any* operation in the bank is failed,
    which is the right default everywhere else and wrong here: this story creates
    a failure on purpose, and that record survives the retry that fixes it.
    """
    texts: list[str] = []
    for _ in range(60):
        texts = await _fact_texts(client, bank)
        if texts == expected:
            return texts
        await asyncio.sleep(1)
    return texts


async def test_a_failed_retain_is_visible_with_its_error(client, llm, bank_id):
    """Extraction fails, and the operation record says so in words.

    An operation that ends `failed` with a null message is barely better than one
    that vanished: whoever is paged has to reproduce it to learn anything.
    """
    llm.on_step("extract_facts").returns_text("this is not the json you asked for")

    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.", retain_async=True)

    failed = await _first_failed(client, bank_id)

    assert failed.error_message, "the failure carries no explanation"
    assert failed.status == "failed"


async def test_a_failed_operation_can_be_retried_after_the_cause_is_fixed(client, llm, bank_id):
    """The recovery path.

    Retry re-runs the stored work rather than asking the caller to re-submit —
    which matters because re-submitting is what creates duplicates. Here the
    model starts answering properly between the failure and the retry, standing
    in for a provider that came back.
    """
    llm.on_step("extract_facts").returns_text("this is not the json you asked for")
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.", retain_async=True)

    failed = await _first_failed(client, bank_id)

    llm.reset()
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.operations.retry_operation(bank_id, failed.id)

    assert await _wait_for_facts(client, bank_id, [BERLIN]) == [BERLIN]


async def test_a_batch_retain_reports_a_parent_over_its_items(client, llm, bank_id, settled):
    """One receipt for the submission, and a record per item underneath it — so
    a fifty-document batch with one bad item is diagnosable without re-running
    the forty-nine that worked."""
    llm.on_step("extract_facts", contains="Berlin").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("extract_facts", contains="cello").returns(
        extracted(fact("Alice plays cello", who="Alice", entities=["Alice", "cello"]))
    )
    llm.on_step("consolidate").returns(consolidation())

    response = await client.aretain_batch(
        bank_id=bank_id,
        items=[{"content": "Alice moved to Berlin."}, {"content": "Alice plays cello."}],
        retain_async=True,
    )
    await settled(bank_id)

    parent = await client.operations.get_operation_status(bank_id, response.operation_id)
    assert parent.status == "completed"
    assert parent.operation_type == "batch_retain"
    assert parent.result_metadata["is_parent"] is True
    assert parent.result_metadata["items_count"] == 2

    assert await _fact_texts(client, bank_id) == sorted([BERLIN, CELLO])


async def test_the_parent_is_excluded_when_a_caller_asks_for_leaves_only(client, llm, bank_id, settled):
    """A batch shows up twice in the operations list — once as the parent, once
    per item — which double-counts every progress bar built on it. The listing
    can drop the parents for exactly that reason."""
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())

    response = await client.aretain_batch(
        bank_id=bank_id, items=[{"content": "Alice moved to Berlin."}], retain_async=True
    )
    await settled(bank_id)

    everything = await client.operations.list_operations(bank_id, limit=100)
    leaves = await client.operations.list_operations(bank_id, limit=100, exclude_parents=True)

    assert response.operation_id in [op.id for op in everything.operations]
    assert response.operation_id not in [op.id for op in leaves.operations]
