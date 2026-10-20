"""A failed re-retain on a store-owned bank keeps the document it failed to replace.

Driven through the real engine against ``InMemoryMemories``, the in-repo store that owns its
memories, because the bug lived in a branch only a store-owned bank takes: the Postgres suite runs
the sibling branch and passed throughout.
"""

import uuid
from collections.abc import Iterator

import pytest

from hindsight_api.engine.memories import set_memories
from hindsight_api.engine.retain import orchestrator
from tests.test_memories_extension import InMemoryMemories


@pytest.fixture
def store() -> Iterator[InMemoryMemories]:
    store = InMemoryMemories({})
    set_memories(store)
    yield store
    set_memories(None)


@pytest.mark.asyncio
async def test_failed_shared_document_reretain_keeps_the_document_text(memory, request_context, store, monkeypatch):
    """Items sharing a document_id take the per-document group path, which used to open no retain
    session. Without one the document record was written before extraction, so a re-retain that
    then failed kept its old memories but replaced the document's text with the failed version."""
    bank_id = f"so-parity-{uuid.uuid4().hex[:8]}"
    document_id = "doc"
    try:
        await memory.retain_async(
            bank_id=bank_id,
            content="Alice works at Google.",
            document_id=document_id,
            request_context=request_context,
        )
        before = await memory.get_document(document_id, bank_id, request_context=request_context)

        async def failing_extract_and_embed(*_args, **_kwargs):
            raise RuntimeError("embeddings quota exceeded")

        monkeypatch.setattr(orchestrator, "_extract_and_embed", failing_extract_and_embed)
        with pytest.raises(RuntimeError, match="embeddings quota exceeded"):
            await memory.retain_batch_async(
                bank_id=bank_id,
                contents=[
                    {"content": "Carol works at Apple.", "document_id": document_id},
                    {"content": "Dave works at Meta.", "document_id": document_id},
                ],
                request_context=request_context,
            )

        after = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert after["original_text"] == before["original_text"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_a_group_cancelled_mid_run_stops_the_remaining_groups(memory, request_context, store, monkeypatch):
    """A bank deleted while one document's group ran must not start the next document's."""
    from hindsight_api.engine.memory_engine import _RetainExecutionResult
    from hindsight_api.engine.response_models import TokenUsage

    ran: list[str | None] = []

    async def cancelled_execution(**kwargs) -> _RetainExecutionResult:
        ran.append(kwargs["document_id"])
        return _RetainExecutionResult(unit_ids=[[]], usage=TokenUsage(), processed_content_tokens=0, cancelled=True)

    monkeypatch.setattr(memory, "_run_retain_execution", cancelled_execution)
    bank_id = f"so-parity-{uuid.uuid4().hex[:8]}"
    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "Alice works at Google.", "document_id": "a"},
                {"content": "Alice lives in Paris.", "document_id": "a"},
                {"content": "Bob works at Microsoft.", "document_id": "b"},
            ],
            request_context=request_context,
        )
        assert ran == ["a"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
