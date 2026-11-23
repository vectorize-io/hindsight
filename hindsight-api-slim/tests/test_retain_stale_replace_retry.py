"""An old replace retried after a newer one finished must not roll the document back (#6763)."""

import json
import uuid

import pytest

from hindsight_api.models import RequestContext


async def _queue_replace(memory, bank_id: str, document_id: str, body: str, age_seconds: int) -> dict:
    operation_id = uuid.uuid4()
    payload = {
        "type": "batch_retain",
        "operation_id": str(operation_id),
        "bank_id": bank_id,
        "contents": [{"content": body, "document_id": document_id}],
    }
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=RequestContext(internal=True))
    pool = await memory._get_backend()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO async_operations (operation_id, bank_id, operation_type, result_metadata, status, "
            "task_payload, serialization_key, created_at) "
            "VALUES ($1, $2, 'retain', '{}'::jsonb, 'pending', $3::jsonb, $4, now() - make_interval(secs => $5))",
            operation_id,
            bank_id,
            json.dumps(payload),
            document_id,
            float(age_seconds),
        )
    return payload


async def _complete(memory, operation_id: str) -> None:
    pool = await memory._get_backend()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE async_operations SET status = 'completed' WHERE operation_id = $1", uuid.UUID(operation_id)
        )


@pytest.mark.asyncio
async def test_an_older_replace_retried_after_a_newer_one_does_not_overwrite_it(memory):
    bank_id = f"test_stale_replace_{uuid.uuid4().hex[:8]}"
    document_id = "conversation"
    old_body = "Alice moved to Paris in 2020."
    new_body = "Alice moved to Paris in 2020. In 2024 she moved again, to Berlin, and started a bakery."

    older = await _queue_replace(memory, bank_id, document_id, old_body, age_seconds=3600)
    newer = await _queue_replace(memory, bank_id, document_id, new_body, age_seconds=0)

    # The older one failed and is waiting out its retry delay, so the newer one runs first.
    await memory._handle_batch_retain(newer)
    await _complete(memory, newer["operation_id"])
    # Then the older one's retry runs.
    await memory._handle_batch_retain(older)

    doc = await memory.get_document(document_id, bank_id, request_context=RequestContext(internal=True))
    assert doc is not None
    assert doc["original_text"] == new_body

    pool = await memory._get_backend()
    async with pool.acquire() as conn:
        metadata = await conn.fetchval(
            "SELECT result_metadata FROM async_operations WHERE operation_id = $1", uuid.UUID(older["operation_id"])
        )
    assert json.loads(metadata)["superseded_by"] == newer["operation_id"]


@pytest.mark.asyncio
async def test_an_older_replace_still_runs_when_nothing_newer_finished(memory):
    """The queue runs a document's replaces oldest first; each must still apply."""
    bank_id = f"test_stale_replace_{uuid.uuid4().hex[:8]}"
    document_id = "conversation"
    first = await _queue_replace(memory, bank_id, document_id, "Bob likes tea.", age_seconds=60)
    second = await _queue_replace(memory, bank_id, document_id, "Bob likes tea. Bob now prefers coffee.", age_seconds=0)

    await memory._handle_batch_retain(first)
    await _complete(memory, first["operation_id"])
    await memory._handle_batch_retain(second)

    doc = await memory.get_document(document_id, bank_id, request_context=RequestContext(internal=True))
    assert doc is not None
    assert doc["original_text"] == "Bob likes tea. Bob now prefers coffee."
