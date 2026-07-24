"""Idempotent async retain via a caller-supplied operation_id.

A client that retries an async retain after a lost or timed-out acknowledgement
must not enqueue a second parent operation. Supplying the same ``operation_id``
returns the original operation and creates no new work; the parent primary key
is the concurrency authority, so no extra schema is involved.
"""

import asyncio
import uuid

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.engine.memory_engine import RetainOperationConflictError

# These tests submit async operations against the shared pool. Share the
# "worker_tests" xdist group with test_async_batch_retain / test_worker so a
# concurrently running poller cannot steal each other's pending rows.
pytestmark = pytest.mark.xdist_group("worker_tests")


@pytest_asyncio.fixture
async def pool(pg0_db_url):
    import asyncpg

    from hindsight_api.pg0 import resolve_database_url

    resolved_url = await resolve_database_url(pg0_db_url)
    p = await asyncpg.create_pool(resolved_url, min_size=1, max_size=5, command_timeout=30)
    yield p
    await p.close()


@pytest_asyncio.fixture
async def api_client(memory):
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def _count_batch_parents(pool, bank_id: str) -> int:
    return await pool.fetchval(
        "SELECT count(*) FROM async_operations WHERE bank_id = $1 AND operation_type = 'batch_retain'",
        bank_id,
    )


# --------------------------------------------------------------------------- #
# Engine-level behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_replay_returns_same_operation_and_no_new_work(memory, request_context, pool):
    """Re-submitting with the same operation_id returns the original, creating no new op."""
    bank_id = "test_retain_idem_replay"
    operation_id = str(uuid.uuid4())
    contents = [{"content": "Alice works at Google", "document_id": "doc1"}]

    first = await memory.submit_async_retain(
        bank_id=bank_id,
        contents=contents,
        request_context=request_context,
        operation_id=operation_id,
    )
    await asyncio.sleep(0.1)
    second = await memory.submit_async_retain(
        bank_id=bank_id,
        contents=contents,
        request_context=request_context,
        operation_id=operation_id,
    )

    assert first["operation_id"] == operation_id
    assert second["operation_id"] == operation_id
    assert second["items_count"] == 1
    # Exactly one parent operation exists — the retry created no duplicate.
    assert await _count_batch_parents(pool, bank_id) == 1


@pytest.mark.asyncio
async def test_no_operation_id_creates_distinct_operations(memory, request_context, pool):
    """Omitting operation_id keeps the legacy create-each-time behaviour."""
    bank_id = "test_retain_idem_legacy"
    contents = [{"content": "Bob went hiking", "document_id": "doc1"}]

    first = await memory.submit_async_retain(bank_id=bank_id, contents=contents, request_context=request_context)
    second = await memory.submit_async_retain(bank_id=bank_id, contents=contents, request_context=request_context)

    assert first["operation_id"] != second["operation_id"]
    assert await _count_batch_parents(pool, bank_id) == 2


@pytest.mark.asyncio
async def test_conflict_when_id_belongs_to_different_bank(memory, request_context):
    """An operation_id owned by another bank cannot be reused (global PK collision)."""
    operation_id = str(uuid.uuid4())
    contents = [{"content": "Shared id content", "document_id": "doc1"}]

    await memory.submit_async_retain(
        bank_id="test_retain_idem_bankA",
        contents=contents,
        request_context=request_context,
        operation_id=operation_id,
    )

    with pytest.raises(RetainOperationConflictError):
        await memory.submit_async_retain(
            bank_id="test_retain_idem_bankB",
            contents=contents,
            request_context=request_context,
            operation_id=operation_id,
        )


@pytest.mark.asyncio
async def test_replay_ignores_payload_differences(memory, request_context, pool):
    """A replay resolves purely by id; a differing payload still returns the original.

    Reusing an id you generated for different content is a client bug, and
    returning the original operation is the safe idempotent answer — no new
    work is enqueued.
    """
    bank_id = "test_retain_idem_payload"
    operation_id = str(uuid.uuid4())

    first = await memory.submit_async_retain(
        bank_id=bank_id,
        contents=[{"content": "original", "document_id": "doc1"}],
        request_context=request_context,
        operation_id=operation_id,
    )
    await asyncio.sleep(0.1)
    second = await memory.submit_async_retain(
        bank_id=bank_id,
        contents=[{"content": "totally different", "document_id": "doc2"}],
        request_context=request_context,
        operation_id=operation_id,
    )

    assert second["operation_id"] == first["operation_id"]
    assert await _count_batch_parents(pool, bank_id) == 1


@pytest.mark.asyncio
async def test_concurrent_submissions_with_same_id_create_one_operation(memory, request_context, pool):
    """Two simultaneous first submissions of the same id resolve to one operation."""
    bank_id = "test_retain_idem_concurrent"
    operation_id = str(uuid.uuid4())
    contents = [{"content": "Concurrent content", "document_id": "doc1"}]

    async def submit():
        return await memory.submit_async_retain(
            bank_id=bank_id,
            contents=contents,
            request_context=request_context,
            operation_id=operation_id,
        )

    results = await asyncio.gather(submit(), submit())

    assert {r["operation_id"] for r in results} == {operation_id}
    assert await _count_batch_parents(pool, bank_id) == 1


# --------------------------------------------------------------------------- #
# HTTP-level validation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_http_invalid_operation_id_is_rejected(api_client):
    bank_id = "test_retain_idem_http_invalid"
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [{"content": "hello", "document_id": "doc1"}],
            "async": True,
            "operation_id": "not-a-uuid",
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_http_replay_returns_same_operation_id(api_client):
    bank_id = "test_retain_idem_http_replay"
    operation_id = str(uuid.uuid4())
    payload = {
        "items": [{"content": "Alice works at Google", "document_id": "doc1"}],
        "async": True,
        "operation_id": operation_id,
    }

    first = await api_client.post(f"/v1/default/banks/{bank_id}/memories", json=payload)
    assert first.status_code == 200
    await asyncio.sleep(0.1)
    second = await api_client.post(f"/v1/default/banks/{bank_id}/memories", json=payload)
    assert second.status_code == 200

    assert first.json()["operation_id"] == operation_id
    assert second.json()["operation_id"] == operation_id
