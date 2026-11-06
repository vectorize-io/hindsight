"""Per-bank vector index DDL on create/delete must survive a transient deadlock.

The test-api shard runs 8 pytest-xdist workers against one shared pg0 database
(``public`` schema), so every bank create/delete does index DDL on the same
``memory_units`` table other workers are writing. A fresh bank builds its
partial indexes with a plain ``CREATE INDEX`` inside the bank-create tx
(ShareLock), and ``delete_bank`` drops them (CONCURRENTLY, post-commit); both
can be chosen as a deadlock victim. These are the exact production paths that
flaked in CI — they must retry the transient deadlock, not surface it.

The deadlock is injected via monkeypatch (one-shot ``DeadlockDetectedError``)
so the retry path is exercised deterministically, without racing real workers.
"""

import asyncio
import uuid

import pytest
from asyncpg.exceptions import DeadlockDetectedError

from hindsight_api import RequestContext
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain import bank_utils


@pytest.mark.asyncio
async def test_bank_create_retries_transient_deadlock(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch
):
    """A deadlock during the lazy bank-create tx retries the whole tx."""
    backend = await memory._get_backend()
    real = bank_utils.get_or_create_bank_profile_on_conn
    calls = 0

    async def flaky(conn, bank_id, *, ops):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise DeadlockDetectedError("deadlock detected")
        return await real(conn, bank_id, ops=ops)

    monkeypatch.setattr(bank_utils, "get_or_create_bank_profile_on_conn", flaky)

    bank_id = f"test-deadlock-{uuid.uuid4().hex[:8]}"
    try:
        result = await bank_utils.get_or_create_bank_profile(backend, bank_id)
        assert calls == 2, "expected exactly one retry after the injected deadlock"
        assert result.created is True
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_bank_delete_retries_transient_deadlock(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch
):
    """A deadlock while dropping per-bank indexes on delete is retried."""
    bank_id = f"test-deadlock-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)

    backend = await memory._get_backend()
    real = backend.ops.drop_bank_vector_indexes
    calls = 0

    async def flaky(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise DeadlockDetectedError("deadlock detected")
        return await real(*args, **kwargs)

    monkeypatch.setattr(backend.ops, "drop_bank_vector_indexes", flaky)

    result = await memory.delete_bank(bank_id, request_context=request_context)
    assert calls == 2, "expected exactly one retry after the injected deadlock"
    assert result["bank_deleted"] is True


@pytest.mark.asyncio
async def test_metadata_only_re_retain_retries_transient_deadlock(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch
):
    """Re-retaining an unchanged document survives a deadlock in its relabel tx (#5752).

    Its queries plan on memory_units, taking AccessShare on every bank's vector
    index, so another bank's DROP INDEX CONCURRENTLY can pick it as the victim.
    """
    from hindsight_api.engine.retain import fact_storage

    bank_id = f"test-deadlock-{uuid.uuid4().hex[:8]}"
    item = {"content": "Alice moved to Berlin in 2021.", "document_id": "doc-1"}
    try:
        await memory.retain_batch_async(bank_id=bank_id, contents=[dict(item)], request_context=request_context)

        real = fact_storage.update_memory_units_metadata_and_tags
        calls = 0

        async def flaky(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise DeadlockDetectedError("deadlock detected")
            return await real(*args, **kwargs)

        monkeypatch.setattr(fact_storage, "update_memory_units_metadata_and_tags", flaky)
        await memory.retain_batch_async(bank_id=bank_id, contents=[dict(item)], request_context=request_context)
        assert calls == 2, "expected exactly one retry after the injected deadlock"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def _create_mental_model(memory: MemoryEngine, bank_id: str, rc: RequestContext) -> None:
    await memory.create_mental_model(bank_id, "Prefs", "what does the user prefer?", "Tea.", request_context=rc)


async def _create_knowledge_folder(memory: MemoryEngine, bank_id: str, rc: RequestContext) -> None:
    await memory.create_knowledge_folder(bank_id, "Notes", request_context=rc)


async def _create_knowledge_page(memory: MemoryEngine, bank_id: str, rc: RequestContext) -> None:
    await memory.create_knowledge_page(bank_id, "Prefs", "what does the user prefer?", "Tea.", request_context=rc)


@pytest.mark.asyncio
@pytest.mark.parametrize("create", [_create_mental_model, _create_knowledge_folder, _create_knowledge_page])
async def test_write_that_creates_its_bank_retries_transient_deadlock(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch, create
):
    """A write that lazily creates its bank on its own tx survives a deadlock in that create (#5830).

    The bank create builds the bank's vector indexes with a plain CREATE INDEX, whose ShareLock
    on memory_units can deadlock with another bank's CONCURRENTLY index build or drop.
    """
    real = bank_utils.create_bank_row_on_conn
    calls = 0

    async def flaky(conn, bank_id, *, ops):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise DeadlockDetectedError("deadlock detected")
        return await real(conn, bank_id, ops=ops)

    monkeypatch.setattr(bank_utils, "create_bank_row_on_conn", flaky)
    bank_id = f"test-deadlock-{uuid.uuid4().hex[:8]}"
    try:
        await create(memory, bank_id, request_context)
        assert calls == 2, "expected exactly one retry after the injected deadlock"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_bank_create_index_does_not_queue_behind_a_held_lock(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch
):
    """A bank create whose CREATE INDEX cannot get its lock gives up fast and retries (#5805).

    Another session holds SHARE UPDATE EXCLUSIVE on memory_units, as a CONCURRENTLY index
    build/drop does. Left to wait, the create's ShareLock request would queue every write to
    memory_units behind it; with the lock_timeout it fails, backs off, and succeeds once free.
    """
    from hindsight_api.engine.db import ops_postgresql
    from hindsight_api.engine.schema import fq_store_table

    if not isinstance(memory._backend.ops, ops_postgresql.PostgreSQLOps):
        pytest.skip("per-bank partial indexes are PostgreSQL-only")
    monkeypatch.setattr(ops_postgresql, "BANK_INDEX_LOCK_TIMEOUT", "100ms")
    backend = await memory._get_backend()
    bank_id = f"test-locktimeout-{uuid.uuid4().hex[:8]}"
    attempts = 0
    real = bank_utils.get_or_create_bank_profile_on_conn

    async def counting(conn, bank_id, *, ops):
        nonlocal attempts
        attempts += 1
        return await real(conn, bank_id, ops=ops)

    monkeypatch.setattr(bank_utils, "get_or_create_bank_profile_on_conn", counting)

    holder_ready = asyncio.Event()

    async def hold_lock():
        async with backend.acquire() as conn:
            async with conn.transaction():
                await conn.execute(f"LOCK TABLE {fq_store_table('memory_units')} IN SHARE UPDATE EXCLUSIVE MODE")
                holder_ready.set()
                await asyncio.sleep(0.4)

    holder = asyncio.create_task(hold_lock())
    try:
        await asyncio.wait_for(holder_ready.wait(), timeout=10)
        result = await bank_utils.get_or_create_bank_profile(backend, bank_id)
        await holder
        assert result.created is True
        assert attempts >= 2, "the first create should have timed out on the held lock and been retried"
    finally:
        holder.cancel()
        await memory.delete_bank(bank_id, request_context=request_context)
