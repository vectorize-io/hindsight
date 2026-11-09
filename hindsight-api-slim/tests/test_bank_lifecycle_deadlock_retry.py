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

import asyncpg
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
@pytest.mark.memory_backend_incompatible
async def test_bank_create_waiting_on_index_lock_does_not_stall_other_banks(
    memory: MemoryEngine, request_context: RequestContext, pg0_db_url: str
):
    """A new bank whose index DDL waits must not hold every other bank's writes up with it (#5805).

    Real locks, no mocks: an open transaction keeps another bank's DROP INDEX CONCURRENTLY
    waiting (it holds a lock that conflicts with CREATE INDEX). A plain CREATE INDEX parked
    behind it would queue every memory_units write in the database; the create must step
    out of the queue between tries, and still finish once the holder ends.
    """
    backend = await memory._get_backend()
    tag = uuid.uuid4().hex[:8]
    other_bank, new_bank = f"test-lockq-other-{tag}", f"test-lockq-new-{tag}"
    holder, dropper, observer = [await asyncpg.connect(pg0_db_url) for _ in range(3)]
    tx = holder.transaction()
    create = drop = None
    try:
        await observer.execute(f"CREATE INDEX idx_lockq_{tag} ON memory_units(bank_id) WHERE bank_id = '{other_bank}'")
        await observer.execute("INSERT INTO banks(bank_id) VALUES($1)", other_bank)
        await tx.start()
        await holder.fetchval("SELECT count(*) FROM memory_units WHERE bank_id = $1", other_bank)
        drop = asyncio.create_task(dropper.execute(f"DROP INDEX CONCURRENTLY idx_lockq_{tag}"))
        await _wait_until_lock_wait(observer, f"DROP INDEX CONCURRENTLY idx_lockq_{tag}")

        create = asyncio.create_task(bank_utils.get_or_create_bank_profile(backend, new_bank))
        await _wait_until_lock_wait(observer, f"WHERE fact_type = 'world' AND bank_id = '{new_bank}'")
        # Another bank's write goes through while the create is still waiting.
        await asyncio.wait_for(
            observer.execute("INSERT INTO memory_units(bank_id, text, fact_type) VALUES($1, 'x', 'world')", other_bank),
            15,
        )
        assert not create.done()

        await tx.rollback()
        await asyncio.wait_for(drop, 15)
        result = await asyncio.wait_for(create, 30)
        assert result.created is True
    finally:
        await holder.close()  # ends its transaction, if the test did not get to
        for task in (create, drop):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(t for t in (create, drop) if t is not None), return_exceptions=True)
        await observer.execute(f"DROP INDEX IF EXISTS idx_lockq_{tag}")
        await observer.execute("DELETE FROM memory_units WHERE bank_id = $1", other_bank)
        await observer.execute("DELETE FROM banks WHERE bank_id = $1", other_bank)
        for conn in (dropper, observer):
            await conn.close()
        await memory.delete_bank(new_bank, request_context=request_context)


async def _wait_until_lock_wait(observer: asyncpg.Connection, query_contains: str) -> None:
    async with asyncio.timeout(30):
        while not await observer.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_stat_activity"
            " WHERE wait_event_type = 'Lock' AND position($1 in query) > 0)",
            query_contains,
        ):
            await asyncio.sleep(0.01)
