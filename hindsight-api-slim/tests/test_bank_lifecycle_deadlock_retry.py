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
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

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
