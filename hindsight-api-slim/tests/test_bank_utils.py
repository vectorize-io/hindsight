import asyncio
from contextlib import asynccontextmanager

import pytest
from asyncpg.exceptions import DeadlockDetectedError

from hindsight_api.engine.retain import bank_utils


class _FailingIndexOps:
    async def create_bank_vector_indexes(self, *args, **kwargs) -> None:
        raise RuntimeError("simulated per-bank vector index DDL failure")


class _DeadlockOnceIndexOps:
    """Raises a deadlock on the first per-bank index DDL, then succeeds."""

    def __init__(self) -> None:
        self.calls = 0

    async def create_bank_vector_indexes(self, *args, **kwargs) -> None:
        self.calls += 1
        if self.calls == 1:
            raise DeadlockDetectedError("deadlock detected")


class _FakeTransaction:
    def __init__(self, conn: "_FakeConnection") -> None:
        self._conn = conn

    async def __aenter__(self) -> None:
        self._conn.in_transaction = True

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self._conn.committed_bank = self._conn.pending_bank
        self._conn.pending_bank = None
        self._conn.in_transaction = False


class _FakeConnection:
    def __init__(self) -> None:
        self.committed_bank: str | None = None
        self.pending_bank: str | None = None
        self.in_transaction = False

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction(self)

    async def fetchrow(self, query: str, bank_id: str):
        visible_bank = self.pending_bank if self.in_transaction else self.committed_bank
        if visible_bank != bank_id:
            return None
        return {
            "name": bank_id,
            "disposition": bank_utils.DEFAULT_DISPOSITION,
            "mission": "",
        }

    async def fetchval(self, query: str, bank_id: str, *args):
        if self.in_transaction:
            self.pending_bank = bank_id
        else:
            self.committed_bank = bank_id
        return bank_id


class _FakePool:
    def __init__(self, conn: _FakeConnection, ops=None) -> None:
        self.conn = conn
        self.ops = ops if ops is not None else _FailingIndexOps()


@pytest.mark.asyncio
async def test_lazy_bank_create_rolls_back_on_vector_index_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed per-bank index DDL must not leave an orphaned bank row."""
    conn = _FakeConnection()
    pool = _FakePool(conn)

    @asynccontextmanager
    async def acquire_without_transaction(*args, **kwargs):
        yield conn

    monkeypatch.setattr(bank_utils, "acquire_with_retry", acquire_without_transaction)

    with pytest.raises(RuntimeError, match="simulated per-bank vector index DDL failure"):
        await bank_utils.get_or_create_bank_profile(pool, "atomicity-test-bank")

    profile = await bank_utils.get_bank_profile_if_exists(pool, "atomicity-test-bank")
    assert profile is None, "bank row should roll back when per-bank vector index creation fails"


@pytest.mark.asyncio
async def test_get_or_create_bank_profile_retries_on_deadlock(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deadlock during per-bank index DDL is retried on a fresh transaction.

    Concurrent bank creation issues CREATE INDEX against the shared memory_units
    table, which can deadlock (asyncpg DeadlockDetectedError). The pool-owning
    get_or_create_bank_profile must retry rather than surface the deadlock.
    """
    conn = _FakeConnection()
    ops = _DeadlockOnceIndexOps()
    pool = _FakePool(conn, ops=ops)

    @asynccontextmanager
    async def acquire(*args, **kwargs):
        yield conn

    monkeypatch.setattr(bank_utils, "acquire_with_retry", acquire)

    async def _noop_sleep(*_args, **_kwargs) -> None:
        return None

    # Keep the retry backoff instant so the test doesn't actually sleep.
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

    result = await bank_utils.get_or_create_bank_profile(pool, "deadlock-retry-bank")

    assert result.created is True
    assert ops.calls == 2, "index DDL should be attempted twice (deadlock, then success)"
    profile = await bank_utils.get_bank_profile_if_exists(pool, "deadlock-retry-bank")
    assert profile is not None, "bank must exist after the retry succeeds"
