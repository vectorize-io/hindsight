import asyncio
from contextlib import asynccontextmanager

import pytest
from asyncpg.exceptions import DeadlockDetectedError

from hindsight_api.engine.retain import bank_utils


class _BankOps:
    """Dialect ops stub.

    Bank creation issues the per-bank index DDL again at the default threshold
    (0 = off), so the stub has to answer for it and record that it was asked.
    """

    def __init__(self) -> None:
        self.create_calls: list[str] = []

    async def create_bank_vector_indexes(self, conn, table, bank_id, internal_id, index_clause, fact_types) -> None:
        self.create_calls.append(bank_id)


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
    def __init__(self, raise_on_insert: BaseException | None = None) -> None:
        self.committed_bank: str | None = None
        self.pending_bank: str | None = None
        self.in_transaction = False
        self.insert_calls = 0
        # Raised by the first INSERT only, then cleared, so a retry succeeds.
        self._raise_on_insert = raise_on_insert

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
        self.insert_calls += 1
        if self._raise_on_insert is not None:
            error, self._raise_on_insert = self._raise_on_insert, None
            raise error
        if self.in_transaction:
            self.pending_bank = bank_id
        else:
            self.committed_bank = bank_id
        return bank_id


class _FakePool:
    def __init__(self, conn: _FakeConnection, ops=None) -> None:
        self.conn = conn
        self.ops = ops if ops is not None else _BankOps()


@pytest.mark.asyncio
async def test_lazy_bank_create_rolls_back_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure inside the bank-create transaction must not leave an orphaned bank row."""
    conn = _FakeConnection(raise_on_insert=RuntimeError("simulated bank insert failure"))
    pool = _FakePool(conn)

    @asynccontextmanager
    async def acquire_without_transaction(*args, **kwargs):
        yield conn

    monkeypatch.setattr(bank_utils, "acquire_with_retry", acquire_without_transaction)

    with pytest.raises(RuntimeError, match="simulated bank insert failure"):
        await bank_utils.get_or_create_bank_profile(pool, "atomicity-test-bank")

    profile = await bank_utils.get_bank_profile_if_exists(pool, "atomicity-test-bank")
    assert profile is None, "bank row should roll back when the create transaction fails"


@pytest.mark.asyncio
async def test_get_or_create_bank_profile_retries_on_deadlock(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deadlock inside the create transaction is retried on a fresh one.

    The retry guards two things: the per-bank CREATE INDEX that runs inline here
    at the default threshold, which takes a ShareLock on the shared memory_units
    table, and the plain bank-row insert, which can lose a deadlock to a
    concurrent writer touching the same row. The body is idempotent
    (INSERT ... ON CONFLICT DO NOTHING + CREATE INDEX IF NOT EXISTS), so it must
    retry rather than surface the deadlock to the caller.
    """
    conn = _FakeConnection(raise_on_insert=DeadlockDetectedError("deadlock detected"))
    pool = _FakePool(conn)

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
    assert conn.insert_calls == 2, "the insert should be attempted twice (deadlock, then success)"
    profile = await bank_utils.get_bank_profile_if_exists(pool, "deadlock-retry-bank")
    assert profile is not None, "bank must exist after the retry succeeds"
