"""Tests for hindsight_api.engine.db_utils.

Regression coverage for the single-yield contract of ``acquire_with_retry`` —
historically a retry loop wrapped the ``yield`` and caused every retryable
user-code exception to surface as
``RuntimeError("generator didn't stop after athrow()")``, masking the real
cause and producing identical failed-op rows in production (see the 1,934
failed consolidations on ``shurick-memory`` in May 2026).

Also covers the self-termination path: when a pool stops producing connections
and never starts again, ``acquire_with_retry`` ends the process so a supervisor
can replace it, since liveness is deliberately database-free and nothing else
would.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

import pytest

from hindsight_api.config import ENV_DB_UNAVAILABLE_EXIT_SECONDS, clear_config_cache
from hindsight_api.engine import db_utils
from hindsight_api.engine.db_utils import _backoff_delay, acquire_with_retry


@pytest.fixture
def exit_window(monkeypatch):
    """Capture ``os._exit`` and set the unavailability window from the env.

    Returns a callable taking the window in seconds and how long ago the last
    successful acquire was, and yielding the list that records exit codes.
    """
    exits: list[int] = []
    monkeypatch.setattr(db_utils.os, "_exit", lambda code: exits.append(code))
    monkeypatch.setattr(db_utils, "_backoff_delay", lambda *a, **k: 0.0)

    def configure(window_s: float, last_success_age_s: float) -> list[int]:
        monkeypatch.setenv(ENV_DB_UNAVAILABLE_EXIT_SECONDS, str(window_s))
        clear_config_cache()
        monkeypatch.setattr(db_utils, "_last_acquire_success", time.monotonic() - last_success_age_s)
        return exits

    yield configure
    # The cache now holds a config built from the patched env; drop it so the
    # next test rebuilds from the real environment.
    clear_config_cache()


class _FakeConnection:
    """Stand-in for a DatabaseConnection that records release events."""

    def __init__(self) -> None:
        self.released = 0


class _FakeBackend:
    """Duck-typed DatabaseBackend that opts in via the ``_wraps_backend`` flag.

    ``acquire_with_retry`` accepts either a real ``DatabaseBackend`` subclass
    or any object with ``_wraps_backend = True``; the flag avoids having to
    stub the full abstract surface for unit tests.
    """

    _wraps_backend = True

    def __init__(self) -> None:
        self.acquired = 0
        self.last_conn: _FakeConnection | None = None

    @asynccontextmanager
    async def acquire(self):
        self.acquired += 1
        conn = _FakeConnection()
        self.last_conn = conn
        try:
            yield conn
        finally:
            conn.released += 1


@pytest.mark.asyncio
async def test_retryable_user_code_exception_propagates_unchanged():
    """A retryable exception inside ``async with`` must propagate as itself.

    Before the single-yield refactor, the retry loop around the ``yield``
    re-entered ``yield conn`` on the next iteration, violating
    ``@asynccontextmanager``'s contract and surfacing as
    ``RuntimeError("generator didn't stop after athrow()")`` — the symptom
    that broke consolidation on large banks.
    """

    backend = _FakeBackend()
    sentinel = asyncio.TimeoutError("query exceeded statement_timeout")

    with pytest.raises(asyncio.TimeoutError) as excinfo:
        async with acquire_with_retry(backend) as conn:
            assert isinstance(conn, _FakeConnection)
            raise sentinel

    # The original exception flows out — not a RuntimeError wrapper.
    assert excinfo.value is sentinel
    assert not isinstance(excinfo.value, RuntimeError)

    # Acquire was called exactly once — user-code failure must not retry.
    assert backend.acquired == 1
    assert backend.last_conn is not None
    assert backend.last_conn.released == 1, "connection must be released exactly once"


def test_backoff_delay_is_jittered_and_bounded():
    """Equal-jitter backoff stays in [ceil/2, ceil] and never exceeds max_delay.

    The jitter exists so concurrent deadlock retriers don't wake in lock-step
    and re-collide (see the entity-prune batch in run_graph_maintenance_job). It must
    still keep a floor (no hot-spin) and honour the max_delay cap once the
    exponential term saturates.
    """
    base, max_delay = 0.5, 5.0

    # Below saturation: ceil = base * 2**attempt, jitter within [ceil/2, ceil].
    for attempt in range(3):
        ceil = base * (2**attempt)
        samples = [_backoff_delay(attempt, base, max_delay) for _ in range(200)]
        assert all(ceil / 2 <= d <= ceil for d in samples)
        # Actually jittered — not a constant.
        assert len(set(samples)) > 1

    # At/after saturation the cap holds: every sample <= max_delay.
    saturated = [_backoff_delay(20, base, max_delay) for _ in range(200)]
    assert all(max_delay / 2 <= d <= max_delay for d in saturated)


class _FailingBackend:
    """Backend whose every acquire raises a retryable connection error."""

    _wraps_backend = True

    @asynccontextmanager
    async def acquire(self):
        raise ConnectionError("pool is not producing connections")
        yield  # pragma: no cover - unreachable, keeps this an async generator


class _SlowThenOkBackend:
    """Backend that fails a few acquires and then succeeds.

    Stands in for a database that is briefly unreachable — a failover, a
    restart — which must NOT be treated as a process that cannot recover.
    """

    _wraps_backend = True

    def __init__(self, failures_before_success: int) -> None:
        self.remaining_failures = failures_before_success

    @asynccontextmanager
    async def acquire(self):
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise ConnectionError("temporarily unreachable")
        yield _FakeConnection()


@pytest.mark.asyncio
async def test_exits_when_no_acquire_has_succeeded_for_the_whole_window(exit_window):
    """A pool that never produces a connection must end the process.

    Nothing else can: liveness is deliberately database-free, so a permanently
    unusable pool otherwise fails readiness forever while the process stays up.
    """
    exits = exit_window(600.0, 601.0)  # last success is older than the window

    with pytest.raises(ConnectionError):
        async with acquire_with_retry(_FailingBackend(), max_retries=1):
            pass  # pragma: no cover - acquire never succeeds

    assert exits == [1], "process should have exited once after the window elapsed"


@pytest.mark.asyncio
async def test_does_not_exit_while_still_inside_the_window(exit_window):
    """Failing acquires alone are not enough — the window must have elapsed.

    This is the case a database restart produces, and exiting there would turn
    a brief outage into a restart loop across every replica at once.
    """
    exits = exit_window(600.0, 5.0)

    with pytest.raises(ConnectionError):
        async with acquire_with_retry(_FailingBackend(), max_retries=1):
            pass  # pragma: no cover - acquire never succeeds

    assert exits == [], "must not exit while the database may still come back"


@pytest.mark.asyncio
async def test_a_successful_acquire_resets_the_window(exit_window):
    """One success clears the clock, so intermittent failures never accumulate.

    Without the reset, a process that fails an acquire now and then would
    eventually exit even though the database is plainly working.
    """
    exits = exit_window(600.0, 599.0)

    # Retries exhaust on the first two attempts, then it connects.
    async with acquire_with_retry(_SlowThenOkBackend(2), max_retries=3) as conn:
        assert conn is not None

    assert exits == []
    # The clock now reads from the success, not from the old timestamp.
    assert time.monotonic() - db_utils._last_acquire_success < 1.0

    # A later total failure therefore starts a fresh window rather than
    # tripping immediately on the pre-existing staleness.
    with pytest.raises(ConnectionError):
        async with acquire_with_retry(_FailingBackend(), max_retries=1):
            pass  # pragma: no cover - acquire never succeeds
    assert exits == []


@pytest.mark.asyncio
async def test_exit_can_be_disabled(exit_window):
    """Zero disables the behaviour outright, for operators who want it off."""
    exits = exit_window(0.0, 86400.0)

    with pytest.raises(ConnectionError):
        async with acquire_with_retry(_FailingBackend(), max_retries=1):
            pass  # pragma: no cover - acquire never succeeds

    assert exits == []


class _LegacyPool:
    """asyncpg.Pool stand-in — no ``_wraps_backend``, so it takes the legacy path.

    ``acquire_with_retry`` still supports raw pools alongside the backend
    abstraction, and that path has its own acquire/retry code, so the exit
    behaviour has to be asserted separately for it.
    """

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def acquire(self):
        raise self.error


@pytest.mark.asyncio
async def test_legacy_pool_path_also_exits(exit_window):
    """The raw-pool path reaches the same dead end and must handle it the same."""
    exits = exit_window(600.0, 601.0)

    with pytest.raises(ConnectionError):
        async with acquire_with_retry(_LegacyPool(ConnectionError("down")), max_retries=1):
            pass  # pragma: no cover - acquire never succeeds

    assert exits == [1]


@pytest.mark.asyncio
async def test_non_retryable_error_does_not_exit(exit_window):
    """A bug in our own code is not evidence the database is unreachable."""
    exits = exit_window(600.0, 601.0)

    with pytest.raises(ValueError):
        async with acquire_with_retry(_LegacyPool(ValueError("bad query")), max_retries=1):
            pass  # pragma: no cover - acquire never succeeds

    assert exits == []
