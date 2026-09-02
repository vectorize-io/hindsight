"""Tests for hindsight_api.engine.db_utils.

Regression coverage for the single-yield contract of ``acquire_with_retry`` —
historically a retry loop wrapped the ``yield`` and caused every retryable
user-code exception to surface as
``RuntimeError("generator didn't stop after athrow()")``, masking the real
cause and producing identical failed-op rows in production (see the 1,934
failed consolidations on ``shurick-memory`` in May 2026).

Also covers what the acquire/retry paths *log*: several asyncpg errors carry no
message, so ``str(e)`` is "" and a failure rendered as
``Database acquire failed after N attempts: `` — an outage with the cause
blanked out.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import pytest

from hindsight_api.engine.db_utils import _backoff_delay, acquire_with_retry, retry_with_backoff


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


class InterfaceError(Exception):
    """Stands in for ``asyncpg.exceptions.InterfaceError``, raised with no args.

    Named to match the driver deliberately: ``_is_retryable`` classifies by
    ``type(exc).__name__``, so only this name reaches the retry-and-log path
    under test. ``InterfaceError`` is also the case that matters most — asyncpg
    raises it both for a dropped connection and for a pool that is closed and
    will never serve again, and several of them carry no message at all.
    """


class _SilentlyFailingBackend:
    """Backend whose acquires fail with an exception that stringifies to ""."""

    _wraps_backend = True

    @asynccontextmanager
    async def acquire(self):
        raise InterfaceError()
        yield  # pragma: no cover - unreachable, keeps this an async generator


@pytest.mark.asyncio
async def test_acquire_failure_logs_name_exception_with_empty_str(monkeypatch, caplog):
    """The acquire logs must name the exception even when it has no message.

    ``str(e)`` is "" for a no-args driver error, which rendered the retry and
    give-up lines as ``Database acquire failed after 2 attempts: `` — dropping
    the single field that distinguishes a dropped connection from a closed pool
    from an acquire timeout, and leaving an outage undiagnosable from the logs.
    """
    monkeypatch.setattr("hindsight_api.engine.db_utils._backoff_delay", lambda *a, **k: 0.0)

    with caplog.at_level(logging.WARNING, logger="hindsight_api.engine.db_utils"):
        with pytest.raises(InterfaceError):
            async with acquire_with_retry(_SilentlyFailingBackend(), max_retries=1):
                pass  # pragma: no cover - the acquire never succeeds

    acquire_lines = [r.getMessage() for r in caplog.records if "Database acquire failed" in r.getMessage()]
    assert acquire_lines, "the acquire path logged nothing at all"
    for line in acquire_lines:
        assert "InterfaceError" in line, f"log line dropped the exception class: {line!r}"


@pytest.mark.asyncio
async def test_retry_with_backoff_logs_name_exception_with_empty_str(monkeypatch, caplog):
    """Same guarantee for the generic operation-retry path."""
    monkeypatch.setattr("hindsight_api.engine.db_utils._backoff_delay", lambda *a, **k: 0.0)

    async def always_fails():
        raise InterfaceError()

    with caplog.at_level(logging.WARNING, logger="hindsight_api.engine.db_utils"):
        with pytest.raises(InterfaceError):
            await retry_with_backoff(always_fails, max_retries=1)

    op_lines = [r.getMessage() for r in caplog.records if "Database operation failed" in r.getMessage()]
    assert op_lines, "the retry path logged nothing at all"
    for line in op_lines:
        assert "InterfaceError" in line, f"log line dropped the exception class: {line!r}"
