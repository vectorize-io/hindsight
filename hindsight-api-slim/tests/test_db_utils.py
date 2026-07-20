"""Tests for hindsight_api.engine.db_utils.

Regression coverage for the single-yield contract of ``acquire_with_retry`` —
historically a retry loop wrapped the ``yield`` and caused every retryable
user-code exception to surface as
``RuntimeError("generator didn't stop after athrow()")``, masking the real
cause and producing identical failed-op rows in production (see the 1,934
failed consolidations on ``shurick-memory`` in May 2026).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from hindsight_api.engine.db_utils import _backoff_delay, acquire_with_retry


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
    and re-collide (see run_graph_maintenance_job's Pass 2/3 sweep). It must
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
