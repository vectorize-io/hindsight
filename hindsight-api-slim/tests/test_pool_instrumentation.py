"""Unit tests for DB pool acquire instrumentation (waiter counter + slow-acquire log).

Deterministic (no DB): we drive the instrumentation with fake acquire context
managers / awaitables and assert the process-wide waiter count is accurate through
success, mid-acquire, and failure, and that a slow acquire logs with pool stats.
"""

import asyncio
import logging

import pytest

from hindsight_api.engine.db.pool_instrumentation import (
    PoolStats,
    acquire_conn,
    instrument_acquire,
    waiting_count,
)


class _GatedAcquire:
    """Async CM whose __aenter__ blocks until released, to observe mid-acquire state."""

    def __init__(self, entered: asyncio.Event, release: asyncio.Event):
        self._entered = entered
        self._release = release

    async def __aenter__(self):
        self._entered.set()
        await self._release.wait()
        return "conn"

    async def __aexit__(self, *exc):
        return False


class _BoomAcquire:
    async def __aenter__(self):
        raise RuntimeError("acquire failed")

    async def __aexit__(self, *exc):
        return False


async def test_instrument_acquire_tracks_waiters():
    assert waiting_count() == 0
    entered = asyncio.Event()
    release = asyncio.Event()

    async def use():
        async with instrument_acquire(_GatedAcquire(entered, release), warn_threshold_s=999) as conn:
            assert conn == "conn"
            # Once acquired, the caller is no longer waiting.
            assert waiting_count() == 0

    task = asyncio.create_task(use())
    await entered.wait()
    # Blocked inside __aenter__ -> counted as one waiter.
    assert waiting_count() == 1
    release.set()
    await task
    assert waiting_count() == 0


async def test_instrument_acquire_decrements_on_failure():
    assert waiting_count() == 0
    with pytest.raises(RuntimeError):
        async with instrument_acquire(_BoomAcquire(), warn_threshold_s=999):
            pass
    # The waiter count must not leak when the acquire itself raises.
    assert waiting_count() == 0


async def test_slow_acquire_logs_with_pool_stats(caplog):
    def stats():
        return PoolStats(in_use=10, max=10, idle=0)

    class _Instant:
        async def __aenter__(self):
            return "conn"

        async def __aexit__(self, *exc):
            return False

    # threshold 0 => any wait (>= 0s) is logged.
    with caplog.at_level(logging.WARNING, logger="hindsight.db.pool"):
        async with instrument_acquire(_Instant(), pool_stats=stats, warn_threshold_s=0.0) as conn:
            assert conn == "conn"

    msgs = [r.getMessage() for r in caplog.records]
    assert any("slow DB pool acquire" in m for m in msgs)
    assert any("in_use=10" in m and "max=10" in m for m in msgs)


async def test_acquire_conn_await_style_tracks_and_returns():
    assert waiting_count() == 0

    async def acquire_awaitable():
        await asyncio.sleep(0)
        return "conn"

    conn = await acquire_conn(acquire_awaitable(), warn_threshold_s=999)
    assert conn == "conn"
    assert waiting_count() == 0


async def test_acquire_conn_decrements_on_failure():
    assert waiting_count() == 0

    async def boom():
        raise RuntimeError("acquire failed")

    with pytest.raises(RuntimeError):
        await acquire_conn(boom(), warn_threshold_s=999)
    assert waiting_count() == 0
