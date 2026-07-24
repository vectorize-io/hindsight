"""Instrumentation for database connection-pool acquisition.

asyncpg exposes pool *size* and *idle* counts, but not how many callers are
currently **queued waiting** for a connection — and that queue depth is the
signal that actually distinguishes a saturated pool from a healthy one. When the
pool is exhausted, ``/health`` (which itself acquires a connection to run
``SELECT 1``) blocks in ``pool.acquire()`` until a connection frees or the acquire
times out, so a liveness probe can fail **with the event loop completely idle**.

This module tracks the process-wide count of in-flight acquisitions that have not
yet obtained a connection, and times each acquire so a slow one logs with full
pool stats. It is the DB-side counterpart to ``loop_watchdog`` (which covers loop
stalls); together, a stuck ``/health`` can be attributed to either a blocked loop
or pool exhaustion from the logs alone.

The counter is a plain int mutated only from the event-loop thread (asyncpg
acquisitions are awaited on the loop), so no lock is needed.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("hindsight.db.pool")

_waiting = 0  # callers currently blocked in pool.acquire(), process-wide


@dataclass(frozen=True, slots=True)
class PoolStats:
    """Point-in-time connection-pool utilization snapshot."""

    in_use: int
    max: int
    idle: int


def waiting_count() -> int:
    """Number of callers currently blocked waiting to acquire a pooled connection."""
    return _waiting


@asynccontextmanager
async def instrument_acquire(
    acquire_cm: Any,
    *,
    pool_stats: Callable[[], PoolStats | None] | None = None,
    warn_threshold_s: float,
) -> AsyncIterator[Any]:
    """Wrap a pool's ``acquire()`` context manager with wait tracking + slow-acquire logging.

    Args:
        acquire_cm: an async context manager yielding a connection (e.g. the object
            returned by ``asyncpg.Pool.acquire()``).
        pool_stats: optional zero-arg callable returning a ``PoolStats`` snapshot for
            the slow-acquire log line.
        warn_threshold_s: log a warning when the acquire itself takes at least this long.

    Yields:
        The acquired connection.
    """
    global _waiting
    _waiting += 1
    start = time.monotonic()
    acquired = False
    try:
        async with acquire_cm as conn:
            acquired = True
            _waiting -= 1
            _record_acquire_wait(time.monotonic() - start, pool_stats, warn_threshold_s)
            yield conn
    finally:
        # If __aenter__ raised (acquire timeout / cancellation), we never
        # decremented above — do it here so the waiter count can't leak.
        if not acquired:
            _waiting -= 1


async def acquire_conn(
    acquire_awaitable: Any,
    *,
    pool_stats: Callable[[], PoolStats | None] | None = None,
    warn_threshold_s: float,
) -> Any:
    """Await a pool acquire that returns a connection, with wait tracking + slow log.

    For pools whose acquire is ``conn = await pool.acquire()`` (oracledb) rather than
    an async context manager (asyncpg — use ``instrument_acquire`` for those). The
    caller is responsible for releasing the returned connection.
    """
    global _waiting
    _waiting += 1
    start = time.monotonic()
    try:
        conn = await acquire_awaitable
    finally:
        _waiting -= 1
    _record_acquire_wait(time.monotonic() - start, pool_stats, warn_threshold_s)
    return conn


def _record_acquire_wait(
    wait_s: float,
    pool_stats: Callable[[], PoolStats | None] | None,
    warn_threshold_s: float,
) -> None:
    try:
        from ...metrics import get_metrics_collector

        get_metrics_collector().record_db_acquire_wait(wait_s)
    except Exception:
        pass

    if wait_s < warn_threshold_s:
        return

    stats: PoolStats | None = None
    if pool_stats is not None:
        try:
            stats = pool_stats()
        except Exception:
            stats = None
    logger.warning(
        "slow DB pool acquire: waited %.3fs for a connection "
        "(in_use=%s max=%s idle=%s waiting=%s). The pool is likely saturated; "
        "/health can stall on connection acquisition while the event loop is free.",
        wait_s,
        stats.in_use if stats else None,
        stats.max if stats else None,
        stats.idle if stats else None,
        _waiting,
    )
