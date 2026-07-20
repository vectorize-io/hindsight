"""
Database utility functions for connection management with retry logic.
"""

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)

# Default retry configuration for database operations
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 0.5  # seconds
DEFAULT_MAX_DELAY = 5.0  # seconds


def _backoff_delay(attempt: int, base_delay: float, max_delay: float) -> float:
    """Exponential backoff with equal jitter.

    Deterministic backoff makes concurrent retriers wake in lock-step and
    re-collide on the very same rows, re-triggering the deadlock they just
    backed off from. "Equal jitter" — half the window fixed, half random —
    keeps a floor (so we don't hot-spin) while decorrelating the wake-ups, so
    two contenders that deadlocked together are very unlikely to retry in sync.
    """
    ceil = min(base_delay * (2**attempt), max_delay)
    return ceil / 2 + random.uniform(0, ceil / 2)


# Retryable exception types (checked by class name to avoid hard imports)
_RETRYABLE_EXCEPTION_NAMES = frozenset(
    {
        "InterfaceError",
        "ConnectionDoesNotExistError",
        "TooManyConnectionsError",
        "DeadlockDetectedError",
    }
)


def _is_oracle_deadlock(exc: BaseException) -> bool:
    """Check if an exception is an Oracle ORA-00060 deadlock."""
    try:
        import oracledb  # type: ignore[import-not-found]
    except ImportError:
        return False
    if isinstance(exc, oracledb.DatabaseError) and exc.args:
        err = exc.args[0]
        return getattr(err, "code", None) == 60  # ORA-00060
    return False


def _is_retryable(exc: BaseException) -> bool:
    """Check if an exception is retryable (transient connection issue)."""
    if isinstance(exc, (OSError, ConnectionError, asyncio.TimeoutError)):
        return True
    if type(exc).__name__ in _RETRYABLE_EXCEPTION_NAMES:
        return True
    return _is_oracle_deadlock(exc)


async def retry_with_backoff(
    func,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
):
    """
    Execute an async function with exponential backoff retry.

    Args:
        func: Async function to execute
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)

    Returns:
        Result of the function

    Raises:
        The last exception if all retries fail
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            if not _is_retryable(e):
                raise
            last_exception = e
            if attempt < max_retries:
                delay = _backoff_delay(attempt, base_delay, max_delay)
                if type(e).__name__ == "DeadlockDetectedError" or _is_oracle_deadlock(e):
                    logger.warning(
                        "Deadlock detected during parallel document processing — "
                        "this is expected and will resolve automatically "
                        f"(attempt {attempt + 1}/{max_retries + 1}, retrying in {delay:.1f}s)"
                    )
                else:
                    logger.warning(
                        f"Database operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                await asyncio.sleep(delay)
            else:
                logger.error(f"Database operation failed after {max_retries + 1} attempts: {e}")
    raise last_exception


@asynccontextmanager
async def acquire_with_retry(backend_or_pool: Any, max_retries: int = DEFAULT_MAX_RETRIES) -> AsyncIterator[Any]:
    """
    Async context manager to acquire a database connection with retry logic.

    Retries the *acquire* itself when it raises a retryable error (connection
    drop, timeout, deadlock detected during acquire). Exceptions raised by
    user code inside the ``async with`` block are NOT retried — they propagate
    as-is. Wrapping retry around the yield would violate the
    ``@asynccontextmanager`` single-yield contract and surface as
    ``RuntimeError("generator didn't stop after athrow()")`` on every
    retryable inner error, masking the real cause.

    Accepts either a DatabaseBackend or a raw asyncpg.Pool for backward compatibility.

    Usage:
        async with acquire_with_retry(backend) as conn:
            await conn.execute(...)

    Args:
        backend_or_pool: A DatabaseBackend instance or asyncpg.Pool
        max_retries: Maximum number of retry attempts for the acquire step

    Yields:
        A DatabaseConnection (if backend) or asyncpg.Connection (if pool)
    """
    from .db.base import DatabaseBackend

    if isinstance(backend_or_pool, DatabaseBackend) or getattr(backend_or_pool, "_wraps_backend", False):
        start = time.time()
        async with AsyncExitStack() as stack:
            conn: Any = None
            for attempt in range(max_retries + 1):
                try:
                    conn = await stack.enter_async_context(backend_or_pool.acquire())
                    break
                except Exception as e:
                    if not _is_retryable(e):
                        raise
                    if attempt < max_retries:
                        delay = _backoff_delay(attempt, DEFAULT_BASE_DELAY, DEFAULT_MAX_DELAY)
                        logger.warning(
                            f"Database acquire failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Database acquire failed after {max_retries + 1} attempts: {e}")
                        raise

            acquire_time = time.time() - start
            if acquire_time > 0.05:
                logger.warning(f"[DB POOL] Slow acquire: {acquire_time:.3f}s")

            yield conn
    else:
        # Legacy path: raw asyncpg.Pool
        pool = backend_or_pool
        start = time.time()

        async def acquire():
            return await pool.acquire()

        conn = await retry_with_backoff(acquire, max_retries=max_retries)
        acquire_time = time.time() - start

        if acquire_time > 0.05:
            pool_size = pool.get_size()
            pool_free = pool.get_idle_size()
            logger.warning(f"[DB POOL] Slow acquire: {acquire_time:.3f}s | size={pool_size}, idle={pool_free}")

        try:
            yield conn
        finally:
            await pool.release(conn)
