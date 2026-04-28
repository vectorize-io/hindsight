"""
Database utility functions for connection management with retry logic.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)

# Default retry configuration for database operations
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 0.5  # seconds
DEFAULT_MAX_DELAY = 5.0  # seconds

# Retryable exception types (checked by class name to avoid hard imports)
_RETRYABLE_EXCEPTION_NAMES = frozenset(
    {
        "InterfaceError",
        "ConnectionDoesNotExistError",
        "TooManyConnectionsError",
        "DeadlockDetectedError",
    }
)


def _is_retryable(exc: BaseException) -> bool:
    """Check if an exception is retryable (transient connection issue)."""
    if isinstance(exc, (OSError, ConnectionError, asyncio.TimeoutError)):
        return True
    return type(exc).__name__ in _RETRYABLE_EXCEPTION_NAMES


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
                delay = min(base_delay * (2**attempt), max_delay)
                if type(e).__name__ == "DeadlockDetectedError":
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

    Accepts either a DatabaseBackend or a raw asyncpg.Pool for backward compatibility.

    Usage:
        async with acquire_with_retry(backend) as conn:
            await conn.execute(...)

    Args:
        backend_or_pool: A DatabaseBackend instance or asyncpg.Pool
        max_retries: Maximum number of retry attempts

    Yields:
        A DatabaseConnection (if backend) or asyncpg.Connection (if pool)
    """
    from .db.base import DatabaseBackend

    if isinstance(backend_or_pool, DatabaseBackend) or getattr(backend_or_pool, "_wraps_backend", False):
        # Use the backend's acquire context manager with retry
        start = time.time()
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                async with backend_or_pool.acquire() as conn:
                    acquire_time = time.time() - start
                    if acquire_time > 0.05:
                        logger.warning(f"[DB POOL] Slow acquire: {acquire_time:.3f}s")
                    yield conn
                    return
            except Exception as e:
                if not _is_retryable(e):
                    raise
                last_exception = e
                if attempt < max_retries:
                    delay = min(DEFAULT_BASE_DELAY * (2**attempt), DEFAULT_MAX_DELAY)
                    logger.warning(
                        f"Database acquire failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Database acquire failed after {max_retries + 1} attempts: {e}")
        raise last_exception
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
