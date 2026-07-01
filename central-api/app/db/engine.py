"""Async engine + session management.

A single lazily-created engine per process. ``get_session`` is the FastAPI
dependency; ``session_scope`` is the imperative context manager used by workers
and tests. ``init_models`` creates tables from metadata (dev/test bootstrap);
production uses Alembic migrations.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings
from app.db.tables import metadata

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine, _sessionmaker
    if _engine is None:
        _engine = create_async_engine(settings.effective_database_url, future=True)
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def _maker() -> async_sessionmaker[AsyncSession]:
    if _sessionmaker is None:
        get_engine()
    assert _sessionmaker is not None
    return _sessionmaker


async def init_models() -> None:
    """Create all control-plane tables (dev/test bootstrap)."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Commit on success, roll back on error — never commit manually inside."""
    async with _maker()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a committed-on-success session."""
    async with session_scope() as session:
        yield session


async def reset_engine() -> None:
    """Dispose the engine (used by tests to swap databases)."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None


def clear_engine() -> None:
    """Synchronously drop engine references so the next call rebuilds it.

    Used by test fixtures between tests (each test gets its own DB URL). We do not
    await ``dispose`` here to avoid cross-event-loop teardown issues with sqlite.
    """
    global _engine, _sessionmaker
    _engine = None
    _sessionmaker = None


async def get_health() -> tuple[bool, int]:
    """Check database connectivity and return (ok: bool, latency_ms: int)."""
    try:
        start = time.time()
        async with session_scope() as session:
            await session.execute(text("SELECT 1"))
        latency_ms = int((time.time() - start) * 1000)
        return True, latency_ms
    except Exception:
        return False, 0
