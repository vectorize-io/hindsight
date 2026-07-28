"""HINDSIGHT_API_DB_ACQUIRE_TIMEOUT must bound the wait it names (#3002).

The value was only passed to ``asyncpg.create_pool(timeout=...)``, which is a
*connect* kwarg — how long establishing a new connection may take. The wait the
knob is named for, ``Pool.acquire()`` blocking until a connection frees up, was
left at asyncpg's default of "wait forever", so pool exhaustion never surfaced
as an error: it just hung, and none of the deployment's configured timeouts
applied.

Deterministic (no DB): a fake pool records the kwargs acquire() is called with.
"""

from contextlib import asynccontextmanager

import pytest

from hindsight_api.engine.db.postgresql import PostgreSQLBackend


class _FakeConn:
    @asynccontextmanager
    async def transaction(self):
        yield


class _FakePool:
    def __init__(self):
        self.acquire_kwargs: list[dict] = []

    def acquire(self, **kwargs):
        self.acquire_kwargs.append(kwargs)

        @asynccontextmanager
        async def _cm():
            yield _FakeConn()

        return _cm()

    def get_size(self):
        return 1

    def get_idle_size(self):
        return 1

    def get_max_size(self):
        return 1


def _backend(acquire_timeout: float | None) -> tuple[PostgreSQLBackend, _FakePool]:
    """Build a backend around a fake pool, skipping initialize()'s real connect."""
    backend = PostgreSQLBackend()
    pool = _FakePool()
    backend._pool = pool
    backend._acquire_timeout_s = acquire_timeout
    return backend, pool


@pytest.mark.asyncio
async def test_acquire_passes_the_configured_timeout():
    backend, pool = _backend(30.0)

    async with backend.acquire():
        pass

    assert pool.acquire_kwargs == [{"timeout": 30.0}]


@pytest.mark.asyncio
async def test_transaction_passes_the_configured_timeout():
    """transaction() acquires too — it must not keep the unbounded default."""
    backend, pool = _backend(30.0)

    async with backend.transaction():
        pass

    assert pool.acquire_kwargs == [{"timeout": 30.0}]


@pytest.mark.asyncio
async def test_zero_restores_the_unbounded_wait():
    """0 is the documented escape hatch for deployments that would rather queue
    than fail; asyncpg reads timeout=None as 'wait forever'."""
    backend, pool = _backend(None)

    async with backend.acquire():
        pass

    assert pool.acquire_kwargs == [{"timeout": None}]
