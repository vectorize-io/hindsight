"""Tests for the optional read-only backend for recall queries."""

from __future__ import annotations

import pytest
import pytest_asyncio

from hindsight_api import MemoryEngine
from hindsight_api.engine.task_backend import SyncTaskBackend


def _make_engine(pg0_db_url: str, embeddings, cross_encoder, query_analyzer) -> MemoryEngine:
    """Build a MemoryEngine for a single test. Tiny pool, no migrations,
    SyncTaskBackend so async tasks resolve inline. Mirrors conftest's
    ``memory`` fixture but lets each test build its own with custom env.
    """
    return MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="none",  # No LLM calls in these tests
        memory_llm_api_key="unused",
        memory_llm_model="unused",
        embeddings=embeddings,
        cross_encoder=cross_encoder,
        query_analyzer=query_analyzer,
        pool_min_size=1,
        pool_max_size=2,
        run_migrations=False,
        task_backend=SyncTaskBackend(),
    )


@pytest_asyncio.fixture
async def engine_no_read_url(monkeypatch, pg0_db_url, embeddings, cross_encoder, query_analyzer):
    """Engine built with READ_DATABASE_URL explicitly unset."""
    from hindsight_api.config import clear_config_cache

    monkeypatch.delenv("HINDSIGHT_API_READ_DATABASE_URL", raising=False)
    clear_config_cache()

    mem = _make_engine(pg0_db_url, embeddings, cross_encoder, query_analyzer)
    await mem.initialize()
    yield mem
    try:
        await mem.close()
    except Exception:
        pass
    clear_config_cache()


@pytest_asyncio.fixture
async def engine_with_read_url(monkeypatch, pg0_db_url, embeddings, cross_encoder, query_analyzer):
    """Engine built with READ_DATABASE_URL set to the same DB. The engine
    can't tell it's the same server — it just sees a second URL and opens
    a second backend, which is what we want to verify.
    """
    from hindsight_api.config import clear_config_cache

    monkeypatch.setenv("HINDSIGHT_API_READ_DATABASE_URL", pg0_db_url)
    clear_config_cache()

    mem = _make_engine(pg0_db_url, embeddings, cross_encoder, query_analyzer)
    await mem.initialize()
    yield mem
    try:
        await mem.close()
    except Exception:
        pass
    clear_config_cache()


@pytest.mark.asyncio
async def test_read_backend_aliases_primary_when_url_unset(engine_no_read_url):
    """Without HINDSIGHT_API_READ_DATABASE_URL, _read_backend is _backend.
    This is the back-compat invariant — every call site that uses
    _get_read_backend() resolves to the same object _get_backend() returns,
    so nothing observable changes.
    """
    assert engine_no_read_url._read_backend is engine_no_read_url._backend


@pytest.mark.asyncio
async def test_read_backend_is_separate_instance_when_url_set(engine_with_read_url):
    """With HINDSIGHT_API_READ_DATABASE_URL set, a SECOND backend object is
    created. Even if both URLs point at the same DB (as in this test), the
    two backends own independent connection pools — connections taken from
    one don't drain the other, and shutting one down doesn't close the
    other's pool.
    """
    primary = engine_with_read_url._backend
    read = engine_with_read_url._read_backend
    assert read is not primary
    # Both backends are independently initialized (each owns a pool)
    assert primary.get_pool() is not None
    assert read.get_pool() is not None
    assert primary.get_pool() is not read.get_pool()


@pytest.mark.asyncio
async def test_get_read_backend_returns_read_backend(engine_with_read_url):
    """The accessor used by recall (`_get_read_backend`) returns the dedicated
    read backend, not the primary. Without this, the env var would have no
    effect.
    """
    backend = await engine_with_read_url._get_read_backend()
    assert backend is engine_with_read_url._read_backend
    assert backend is not engine_with_read_url._backend


@pytest.mark.asyncio
async def test_get_read_backend_returns_primary_when_unset(engine_no_read_url):
    """The accessor falls through to the primary when no read URL is set,
    so callers don't need to handle a None case.
    """
    backend = await engine_no_read_url._get_read_backend()
    assert backend is engine_no_read_url._backend


@pytest.mark.asyncio
async def test_close_terminates_distinct_read_backend(engine_with_read_url):
    """When the read backend is distinct, close() must shut it down too,
    not just the primary. Otherwise we leak the read pool when the engine
    is recycled (e.g. across pytest sessions or in app shutdown).
    """
    primary_before = engine_with_read_url._backend
    read_before = engine_with_read_url._read_backend
    assert read_before is not primary_before

    await engine_with_read_url.close()

    # Both backends should be cleared on close. The exact post-close state
    # is "primary _backend cleared, _read_backend cleared". The shutdown
    # method on the read backend is called — we verify by checking the
    # engine no longer references either.
    assert engine_with_read_url._backend is None
    assert engine_with_read_url._read_backend is None
