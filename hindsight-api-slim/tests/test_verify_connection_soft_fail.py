"""
Tests that LLM connection verification failures don't crash server startup.

When the LLM provider is unavailable (e.g. 429 quota exhaustion), the server
should log a warning and continue booting rather than crash-looping.
See: https://github.com/vectorize-io/hindsight/issues/1147
"""

import logging
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from hindsight_api import MemoryEngine
from hindsight_api.engine.task_backend import SyncTaskBackend


@pytest_asyncio.fixture(scope="function")
async def engine_with_failing_llm(pg0_db_url, embeddings, cross_encoder, query_analyzer):
    """Create a MemoryEngine whose LLM verify_connection raises."""
    mem = MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="mock",
        memory_llm_api_key="",
        memory_llm_model="mock",
        embeddings=embeddings,
        cross_encoder=cross_encoder,
        query_analyzer=query_analyzer,
        pool_min_size=1,
        pool_max_size=5,
        run_migrations=False,
        task_backend=SyncTaskBackend(),
        skip_llm_verification=False,  # Enable verification — we want to test the soft-fail path
    )
    yield mem
    try:
        if mem._pool and not mem._pool._closing:
            await mem.close()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_initialize_succeeds_when_verify_connection_fails(
    engine_with_failing_llm,
    caplog,
):
    """Server should start even if LLM verify_connection raises (e.g. 429)."""
    engine = engine_with_failing_llm

    # Patch the mock provider's verify_connection to simulate a 429 error
    with patch.object(
        engine._llm_config._provider_impl,
        "verify_connection",
        new_callable=AsyncMock,
        side_effect=RuntimeError("429 RESOURCE_EXHAUSTED: Quota exceeded"),
    ):
        with caplog.at_level(logging.WARNING):
            # Should NOT raise — the server boots despite the LLM being unavailable
            await engine.initialize()

    # Verify the warning was logged
    assert any("LLM connection verification failed" in record.message for record in caplog.records)
    assert any("429 RESOURCE_EXHAUSTED" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_initialize_logs_warning_per_failing_config(
    pg0_db_url,
    embeddings,
    cross_encoder,
    query_analyzer,
    caplog,
):
    """Each distinct LLM config that fails verification gets its own warning."""
    engine = MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="mock",
        memory_llm_api_key="",
        memory_llm_model="default-model",
        retain_llm_provider="mock",
        retain_llm_model="retain-model",  # Different model → separate verification
        embeddings=embeddings,
        cross_encoder=cross_encoder,
        query_analyzer=query_analyzer,
        pool_min_size=1,
        pool_max_size=5,
        run_migrations=False,
        task_backend=SyncTaskBackend(),
        skip_llm_verification=False,
    )

    # Both default and retain verify_connection will raise
    with (
        patch.object(
            engine._llm_config._provider_impl,
            "verify_connection",
            new_callable=AsyncMock,
            side_effect=RuntimeError("429 quota exceeded for default"),
        ),
        patch.object(
            engine._retain_llm_config._provider_impl,
            "verify_connection",
            new_callable=AsyncMock,
            side_effect=RuntimeError("429 quota exceeded for retain"),
        ),
    ):
        with caplog.at_level(logging.WARNING):
            await engine.initialize()

    warning_messages = [r.message for r in caplog.records if "LLM connection verification failed" in r.message]
    assert len(warning_messages) == 2
    assert any("'default'" in msg for msg in warning_messages)
    assert any("'retain'" in msg for msg in warning_messages)

    try:
        if engine._pool and not engine._pool._closing:
            await engine.close()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_initialize_succeeds_when_verify_connection_succeeds(
    pg0_db_url,
    embeddings,
    cross_encoder,
    query_analyzer,
    caplog,
):
    """Verify the happy path still works — no warnings when verification passes."""
    engine = MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="mock",
        memory_llm_api_key="",
        memory_llm_model="mock",
        embeddings=embeddings,
        cross_encoder=cross_encoder,
        query_analyzer=query_analyzer,
        pool_min_size=1,
        pool_max_size=5,
        run_migrations=False,
        task_backend=SyncTaskBackend(),
        skip_llm_verification=False,
    )

    with caplog.at_level(logging.WARNING):
        await engine.initialize()

    assert not any("LLM connection verification failed" in r.message for r in caplog.records)

    try:
        if engine._pool and not engine._pool._closing:
            await engine.close()
    except Exception:
        pass
