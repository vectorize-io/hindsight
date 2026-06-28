"""
Unit tests for CrossEncoderReranker.ensure_initialized concurrency guard.

Verifies that concurrent callers do not double-initialize the cross-encoder
model (race condition where both see _initialized=False and both call initialize).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.search.reranking import CrossEncoderReranker


def _make_cross_encoder():
    ce = AsyncMock()
    ce.provider_name = "local"
    ce.initialize = AsyncMock()
    return ce


@pytest.mark.asyncio
async def test_concurrent_ensure_initialized_calls_initialize_once():
    """Concurrent ensure_initialized calls must only initialize the model once."""
    ce = _make_cross_encoder()
    reranker = CrossEncoderReranker(cross_encoder=ce)

    await asyncio.gather(
        reranker.ensure_initialized(),
        reranker.ensure_initialized(),
        reranker.ensure_initialized(),
    )

    ce.initialize.assert_awaited_once()
    assert reranker._initialized is True


@pytest.mark.asyncio
async def test_second_call_after_init_skips_initialize():
    """After first initialization, subsequent calls return immediately."""
    ce = _make_cross_encoder()
    reranker = CrossEncoderReranker(cross_encoder=ce)

    await reranker.ensure_initialized()
    await reranker.ensure_initialized()

    ce.initialize.assert_awaited_once()
