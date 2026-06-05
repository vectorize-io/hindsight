"""
Startup/lazy model-init must fail fast instead of hanging forever.

Covers issue #1897: if a model load blocks (e.g. an offline HuggingFace
download or an unreachable provider), initialization is capped by a wall-clock
timeout (HINDSIGHT_API_MODEL_INIT_TIMEOUT) and raises a clear RuntimeError
rather than leaving the daemon/request stuck in a third state — neither
started nor errored.
"""

import asyncio
from dataclasses import fields
from unittest.mock import patch

import pytest

from hindsight_api.config import HindsightConfig
from hindsight_api.engine.search.reranking import CrossEncoderReranker


def _make_config(**overrides) -> HindsightConfig:
    """Build a HindsightConfig with field-type-appropriate zero values plus overrides."""
    defaults: dict = {}
    for f in fields(HindsightConfig):
        if f.type == "str":
            defaults[f.name] = ""
        elif f.type == "str | None":
            defaults[f.name] = None
        elif f.type == "int":
            defaults[f.name] = 0
        elif f.type == "int | None":
            defaults[f.name] = None
        elif f.type == "float":
            defaults[f.name] = 0.0
        elif f.type == "float | None":
            defaults[f.name] = None
        elif f.type == "bool":
            defaults[f.name] = False
        else:
            defaults[f.name] = None
    defaults.update(overrides)
    return HindsightConfig(**defaults)


class _HangingCrossEncoder:
    """Cross-encoder whose initialize() never returns (simulates a stuck download)."""

    provider_name = "remote"  # non-local: stays on the event loop, no executor thread

    async def initialize(self) -> None:
        await asyncio.Event().wait()  # hangs forever


class _FastCrossEncoder:
    provider_name = "remote"

    def __init__(self) -> None:
        self.initialized = False

    async def initialize(self) -> None:
        self.initialized = True


def test_default_model_init_timeout_is_300s():
    """Unset env keeps a generous default that still covers first-time downloads."""
    config = HindsightConfig.from_env()
    assert config.model_init_timeout == 300.0


@pytest.mark.asyncio
async def test_lazy_reranker_init_fails_fast_on_hang():
    """A stuck cross-encoder load raises RuntimeError within the timeout, not forever."""
    reranker = CrossEncoderReranker(cross_encoder=_HangingCrossEncoder())
    config = _make_config(model_init_timeout=0.1)

    with patch("hindsight_api.config.get_config", return_value=config):
        with pytest.raises(RuntimeError, match="did not complete within"):
            await asyncio.wait_for(reranker.ensure_initialized(), timeout=5.0)

    assert reranker._initialized is False


@pytest.mark.asyncio
async def test_lazy_reranker_init_succeeds_within_timeout():
    """A fast init completes normally and marks the reranker initialized."""
    encoder = _FastCrossEncoder()
    reranker = CrossEncoderReranker(cross_encoder=encoder)
    config = _make_config(model_init_timeout=10.0)

    with patch("hindsight_api.config.get_config", return_value=config):
        await reranker.ensure_initialized()

    assert encoder.initialized is True
    assert reranker._initialized is True
