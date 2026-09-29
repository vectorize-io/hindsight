"""
Regression test for the JinaMLXCrossEncoder post-batch memory release.

`JinaMLXCrossEncoder._predict_sync` did not call `release_local_inference_memory`,
so MLX kept every freed Metal buffer in its own cache. That cache is bounded by
`mx.set_cache_limit`, which defaults to the device memory limit (121.60 GB measured
on a 128 GB Mac), so it grew for the life of the process — 42.7 GB of IOAccelerator
memory after 486 rerank calls on a laptop.

The other two in-process rerankers already release: LocalSTCrossEncoder and
FlashRankCrossEncoder both call the shared helper in a `finally`.

These tests verify:
1. Loading the model marks the provider as an mlx device.
2. A successful batch releases, and passes the provider's own device type.
3. A batch that raises still releases.
4. An empty batch does not release — nothing was allocated.
"""

import types
from unittest.mock import patch

import pytest

from hindsight_api.engine.cross_encoder import JinaMLXCrossEncoder


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _encoder_with_fake_reranker(*, raises=False):
    """Build a JinaMLXCrossEncoder wired to a stub reranker, skipping model load."""
    enc = JinaMLXCrossEncoder.__new__(JinaMLXCrossEncoder)
    enc._device_type = "mlx"
    enc._mlx_lock = _NullLock()

    def rerank(query, docs):
        if raises:
            raise RuntimeError("Metal command buffer failed")
        return [{"index": i, "relevance_score": 1.0 - i * 0.1} for i in range(len(docs))]

    enc._reranker = types.SimpleNamespace(rerank=rerank)
    return enc


PAIRS = [("q", "first document"), ("q", "second document")]


def test_successful_batch_releases_with_the_mlx_device_type():
    calls = []
    enc = _encoder_with_fake_reranker()
    with patch(
        "hindsight_api.engine.cross_encoder.release_local_inference_memory",
        lambda device_type=None: calls.append(device_type),
    ):
        scores = enc._predict_sync(PAIRS)

    assert len(scores) == len(PAIRS)
    assert calls == ["mlx"]


def test_release_runs_even_when_scoring_raises():
    calls = []
    enc = _encoder_with_fake_reranker(raises=True)
    with patch(
        "hindsight_api.engine.cross_encoder.release_local_inference_memory",
        lambda device_type=None: calls.append(device_type),
    ):
        with pytest.raises(RuntimeError):
            enc._predict_sync(PAIRS)

    assert calls == ["mlx"]


def test_empty_batch_does_not_release():
    calls = []
    enc = _encoder_with_fake_reranker()
    with patch(
        "hindsight_api.engine.cross_encoder.release_local_inference_memory",
        lambda device_type=None: calls.append(device_type),
    ):
        assert enc._predict_sync([]) == []

    assert calls == []


def test_load_model_marks_the_provider_as_mlx():
    """Without this the release call reaches _empty_gpu_cache with the wrong device type,
    which falls through to the torch lookup and frees nothing."""
    enc = JinaMLXCrossEncoder.__new__(JinaMLXCrossEncoder)
    enc.model_path = "/tmp/does-not-matter"

    with patch(
        "hindsight_api.engine.jina_mlx_reranker.MLXReranker",
        lambda **kwargs: object(),
    ):
        enc._load_model()

    assert enc._device_type == "mlx"
