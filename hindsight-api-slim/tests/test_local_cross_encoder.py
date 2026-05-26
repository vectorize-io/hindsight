"""
Tests for LocalSTCrossEncoder, FlashRankCrossEncoder, and the glibc malloc_trim
helper that releases heap pages after each rerank batch (issue #1717).

These tests use mocked models — they do not load real SentenceTransformers or
FlashRank weights, so they run fast in CI without network access.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from hindsight_api.engine import cross_encoder as ce_module
from hindsight_api.engine.cross_encoder import (
    FlashRankCrossEncoder,
    LocalSTCrossEncoder,
    _resolve_malloc_trim,
)


class TestResolveMallocTrim:
    """The malloc_trim resolver must be safe to call on every platform."""

    def test_returns_callable(self):
        trim = _resolve_malloc_trim()
        assert callable(trim)

    def test_callable_returns_none_or_int(self):
        # Linux/glibc returns the int from malloc_trim; everywhere else the
        # no-op returns None. Either is acceptable — the call must not raise.
        result = _resolve_malloc_trim()()
        assert result is None or isinstance(result, int)

    def test_non_linux_is_noop(self):
        """On non-Linux platforms the resolver must short-circuit to a no-op."""
        with patch.object(sys, "platform", "darwin"):
            trim = _resolve_malloc_trim()
        assert trim() is None

    def test_module_level_trim_is_resolved(self):
        """The module caches the resolved callable at import time."""
        assert callable(ce_module._malloc_trim)


class TestLocalSTCrossEncoder:
    """Unit tests for the SentenceTransformers-backed local reranker."""

    def _make_encoder(self, *, bucket_batching: bool = False, batch_size: int = 32):
        encoder = LocalSTCrossEncoder(
            model_name="test-model",
            bucket_batching=bucket_batching,
            batch_size=batch_size,
        )
        # Bypass initialize() — we don't want to download or load real weights.
        encoder._model = MagicMock()
        return encoder

    def test_provider_name(self):
        assert LocalSTCrossEncoder().provider_name == "local"

    async def test_predict_returns_scores_in_input_order(self):
        encoder = self._make_encoder()
        # Mock returns a numpy-array-like object with .tolist()
        mock_scores = MagicMock()
        mock_scores.tolist.return_value = [0.9, 0.1, 0.5]
        encoder._model.predict.return_value = mock_scores

        pairs = [
            ("q", "doc-a"),
            ("q", "doc-b"),
            ("q", "doc-c"),
        ]
        scores = await encoder.predict(pairs)

        assert scores == [0.9, 0.1, 0.5]
        encoder._model.predict.assert_called_once_with(pairs, batch_size=32, show_progress_bar=False)

    async def test_predict_accepts_plain_list_scores(self):
        """Backend may return a plain list instead of numpy — must still work."""
        encoder = self._make_encoder()
        encoder._model.predict.return_value = [0.3, 0.7]

        scores = await encoder.predict([("q", "a"), ("q", "b")])
        assert scores == [0.3, 0.7]

    async def test_predict_uses_configured_batch_size(self):
        encoder = self._make_encoder(batch_size=128)
        encoder._model.predict.return_value = [0.5]

        await encoder.predict([("q", "doc")])

        encoder._model.predict.assert_called_once()
        assert encoder._model.predict.call_args.kwargs["batch_size"] == 128

    async def test_predict_bucket_batching_restores_original_order(self):
        """With bucket_batching, pairs are sorted by length internally but
        scores must be returned in the caller's original order."""
        encoder = self._make_encoder(bucket_batching=True)

        # Pairs ordered long -> short. The encoder should reorder to short -> long
        # before calling .predict, then unscramble the result.
        pairs = [
            ("q", "long document " * 10),  # idx 0, longest
            ("q", "short"),  # idx 1, shortest
            ("q", "medium doc here"),  # idx 2, middle
        ]

        # Capture the sorted order that .predict actually receives, and return
        # scores keyed to that order so we can verify unscrambling. Use integer
        # scores to avoid float-precision noise in the assertion.
        def fake_predict(sorted_pairs, batch_size, show_progress_bar):
            return [float(i + 1) for i in range(len(sorted_pairs))]

        encoder._model.predict.side_effect = fake_predict

        scores = await encoder.predict(pairs)

        # Sorted by total length asc -> [short(1), medium(2), long(0)]
        # so fake_predict assigned: short=1.0, medium=2.0, long=3.0
        # In original order: [long=3.0, short=1.0, medium=2.0]
        assert scores == [3.0, 1.0, 2.0]

    async def test_predict_not_initialized_raises(self):
        encoder = LocalSTCrossEncoder()
        with pytest.raises(RuntimeError, match="not initialized"):
            await encoder.predict([("q", "d")])

    async def test_predict_calls_malloc_trim_after_success(self):
        """The trim callable must run after every successful predict batch."""
        encoder = self._make_encoder()
        encoder._model.predict.return_value = [0.5]

        trim_calls = []
        with patch.object(ce_module, "_malloc_trim", lambda: trim_calls.append("trim")):
            await encoder.predict([("q", "doc")])

        assert trim_calls == ["trim"]

    async def test_predict_calls_malloc_trim_even_on_exception(self):
        """`finally` semantics: trim must run when the model raises mid-batch."""
        encoder = self._make_encoder()
        encoder._model.predict.side_effect = RuntimeError("boom")

        trim_calls = []
        with patch.object(ce_module, "_malloc_trim", lambda: trim_calls.append("trim")):
            with pytest.raises(RuntimeError, match="boom"):
                await encoder.predict([("q", "doc")])

        assert trim_calls == ["trim"]


class TestFlashRankCrossEncoder:
    """Unit tests for the FlashRank ONNX reranker."""

    def _make_encoder(self):
        encoder = FlashRankCrossEncoder(model_name="ms-marco-MiniLM-L-12-v2")
        # Bypass initialize() — no model load, no executor needed (we call
        # _predict_sync directly).
        encoder._ranker = MagicMock()
        return encoder

    def test_provider_name(self):
        assert FlashRankCrossEncoder().provider_name == "flashrank"

    def test_predict_sync_empty_pairs(self):
        encoder = self._make_encoder()
        assert encoder._predict_sync([]) == []
        encoder._ranker.rerank.assert_not_called()

    def test_predict_sync_single_query_preserves_order(self):
        encoder = self._make_encoder()

        # FlashRank returns results in score-descending order, identified by the
        # "id" we assigned in the passages list. The encoder must map them back
        # to the original pair positions.
        def fake_rerank(request):
            # Score in reverse: last passage scores highest.
            return [{"id": i, "score": float(len(request.passages) - i)} for i in range(len(request.passages))]

        encoder._ranker.rerank.side_effect = fake_rerank

        # Patch sys.modules so the inline `from flashrank import RerankRequest`
        # in _predict_sync resolves to a lightweight stand-in.
        fake_flashrank = MagicMock()
        fake_flashrank.RerankRequest = lambda query, passages: MagicMock(query=query, passages=passages)

        with patch.dict("sys.modules", {"flashrank": fake_flashrank}):
            scores = encoder._predict_sync([("q", "a"), ("q", "b"), ("q", "c")])

        assert scores == [3.0, 2.0, 1.0]

    def test_predict_sync_multiple_queries_grouped(self):
        encoder = self._make_encoder()

        def fake_rerank(request):
            # Score everything 0.5 — we just want to verify grouping.
            return [{"id": i, "score": 0.5} for i in range(len(request.passages))]

        encoder._ranker.rerank.side_effect = fake_rerank

        fake_flashrank = MagicMock()
        fake_flashrank.RerankRequest = lambda query, passages: MagicMock(query=query, passages=passages)

        pairs = [
            ("q1", "a"),
            ("q2", "b"),
            ("q1", "c"),
        ]
        with patch.dict("sys.modules", {"flashrank": fake_flashrank}):
            scores = encoder._predict_sync(pairs)

        assert scores == [0.5, 0.5, 0.5]
        # Two unique queries -> two rerank calls.
        assert encoder._ranker.rerank.call_count == 2

    def test_predict_sync_calls_malloc_trim_after_success(self):
        encoder = self._make_encoder()
        encoder._ranker.rerank.return_value = [{"id": 0, "score": 0.5}]

        fake_flashrank = MagicMock()
        fake_flashrank.RerankRequest = lambda query, passages: MagicMock()

        trim_calls = []
        with patch.dict("sys.modules", {"flashrank": fake_flashrank}):
            with patch.object(ce_module, "_malloc_trim", lambda: trim_calls.append("trim")):
                encoder._predict_sync([("q", "doc")])

        assert trim_calls == ["trim"]

    def test_predict_sync_calls_malloc_trim_even_on_exception(self):
        encoder = self._make_encoder()
        encoder._ranker.rerank.side_effect = RuntimeError("flashrank boom")

        fake_flashrank = MagicMock()
        fake_flashrank.RerankRequest = lambda query, passages: MagicMock()

        trim_calls = []
        with patch.dict("sys.modules", {"flashrank": fake_flashrank}):
            with patch.object(ce_module, "_malloc_trim", lambda: trim_calls.append("trim")):
                with pytest.raises(RuntimeError, match="flashrank boom"):
                    encoder._predict_sync([("q", "doc")])

        assert trim_calls == ["trim"]

    def test_predict_sync_empty_pairs_does_not_trim(self):
        """The early `if not pairs: return []` short-circuits before the
        try/finally, so trim doesn't fire on a no-op call. This is intentional
        — nothing was allocated."""
        encoder = self._make_encoder()

        trim_calls = []
        with patch.object(ce_module, "_malloc_trim", lambda: trim_calls.append("trim")):
            encoder._predict_sync([])

        assert trim_calls == []
