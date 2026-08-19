"""
Tests for LocalSTCrossEncoder and FlashRankCrossEncoder. The post-batch memory
release (heap trim + GPU empty_cache) is exercised here via its call sites; the
release helper itself is unit-tested in test_local_device.py.

These tests use mocked models — they do not load real SentenceTransformers or
FlashRank weights, so they run fast in CI without network access.
"""

from unittest.mock import MagicMock, patch

import pytest

from hindsight_api.config import DEFAULT_RERANKER_FLASHRANK_BATCH_SIZE
from hindsight_api.engine import cross_encoder as ce_module
from hindsight_api.engine.cross_encoder import (
    FlashRankCrossEncoder,
    LocalSTCrossEncoder,
)


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

    async def test_predict_releases_rerank_heap_after_success(self):
        """The cleanup hook must run after every successful predict batch."""
        encoder = self._make_encoder()
        encoder._model.predict.return_value = [0.5]

        cleanup_calls = []
        with patch.object(ce_module, "release_local_inference_memory", lambda *a: cleanup_calls.append("cleanup")):
            await encoder.predict([("q", "doc")])

        assert cleanup_calls == ["cleanup"]

    async def test_predict_releases_rerank_heap_even_on_exception(self):
        """`finally` semantics: cleanup must run when the model raises mid-batch."""
        encoder = self._make_encoder()
        encoder._model.predict.side_effect = RuntimeError("boom")

        cleanup_calls = []
        with patch.object(ce_module, "release_local_inference_memory", lambda *a: cleanup_calls.append("cleanup")):
            with pytest.raises(RuntimeError, match="boom"):
                await encoder.predict([("q", "doc")])

        assert cleanup_calls == ["cleanup"]


class TestFlashRankCrossEncoder:
    """Unit tests for the FlashRank ONNX reranker."""

    def _make_encoder(self, *, batch_size: int = DEFAULT_RERANKER_FLASHRANK_BATCH_SIZE):
        encoder = FlashRankCrossEncoder(model_name="ms-marco-MiniLM-L-12-v2", batch_size=batch_size)
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

    def test_predict_sync_releases_rerank_heap_after_success(self):
        encoder = self._make_encoder()
        encoder._ranker.rerank.return_value = [{"id": 0, "score": 0.5}]

        fake_flashrank = MagicMock()
        fake_flashrank.RerankRequest = lambda query, passages: MagicMock()

        cleanup_calls = []
        with patch.dict("sys.modules", {"flashrank": fake_flashrank}):
            with patch.object(ce_module, "release_local_inference_memory", lambda *a: cleanup_calls.append("cleanup")):
                encoder._predict_sync([("q", "doc")])

        assert cleanup_calls == ["cleanup"]

    def test_predict_sync_releases_rerank_heap_even_on_exception(self):
        encoder = self._make_encoder()
        encoder._ranker.rerank.side_effect = RuntimeError("flashrank boom")

        fake_flashrank = MagicMock()
        fake_flashrank.RerankRequest = lambda query, passages: MagicMock()

        cleanup_calls = []
        with patch.dict("sys.modules", {"flashrank": fake_flashrank}):
            with patch.object(ce_module, "release_local_inference_memory", lambda *a: cleanup_calls.append("cleanup")):
                with pytest.raises(RuntimeError, match="flashrank boom"):
                    encoder._predict_sync([("q", "doc")])

        assert cleanup_calls == ["cleanup"]

    def test_predict_sync_empty_pairs_does_not_release_heap(self):
        """The early `if not pairs: return []` short-circuits before the
        try/finally, so cleanup doesn't fire on a no-op call. This is intentional
        — nothing was allocated."""
        encoder = self._make_encoder()

        cleanup_calls = []
        with patch.object(ce_module, "release_local_inference_memory", lambda *a: cleanup_calls.append("cleanup")):
            encoder._predict_sync([])

        assert cleanup_calls == []

    def test_predict_sync_splits_into_batches(self):
        """A candidate pool larger than batch_size must never reach FlashRank as a
        single request: one forward pass allocates attention tensors sized
        batch * heads * seq^2, which OOM-killed containers on large banks (#3355).
        """
        encoder = self._make_encoder(batch_size=32)

        batch_sizes = []

        def fake_rerank(request):
            batch_sizes.append(len(request.passages))
            return [{"id": i, "score": 0.5} for i in range(len(request.passages))]

        encoder._ranker.rerank.side_effect = fake_rerank

        fake_flashrank = MagicMock()
        fake_flashrank.RerankRequest = lambda query, passages: MagicMock(query=query, passages=passages)

        pairs = [("q", f"doc-{i}") for i in range(300)]
        with patch.dict("sys.modules", {"flashrank": fake_flashrank}):
            scores = encoder._predict_sync(pairs)

        assert len(scores) == 300
        # ceil(300 / 32) == 10 passes, none exceeding the batch size.
        assert batch_sizes == [32] * 9 + [12]

    def test_predict_sync_batching_preserves_pair_positions(self):
        """Batch-local FlashRank ids must be shifted back onto the caller's
        positions, or scores land on the wrong candidates past the first batch."""
        encoder = self._make_encoder(batch_size=2)

        def fake_rerank(request):
            # Score by passage text so a misplaced score is detectable, and return
            # them out of order the way FlashRank does (score-descending).
            scored = [{"id": i, "score": float(p["text"])} for i, p in enumerate(request.passages)]
            return sorted(scored, key=lambda r: r["score"], reverse=True)

        encoder._ranker.rerank.side_effect = fake_rerank

        fake_flashrank = MagicMock()
        fake_flashrank.RerankRequest = lambda query, passages: MagicMock(query=query, passages=passages)

        pairs = [("q", str(i)) for i in range(5)]
        with patch.dict("sys.modules", {"flashrank": fake_flashrank}):
            scores = encoder._predict_sync(pairs)

        assert scores == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_predict_sync_batches_each_query_group_independently(self):
        """Grouping by query and batching within a group compose: two queries of
        three passages at batch_size=2 give two passes each, not three overall."""
        encoder = self._make_encoder(batch_size=2)

        seen = []

        def fake_rerank(request):
            seen.append((request.query, len(request.passages)))
            return [{"id": i, "score": 0.5} for i in range(len(request.passages))]

        encoder._ranker.rerank.side_effect = fake_rerank

        fake_flashrank = MagicMock()
        fake_flashrank.RerankRequest = lambda query, passages: MagicMock(query=query, passages=passages)

        pairs = [("q1", "a"), ("q2", "d"), ("q1", "b"), ("q2", "e"), ("q1", "c"), ("q2", "f")]
        with patch.dict("sys.modules", {"flashrank": fake_flashrank}):
            scores = encoder._predict_sync(pairs)

        assert scores == [0.5] * 6
        assert seen == [("q1", 2), ("q1", 1), ("q2", 2), ("q2", 1)]

    @pytest.mark.parametrize("configured", [0, -1])
    def test_non_positive_batch_size_clamps_to_one(self, configured):
        """A misconfigured 0/-1 must not silently restore the unbounded single pass."""
        assert FlashRankCrossEncoder(batch_size=configured).batch_size == 1

    def test_default_batch_size_matches_config(self):
        assert FlashRankCrossEncoder().batch_size == DEFAULT_RERANKER_FLASHRANK_BATCH_SIZE
