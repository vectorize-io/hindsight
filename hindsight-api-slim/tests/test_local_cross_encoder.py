"""
Tests for LocalSTCrossEncoder and FlashRankCrossEncoder. The post-batch memory
release (heap trim + GPU empty_cache) is exercised here via its call sites; the
release helper itself is unit-tested in test_local_device.py.

These tests use mocked models — they do not load real SentenceTransformers or
FlashRank weights, so they run fast in CI without network access.

The MiniLM stress test at the bottom is skip-by-default (HS_RERANKER_STRESS=1).
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from hindsight_api.config import DEFAULT_RERANKER_FLASHRANK_BATCH_SIZE
from hindsight_api.engine import cross_encoder as ce_module
from hindsight_api.engine.cross_encoder import (
    _CE_MAX_LENGTH_CEILING,
    FlashRankCrossEncoder,
    LocalSTCrossEncoder,
    TokenizerMaxLengthWrapper,
    _bind_tokenizer_max_length,
    query_exceeds_max_length,
    resolve_ce_max_length,
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
        # Production predict never reads _model; the per-thread loader is stubbed
        # so existing call-count assertions on this mock still work.
        mock = MagicMock()
        encoder._model = mock
        encoder._initialized = True
        encoder._load_model_instance = lambda: mock
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

    def test_resolve_ce_max_length_reads_tokenizer_not_hardcoded(self):
        model = MagicMock()
        model.max_length = None
        model.tokenizer.model_max_length = 384
        assert resolve_ce_max_length(model) == 384
        model.tokenizer.model_max_length = 512
        assert resolve_ce_max_length(model) == 512

    def test_query_exceeds_max_length_uses_tokenizer_encode(self):
        tok = MagicMock()
        tok.encode.return_value = [1] * 600
        assert query_exceeds_max_length(tok, "long query", 512) is True
        tok.encode.return_value = [1, 2, 3]
        assert query_exceeds_max_length(tok, "short", 512) is False

    async def test_long_query_scores_and_warns_once(self, caplog):
        ce_module._query_truncation_last_warn_at = 0.0
        encoder = self._make_encoder()
        encoder._model.predict.return_value = [0.4, 0.6]
        encoder._model.max_length = None
        tok = MagicMock()
        tok.model_max_length = 512
        tok.encode.return_value = [1] * 2000
        encoder._model.tokenizer = tok
        long_query = "x" * 7500
        with caplog.at_level("WARNING", logger="hindsight_api.engine.cross_encoder"):
            scores = await encoder.predict([(long_query, "doc-a"), (long_query, "doc-b")])
        assert scores == [0.4, 0.6]
        warnings = [r for r in caplog.records if "query truncated" in r.getMessage()]
        assert len(warnings) == 1
        assert "512" in warnings[0].getMessage()

    async def test_long_passage_is_scored(self):
        encoder = self._make_encoder()
        encoder._model.predict.return_value = [0.22]
        encoder._model.max_length = None
        tok = MagicMock()
        tok.model_max_length = 512
        tok.encode.return_value = [1, 2, 3]
        encoder._model.tokenizer = tok
        long_passage = "passage " * 600
        scores = await encoder.predict([("short query", long_passage)])
        assert scores == [0.22]
        encoder._model.predict.assert_called_once()
        called_pairs = encoder._model.predict.call_args.args[0]
        assert called_pairs[0][1] == long_passage

    async def test_short_pairs_unchanged_and_no_truncation_warning(self, caplog):
        encoder = self._make_encoder()
        encoder._model.predict.return_value = [0.9, 0.1]
        encoder._model.max_length = None
        tok = MagicMock()
        tok.model_max_length = 512
        tok.encode.return_value = [1, 2, 3, 4]
        encoder._model.tokenizer = tok
        pairs = [("q", "doc-a"), ("q", "doc-b")]
        with caplog.at_level("WARNING", logger="hindsight_api.engine.cross_encoder"):
            scores = await encoder.predict(pairs)
        assert scores == [0.9, 0.1]
        assert encoder._model.predict.call_args.args[0] == pairs
        assert not any("query truncated" in r.getMessage() for r in caplog.records)

    async def test_overflow_value_error_does_not_raise(self):
        encoder = self._make_encoder()
        encoder._model.predict.side_effect = [
            ValueError("Unable to create tensor: activate truncation and/or padding"),
            [0.3],
            [0.7],
        ]
        scores = await encoder.predict([("q", "a"), ("q", "b")])
        assert scores == [0.3, 0.7]

    def test_tokenizer_wrapper_injects_truncation_and_max_length(self):
        inner = MagicMock(return_value={"input_ids": [[1, 2]]})
        wrapped = TokenizerMaxLengthWrapper(inner, 512)
        wrapped(["pair"], return_tensors="pt")
        kwargs = inner.call_args.kwargs
        assert kwargs["truncation"] is True
        assert kwargs["padding"] is True
        assert kwargs["max_length"] == 512

    def test_scores_from_model_binds_tokenizer_wrapper(self):
        encoder = self._make_encoder()
        encoder._model.predict.return_value = [0.22]
        tok = MagicMock()
        tok.model_max_length = 512
        encoder._model.tokenizer = tok
        encoder._model.max_length = None
        encoder._scores_from_model(encoder._model, [("short query", "passage " * 600)])
        assert isinstance(encoder._model.tokenizer, TokenizerMaxLengthWrapper)
        assert encoder._model.tokenizer._max_length == 512

    async def test_pair_overflow_scores_negative_inf(self):
        encoder = self._make_encoder()
        encoder._model.predict.side_effect = [
            ValueError("Unable to create tensor: activate truncation and/or padding"),
            ValueError("Unable to create tensor: activate truncation and/or padding"),
            [0.7],
        ]
        scores = await encoder.predict([("q", "a"), ("q", "b")])
        assert scores[0] == float("-inf")
        assert scores[1] == 0.7

    async def test_unexpected_value_error_is_reraised(self):
        encoder = self._make_encoder()
        encoder._model.predict.side_effect = ValueError("shape mismatch: expected (2, 512)")
        with pytest.raises(ValueError, match="shape mismatch"):
            await encoder.predict([("q", "a")])

    async def test_bind_tokenizer_logs_when_tokenizer_is_read_only(self, caplog):
        class ReadOnlyTokenizerModel:
            def __init__(self) -> None:
                self._tok = MagicMock()
                self._tok.model_max_length = 512
                self._tok.encode.return_value = [1, 2, 3]
                self.predict = MagicMock(return_value=[0.5])
                self.max_length = None

            @property
            def tokenizer(self) -> MagicMock:
                return self._tok

        model = ReadOnlyTokenizerModel()
        encoder = self._make_encoder()
        encoder._model = model
        encoder._load_model_instance = lambda: model
        with caplog.at_level("WARNING", logger="hindsight_api.engine.cross_encoder"):
            scores = await encoder.predict([("q", "d")])
        assert scores == [0.5]
        assert any("could not bind tokenizer wrapper" in r.getMessage() for r in caplog.records)

    def test_resolve_ce_max_length_clamps_out_of_range(self, caplog):
        model = MagicMock()
        model.max_length = None
        model.tokenizer.model_max_length = 999999
        with caplog.at_level("WARNING", logger="hindsight_api.engine.cross_encoder"):
            assert resolve_ce_max_length(model) == _CE_MAX_LENGTH_CEILING
        assert any("clamping" in r.getMessage() for r in caplog.records)
        model.tokenizer.model_max_length = 4
        caplog.clear()
        with caplog.at_level("WARNING", logger="hindsight_api.engine.cross_encoder"):
            assert resolve_ce_max_length(model) == 512
        assert any("below floor" in r.getMessage() for r in caplog.records)

    def test_bind_tokenizer_is_idempotent_under_threads(self):
        model = MagicMock()
        inner = MagicMock()
        model.tokenizer = inner

        def bind() -> None:
            _bind_tokenizer_max_length(model, 512)

        threads = [threading.Thread(target=bind) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert isinstance(model.tokenizer, TokenizerMaxLengthWrapper)
        assert model.tokenizer._inner is inner


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


# ---------------------------------------------------------------------------
# Per-thread CrossEncoder isolation (MB1c2)
# ---------------------------------------------------------------------------


class _TrackingStub:
    """Stub CrossEncoder that records overlapping predict() on THIS instance."""

    def __init__(self, barrier: threading.Barrier | None = None) -> None:
        self.barrier = barrier
        self.inside = 0
        self.max_inside = 0
        self.thread_ids: list[int] = []
        self.predict_calls = 0
        self.seen_pairs: list[list[tuple[str, str]]] = []
        self._lock = threading.Lock()

    def predict(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> list[float]:
        del batch_size, show_progress_bar
        with self._lock:
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)
            self.thread_ids.append(threading.get_ident())
            self.predict_calls += 1
            self.seen_pairs.append(list(pairs))
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
            # Stay inside predict long enough that a shared instance would show
            # overlapping callers if they also waited on this barrier.
            time.sleep(0.05)
        with self._lock:
            self.inside -= 1
        return [_deterministic_score(q, d) for q, d in pairs]


def _deterministic_score(query: str, document: str) -> float:
    """Stable stand-in for a CE logit so identity tests do not need MiniLM."""
    return float(len(query) + 2 * len(document) + sum(ord(c) for c in document[:8]))


class LegacySharedLocalSTCrossEncoder(LocalSTCrossEncoder):
    """Pre-MB1c2 LocalSTCrossEncoder: one shared model on every executor thread.

    Used so the attacking test goes RED against the old design (parametrized
    as ``legacy``) without depending on a git checkout of the previous class.
    """

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._ensure_executor()
        # One instance, assigned to _model, used by every worker - the race.
        self._model = self._load_model_instance()
        if self._model is None:
            raise RuntimeError("legacy loader returned None")
        self._initialized = True

    def _predict_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        try:
            if self.bucket_batching and len(pairs) > 1:
                lengths = [len(pairs[i][0]) + len(pairs[i][1]) for i in range(len(pairs))]
                sorted_indices = sorted(range(len(pairs)), key=lambda i: lengths[i])
                sorted_pairs = [pairs[i] for i in sorted_indices]
                sorted_scores = self._model.predict(sorted_pairs, batch_size=self.batch_size, show_progress_bar=False)
                sorted_scores = sorted_scores.tolist() if hasattr(sorted_scores, "tolist") else list(sorted_scores)
                scores = [0.0] * len(pairs)
                for new_pos, orig_idx in enumerate(sorted_indices):
                    scores[orig_idx] = sorted_scores[new_pos]
                return scores
            scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
            return scores.tolist() if hasattr(scores, "tolist") else list(scores)
        finally:
            ce_module.release_local_inference_memory(self._device_type)


@pytest.fixture
def isolated_reranker_executor():
    """Swap the class-level pool so these tests cannot starve a session fixture."""
    old_executor = LocalSTCrossEncoder._executor
    old_max = LocalSTCrossEncoder._max_concurrent
    pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="reranker-test")
    LocalSTCrossEncoder._executor = pool
    LocalSTCrossEncoder._max_concurrent = 4
    try:
        yield pool
    finally:
        pool.shutdown(wait=True)
        LocalSTCrossEncoder._executor = old_executor
        LocalSTCrossEncoder._max_concurrent = old_max


def _install_stub_loader(encoder: LocalSTCrossEncoder, make_stub) -> list[_TrackingStub]:
    created: list[_TrackingStub] = []

    def _load() -> _TrackingStub:
        stub = make_stub()
        created.append(stub)
        return stub

    encoder._load_model_instance = _load  # type: ignore[method-assign]
    return created


class TestLocalSTPerThreadIsolation:
    """Attacking concurrency tests for the shared-tokenizer defect."""

    @pytest.mark.parametrize(
        "impl,expect_exclusive",
        [
            ("legacy", False),
            ("current", True),
        ],
    )
    async def test_concurrent_predict_instance_exclusivity(
        self, isolated_reranker_executor, impl: str, expect_exclusive: bool
    ):
        """Under max_workers=4, the current class never interleaves two threads
        inside the same instance's predict(). The pre-change (legacy) class
        does - this parametrization goes RED on exclusive if pointed at the
        old _predict_sync (one shared _model).
        """
        n = 4
        barrier = threading.Barrier(n)
        cls = LegacySharedLocalSTCrossEncoder if impl == "legacy" else LocalSTCrossEncoder
        encoder = cls(model_name="test-model", max_concurrent=n)
        # If production were reverted to ``self._model.predict``, this one
        # shared stub is what every thread would enter.
        shared_fallback = _TrackingStub(barrier=barrier)
        encoder._model = shared_fallback

        created = _install_stub_loader(encoder, lambda: _TrackingStub(barrier=barrier))
        await encoder.initialize()

        # One pair-list per concurrent call so each worker is inside predict at once.
        pairs_by_call = [[(f"q{i}", "x" * (10 + i))] for i in range(n)]

        scores = await asyncio.gather(*[encoder.predict(p) for p in pairs_by_call])
        assert len(scores) == n
        for batch in scores:
            assert len(batch) == 1

        if expect_exclusive:
            assert len(created) == n, created
            ids = {id(stub) for stub in created}
            assert len(ids) == n
            for stub in created:
                assert stub.max_inside == 1, f"two threads entered the same per-thread instance: {stub.max_inside}"
                assert len(set(stub.thread_ids)) == 1
            # Production must not have used the shared fallback even though
            # _model was planted (that is the old code path).
            assert shared_fallback.max_inside == 0
            assert shared_fallback.predict_calls == 0
        else:
            # Legacy: one shared instance, four threads inside predict at once.
            assert encoder._model is shared_fallback or len(created) == 1
            raced = shared_fallback if shared_fallback.predict_calls else created[0]
            assert raced.max_inside > 1, (
                "legacy shared instance did not interleave; the attacking test cannot prove the old race shape"
            )

    @pytest.mark.parametrize("bucket_batching", [False, True])
    async def test_plain_and_bucket_arms_use_per_thread_instance(
        self, isolated_reranker_executor, bucket_batching: bool
    ):
        encoder = LocalSTCrossEncoder(
            model_name="test-model",
            max_concurrent=4,
            bucket_batching=bucket_batching,
        )
        created = _install_stub_loader(encoder, _TrackingStub)
        await encoder.initialize()

        pairs = [
            ("q", "long document " * 10),
            ("q", "short"),
            ("q", "medium doc here"),
        ]
        scores = await encoder.predict(pairs)
        assert len(scores) == 3
        assert len(created) >= 1
        used = [s for s in created if s.predict_calls]
        assert len(used) == 1
        stub = used[0]
        assert stub.predict_calls == 1
        if bucket_batching:
            received = stub.seen_pairs[0]
            received_lengths = [len(a) + len(b) for a, b in received]
            assert received_lengths == sorted(received_lengths)
            # Caller order is restored even though predict saw sorted pairs.
            assert scores == [_deterministic_score(q, d) for q, d in pairs]
        else:
            assert stub.seen_pairs[0] == pairs

    async def test_scores_match_single_instance_path(self, isolated_reranker_executor):
        encoder = LocalSTCrossEncoder(model_name="test-model", max_concurrent=4)
        _install_stub_loader(encoder, _TrackingStub)
        await encoder.initialize()

        pairs = [("what is rust?", "Rust is a language"), ("what is rust?", "Bananas are fruit")]
        serial = await encoder.predict(pairs)

        concurrent = await asyncio.gather(*[encoder.predict(pairs) for _ in range(4)])
        for got in concurrent:
            assert got == serial
        # Rank order is the same as a single-instance deterministic scorer.
        expected = [_deterministic_score(q, d) for q, d in pairs]
        assert serial == expected

    async def test_load_failure_raises_and_does_not_share(self, isolated_reranker_executor):
        encoder = LocalSTCrossEncoder(model_name="test-model", max_concurrent=4)
        shared = MagicMock(name="shared-must-not-be-used")
        encoder._model = shared

        def boom() -> None:
            raise RuntimeError("simulated HF load failure")

        encoder._load_model_instance = boom  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="per-thread LocalSTCrossEncoder|simulated HF load failure"):
            await encoder.initialize()

        assert encoder._initialized is False
        shared.predict.assert_not_called()
        with pytest.raises(RuntimeError, match="not initialized"):
            await encoder.predict([("q", "d")])
        shared.predict.assert_not_called()

    async def test_warmup_matches_existing_pool_size_not_later_max(self):
        """A later LocalSTCrossEncoder(max_concurrent=4) must not barrier-wait
        on a pool that was created with 2 workers. That hung initialize()
        before warmup used executor._max_workers.
        """
        old_executor = LocalSTCrossEncoder._executor
        old_max = LocalSTCrossEncoder._max_concurrent
        pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="reranker-small")
        LocalSTCrossEncoder._executor = pool
        try:
            encoder = LocalSTCrossEncoder(model_name="test-model", max_concurrent=4)
            created = _install_stub_loader(encoder, _TrackingStub)
            await asyncio.wait_for(encoder.initialize(), timeout=5)
            assert encoder._initialized is True
            assert len(created) == 2
        finally:
            pool.shutdown(wait=True)
            LocalSTCrossEncoder._executor = old_executor
            LocalSTCrossEncoder._max_concurrent = old_max

    async def test_load_none_raises_not_fallback(self, isolated_reranker_executor):
        encoder = LocalSTCrossEncoder(model_name="test-model", max_concurrent=2)
        shared = MagicMock(name="shared-none-fallback")
        encoder._model = shared
        encoder._load_model_instance = lambda: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="Refusing to share"):
            await encoder.initialize()
        assert encoder._initialized is False
        shared.predict.assert_not_called()


_STRESS_REASON = (
    "Real MiniLM 4x concurrent stress; run with: "
    "HS_RERANKER_STRESS=1 uv run pytest "
    "tests/test_local_cross_encoder.py::TestLocalSTMiniLMStress::test_real_minilm_four_concurrent "
    "-v -n0 --timeout=600"
)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("HS_RERANKER_STRESS") != "1", reason=_STRESS_REASON)
class TestLocalSTMiniLMStress:
    async def test_real_minilm_four_concurrent(self, isolated_reranker_executor):
        """Four overlapping CrossEncoder.predict calls on the real MiniLM.

        This is a soak, not a guaranteed reproduction of the tokenizer race
        (planner measurement: 0 errors in ~2.5 min). After the fix it must
        complete without ValueError/TypeError from the tokenizer.
        """
        pytest.importorskip("sentence_transformers")

        encoder = LocalSTCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_concurrent=4,
            force_cpu=True,
            batch_size=64,
            bucket_batching=True,
        )
        await encoder.initialize()

        query = ("BRIEF MB1c2 reranker concurrency query about git hooks. " * 20).strip()
        docs = [f"memory fact {i} about git hooks and guards. " * 15 for i in range(64)]
        pairs = [(query, doc) for doc in docs]

        results = await asyncio.gather(*[encoder.predict(pairs) for _ in range(4)])
        assert len(results) == 4
        for scores in results:
            assert len(scores) == 64
            assert all(isinstance(s, float) for s in scores)
        # Same input, same model family: values should match within float noise.
        reference = results[0]
        for other in results[1:]:
            for a, b in zip(reference, other, strict=True):
                assert abs(a - b) < 1e-4
