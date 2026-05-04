"""
Regression tests for issue #1369: cross-encoder ``predict()`` returns 2-D
shape ``(n, 1)`` and the recall pipeline crashes downstream with
``TypeError: bad operand type for unary -: list``.

Two layers of defense:
1. ``LocalSTCrossEncoder._predict_sync`` flattens at the boundary, so the
   declared ``-> list[float]`` contract on ``predict`` is honored regardless
   of whether sentence-transformers' underlying model emitted 1-D or 2-D.
2. ``CrossEncoderReranker.rerank`` also flattens defensively, so a future
   custom backend that slips a shape past its own boundary still produces a
   sane ranking instead of crashing the request.

These are pure unit tests — no DB, no real model, no network — so they run
under any environment that can import ``hindsight_api``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from hindsight_api.engine.cross_encoder import LocalSTCrossEncoder, _coerce_scores_1d
from hindsight_api.engine.search.reranking import CrossEncoderReranker
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult

# --- _coerce_scores_1d ----------------------------------------------------


class TestCoerceScores1D:
    """The score-shape boundary helper used by ``LocalSTCrossEncoder``."""

    def test_1d_ndarray_passes_through(self):
        scores = np.asarray([0.1, 0.5, -0.3], dtype=np.float32)
        assert _coerce_scores_1d(scores) == pytest.approx([0.1, 0.5, -0.3], rel=1e-5)

    def test_2d_ndarray_n_by_1_flattened(self):
        # The shape sentence-transformers returns when the underlying model
        # has num_labels=1 but ``activation_fn`` skips the squeeze.
        scores = np.asarray([[0.5], [0.8], [-1.2]], dtype=np.float32)
        out = _coerce_scores_1d(scores)
        assert out == pytest.approx([0.5, 0.8, -1.2], rel=1e-5)
        assert all(isinstance(x, float) for x in out)

    def test_python_list_of_lists_flattened(self):
        # Some custom subclasses may convert to nested lists before returning.
        out = _coerce_scores_1d([[0.5], [0.8]])
        assert out == [0.5, 0.8]
        assert all(isinstance(x, float) for x in out)

    def test_plain_list_passes_through(self):
        assert _coerce_scores_1d([0.5, -0.2, 1.1]) == [0.5, -0.2, 1.1]

    def test_empty_input(self):
        assert _coerce_scores_1d(np.asarray([], dtype=np.float32)) == []
        assert _coerce_scores_1d([]) == []

    def test_returned_values_are_python_floats(self):
        # Reranking-side ``float()`` calls and JSON serialization both rely on
        # plain Python floats — ndarray scalars satisfy ``float()`` but their
        # ``isinstance`` identity differs.
        scores = np.asarray([[0.7]], dtype=np.float64)
        out = _coerce_scores_1d(scores)
        assert isinstance(out[0], float) and not isinstance(out[0], np.generic)


# --- LocalSTCrossEncoder._predict_sync ------------------------------------


def _build_encoder_with_mock_model(model_predict_return_value):
    """Construct a ``LocalSTCrossEncoder`` with a mocked underlying model.

    Bypasses ``initialize()`` entirely — we only exercise ``_predict_sync``,
    which is the path where shape-coercion lives.
    """
    encoder = LocalSTCrossEncoder.__new__(LocalSTCrossEncoder)
    encoder._model = MagicMock()
    encoder._model.predict = MagicMock(return_value=model_predict_return_value)
    encoder.bucket_batching = False
    encoder.batch_size = 32
    encoder.fp16 = False
    return encoder


class TestLocalSTPredictSync:
    """Per-issue regression tests for the underlying-model output shape."""

    def test_1d_output_unchanged(self):
        enc = _build_encoder_with_mock_model(np.asarray([0.1, 0.5, -0.3], dtype=np.float32))
        out = enc._predict_sync([("q", "d1"), ("q", "d2"), ("q", "d3")])
        assert out == pytest.approx([0.1, 0.5, -0.3], rel=1e-5)
        assert all(isinstance(x, float) for x in out)

    def test_2d_n_by_1_flattened(self):
        # Issue #1369 reproducer: model returns shape (n, 1).
        enc = _build_encoder_with_mock_model(np.asarray([[0.5], [0.8], [-1.2]], dtype=np.float32))
        out = enc._predict_sync([("q", "d1"), ("q", "d2"), ("q", "d3")])
        assert out == pytest.approx([0.5, 0.8, -1.2], rel=1e-5)
        # Every element must be a scalar float — not a list / ndarray slice —
        # so the downstream ``-x`` in sigmoid does not raise.
        for x in out:
            assert isinstance(x, float)
            # ``-x`` must not raise.
            _ = -x

    def test_2d_with_bucket_batching_preserves_order(self):
        # bucket_batching reorders pairs by length and must restore original
        # order. Combined with 2-D output, this exercises both branches.
        long_pair = ("query", "x" * 200)
        short_pair = ("query", "y")
        mid_pair = ("query", "z" * 50)
        pairs = [long_pair, short_pair, mid_pair]

        # The mock receives sorted_pairs (by length) — short, mid, long.
        # We return scores in that sorted order; _predict_sync must restore.
        sorted_scores_2d = np.asarray([[0.1], [0.2], [0.3]], dtype=np.float32)
        enc = _build_encoder_with_mock_model(sorted_scores_2d)
        enc.bucket_batching = True

        out = enc._predict_sync(pairs)
        # original order: long(0.3), short(0.1), mid(0.2)
        assert out == pytest.approx([0.3, 0.1, 0.2], rel=1e-5)
        assert all(isinstance(x, float) for x in out)

    def test_empty_pairs_does_not_crash(self):
        enc = _build_encoder_with_mock_model(np.asarray([], dtype=np.float32))
        assert enc._predict_sync([]) == []


# --- CrossEncoderReranker.rerank end-to-end -------------------------------


def _make_candidate(text: str = "fact text") -> MergedCandidate:
    """Build a minimal MergedCandidate for reranker tests."""
    retrieval = RetrievalResult(
        id="00000000-0000-0000-0000-000000000001",
        text=text,
        fact_type="world",
    )
    return MergedCandidate(retrieval=retrieval, rrf_score=0.1)


class TestRerankerHandlesMisbehavingBackend:
    """End-to-end sanity that ``rerank`` survives a 2-D leaking backend.

    Even with the boundary fix in ``LocalSTCrossEncoder._predict_sync``, a
    custom subclass or a future backend can violate the contract. The
    reranker's defensive flatten guards against that — the recall request
    must still succeed.
    """

    @pytest.mark.asyncio
    async def test_rerank_with_2d_scores_does_not_raise_typeerror(self):
        # Mock cross_encoder whose predict returns 2-D shape (n, 1).
        cross_encoder = MagicMock()
        cross_encoder.predict = AsyncMock(return_value=np.asarray([[0.5], [0.8]], dtype=np.float32))

        reranker = CrossEncoderReranker(cross_encoder=cross_encoder)
        reranker._initialized = True  # skip ensure_initialized

        candidates = [_make_candidate("a"), _make_candidate("b")]
        # Pre-fix: sigmoid raises ``TypeError: bad operand type for unary -: list``.
        results = await reranker.rerank("query", candidates)

        assert len(results) == 2
        # Post-sigmoid, weights must be plain floats in (0, 1).
        for r in results:
            assert isinstance(r.weight, float)
            assert 0.0 < r.weight < 1.0
        # Sort order must reflect the higher logit (0.8) ranking above 0.5.
        assert results[0].cross_encoder_score == pytest.approx(0.8, rel=1e-5)
        assert results[1].cross_encoder_score == pytest.approx(0.5, rel=1e-5)

    @pytest.mark.asyncio
    async def test_rerank_with_1d_scores_unchanged(self):
        cross_encoder = MagicMock()
        cross_encoder.predict = AsyncMock(return_value=[0.5, 0.8])

        reranker = CrossEncoderReranker(cross_encoder=cross_encoder)
        reranker._initialized = True

        candidates = [_make_candidate("a"), _make_candidate("b")]
        results = await reranker.rerank("query", candidates)

        assert len(results) == 2
        assert results[0].cross_encoder_score == pytest.approx(0.8, rel=1e-5)
        assert results[1].cross_encoder_score == pytest.approx(0.5, rel=1e-5)

    @pytest.mark.asyncio
    async def test_rerank_empty_candidates_returns_empty(self):
        cross_encoder = MagicMock()
        cross_encoder.predict = AsyncMock(return_value=np.asarray([], dtype=np.float32))

        reranker = CrossEncoderReranker(cross_encoder=cross_encoder)
        reranker._initialized = True

        results = await reranker.rerank("query", [])
        assert results == []
        # Empty short-circuit: predict must NOT have been called.
        cross_encoder.predict.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rerank_with_python_list_of_lists_does_not_raise(self):
        # Some custom backends may return nested Python lists rather than
        # ndarrays — same crash class, same defense.
        cross_encoder = MagicMock()
        cross_encoder.predict = AsyncMock(return_value=[[0.5], [0.8]])

        reranker = CrossEncoderReranker(cross_encoder=cross_encoder)
        reranker._initialized = True

        candidates = [_make_candidate("a"), _make_candidate("b")]
        results = await reranker.rerank("query", candidates)

        assert len(results) == 2
        for r in results:
            assert isinstance(r.weight, float)
            assert 0.0 < r.weight < 1.0
