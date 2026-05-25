"""
Unit tests for CrossEncoderReranker.rerank() score normalization logic.

Covers:
1. Rank-based normalization when scores are already in [0, 1].
2. Tied scores receiving identical normalized values.
3. Sigmoid normalization when scores are logits outside [0, 1].
4. Empty candidates returning an empty list without calling predict().
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from hindsight_api.engine.search.reranking import CrossEncoderReranker
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidates(n: int) -> list[MergedCandidate]:
    """Create *n* minimal MergedCandidate objects."""
    candidates = []
    for i in range(n):
        retrieval = RetrievalResult(
            id=str(uuid4()),
            text=f"Document {i}",
            fact_type="world",
            occurred_start=None,
            occurred_end=None,
        )
        candidates.append(
            MergedCandidate(retrieval=retrieval, rrf_score=1.0 / (i + 1))
        )
    return candidates


def _make_cross_encoder(predict_return: list[float]):
    """Return a fake cross-encoder whose `predict` is an AsyncMock."""
    ce = AsyncMock()
    ce.predict = AsyncMock(return_value=predict_return)
    ce.provider_name = "local"
    ce.initialize = AsyncMock()
    return ce


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rank_normalization_for_0_1_scores():
    """Scores already in [0, 1] should use rank-based normalization."""
    # Scores in [0, 1] but NOT logit-shaped — rank normalization preserves order
    raw_scores = [0.1, 0.5, 0.9]
    ce = _make_cross_encoder(raw_scores)
    reranker = CrossEncoderReranker(cross_encoder=ce)
    reranker._initialized = True

    candidates = _make_candidates(3)
    results = await reranker.rerank("test query", candidates)

    assert len(results) == 3
    # Top-ranked (0.9) gets normalized 1.0
    assert results[0].cross_encoder_score == pytest.approx(0.9)
    assert results[0].cross_encoder_score_normalized == pytest.approx(1.0)
    # Bottom-ranked (0.1) gets normalized 0.0
    assert results[-1].cross_encoder_score == pytest.approx(0.1)
    assert results[-1].cross_encoder_score_normalized == pytest.approx(0.0)
    # Middle gets 0.5
    assert results[1].cross_encoder_score_normalized == pytest.approx(0.5)
    # predict was called exactly once
    ce.predict.assert_awaited_once()


@pytest.mark.asyncio
async def test_tied_scores_get_same_normalized_value():
    """When raw scores contain ties, tied entries must receive the same normalized score."""
    # Two scores tied at 0.7, one distinct at 0.3
    raw_scores = [0.7, 0.3, 0.7]
    ce = _make_cross_encoder(raw_scores)
    reranker = CrossEncoderReranker(cross_encoder=ce)
    reranker._initialized = True

    candidates = _make_candidates(3)
    results = await reranker.rerank("test query", candidates)

    # The two 0.7 entries should share the same normalized value.
    # With rank-based normalization over 3 items (indices 0,1,2 sorted desc):
    #   rank 0 & 1 tied → avg_rank = 0.5 → norm = 1 - 0.5/2 = 0.75
    #   rank 2          → norm = 1 - 2/2 = 0.0
    # After sorting by normalized score descending, the two 0.7 entries are
    # at indices 0 and 1; the 0.3 entry is at index 2.
    scores_by_raw = {r.cross_encoder_score: r.cross_encoder_score_normalized for r in results}
    tied_norm = scores_by_raw[0.7]
    assert tied_norm == pytest.approx(0.75)
    assert scores_by_raw[0.3] == pytest.approx(0.0)
    # Both 0.7 results have identical normalized scores
    assert results[0].cross_encoder_score_normalized == pytest.approx(
        results[1].cross_encoder_score_normalized
    )


@pytest.mark.asyncio
async def test_sigmoid_normalization_for_logits():
    """When scores are outside [0, 1] (logits), sigmoid normalization is used."""
    raw_scores = [2.0, -1.0, 0.0]
    ce = _make_cross_encoder(raw_scores)
    reranker = CrossEncoderReranker(cross_encoder=ce)
    reranker._initialized = True

    candidates = _make_candidates(3)
    results = await reranker.rerank("test query", candidates)

    assert len(results) == 3
    import math

    # Results are sorted by weight descending
    expected_sigmoid = [1 / (1 + math.exp(-s)) for s in raw_scores]
    expected_sorted = sorted(expected_sigmoid, reverse=True)

    for result, expected in zip(results, expected_sorted):
        assert result.cross_encoder_score_normalized == pytest.approx(expected, rel=1e-6)

    # Verify the highest logit (2.0) maps to the highest normalized score
    assert results[0].cross_encoder_score == pytest.approx(2.0)
    assert results[0].cross_encoder_score_normalized > 0.5


@pytest.mark.asyncio
async def test_empty_candidates_returns_empty_without_predict():
    """When candidates are empty, rerank must return [] without calling predict()."""
    ce = _make_cross_encoder([])
    reranker = CrossEncoderReranker(cross_encoder=ce)
    reranker._initialized = True

    results = await reranker.rerank("test query", [])

    assert results == []
    ce.predict.assert_not_awaited()


@pytest.mark.asyncio
async def test_mixed_boundary_scores_use_rank():
    """Boundary values 0.0 and 1.0 (still in [0,1]) should trigger rank normalization."""
    raw_scores = [0.0, 1.0, 0.5]
    ce = _make_cross_encoder(raw_scores)
    reranker = CrossEncoderReranker(cross_encoder=ce)
    reranker._initialized = True

    candidates = _make_candidates(3)
    results = await reranker.rerank("test query", candidates)

    # Rank-based: 1.0 → 1.0, 0.5 → 0.5, 0.0 → 0.0
    by_score = {r.cross_encoder_score: r.cross_encoder_score_normalized for r in results}
    assert by_score[1.0] == pytest.approx(1.0)
    assert by_score[0.5] == pytest.approx(0.5)
    assert by_score[0.0] == pytest.approx(0.0)
