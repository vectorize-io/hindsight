"""
Unit tests for CrossEncoderReranker.rerank() score normalization logic.

Covers:
1. Passthrough normalization when scores are already in [0, 1].
2. Sigmoid normalization when scores are logits outside [0, 1].
3. Empty candidates returning an empty list without calling predict().
4. Low-confidence scores are preserved (not inflated by rank normalization).
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
        candidates.append(MergedCandidate(retrieval=retrieval, rrf_score=1.0 / (i + 1)))
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
async def test_passthrough_for_0_1_scores():
    """Scores already in [0, 1] should be passed through as-is."""
    raw_scores = [0.1, 0.5, 0.9]
    ce = _make_cross_encoder(raw_scores)
    reranker = CrossEncoderReranker(cross_encoder=ce)
    reranker._initialized = True

    candidates = _make_candidates(3)
    results = await reranker.rerank("test query", candidates)

    assert len(results) == 3
    # Results sorted by score descending; normalized == raw
    assert results[0].cross_encoder_score == pytest.approx(0.9)
    assert results[0].cross_encoder_score_normalized == pytest.approx(0.9)
    assert results[1].cross_encoder_score == pytest.approx(0.5)
    assert results[1].cross_encoder_score_normalized == pytest.approx(0.5)
    assert results[2].cross_encoder_score == pytest.approx(0.1)
    assert results[2].cross_encoder_score_normalized == pytest.approx(0.1)
    ce.predict.assert_awaited_once()


@pytest.mark.asyncio
async def test_low_confidence_scores_preserved():
    """When all candidates have low scores, they should stay low (not inflated)."""
    # Simulates a recall where no candidate is truly relevant
    raw_scores = [0.0077, 0.0033, 0.0021, 0.0003]
    ce = _make_cross_encoder(raw_scores)
    reranker = CrossEncoderReranker(cross_encoder=ce)
    reranker._initialized = True

    candidates = _make_candidates(4)
    results = await reranker.rerank("test query", candidates)

    # All normalized scores should remain low — not inflated to 1.0
    for result in results:
        assert result.cross_encoder_score_normalized < 0.01
    # Top candidate stays at its raw score
    assert results[0].cross_encoder_score_normalized == pytest.approx(0.0077)


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
async def test_boundary_scores_passthrough():
    """Boundary values 0.0 and 1.0 (still in [0,1]) should pass through."""
    raw_scores = [0.0, 1.0, 0.5]
    ce = _make_cross_encoder(raw_scores)
    reranker = CrossEncoderReranker(cross_encoder=ce)
    reranker._initialized = True

    candidates = _make_candidates(3)
    results = await reranker.rerank("test query", candidates)

    by_score = {r.cross_encoder_score: r.cross_encoder_score_normalized for r in results}
    assert by_score[1.0] == pytest.approx(1.0)
    assert by_score[0.5] == pytest.approx(0.5)
    assert by_score[0.0] == pytest.approx(0.0)
