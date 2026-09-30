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

from hindsight_api.engine.cross_encoder import RerankTimeoutError, ScoreSemantics, ServedReranker
from hindsight_api.engine.search.reranking import CrossEncoderReranker, RerankResult
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
    # Set explicitly: an AsyncMock answers every unset attribute with a truthy Mock,
    # so leaving this out would make the fake claim it prunes and silently drop any
    # candidate scoring 0.0.
    ce.prunes_candidates = False
    ce.score_semantics = ScoreSemantics.POINTWISE
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
    results = (await reranker.rerank("test query", candidates)).results

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
    results = (await reranker.rerank("test query", candidates)).results

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
    results = (await reranker.rerank("test query", candidates)).results

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

    assert results.results == []
    assert results.served is None
    ce.predict.assert_not_awaited()


@pytest.mark.asyncio
async def test_duck_typed_cross_encoder_uses_legacy_capability_defaults() -> None:
    class DuckTypedCrossEncoder:
        provider_name = "custom"

        async def initialize(self) -> None:
            pass

        async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [0.8] * len(pairs)

    reranked = await CrossEncoderReranker(cross_encoder=DuckTypedCrossEncoder()).rerank(
        "test query", _make_candidates(1)
    )

    assert reranked.served == ServedReranker("custom", ScoreSemantics.POINTWISE, False)
    assert reranked.results[0].cross_encoder_score_normalized == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_boundary_scores_passthrough():
    """Boundary values 0.0 and 1.0 (still in [0,1]) should pass through."""
    raw_scores = [0.0, 1.0, 0.5]
    ce = _make_cross_encoder(raw_scores)
    reranker = CrossEncoderReranker(cross_encoder=ce)
    reranker._initialized = True

    candidates = _make_candidates(3)
    results = (await reranker.rerank("test query", candidates)).results

    by_score = {r.cross_encoder_score: r.cross_encoder_score_normalized for r in results}
    assert by_score[1.0] == pytest.approx(1.0)
    assert by_score[0.5] == pytest.approx(0.5)
    assert by_score[0.0] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_zero_scores_survive_an_ordinary_reranker():
    """A low score from a plain cross-encoder means "least bad", never "discard"."""
    ce = _make_cross_encoder([0.5, 0.0, 0.2])
    ce.prunes_candidates = False
    results = (await CrossEncoderReranker(cross_encoder=ce).rerank("q", _make_candidates(3))).results
    assert len(results) == 3


@pytest.mark.asyncio
async def test_zero_scores_are_dropped_when_the_reranker_judges_relevance():
    """A reranker that decides relevance marks a discard with exactly 0.0."""
    ce = _make_cross_encoder([0.5, 0.0, 0.2])
    ce.prunes_candidates = True
    results = (await CrossEncoderReranker(cross_encoder=ce).rerank("q", _make_candidates(3))).results
    assert [r.weight for r in results] == [0.5, 0.2]


@pytest.mark.asyncio
@pytest.mark.parametrize("first_query", ["primary", "fallback"])
async def test_concurrent_reranks_capture_the_serving_member(first_query: str) -> None:
    """The first predict pauses while another request changes the chain's cursor."""
    import asyncio

    from hindsight_api.engine.cross_encoder import CrossEncoderModel, MultiCrossEncoder, RRFPassthroughCrossEncoder

    first_scored = asyncio.Event()
    second_scored = asyncio.Event()

    class _Primary(CrossEncoderModel):
        score_semantics = ScoreSemantics.POINTWISE
        prunes_candidates = True

        @property
        def provider_name(self) -> str:
            return "tei"

        async def initialize(self) -> None:
            pass

        async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            if pairs[0][0] == "fallback":
                raise RuntimeError("member down")
            return [0.4, 0.0]

    class _InterleavedChain(MultiCrossEncoder):
        async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            scores = await super().predict(pairs)
            # Yield AFTER the member answered and BEFORE rerank reads its metadata.
            # Reading the shared _active cursor here would label the first request
            # with the second request's provider, in either failover direction.
            if pairs[0][0] == first_query:
                first_scored.set()
                await second_scored.wait()
            else:
                second_scored.set()
            return scores

    chain = _InterleavedChain([_Primary(), RRFPassthroughCrossEncoder()])
    reranker = CrossEncoderReranker(cross_encoder=chain)
    second_query = "fallback" if first_query == "primary" else "primary"

    async def second_rerank() -> RerankResult:
        await first_scored.wait()
        return await reranker.rerank(second_query, _make_candidates(2))

    results = await asyncio.wait_for(
        asyncio.gather(reranker.rerank(first_query, _make_candidates(2)), second_rerank()), timeout=5
    )
    expected = {
        "primary": ServedReranker("tei", ScoreSemantics.POINTWISE, True),
        "fallback": ServedReranker("rrf", ScoreSemantics.ORDINAL, False),
    }
    assert [result.served for result in results] == [expected[query] for query in [first_query, second_query]]
    assert [len(result.results) for result in results] == [
        1 if query == "primary" else 2 for query in [first_query, second_query]
    ]
    assert chain.provider_name == results[1].served.provider_name
    assert chain.provider_name != results[0].served.provider_name


@pytest.mark.asyncio
async def test_rerank_timeout_keeps_unscored_candidates_in_rrf_order():
    """When the local model runs out of wall-clock (#4696), recall still returns:
    the scored candidates rank normally and the unscored tail keeps RRF order."""
    # Candidates arrive in RRF order (index 0 is the strongest). The model scored
    # only indices 0 and 1 — and scored them in the opposite order — before expiring.
    ce = _make_cross_encoder([])
    ce.predict = AsyncMock(
        side_effect=RerankTimeoutError([1.0, 5.0, None, None, None], timeout=300.0, model_name="big-reranker")
    )
    reranker = CrossEncoderReranker(cross_encoder=ce)
    reranker._initialized = True

    reranked = await reranker.rerank("test query", _make_candidates(5))
    assert reranked.served == ServedReranker("local", ScoreSemantics.POINTWISE, False)
    results = reranked.results

    # Nothing is dropped.
    assert len(results) == 5
    # Scored pair first, reordered by score (5.0 > 1.0); unscored tail in RRF order.
    assert [r.candidate.retrieval.text for r in results] == [
        "Document 1",
        "Document 0",
        "Document 2",
        "Document 3",
        "Document 4",
    ]
    # The unscored tail carries no invented confidence: 0.0, not sigmoid(0)=0.5, so a
    # min_reranker_score threshold excludes them instead of ranking them mid-field.
    assert [r.weight for r in results[2:]] == [0.0, 0.0, 0.0]
    assert [r.cross_encoder_score_normalized for r in results[2:]] == [0.0, 0.0, 0.0]
