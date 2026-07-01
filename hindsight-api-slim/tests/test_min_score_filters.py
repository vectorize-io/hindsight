"""Unit tests for post-query recall min_scores filters."""

from hindsight_api.engine.response_models import MinScores, RecallScores
from hindsight_api.engine.search.reranking import filter_scored_results_by_min_scores
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult, ScoredResult


def _scored_result(
    result_id: str,
    *,
    raw: float,
    normalized: float,
    final: float,
) -> ScoredResult:
    return ScoredResult(
        candidate=MergedCandidate(
            retrieval=RetrievalResult(id=result_id, text=result_id, fact_type="world"),
            rrf_score=1.0,
        ),
        cross_encoder_score=raw,
        cross_encoder_score_normalized=normalized,
        combined_score=final,
        weight=final,
    )


def test_min_scores_models_accept_reranker_raw() -> None:
    assert MinScores(reranker_raw=0.05).reranker_raw == 0.05
    assert RecallScores(final=0.9, reranker=0.8, reranker_raw=0.05).reranker_raw == 0.05


def test_reranker_raw_floor_uses_raw_score_not_normalized() -> None:
    low_raw_high_norm = _scored_result("low-raw-high-norm", raw=0.01, normalized=0.95, final=0.9)
    high_raw_low_norm = _scored_result("high-raw-low-norm", raw=0.2, normalized=0.2, final=0.8)

    filtered = filter_scored_results_by_min_scores(
        [low_raw_high_norm, high_raw_low_norm],
        min_reranker_raw=0.1,
    )

    assert [result.id for result in filtered] == ["high-raw-low-norm"]


def test_reranker_floor_still_uses_normalized_score() -> None:
    low_raw_high_norm = _scored_result("low-raw-high-norm", raw=0.01, normalized=0.95, final=0.9)
    high_raw_low_norm = _scored_result("high-raw-low-norm", raw=0.2, normalized=0.2, final=0.8)

    filtered = filter_scored_results_by_min_scores(
        [low_raw_high_norm, high_raw_low_norm],
        min_reranker=0.9,
    )

    assert [result.id for result in filtered] == ["low-raw-high-norm"]


def test_final_floor_still_uses_final_weight() -> None:
    low_final = _scored_result("low-final", raw=0.2, normalized=0.9, final=0.4)
    high_final = _scored_result("high-final", raw=0.1, normalized=0.5, final=0.8)

    filtered = filter_scored_results_by_min_scores([low_final, high_final], min_final=0.5)

    assert [result.id for result in filtered] == ["high-final"]
