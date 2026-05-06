"""Tests for raw cross-encoder threshold filtering."""

import math
from datetime import datetime, timezone

from hindsight_api.config import (
    DEFAULT_RERANKER_THRESHOLD,
    ENV_RERANKER_THRESHOLD,
    HindsightConfig,
)
from hindsight_api.engine.search.reranking import apply_combined_scoring, filter_by_reranker_threshold
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult, ScoredResult

UTC = timezone.utc
NOW = datetime(2026, 4, 21, tzinfo=UTC)


def _make_result(
    result_id: str,
    raw_ce: float,
    *,
    ce_norm: float = 0.5,
    occurred_start: datetime | None = None,
) -> ScoredResult:
    retrieval = RetrievalResult(
        id=result_id,
        text=f"fact {result_id}",
        fact_type="world",
        occurred_start=occurred_start,
    )
    candidate = MergedCandidate(
        retrieval=retrieval,
        rrf_score=0.1,
    )
    return ScoredResult(
        candidate=candidate,
        cross_encoder_score=raw_ce,
        cross_encoder_score_normalized=ce_norm,
        weight=ce_norm,
    )


def test_filters_results_below_raw_cross_encoder_threshold():
    high = _make_result("high", 0.86)
    low = _make_result("low", 0.0002)

    filtered = filter_by_reranker_threshold([high, low], 0.01)

    assert [sr.id for sr in filtered] == ["high"]


def test_threshold_uses_raw_score_not_normalized_score():
    low_raw_high_norm = _make_result("low-raw", 0.0002, ce_norm=0.99)
    high_raw_low_norm = _make_result("high-raw", 0.86, ce_norm=0.51)

    filtered = filter_by_reranker_threshold([low_raw_high_norm, high_raw_low_norm], 0.01)

    assert [sr.id for sr in filtered] == ["high-raw"]


def test_threshold_allows_empty_results():
    results = [
        _make_result("noise-1", 0.00002),
        _make_result("noise-2", 0.00016),
    ]

    assert filter_by_reranker_threshold(results, 0.01) == []


def test_threshold_none_and_disabled_return_original_list():
    results = [_make_result("low", 0.0002)]

    assert filter_by_reranker_threshold(results, None) is results
    assert filter_by_reranker_threshold(results, 0.01, enabled=False) is results


def test_threshold_drops_nan_when_enabled():
    nan_result = _make_result("nan", math.nan)
    high = _make_result("high", 0.95)

    filtered = filter_by_reranker_threshold([nan_result, high], 0.01)

    assert [sr.id for sr in filtered] == ["high"]


def test_threshold_filters_after_combined_scoring_without_recency_leaking_noise():
    high_relevance_old = _make_result("relevant", 0.96, ce_norm=0.72, occurred_start=NOW.replace(year=2025))
    low_relevance_recent = _make_result("noise", 0.0001, ce_norm=0.5, occurred_start=NOW)

    apply_combined_scoring([high_relevance_old, low_relevance_recent], now=NOW)
    scored = sorted([high_relevance_old, low_relevance_recent], key=lambda sr: sr.weight, reverse=True)

    filtered = filter_by_reranker_threshold(scored, 0.01)

    assert [sr.id for sr in filtered] == ["relevant"]


def test_low_threshold_preserves_weak_but_nontrivial_relevance():
    weak_match = _make_result("weak-match", 0.02)
    noise = _make_result("noise", 0.0002)

    filtered = filter_by_reranker_threshold([weak_match, noise], 0.01)

    assert [sr.id for sr in filtered] == ["weak-match"]


def test_config_reads_reranker_threshold_from_env(monkeypatch):
    monkeypatch.setenv(ENV_RERANKER_THRESHOLD, "0.01")

    config = HindsightConfig.from_env()

    assert config.reranker_threshold == 0.01


def test_config_defaults_reranker_threshold(monkeypatch):
    monkeypatch.delenv(ENV_RERANKER_THRESHOLD, raising=False)

    config = HindsightConfig.from_env()

    assert config.reranker_threshold == DEFAULT_RERANKER_THRESHOLD


def test_reranker_threshold_is_static_server_config():
    assert "reranker_threshold" not in HindsightConfig.get_configurable_fields()
