"""
Tests for combined scoring (apply_combined_scoring).

The function applies multiplicative recency/temporal boosts to the cross-encoder
score so that the relative influence of these signals is proportional to the base
relevance score, independent of the cross-encoder model's score calibration.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from hindsight_api.engine.search.reranking import apply_combined_scoring, _RECENCY_ALPHA, _TEMPORAL_ALPHA
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult, ScoredResult

UTC = timezone.utc
NOW = datetime(2024, 6, 1, tzinfo=UTC)


def _make_result(
    ce_norm: float,
    occurred_start: datetime | None = None,
    temporal_proximity: float | None = None,
) -> ScoredResult:
    retrieval = MagicMock(spec=RetrievalResult)
    retrieval.occurred_start = occurred_start
    retrieval.temporal_proximity = temporal_proximity

    candidate = MagicMock(spec=MergedCandidate)
    candidate.retrieval = retrieval
    candidate.rrf_score = 0.05

    return ScoredResult(
        candidate=candidate,
        cross_encoder_score=1.0,
        cross_encoder_score_normalized=ce_norm,
        weight=ce_norm,
    )


class TestBoostFormula:
    def test_neutral_signals_leave_score_unchanged(self):
        """recency=0.5 and temporal=0.5 both produce boost=1.0, so weight == ce."""
        sr = _make_result(ce_norm=0.6)
        apply_combined_scoring([sr], now=NOW)
        assert abs(sr.weight - 0.6) < 1e-9

    def test_max_recency_boost(self):
        """A memory from today (recency≈1.0) should boost by (1 + alpha*0.5)."""
        sr = _make_result(ce_norm=0.5, occurred_start=NOW)
        apply_combined_scoring([sr], now=NOW)
        expected = 0.5 * (1.0 + _RECENCY_ALPHA * 0.5) * 1.0  # temporal neutral
        assert abs(sr.weight - expected) < 1e-6

    def test_min_recency_penalty(self):
        """A memory from >365 days ago (recency=0.1) should penalise score."""
        old = NOW - timedelta(days=400)
        sr = _make_result(ce_norm=0.5, occurred_start=old)
        apply_combined_scoring([sr], now=NOW)
        expected = 0.5 * (1.0 + _RECENCY_ALPHA * (0.1 - 0.5)) * 1.0
        assert abs(sr.weight - expected) < 1e-6

    def test_max_temporal_boost(self):
        """temporal_proximity=1.0 should boost by (1 + alpha*0.5)."""
        sr = _make_result(ce_norm=0.5, temporal_proximity=1.0)
        apply_combined_scoring([sr], now=NOW)
        expected = 0.5 * 1.0 * (1.0 + _TEMPORAL_ALPHA * 0.5)  # recency neutral
        assert abs(sr.weight - expected) < 1e-6

    def test_temporal_none_is_neutral(self):
        """temporal_proximity=None must be treated as 0.5 (no boost/penalty)."""
        sr_none = _make_result(ce_norm=0.5, temporal_proximity=None)
        sr_half = _make_result(ce_norm=0.5, temporal_proximity=0.5)
        apply_combined_scoring([sr_none], now=NOW)
        apply_combined_scoring([sr_half], now=NOW)
        assert abs(sr_none.weight - sr_half.weight) < 1e-9

    def test_both_signals_combined(self):
        """Both boosts are applied multiplicatively."""
        sr = _make_result(ce_norm=0.5, occurred_start=NOW, temporal_proximity=1.0)
        apply_combined_scoring([sr], now=NOW)
        recency_boost = 1.0 + _RECENCY_ALPHA * (1.0 - 0.5)
        temporal_boost = 1.0 + _TEMPORAL_ALPHA * (1.0 - 0.5)
        expected = 0.5 * recency_boost * temporal_boost
        assert abs(sr.weight - expected) < 1e-6

    def test_boost_is_proportional_to_ce(self):
        """The absolute boost from recency scales with the CE score."""
        sr_high = _make_result(ce_norm=0.9, occurred_start=NOW)
        sr_low = _make_result(ce_norm=0.3, occurred_start=NOW)
        apply_combined_scoring([sr_high, sr_low], now=NOW)

        # Both get the same recency boost factor — absolute gain is proportional to CE
        boost_factor = 1.0 + _RECENCY_ALPHA * 0.5
        assert abs(sr_high.weight - 0.9 * boost_factor) < 1e-6
        assert abs(sr_low.weight - 0.3 * boost_factor) < 1e-6

    def test_boost_capped(self):
        """Max boost: recency=1.0 + temporal=1.0 gives ≤21% uplift on CE."""
        sr = _make_result(ce_norm=1.0, occurred_start=NOW, temporal_proximity=1.0)
        apply_combined_scoring([sr], now=NOW)
        assert sr.weight <= 1.0 * (1 + _RECENCY_ALPHA / 2) * (1 + _TEMPORAL_ALPHA / 2) + 1e-9

    def test_rrf_normalized_always_zero(self):
        """RRF is excluded from scoring; rrf_normalized is set to 0.0 for trace clarity."""
        sr = _make_result(ce_norm=0.5)
        apply_combined_scoring([sr], now=NOW)
        assert sr.rrf_normalized == 0.0

    def test_combined_score_equals_weight(self):
        """combined_score and weight must stay in sync."""
        sr = _make_result(ce_norm=0.7, occurred_start=NOW, temporal_proximity=0.8)
        apply_combined_scoring([sr], now=NOW)
        assert sr.combined_score == sr.weight

    def test_model_calibration_independence(self):
        """
        A low-calibration model (low CE scores) and a high-calibration model
        (high CE scores) should produce the same ranking for identical content.

        With additive scoring the recency term would dominate for low-CE models;
        with multiplicative boosting the relative ranking is stable.
        """
        recent = NOW - timedelta(days=10)
        old = NOW - timedelta(days=300)

        # High-calibration model: clear winner is #1 (more relevant, slightly older)
        h_relevant = _make_result(ce_norm=0.85, occurred_start=old)
        h_recent = _make_result(ce_norm=0.60, occurred_start=recent)
        apply_combined_scoring([h_relevant, h_recent], now=NOW)
        assert h_relevant.weight > h_recent.weight, "High-CE model: relevance should win"

        # Low-calibration model: same relative difference, just compressed scores
        l_relevant = _make_result(ce_norm=0.34, occurred_start=old)
        l_recent = _make_result(ce_norm=0.24, occurred_start=recent)
        apply_combined_scoring([l_relevant, l_recent], now=NOW)
        assert l_relevant.weight > l_recent.weight, "Low-CE model: relevance should still win"

    def test_no_occurred_start_defaults_recency_neutral(self):
        """Missing occurred_start → recency=0.5 → no boost/penalty."""
        sr = _make_result(ce_norm=0.5, occurred_start=None)
        apply_combined_scoring([sr], now=NOW)
        assert sr.recency == 0.5
        assert abs(sr.weight - 0.5) < 1e-9

    def test_timezone_naive_occurred_start_handled(self):
        """Naive datetimes in occurred_start should not raise."""
        naive_date = datetime(2024, 1, 1)  # no tzinfo
        sr = _make_result(ce_norm=0.5, occurred_start=naive_date)
        apply_combined_scoring([sr], now=NOW)  # must not raise
        assert 0.0 < sr.weight < 1.0

    def test_custom_alpha_values(self):
        """Custom alpha parameters are respected."""
        sr = _make_result(ce_norm=0.5, occurred_start=NOW)
        apply_combined_scoring([sr], now=NOW, recency_alpha=0.4, temporal_alpha=0.0)
        expected = 0.5 * (1.0 + 0.4 * 0.5) * 1.0
        assert abs(sr.weight - expected) < 1e-6

    def test_future_event_recency_capped_at_one(self):
        """Events in the future must not produce recency > 1.0, keeping boost within bounds."""
        future = NOW + timedelta(days=180)
        sr = _make_result(ce_norm=0.5, occurred_start=future)
        apply_combined_scoring([sr], now=NOW)
        assert sr.recency == 1.0
        expected_max_boost = 1.0 + _RECENCY_ALPHA * 0.5
        assert sr.weight <= 0.5 * expected_max_boost + 1e-9

    def test_empty_list_is_noop(self):
        apply_combined_scoring([], now=NOW)  # must not raise
