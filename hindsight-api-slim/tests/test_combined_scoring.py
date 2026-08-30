"""
Tests for combined scoring (apply_combined_scoring).

The function applies multiplicative recency/temporal boosts to the cross-encoder
score so that the relative influence of these signals is proportional to the base
relevance score, independent of the cross-encoder model's score calibration.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from hindsight_api.engine.search.reranking import (
    _RECENCY_ALPHA,
    _TEMPORAL_ALPHA,
    apply_combined_scoring,
    compute_occurrence_recency,
    compute_recency_decay,
)
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult, ScoredResult
from hindsight_api.engine.temporal_precision import OCCURRENCE_PRECISION_METADATA_KEY

UTC = timezone.utc
NOW = datetime(2024, 6, 1, tzinfo=UTC)


def _make_result(
    ce_norm: float,
    occurred_start: datetime | None = None,
    temporal_proximity: float | None = None,
    mentioned_at: datetime | None = None,
    occurred_end: datetime | None = None,
    metadata: dict[str, str] | None = None,
    text: str = "test",
    result_id: str = "test",
    fact_type: str = "world",
    proof_count: int | None = None,
    rrf_score: float = 0.05,
) -> ScoredResult:
    retrieval = RetrievalResult(
        id=result_id,
        text=text,
        fact_type=fact_type,
        occurred_start=occurred_start,
        occurred_end=occurred_end,
        mentioned_at=mentioned_at,
        metadata=metadata,
        proof_count=proof_count,
        temporal_proximity=temporal_proximity,
    )

    candidate = MergedCandidate(
        retrieval=retrieval,
        rrf_score=rrf_score,
    )

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

    def test_no_effective_time_defaults_recency_neutral(self):
        """No effective time at all (occurred_start/mentioned_at/occurred_end) → recency=0.5."""
        sr = _make_result(ce_norm=0.5)
        apply_combined_scoring([sr], now=NOW)
        assert sr.recency == 0.5
        assert abs(sr.weight - 0.5) < 1e-9

    def test_mentioned_at_drives_recency_when_no_occurred_start(self):
        """A memory with only mentioned_at must derive recency from it, not stay neutral."""
        sr = _make_result(ce_norm=0.5, mentioned_at=NOW)
        apply_combined_scoring([sr], now=NOW)
        assert sr.recency == 1.0
        assert sr.weight > 0.5

    def test_occurred_end_is_last_recency_fallback(self):
        """occurred_end feeds recency when neither occurred_start nor mentioned_at is set."""
        old = NOW - timedelta(days=400)
        sr = _make_result(ce_norm=0.5, occurred_end=old)
        apply_combined_scoring([sr], now=NOW)
        assert sr.recency == 0.1
        assert sr.weight < 0.5

    def test_occurred_start_takes_precedence_over_mentioned_at(self):
        """occurred_start wins over mentioned_at (matches _coalesce_date COALESCE order)."""
        recent = NOW - timedelta(days=10)
        old = NOW - timedelta(days=400)
        sr = _make_result(ce_norm=0.5, occurred_start=recent, mentioned_at=old)
        apply_combined_scoring([sr], now=NOW)
        assert sr.recency > 0.9

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


class TestRecencyDecayFunction:
    """The configurable age→freshness curve (compute_recency_decay)."""

    def test_linear_is_default_and_unchanged(self):
        """Default function reproduces the historical linear decay over 365 days."""
        assert compute_recency_decay(0) == 1.0
        assert abs(compute_recency_decay(182.5) - 0.5) < 1e-6  # neutral at half the window
        assert compute_recency_decay(400) == 0.1  # floored past the window

    def test_linear_window_is_configurable(self):
        """A custom window moves the neutral crossing; 730d window → neutral at 365d."""
        assert abs(compute_recency_decay(365, "linear", linear_window_days=730) - 0.5) < 1e-6

    def test_exponential_neutral_at_halflife(self):
        """Exponential decay is exactly neutral (0.5) at the configured half-life."""
        assert compute_recency_decay(0, "exponential", halflife_days=90) == 1.0
        assert abs(compute_recency_decay(90, "exponential", halflife_days=90) - 0.5) < 1e-9
        assert abs(compute_recency_decay(180, "exponential", halflife_days=90) - 0.25) < 1e-9

    def test_exponential_penalises_old_less_harshly_than_linear(self):
        """A 1-year-old memory keeps more freshness under a 90d-halflife exponential
        than under the linear floor — the curve never hard-cuts to 0.1."""
        lin = compute_recency_decay(365, "linear")
        exp = compute_recency_decay(365, "exponential", halflife_days=180)
        assert exp > lin

    def test_none_is_always_neutral(self):
        """'none' disables the recency signal — always neutral, no boost."""
        assert compute_recency_decay(0, "none") == 0.5
        assert compute_recency_decay(10_000, "none") == 0.5

    def test_future_dates_clamp_to_max(self):
        """Negative ages (future-dated memories) never exceed full freshness."""
        assert compute_recency_decay(-100, "linear") == 1.0
        assert compute_recency_decay(-100, "exponential", halflife_days=90) == 1.0

    def test_nonpositive_halflife_falls_back_to_neutral(self):
        """A misconfigured (<=0) half-life degrades to neutral rather than dividing by zero."""
        assert compute_recency_decay(30, "exponential", halflife_days=0) == 0.5

    def test_function_threads_through_apply_combined_scoring(self):
        """The decay function chosen at the call site is what scores sr.recency."""
        old = NOW - timedelta(days=180)
        sr = _make_result(ce_norm=0.5, occurred_start=old)
        apply_combined_scoring([sr], now=NOW, recency_decay_function="none")
        assert sr.recency == 0.5
        assert abs(sr.weight - 0.5) < 1e-9  # neutral → no recency boost


class TestCoarseOccurrenceRecency:
    # The issue trace was captured shortly after 01:34 UTC on 2026-08-30,
    # making the exact 2026-08-24 candidate about 6.066 days old.
    ISSUE_NOW = datetime(2026, 8, 30, 1, 34, 50, 950707, tzinfo=UTC)

    @staticmethod
    def _precision_metadata(precision: str) -> dict[str, str]:
        return {OCCURRENCE_PRECISION_METADATA_KEY: precision}

    def test_issue_3893_stronger_summit_match_remains_top_one(self):
        summit = _make_result(
            ce_norm=0.999924,
            occurred_start=datetime(2026, 1, 1, tzinfo=UTC),
            occurred_end=datetime(2026, 1, 1, tzinfo=UTC),
            metadata=self._precision_metadata("year"),
            text="用户在杭州开源峰会分享了时间感知记忆 | When: 2026 | Involving: 用户",
        )
        theme = _make_result(
            ce_norm=0.900004,
            occurred_start=datetime(2026, 8, 24, tzinfo=UTC),
            occurred_end=datetime(2026, 8, 24, tzinfo=UTC),
            metadata=self._precision_metadata("day"),
            text="用户切换了浅色主题 | When: August 24, 2026",
        )

        apply_combined_scoring([summit, theme], now=self.ISSUE_NOW)

        assert summit.recency == 0.5
        assert theme.recency == pytest.approx(0.9833811849725)
        assert summit.weight == pytest.approx(0.999924)
        assert theme.weight == pytest.approx(0.987013, abs=1e-6)
        assert summit.weight > theme.weight

    @pytest.mark.parametrize(
        ("precision", "occurred_start", "function", "expected_relation"),
        [
            ("year", datetime(2026, 1, 1, tzinfo=UTC), "linear", "neutral"),
            ("year", datetime(2020, 1, 1, tzinfo=UTC), "linear", "old"),
            ("month", datetime(2026, 8, 1, tzinfo=UTC), "linear", "recent"),
            ("year", datetime(2026, 1, 1, tzinfo=UTC), "exponential", "neutral"),
            ("year", datetime(2020, 1, 1, tzinfo=UTC), "exponential", "old"),
            ("month", datetime(2026, 8, 1, tzinfo=UTC), "exponential", "recent"),
            ("month", datetime(2026, 8, 1, tzinfo=UTC), "none", "neutral"),
        ],
    )
    def test_uncertainty_envelope_is_conservative(self, precision, occurred_start, function, expected_relation):
        recency = compute_occurrence_recency(
            occurred_start,
            self.ISSUE_NOW,
            precision,
            function=function,
            halflife_days=90,
        )

        if expected_relation == "neutral":
            assert recency == 0.5
        elif expected_relation == "old":
            assert recency < 0.5
        else:
            assert 0.5 < recency < 1.0

    def test_recent_mention_does_not_refresh_an_old_coarse_occurrence(self):
        result = _make_result(
            ce_norm=0.5,
            occurred_start=datetime(2020, 1, 1, tzinfo=UTC),
            occurred_end=datetime(2020, 1, 1, tzinfo=UTC),
            mentioned_at=self.ISSUE_NOW,
            metadata=self._precision_metadata("year"),
            text="Historical event | When: 2020",
        )

        apply_combined_scoring([result], now=self.ISSUE_NOW)

        assert result.recency == 0.1
        assert result.weight < 0.5

    @pytest.mark.parametrize("metadata", [None, {OCCURRENCE_PRECISION_METADATA_KEY: "day"}])
    def test_genuine_january_first_preserves_exact_date_behavior(self, metadata):
        result = _make_result(
            ce_norm=0.5,
            occurred_start=datetime(2026, 1, 1, tzinfo=UTC),
            occurred_end=datetime(2026, 1, 1, tzinfo=UTC),
            metadata=metadata,
            text="New year event | When: January 1, 2026",
        )

        apply_combined_scoring([result], now=self.ISSUE_NOW)

        exact_age_days = (self.ISSUE_NOW - datetime(2026, 1, 1, tzinfo=UTC)).total_seconds() / 86400
        assert result.recency == pytest.approx(compute_recency_decay(exact_age_days))
        assert result.recency < 0.5

    def test_metadata_day_prevents_legacy_when_from_reclassifying_exact_event(self):
        result = _make_result(
            ce_norm=0.5,
            occurred_start=datetime(2026, 1, 1, tzinfo=UTC),
            occurred_end=datetime(2026, 1, 1, tzinfo=UTC),
            metadata=self._precision_metadata("day"),
            text="New year event | When: 2026",
        )

        apply_combined_scoring([result], now=self.ISSUE_NOW)

        assert result.recency < 0.5

    def test_legacy_canonical_when_gets_coarse_scoring_without_metadata(self):
        result = _make_result(
            ce_norm=0.5,
            occurred_start=datetime(2026, 1, 1, 0, 0, 0, 10_000, tzinfo=UTC),
            occurred_end=datetime(2026, 1, 1, 0, 0, 0, 10_000, tzinfo=UTC),
            text="Summit talk | When: 2026 | Involving: user",
        )

        apply_combined_scoring([result], now=self.ISSUE_NOW)

        assert result.recency == 0.5

    def test_postgres_json_metadata_is_normalized_before_coarse_scoring(self):
        occurred = datetime(2026, 1, 1, tzinfo=UTC)
        retrieval = RetrievalResult.from_db_row(
            {
                "id": "postgres-jsonb",
                "text": "Verbatim source without a canonical temporal segment",
                "fact_type": "world",
                "occurred_start": occurred,
                "occurred_end": occurred,
                "metadata": json.dumps(self._precision_metadata("year")),
            }
        )
        result = ScoredResult(
            candidate=MergedCandidate(retrieval=retrieval, rrf_score=0.05),
            cross_encoder_score=1.0,
            cross_encoder_score_normalized=0.5,
            weight=0.5,
        )

        apply_combined_scoring([result], now=self.ISSUE_NOW)

        assert retrieval.metadata == self._precision_metadata("year")
        assert result.recency == 0.5

    def test_arbitrary_year_prose_is_not_legacy_precision_evidence(self):
        result = _make_result(
            ce_norm=0.5,
            occurred_start=datetime(2026, 1, 1, tzinfo=UTC),
            occurred_end=datetime(2026, 1, 1, tzinfo=UTC),
            text="The summit happened in 2026",
        )

        apply_combined_scoring([result], now=self.ISSUE_NOW)

        assert result.recency < 0.5

    def test_derived_observation_does_not_use_legacy_when_recovery(self):
        result = _make_result(
            ce_norm=0.5,
            occurred_start=datetime(2026, 1, 1, tzinfo=UTC),
            occurred_end=datetime(2026, 1, 1, tzinfo=UTC),
            text="Derived observation | When: 2026",
            fact_type="observation",
        )

        apply_combined_scoring([result], now=self.ISSUE_NOW)

        exact_age_days = (self.ISSUE_NOW - datetime(2026, 1, 1, tzinfo=UTC)).total_seconds() / 86400
        assert result.recency == pytest.approx(compute_recency_decay(exact_age_days))

    def test_genuine_range_keeps_existing_start_based_recency(self):
        result = _make_result(
            ce_norm=0.5,
            occurred_start=datetime(2026, 1, 1, tzinfo=UTC),
            occurred_end=datetime(2026, 12, 31, tzinfo=UTC),
            metadata=self._precision_metadata("range"),
            text="Project ran throughout 2026 | When: 2026",
        )

        apply_combined_scoring([result], now=self.ISSUE_NOW)

        exact_age_days = (self.ISSUE_NOW - datetime(2026, 1, 1, tzinfo=UTC)).total_seconds() / 86400
        assert result.recency == pytest.approx(compute_recency_decay(exact_age_days))

    @pytest.mark.parametrize("precision", ["day", "instant", "range", "unknown"])
    @pytest.mark.parametrize("function", ["linear", "exponential", "none"])
    def test_exact_precision_modes_keep_historical_decay(self, precision, function):
        occurred = datetime(2026, 1, 1, 12, 30, tzinfo=UTC)
        age_days = (self.ISSUE_NOW - occurred).total_seconds() / 86400

        actual = compute_occurrence_recency(
            occurred,
            self.ISSUE_NOW,
            precision,
            function=function,
            linear_window_days=730,
            halflife_days=45,
        )
        expected = compute_recency_decay(
            age_days,
            function=function,
            linear_window_days=730,
            halflife_days=45,
        )

        assert actual == pytest.approx(expected)

    def test_naive_and_aware_exact_datetimes_match_for_the_same_instant(self):
        naive_utc = datetime(2026, 8, 20, 12, 0)
        aware_local = datetime(2026, 8, 20, 20, 0, tzinfo=timezone(timedelta(hours=8)))

        naive_recency = compute_occurrence_recency(naive_utc, self.ISSUE_NOW, "instant")
        aware_recency = compute_occurrence_recency(aware_local, self.ISSUE_NOW, "instant")

        assert naive_recency == pytest.approx(aware_recency)

    @pytest.mark.parametrize("function", ["linear", "exponential", "none"])
    def test_old_current_and_future_coarse_periods_across_calendar_boundaries(self, function):
        recencies = {
            "old_year": compute_occurrence_recency(
                datetime(2020, 1, 1, tzinfo=UTC),
                self.ISSUE_NOW,
                "year",
                function=function,
                halflife_days=90,
            ),
            "current_year": compute_occurrence_recency(
                datetime(2026, 1, 1, tzinfo=UTC),
                self.ISSUE_NOW,
                "year",
                function=function,
                halflife_days=90,
            ),
            "future_year": compute_occurrence_recency(
                datetime(2027, 1, 1, tzinfo=UTC),
                self.ISSUE_NOW,
                "year",
                function=function,
                halflife_days=90,
            ),
            "previous_december": compute_occurrence_recency(
                datetime(2025, 12, 1, tzinfo=UTC),
                self.ISSUE_NOW,
                "month",
                function=function,
                halflife_days=90,
            ),
            "leap_february": compute_occurrence_recency(
                datetime(2024, 2, 1, tzinfo=UTC),
                self.ISSUE_NOW,
                "month",
                function=function,
                halflife_days=90,
            ),
            "current_month": compute_occurrence_recency(
                datetime(2026, 8, 1, tzinfo=UTC),
                self.ISSUE_NOW,
                "month",
                function=function,
                halflife_days=90,
            ),
            "future_month": compute_occurrence_recency(
                datetime(2026, 9, 1, tzinfo=UTC),
                self.ISSUE_NOW,
                "month",
                function=function,
                halflife_days=90,
            ),
        }

        if function == "none":
            assert set(recencies.values()) == {0.5}
            return

        assert recencies["old_year"] < 0.5
        assert recencies["previous_december"] < 0.5
        assert recencies["leap_february"] < 0.5
        assert recencies["current_year"] == 0.5
        assert 0.5 < recencies["current_month"] < 1.0
        assert recencies["future_year"] == 1.0
        assert recencies["future_month"] == 1.0

    def test_mixed_precision_candidates_rank_together_with_near_tie(self):
        candidates = [
            _make_result(
                ce_norm=0.99,
                occurred_start=datetime(2026, 1, 1, tzinfo=UTC),
                occurred_end=datetime(2026, 1, 1, tzinfo=UTC),
                mentioned_at=self.ISSUE_NOW,
                metadata=self._precision_metadata("year"),
                result_id="year",
            ),
            _make_result(
                ce_norm=0.95,
                occurred_start=datetime(2026, 8, 1, tzinfo=UTC),
                occurred_end=datetime(2026, 8, 1, tzinfo=UTC),
                metadata=self._precision_metadata("month"),
                result_id="month",
            ),
            _make_result(
                ce_norm=0.94,
                occurred_start=datetime(2026, 8, 29, tzinfo=UTC),
                occurred_end=datetime(2026, 8, 29, tzinfo=UTC),
                metadata=self._precision_metadata("day"),
                result_id="day",
            ),
            _make_result(
                ce_norm=0.939,
                occurred_start=datetime(2026, 8, 29, 12, 0, tzinfo=UTC),
                occurred_end=datetime(2026, 8, 29, 12, 0, tzinfo=UTC),
                metadata=self._precision_metadata("instant"),
                result_id="instant",
            ),
            _make_result(
                ce_norm=0.98,
                occurred_start=datetime(2026, 1, 1, tzinfo=UTC),
                occurred_end=datetime(2026, 6, 30, tzinfo=UTC),
                metadata=self._precision_metadata("range"),
                result_id="range",
            ),
            _make_result(
                ce_norm=0.9,
                mentioned_at=self.ISSUE_NOW,
                result_id="undated",
            ),
        ]

        apply_combined_scoring(candidates, now=self.ISSUE_NOW)
        by_id = {candidate.id: candidate for candidate in candidates}
        ranked_ids = [candidate.id for candidate in sorted(candidates, key=lambda item: item.weight, reverse=True)]

        assert ranked_ids == ["day", "instant", "month", "undated", "year", "range"]
        assert by_id["year"].recency == 0.5
        assert by_id["month"].recency > 0.5
        assert by_id["day"].recency < by_id["instant"].recency
        assert by_id["range"].recency == pytest.approx(
            compute_recency_decay((self.ISSUE_NOW - datetime(2026, 1, 1, tzinfo=UTC)).total_seconds() / 86400)
        )
        assert by_id["undated"].recency == 1.0
        assert by_id["undated"].weight == pytest.approx(by_id["year"].weight)
        assert all(candidate.combined_score == candidate.weight for candidate in candidates)

    def test_none_decay_keeps_mixed_precision_weights_equal_to_ce_scores(self):
        candidates = [
            _make_result(
                ce_norm=0.91,
                occurred_start=datetime(2020, 1, 1, tzinfo=UTC),
                metadata=self._precision_metadata("year"),
                result_id="year",
            ),
            _make_result(
                ce_norm=0.87,
                occurred_start=datetime(2026, 9, 1, tzinfo=UTC),
                metadata=self._precision_metadata("month"),
                result_id="month",
            ),
            _make_result(
                ce_norm=0.83,
                occurred_start=datetime(2026, 1, 1, tzinfo=UTC),
                metadata=self._precision_metadata("day"),
                result_id="day",
            ),
            _make_result(
                ce_norm=0.79,
                occurred_start=datetime(2026, 8, 30, 1, tzinfo=UTC),
                metadata=self._precision_metadata("instant"),
                result_id="instant",
            ),
            _make_result(
                ce_norm=0.75,
                occurred_start=datetime(2024, 1, 1, tzinfo=UTC),
                occurred_end=datetime(2024, 12, 31, tzinfo=UTC),
                metadata=self._precision_metadata("range"),
                result_id="range",
            ),
            _make_result(ce_norm=0.71, mentioned_at=self.ISSUE_NOW, result_id="undated"),
        ]

        apply_combined_scoring(candidates, now=self.ISSUE_NOW, recency_decay_function="none")

        assert [candidate.recency for candidate in candidates] == [0.5] * len(candidates)
        assert [candidate.weight for candidate in candidates] == pytest.approx(
            [candidate.cross_encoder_score_normalized for candidate in candidates]
        )

    def test_passthrough_reranker_keeps_rrf_order_with_mixed_dates(self):
        candidates = [
            _make_result(
                ce_norm=0.5,
                occurred_start=datetime(2027, 1, 1, tzinfo=UTC),
                metadata=self._precision_metadata("year"),
                result_id="rrf-low",
                rrf_score=0.1,
            ),
            _make_result(
                ce_norm=0.5,
                mentioned_at=self.ISSUE_NOW,
                result_id="rrf-mid",
                rrf_score=0.5,
            ),
            _make_result(
                ce_norm=0.5,
                occurred_start=datetime(2020, 1, 1, tzinfo=UTC),
                metadata=self._precision_metadata("year"),
                result_id="rrf-high",
                rrf_score=0.9,
            ),
        ]

        apply_combined_scoring(candidates, now=self.ISSUE_NOW, is_passthrough_reranker=True)
        ranked_ids = [candidate.id for candidate in sorted(candidates, key=lambda item: item.weight, reverse=True)]

        assert ranked_ids == ["rrf-high", "rrf-mid", "rrf-low"]
