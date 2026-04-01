"""Tests for link_utils datetime handling, temporal link computation, and semantic link splitting."""
import numpy as np
import pytest
from datetime import datetime, timezone, timedelta

from hindsight_api.engine.retain.link_utils import (
    _normalize_datetime,
    _cap_links_per_unit,
    compute_temporal_links,
    compute_temporal_query_bounds,
    compute_semantic_links_within_batch,
    MAX_TEMPORAL_LINKS_PER_UNIT,
)


class TestNormalizeDatetime:
    """Tests for the _normalize_datetime helper function."""

    def test_none_returns_none(self):
        """Test that None input returns None."""
        assert _normalize_datetime(None) is None

    def test_naive_datetime_becomes_utc(self):
        """Test that naive datetimes are converted to UTC."""
        naive_dt = datetime(2024, 6, 15, 10, 30, 0)
        result = _normalize_datetime(naive_dt)

        assert result.tzinfo is not None
        assert result.tzinfo == timezone.utc
        assert result.year == 2024
        assert result.month == 6
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30

    def test_aware_datetime_unchanged(self):
        """Test that timezone-aware datetimes are returned unchanged."""
        aware_dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = _normalize_datetime(aware_dt)

        assert result == aware_dt
        assert result.tzinfo == timezone.utc

    def test_mixed_datetimes_can_be_compared(self):
        """Test that normalized naive and aware datetimes can be compared."""
        naive_dt = datetime(2024, 6, 15, 10, 30, 0)
        aware_dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)

        normalized_naive = _normalize_datetime(naive_dt)
        normalized_aware = _normalize_datetime(aware_dt)

        # Should be able to compare without TypeError
        assert normalized_naive == normalized_aware


class TestComputeTemporalQueryBounds:
    """Tests for compute_temporal_query_bounds function."""

    def test_empty_units_returns_none(self):
        """Test that empty input returns (None, None)."""
        min_date, max_date = compute_temporal_query_bounds({})
        assert min_date is None
        assert max_date is None

    def test_single_unit_normal_date(self):
        """Test bounds for a single unit with normal date."""
        units = {"unit-1": datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)}
        min_date, max_date = compute_temporal_query_bounds(units, time_window_hours=24)

        assert min_date == datetime(2024, 6, 14, 12, 0, 0, tzinfo=timezone.utc)
        assert max_date == datetime(2024, 6, 16, 12, 0, 0, tzinfo=timezone.utc)

    def test_multiple_units(self):
        """Test bounds span across multiple units."""
        units = {
            "unit-1": datetime(2024, 6, 10, 12, 0, 0, tzinfo=timezone.utc),
            "unit-2": datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
            "unit-3": datetime(2024, 6, 20, 12, 0, 0, tzinfo=timezone.utc),
        }
        min_date, max_date = compute_temporal_query_bounds(units, time_window_hours=24)

        # min should be Jun 10 - 24h = Jun 9
        assert min_date == datetime(2024, 6, 9, 12, 0, 0, tzinfo=timezone.utc)
        # max should be Jun 20 + 24h = Jun 21
        assert max_date == datetime(2024, 6, 21, 12, 0, 0, tzinfo=timezone.utc)

    def test_mixed_naive_and_aware_datetimes(self):
        """Test that mixed naive/aware datetimes work correctly."""
        units = {
            "unit-1": datetime(2024, 6, 10, 12, 0, 0),  # naive
            "unit-2": datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc),  # aware
        }
        # Should not raise TypeError
        min_date, max_date = compute_temporal_query_bounds(units, time_window_hours=24)

        assert min_date is not None
        assert max_date is not None
        assert min_date.tzinfo is not None
        assert max_date.tzinfo is not None

    def test_overflow_near_datetime_min(self):
        """Test overflow protection near datetime.min."""
        units = {"unit-1": datetime(1, 1, 2, 0, 0, tzinfo=timezone.utc)}
        min_date, max_date = compute_temporal_query_bounds(units, time_window_hours=48)

        # Should handle overflow gracefully
        assert min_date == datetime.min.replace(tzinfo=timezone.utc)
        assert max_date is not None

    def test_overflow_near_datetime_max(self):
        """Test overflow protection near datetime.max."""
        units = {"unit-1": datetime(9999, 12, 30, 0, 0, tzinfo=timezone.utc)}
        min_date, max_date = compute_temporal_query_bounds(units, time_window_hours=48)

        # Should handle overflow gracefully
        assert min_date is not None
        assert max_date == datetime.max.replace(tzinfo=timezone.utc)


class TestComputeTemporalLinks:
    """Tests for compute_temporal_links function."""

    def test_empty_units_returns_empty(self):
        """Test that empty input returns empty list."""
        links = compute_temporal_links({}, [])
        assert links == []

    def test_no_candidates_returns_empty(self):
        """Test that no candidates means no links."""
        units = {"unit-1": datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)}
        links = compute_temporal_links(units, [])
        assert links == []

    def test_candidate_within_window_creates_link(self):
        """Test that candidates within time window create links."""
        units = {"unit-1": datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)}
        candidates = [
            {"id": "candidate-1", "event_date": datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc)},
        ]

        links = compute_temporal_links(units, candidates, time_window_hours=24)

        assert len(links) == 1
        assert links[0][0] == "unit-1"
        assert links[0][1] == "candidate-1"
        assert links[0][2] == "temporal"
        assert links[0][4] is None
        # Weight should be high since they're close (2 hours apart)
        assert links[0][3] > 0.9

    def test_candidate_outside_window_no_link(self):
        """Test that candidates outside time window don't create links."""
        units = {"unit-1": datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)}
        candidates = [
            {"id": "candidate-1", "event_date": datetime(2024, 6, 10, 12, 0, 0, tzinfo=timezone.utc)},
        ]

        links = compute_temporal_links(units, candidates, time_window_hours=24)

        assert len(links) == 0

    def test_weight_decreases_with_distance(self):
        """Test that weight decreases as time difference increases."""
        units = {"unit-1": datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)}
        candidates = [
            {"id": "close", "event_date": datetime(2024, 6, 15, 11, 0, 0, tzinfo=timezone.utc)},  # 1 hour
            {"id": "far", "event_date": datetime(2024, 6, 14, 18, 0, 0, tzinfo=timezone.utc)},  # 18 hours
        ]

        links = compute_temporal_links(units, candidates, time_window_hours=24)

        assert len(links) == 2
        close_link = next(l for l in links if l[1] == "close")
        far_link = next(l for l in links if l[1] == "far")

        assert close_link[3] > far_link[3]

    def test_max_10_links_per_unit(self):
        """Test that at most 10 links are created per unit."""
        units = {"unit-1": datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)}
        # Create 15 candidates all within window
        candidates = [
            {"id": f"candidate-{i}", "event_date": datetime(2024, 6, 15, 11, 0, 0, tzinfo=timezone.utc)}
            for i in range(15)
        ]

        links = compute_temporal_links(units, candidates, time_window_hours=24)

        assert len(links) == 10

    def test_multiple_units_multiple_candidates(self):
        """Test with multiple units and candidates."""
        units = {
            "unit-1": datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
            "unit-2": datetime(2024, 6, 20, 12, 0, 0, tzinfo=timezone.utc),
        }
        candidates = [
            {"id": "c1", "event_date": datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc)},  # near unit-1
            {"id": "c2", "event_date": datetime(2024, 6, 20, 10, 0, 0, tzinfo=timezone.utc)},  # near unit-2
            {"id": "c3", "event_date": datetime(2024, 6, 17, 12, 0, 0, tzinfo=timezone.utc)},  # between, near neither
        ]

        links = compute_temporal_links(units, candidates, time_window_hours=24)

        # unit-1 should link to c1 only
        # unit-2 should link to c2 only
        unit1_links = [l for l in links if l[0] == "unit-1"]
        unit2_links = [l for l in links if l[0] == "unit-2"]

        assert len(unit1_links) == 1
        assert unit1_links[0][1] == "c1"

        assert len(unit2_links) == 1
        assert unit2_links[0][1] == "c2"

    def test_mixed_naive_and_aware_datetimes(self):
        """Test that mixed naive/aware datetimes work correctly."""
        units = {"unit-1": datetime(2024, 6, 15, 12, 0, 0)}  # naive
        candidates = [
            {"id": "c1", "event_date": datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc)},  # aware
        ]

        # Should not raise TypeError
        links = compute_temporal_links(units, candidates, time_window_hours=24)
        assert len(links) == 1

    def test_overflow_near_datetime_min(self):
        """Test overflow protection when unit date is near datetime.min."""
        units = {"unit-1": datetime(1, 1, 2, 0, 0, tzinfo=timezone.utc)}
        candidates = [
            {"id": "c1", "event_date": datetime(1, 1, 1, 12, 0, 0, tzinfo=timezone.utc)},
        ]

        # Should not raise OverflowError
        links = compute_temporal_links(units, candidates, time_window_hours=48)
        assert len(links) == 1

    def test_overflow_near_datetime_max(self):
        """Test overflow protection when unit date is near datetime.max."""
        units = {"unit-1": datetime(9999, 12, 30, 0, 0, tzinfo=timezone.utc)}
        candidates = [
            {"id": "c1", "event_date": datetime(9999, 12, 31, 12, 0, 0, tzinfo=timezone.utc)},
        ]

        # Should not raise OverflowError
        links = compute_temporal_links(units, candidates, time_window_hours=48)
        assert len(links) == 1

    def test_weight_minimum_is_0_3(self):
        """Test that weight doesn't go below 0.3."""
        units = {"unit-1": datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)}
        candidates = [
            # 23 hours apart - should be just within 24h window but low weight
            {"id": "c1", "event_date": datetime(2024, 6, 14, 13, 0, 0, tzinfo=timezone.utc)},
        ]

        links = compute_temporal_links(units, candidates, time_window_hours=24)

        assert len(links) == 1
        assert links[0][3] >= 0.3


class TestCapLinksPerUnit:
    """Tests for the _cap_links_per_unit helper function."""

    def test_empty_links(self):
        assert _cap_links_per_unit([]) == []

    def test_under_cap_unchanged(self):
        links = [
            ("unit_a", "unit_x", "temporal", 0.9, None),
            ("unit_a", "unit_y", "temporal", 0.8, None),
        ]
        result = _cap_links_per_unit(links, max_per_unit=5)
        assert len(result) == 2

    def test_caps_to_max_per_unit(self):
        # Create 30 links from the same unit with descending weights
        links = [("unit_a", f"unit_{i}", "temporal", 1.0 - i * 0.01, None) for i in range(30)]
        result = _cap_links_per_unit(links, max_per_unit=10)
        assert len(result) == 10
        # Should keep the highest-weight links
        weights = [lnk[3] for lnk in result]
        assert weights == sorted(weights, reverse=True)
        assert weights[0] == 1.0  # Highest weight kept

    def test_caps_independently_per_unit(self):
        links_a = [("unit_a", f"target_{i}", "temporal", 0.9 - i * 0.01, None) for i in range(10)]
        links_b = [("unit_b", f"target_{i}", "temporal", 0.8 - i * 0.01, None) for i in range(10)]
        result = _cap_links_per_unit(links_a + links_b, max_per_unit=5)
        # 5 from unit_a + 5 from unit_b
        assert len(result) == 10
        from_a = [lnk for lnk in result if lnk[0] == "unit_a"]
        from_b = [lnk for lnk in result if lnk[0] == "unit_b"]
        assert len(from_a) == 5
        assert len(from_b) == 5

    def test_default_max_is_temporal_constant(self):
        links = [("unit_a", f"target_{i}", "temporal", 1.0 - i * 0.01, None) for i in range(50)]
        result = _cap_links_per_unit(links)
        assert len(result) == MAX_TEMPORAL_LINKS_PER_UNIT

    def test_preserves_tuple_structure(self):
        links = [("from_id", "to_id", "temporal", 0.95, "entity_id")]
        result = _cap_links_per_unit(links, max_per_unit=5)
        assert result[0] == ("from_id", "to_id", "temporal", 0.95, "entity_id")


class TestComputeSemanticLinksWithinBatch:
    """Tests for compute_semantic_links_within_batch.

    This function computes semantic links between units in the same batch
    using numpy dot product (no DB access). It runs in Phase 2 (write
    transaction) while the expensive ANN search against existing units runs
    in Phase 1 on a separate connection to avoid TimeoutErrors from HNSW
    index contention under concurrent load.
    """

    def test_empty_returns_empty(self):
        assert compute_semantic_links_within_batch([], []) == []

    def test_single_unit_returns_empty(self):
        emb = [np.random.randn(384).tolist()]
        assert compute_semantic_links_within_batch(["u1"], emb) == []

    def test_identical_embeddings_produce_links(self):
        """Two identical embeddings should have similarity=1.0 (above 0.7 threshold)."""
        emb = [0.1] * 384
        links = compute_semantic_links_within_batch(["u1", "u2"], [emb, emb])
        assert len(links) == 2  # bidirectional: u1→u2, u2→u1
        from_ids = {lnk[0] for lnk in links}
        to_ids = {lnk[1] for lnk in links}
        assert from_ids == {"u1", "u2"}
        assert to_ids == {"u1", "u2"}
        for lnk in links:
            assert lnk[2] == "semantic"
            assert lnk[3] >= 0.99  # near-1.0 similarity
            assert lnk[4] is None  # no entity_id

    def test_orthogonal_embeddings_no_links(self):
        """Orthogonal embeddings should have similarity=0 (below 0.7 threshold)."""
        emb1 = [1.0] + [0.0] * 383
        emb2 = [0.0] + [1.0] + [0.0] * 382
        links = compute_semantic_links_within_batch(["u1", "u2"], [emb1, emb2])
        assert len(links) == 0

    def test_respects_threshold(self):
        """Links below threshold should be excluded."""
        emb1 = np.random.randn(384).tolist()
        # Create a slightly similar embedding (add noise)
        emb2 = [x + np.random.randn() * 0.5 for x in emb1]
        # Normalize both
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        emb1 = [x / norm1 for x in emb1]
        emb2 = [x / norm2 for x in emb2]

        links_low = compute_semantic_links_within_batch(["u1", "u2"], [emb1, emb2], threshold=0.0)
        links_high = compute_semantic_links_within_batch(["u1", "u2"], [emb1, emb2], threshold=0.99)
        # Low threshold should have more links than high threshold
        assert len(links_low) >= len(links_high)

    def test_top_k_limits_per_unit(self):
        """Each unit should link to at most top_k other units."""
        n = 10
        # Create similar embeddings (all close to the same vector)
        base = np.random.randn(384)
        base = base / np.linalg.norm(base)
        embs = [(base + np.random.randn(384) * 0.01).tolist() for _ in range(n)]
        unit_ids = [f"u{i}" for i in range(n)]

        links = compute_semantic_links_within_batch(unit_ids, embs, top_k=3, threshold=0.5)
        # Each unit should have at most 3 outgoing links
        from collections import Counter
        from_counts = Counter(lnk[0] for lnk in links)
        for count in from_counts.values():
            assert count <= 3

    def test_link_tuple_structure(self):
        """Verify the tuple format matches what _bulk_insert_links expects."""
        emb = [0.1] * 384
        links = compute_semantic_links_within_batch(["u1", "u2"], [emb, emb])
        for lnk in links:
            assert len(lnk) == 5
            from_id, to_id, link_type, weight, entity_id = lnk
            assert isinstance(from_id, str)
            assert isinstance(to_id, str)
            assert link_type == "semantic"
            assert 0.0 <= weight <= 1.0
            assert entity_id is None
