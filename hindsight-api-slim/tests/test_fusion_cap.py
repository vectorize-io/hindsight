"""Tests for per-source candidate capping before RRF fusion."""

from hindsight_api.engine.search.fusion import cap_per_source
from hindsight_api.engine.search.types import RetrievalResult


def _results(n: int) -> list[RetrievalResult]:
    return [RetrievalResult(id=str(i), text=f"r{i}", fact_type="world") for i in range(n)]


def test_cap_truncates_to_top_n():
    results = _results(10)
    capped = cap_per_source(results, 3)
    assert [r.id for r in capped] == ["0", "1", "2"]


def test_cap_preserves_order():
    """Capping must keep the caller's best-first ordering (it only slices)."""
    results = _results(5)
    capped = cap_per_source(results, 2)
    assert capped == results[:2]


def test_cap_zero_disables():
    results = _results(5)
    # 0 means "unlimited" — return the list untouched (same object, no copy).
    assert cap_per_source(results, 0) is results


def test_cap_negative_disables():
    results = _results(5)
    assert cap_per_source(results, -1) is results


def test_cap_at_or_above_length_is_noop():
    results = _results(4)
    assert cap_per_source(results, 4) is results
    assert cap_per_source(results, 10) is results


def test_cap_empty_list():
    assert cap_per_source([], 5) == []
