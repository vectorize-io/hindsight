"""Unit tests for the per-budget reranker candidate cap (issue #3107).

The cross-encoder always pre-filtered to a flat 300 candidates regardless of the recall budget.
`_resolve_reranker_max_candidates` lets an operator scale that cap by the budget level via env config,
while defaulting (per-level = 0) to the flat `reranker_max_candidates` for full backwards compatibility.
Pure function — no DB, no LLM.
"""

from types import SimpleNamespace

from hindsight_api.engine.memory_engine import Budget, _resolve_reranker_max_candidates


def _config(low: int = 0, mid: int = 0, high: int = 0, flat: int = 300) -> SimpleNamespace:
    return SimpleNamespace(
        reranker_max_candidates=flat,
        reranker_max_candidates_low=low,
        reranker_max_candidates_mid=mid,
        reranker_max_candidates_high=high,
    )


def test_all_levels_unset_falls_back_to_flat_default():
    cfg = _config(flat=300)
    for budget in (Budget.LOW, Budget.MID, Budget.HIGH, None):
        assert _resolve_reranker_max_candidates(cfg, budget) == 300


def test_per_level_override_applied_when_set():
    cfg = _config(low=50, mid=200, high=1000, flat=300)
    assert _resolve_reranker_max_candidates(cfg, Budget.LOW) == 50
    assert _resolve_reranker_max_candidates(cfg, Budget.MID) == 200
    assert _resolve_reranker_max_candidates(cfg, Budget.HIGH) == 1000


def test_partial_mapping_falls_back_per_level():
    """An unset level (0) uses the flat default even when other levels are set."""
    cfg = _config(low=50, mid=0, high=1000, flat=300)
    assert _resolve_reranker_max_candidates(cfg, Budget.LOW) == 50
    assert _resolve_reranker_max_candidates(cfg, Budget.MID) == 300  # unset → flat
    assert _resolve_reranker_max_candidates(cfg, Budget.HIGH) == 1000


def test_none_budget_uses_mid_level():
    cfg = _config(low=50, mid=200, high=1000, flat=300)
    assert _resolve_reranker_max_candidates(cfg, None) == 200
