"""Tests for per-strategy recall boosting (config parsing + boost math)."""

import pytest

from hindsight_api.config import RECALL_BOOST_LEVELS, _parse_strategy_boosts
from hindsight_api.engine.search.recall_boost import (
    BOOST_LEVELS,
    additive_strategy_boost,
    boosted_rrf_score,
)
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult


def _candidate(rrf_score: float, source_ranks: dict[str, int]) -> MergedCandidate:
    retrieval = RetrievalResult(id="x", text="t", fact_type="world")
    return MergedCandidate(retrieval=retrieval, rrf_score=rrf_score, source_ranks=source_ranks)


# --- level table integrity ----------------------------------------------------


def test_config_levels_match_boost_table():
    """The user-facing level names in config must match the weights table keys."""
    assert set(RECALL_BOOST_LEVELS) == set(BOOST_LEVELS)


def test_levels_are_monotonic():
    """Higher levels must boost more in both stages, or the names lie."""
    low, medium, high = (BOOST_LEVELS[lvl] for lvl in ("low", "medium", "high"))
    assert low.rrf < medium.rrf < high.rrf
    assert low.additive < medium.additive < high.additive


# --- _parse_strategy_boosts ---------------------------------------------------


def test_parse_empty_is_noop():
    assert _parse_strategy_boosts("") == {}
    assert _parse_strategy_boosts(None) == {}
    assert _parse_strategy_boosts("   ") == {}


def test_parse_single_and_multiple():
    assert _parse_strategy_boosts("graph:high") == {"graph": "high"}
    assert _parse_strategy_boosts("graph:high,semantic:low") == {"graph": "high", "semantic": "low"}


def test_parse_is_case_insensitive_and_strips_whitespace():
    assert _parse_strategy_boosts(" GRAPH : HIGH , BM25:Low ") == {"graph": "high", "bm25": "low"}


def test_parse_skips_unknown_strategy():
    assert _parse_strategy_boosts("graphh:high,graph:low") == {"graph": "low"}


def test_parse_skips_unknown_level():
    # A raw number (the old format) is now an invalid level and skipped.
    assert _parse_strategy_boosts("graph:0.1,semantic:medium") == {"semantic": "medium"}
    assert _parse_strategy_boosts("graph:huge") == {}


def test_parse_bare_strategy_defaults_to_medium():
    # A strategy with no level (or a trailing colon) defaults to medium.
    assert _parse_strategy_boosts("graph") == {"graph": "medium"}
    assert _parse_strategy_boosts("graph:") == {"graph": "medium"}
    assert _parse_strategy_boosts("graph,semantic:high") == {"graph": "medium", "semantic": "high"}


def test_parse_skips_empty_name():
    assert _parse_strategy_boosts(":high") == {}


# --- boosted_rrf_score (pre-rerank, rank-aware) -------------------------------


def test_boosted_rrf_noop_when_no_boosts():
    cand = _candidate(0.5, {"graph_rank": 1})
    assert boosted_rrf_score(cand, {}) == 0.5


def test_boosted_rrf_adds_weighted_contribution():
    cand = _candidate(0.5, {"graph_rank": 1})
    expected = 0.5 + BOOST_LEVELS["high"].rrf * (1.0 / 61)
    assert boosted_rrf_score(cand, {"graph": "high"}, k=60) == expected


def test_boosted_rrf_higher_level_boosts_more():
    cand = _candidate(0.5, {"graph_rank": 5})
    low = boosted_rrf_score(cand, {"graph": "low"})
    high = boosted_rrf_score(cand, {"graph": "high"})
    assert high > low > 0.5


def test_boosted_rrf_is_rank_aware():
    """A better rank in the boosted arm yields a larger boost."""
    top = _candidate(0.5, {"graph_rank": 1})
    deep = _candidate(0.5, {"graph_rank": 200})
    assert boosted_rrf_score(top, {"graph": "high"}) > boosted_rrf_score(deep, {"graph": "high"})


def test_boosted_rrf_ignores_non_matching_arm():
    # Candidate only came from semantic; a graph boost must not touch it.
    cand = _candidate(0.5, {"semantic_rank": 3})
    assert boosted_rrf_score(cand, {"graph": "high"}) == 0.5


# --- additive_strategy_boost (post-rerank, flat) ------------------------------


def test_additive_noop_when_no_boosts():
    assert additive_strategy_boost({"graph_rank": 1}, {}) == 0.0


def test_additive_is_flat_regardless_of_rank():
    assert additive_strategy_boost({"graph_rank": 1}, {"graph": "high"}) == BOOST_LEVELS["high"].additive
    assert additive_strategy_boost({"graph_rank": 999}, {"graph": "high"}) == BOOST_LEVELS["high"].additive


def test_additive_sums_matched_arms():
    ranks = {"graph_rank": 2, "semantic_rank": 5}
    expected = BOOST_LEVELS["high"].additive + BOOST_LEVELS["low"].additive
    assert additive_strategy_boost(ranks, {"graph": "high", "semantic": "low"}) == pytest.approx(expected)


def test_additive_ignores_unmatched_arm():
    assert additive_strategy_boost({"semantic_rank": 1}, {"graph": "high"}) == 0.0
