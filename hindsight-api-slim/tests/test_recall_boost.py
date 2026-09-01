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
    assert low.rank_divisor < medium.rank_divisor < high.rank_divisor
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


def test_boosted_rrf_promotes_the_arm_in_rank_space():
    """The boosted arm contributes as if it had placed rank/divisor."""
    cand = _candidate(0.5, {"graph_rank": 8})
    divisor = BOOST_LEVELS["high"].rank_divisor
    # rank 8 at divisor 8 contributes as rank 1, replacing its rank-8 contribution.
    expected = 0.5 + (1.0 / (60 + 8 / divisor)) - (1.0 / 68)
    assert boosted_rrf_score(cand, {"graph": "high"}, k=60) == pytest.approx(expected)


def test_boosted_rrf_higher_level_boosts_more():
    cand = _candidate(0.5, {"graph_rank": 5})
    low = boosted_rrf_score(cand, {"graph": "low"})
    high = boosted_rrf_score(cand, {"graph": "high"})
    assert high > low > 0.5


def test_boosted_rrf_is_rank_aware():
    """Boosting preserves the boosted arm's internal order.

    Note the boost *delta* is deliberately largest deep in the arm, where the
    reranker cut bites — a rank-1 candidate needs no rescuing. So this asserts
    the invariant that matters, final-score monotonicity, using base scores
    consistent with the ranks (as fusion produces them) rather than a flat stub.
    """
    top = _candidate(1.0 / 61, {"graph_rank": 1})
    deep = _candidate(1.0 / 260, {"graph_rank": 200})
    assert boosted_rrf_score(top, {"graph": "high"}) > boosted_rrf_score(deep, {"graph": "high"})


def test_boost_delta_is_largest_where_the_cut_bites():
    """The rescue is aimed at candidates near the cut, not at the arm's head."""
    top = _candidate(1.0 / 61, {"graph_rank": 1})
    deep = _candidate(1.0 / 260, {"graph_rank": 200})
    top_delta = boosted_rrf_score(top, {"graph": "high"}) - top.rrf_score
    deep_delta = boosted_rrf_score(deep, {"graph": "high"}) - deep.rrf_score
    assert deep_delta > top_delta


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


# --- #3956: the boost must not monopolise the reranker cap --------------------


def _cut(level: str | None, cap: int = 300, arm_depth: int = 1000, k: int = 60) -> list[tuple[str, int]]:
    """Merge a boosted arm and an unboosted arm, sort as recall does, take top ``cap``.

    Mirrors ``memory_engine`` step 4's pre-filter: build the merged pool, sort by
    ``boosted_rrf_score``, slice to the reranker candidate budget. Returns the
    surviving ``(arm, rank)`` pairs.
    """
    boosts = {"graph": level} if level else {}
    pool = [_candidate(1.0 / (k + r), {"graph_rank": r}) for r in range(1, arm_depth + 1)]
    pool += [_candidate(1.0 / (k + s), {"semantic_rank": s}) for s in range(1, arm_depth + 1)]
    pool.sort(key=lambda mc: boosted_rrf_score(mc, boosts, k=k), reverse=True)
    survivors = []
    for mc in pool[:cap]:
        arm, rank = next(iter(mc.source_ranks.items()))
        survivors.append((arm.removesuffix("_rank"), rank))
    return survivors


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_boost_never_starves_the_other_arm_at_the_cap(level):
    """Regression for #3956: `graph:high` left zero semantic-only survivors.

    The score-space form multiplied the arm's contribution by a weight larger
    than RRF's whole dynamic range over the cap window, so the sort degenerated
    to "boosted arm first" and the cut kept 300/300 graph candidates.
    """
    semantic = [rank for arm, rank in _cut(level) if arm == "semantic"]
    assert semantic, f"{level} starved the unboosted arm out of the reranker budget"


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_boost_never_displaces_the_head_of_the_other_arm(level):
    """No level may push the *top* unboosted hit out of the reranker budget.

    This is the property that makes the boost safe on banks whose merged pool is
    far larger than ``RERANKER_MAX_CANDIDATES``: displacement is proportional to
    rank, so the head of every arm is preserved whatever the pool size.
    """
    assert ("semantic", 1) in _cut(level)


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_boost_still_protects_the_arm_from_the_cut(level):
    """The feature must still do its job: reach deeper into the boosted arm."""
    unboosted_depth = max(rank for arm, rank in _cut(None) if arm == "graph")
    boosted_depth = max(rank for arm, rank in _cut(level) if arm == "graph")
    assert boosted_depth > unboosted_depth


def test_higher_levels_protect_the_arm_more_deeply():
    depths = [max(rank for arm, rank in _cut(lvl) if arm == "graph") for lvl in ("low", "medium", "high")]
    assert depths[0] < depths[1] < depths[2]


def test_rank_boost_crossover_is_independent_of_k():
    """`r < divisor * s` must hold whatever RRF constant fusion was run with.

    The score-space form's crossover carried a `w*k` term, which is why the
    damage scaled with k and surprised on real banks; the rank-space form must
    not depend on k at all.
    """
    divisor = BOOST_LEVELS["medium"].rank_divisor
    for k in (10, 60, 200):
        boosted_wins = _candidate(1.0 / (k + 39), {"graph_rank": 39})
        boosted_loses = _candidate(1.0 / (k + 41), {"graph_rank": 41})
        rival = _candidate(1.0 / (k + 10), {"semantic_rank": 10})  # crossover at r = 4*10 = 40
        assert boosted_rrf_score(boosted_wins, {"graph": "medium"}, k=k) > boosted_rrf_score(rival, {}, k=k)
        assert boosted_rrf_score(boosted_loses, {"graph": "medium"}, k=k) < boosted_rrf_score(rival, {}, k=k)
        assert divisor == 4.0
