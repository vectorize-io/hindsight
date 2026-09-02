"""Per-strategy recall boosting.

A deployment can prioritise one retrieval arm (semantic, bm25, graph, temporal)
over the others via ``HINDSIGHT_API_RECALL_STRATEGY_BOOSTS``, expressed as a
human priority *level* rather than an opaque number — e.g. ``graph:high`` to
strongly favour graph hits.

A level is chosen instead of a raw weight because the boost is applied in two
structurally different places that live on different score scales, so a single
number could not mean the same thing in both. The level maps to a tuned
:class:`BoostWeights` pair:

1. **Before the reranker cap** — :func:`boosted_rrf_score` promotes the boosted
   arm in *rank space*: the arm's RRF contribution is recomputed as if the
   candidate had placed ``rank / rank_divisor`` instead of ``rank``, so its
   candidates survive the global reranker candidate budget instead of being
   trimmed by raw RRF score. Rank-aware: a candidate ranked #1 in the boosted
   arm is protected more than one ranked #200.

2. **After the reranker** — :func:`additive_strategy_boost` uses
   ``BoostWeights.additive`` as a flat bump to the final ranking weight (which
   sits in ~[0, 1] after cross-encoder + recency/temporal scoring), nudging the
   boosted arm's candidates up the final ordering.

Both functions are no-ops when ``boosts`` is empty, preserving current behaviour.

Why stage 1 boosts the rank and not the score
---------------------------------------------
The original implementation multiplied the arm's ``1/(k+rank)`` contribution by
a weight ``w``. That is standard weighted RRF, but it interacts badly with the
hard ``RERANKER_MAX_CANDIDATES`` cut that immediately follows it (issue #3956).

RRF with ``k=60`` is deliberately flat: across the whole 300-candidate cap window
the score only spans ``1/61 -> 1/360``, a factor of 5.9. Any ``w`` above that
spread exceeds the entire dynamic range of the rank term, so the sort degenerates
into a *lexicographic* one — boosted arm first, rank merely a tiebreaker. ``high``
was ``w=7``, over the line, and on a bank whose merged pool is far larger than the
cap the boosted arm then filled all 300 slots and no semantic-only candidate ever
reached the cross-encoder (measured: recall@20 0.97 -> 0.40).

The culprit is the ``k`` term. In score space the displacement reach is
``r_max = w*(k+s) - k``, so at the head of the ranking the constant ``w*k``
dominates and the boosted arm's ~360th hit outranks the other arm's *first*.
Boosting the rank instead — ``1/(k + rank/w)`` — cancels ``k``: the boosted arm's
rank ``r`` beats another arm's rank ``s`` iff ``r < w*s``. Displacement becomes
strictly proportional rather than an absolute offset, so it can never invert the
head of the ranking, and the behaviour no longer depends on the pool size.
"""

from dataclasses import dataclass

from .types import MergedCandidate


@dataclass(frozen=True)
class BoostWeights:
    """Per-stage boost magnitudes for one priority level.

    The two fields live on different scales on purpose (see module docstring):
    ``rank_divisor`` divides an arm's rank before the ``1/(k+rank)`` RRF
    contribution is computed; ``additive`` is added directly to the post-rerank
    weight in ~[0, 1].
    """

    rank_divisor: float
    additive: float


# Priority level -> per-stage boost magnitudes.
#
# Stage 1 (rank_divisor, applied in rank space: the arm's contribution becomes
# 1/(k + rank/divisor)). A boosted candidate at arm-rank r outranks an unboosted
# candidate at arm-rank s exactly when r < divisor * s, independent of k, of the
# cap, and of the merged pool size. Simulated against 1000-deep arms and the
# default 300-cap: the share of the reranker budget left to unboosted-only
# candidates, and how deep into the boosted arm the cut still reaches:
#   (unboosted baseline: 150 slots each, boosted arm protected to rank 150)
#   low=2.0    100 slots left to other arms; boosted arm protected to rank 200.
#   medium=4.0  60 slots left;               protected to rank 240.
#   high=8.0    33 slots left;               protected to rank 267.
# Every level keeps the *head* of every other arm — the top-ranked semantic hit
# is only ever displaced by boosted hits from the arm's own top `divisor` ranks —
# which is the property the score-space form could not offer at `high`.
#
# Stage 2 (additive, flat bump to the post-rerank weight in [0, 1]). The local
# cross-encoder is sharply bimodal: strong direct matches score 0.5-0.999, while
# everything else — including graph hits the CE undervalues, which is exactly
# what we boost — collapses near 0. So the additive lifts a ~0 candidate up the
# weight scale. Levels are calibrated as relevance thresholds it can outrank:
#   low=0.05  nudges above the near-0 tail; loses to any real CE match.
#   medium=0.2 competes with weak/moderate matches.
#   high=0.5  wins over most semantic matches (honouring "prioritise graph over
#             semantic"); only a strong direct match (>0.5 normalized) still wins.
#
# The keys are the user-facing contract; config.py validates env input against
# them (kept in sync by a guard test).
BOOST_LEVELS: dict[str, BoostWeights] = {
    "low": BoostWeights(rank_divisor=2.0, additive=0.05),
    "medium": BoostWeights(rank_divisor=4.0, additive=0.2),
    "high": BoostWeights(rank_divisor=8.0, additive=0.5),
}


def boosted_rrf_score(candidate: MergedCandidate, boosts: dict[str, str], k: int = 60) -> float:
    """Return ``candidate``'s RRF score with boosted arms promoted in rank space.

    For each boosted arm the candidate appeared in, replaces that arm's
    ``1/(k+rank)`` contribution with ``1/(k + rank/divisor)`` — expressed as a
    delta so ``rrf_score`` stays authoritative and unboosted arms are untouched.

    Args:
        candidate: Merged candidate carrying ``rrf_score`` and ``source_ranks``.
        boosts: Map of strategy name -> priority level. Empty means no boost.
        k: RRF constant; must match the value used during fusion.

    Returns:
        The (possibly) boosted score to sort by. Equal to ``rrf_score`` when no
        boosted arm surfaced this candidate.
    """
    if not boosts:
        return candidate.rrf_score
    delta = 0.0
    for strategy, level in boosts.items():
        rank = candidate.source_ranks.get(f"{strategy}_rank")
        if rank is not None:
            divisor = BOOST_LEVELS[level].rank_divisor
            delta += 1.0 / (k + rank / divisor) - 1.0 / (k + rank)
    return candidate.rrf_score + delta


def additive_strategy_boost(source_ranks: dict[str, int], boosts: dict[str, str]) -> float:
    """Return the flat additive boost for a candidate given its source ranks.

    Sums the ``additive`` magnitude of every boosted arm that surfaced the
    candidate. Flat by design: the bump does not depend on the candidate's rank
    within the arm, matching the post-rerank "additive boost" semantics.

    Args:
        source_ranks: ``{"graph_rank": 3, "semantic_rank": 50, ...}`` from RRF.
        boosts: Map of strategy name -> priority level. Empty means no boost.

    Returns:
        The additive boost (0.0 when no boosted arm surfaced this candidate).
    """
    if not boosts:
        return 0.0
    return sum(BOOST_LEVELS[level].additive for strategy, level in boosts.items() if f"{strategy}_rank" in source_ranks)
