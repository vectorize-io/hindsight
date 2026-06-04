"""Per-strategy recall boosting.

A deployment can prioritise one retrieval arm (semantic, bm25, graph, temporal)
over the others via ``HINDSIGHT_API_RECALL_STRATEGY_BOOSTS``, expressed as a
human priority *level* rather than an opaque number — e.g. ``graph:high`` to
strongly favour graph hits.

A level is chosen instead of a raw weight because the boost is applied in two
structurally different places that live on different score scales, so a single
number could not mean the same thing in both. The level maps to a tuned
:class:`BoostWeights` pair:

1. **Before the reranker cap** — :func:`boosted_rrf_score` uses ``BoostWeights.rrf``
   as a weighted-RRF multiplier on the boosted arm's rank contribution, so its
   candidates survive the global reranker candidate budget instead of being
   trimmed by raw RRF score. Rank-aware: a candidate ranked #1 in the boosted
   arm is protected more than one ranked #200.

2. **After the reranker** — :func:`additive_strategy_boost` uses
   ``BoostWeights.additive`` as a flat bump to the final ranking weight (which
   sits in ~[0, 1] after cross-encoder + recency/temporal scoring), nudging the
   boosted arm's candidates up the final ordering.

Both functions are no-ops when ``boosts`` is empty, preserving current behaviour.
"""

from dataclasses import dataclass

from .types import MergedCandidate


@dataclass(frozen=True)
class BoostWeights:
    """Per-stage boost magnitudes for one priority level.

    The two fields live on different scales on purpose (see module docstring):
    ``rrf`` multiplies an arm's ``1/(k+rank)`` RRF contribution; ``additive`` is
    added directly to the post-rerank weight in ~[0, 1].
    """

    rrf: float
    additive: float


# Priority level -> per-stage boost magnitudes. Tuned against real recall traces
# (LoCoMo bank, 336 merged candidates → 300-cap, local ms-marco cross-encoder):
#
# Stage 1 (rrf, weighted-RRF multiplier on the arm's 1/(k+rank) contribution).
# The observed 300-cap boundary RRF score was ~0.0055; a graph-only candidate
# falls below it past graph-rank ~120. The multipliers map to that boundary:
#   low=1.0   doubles the arm's vote — rescues at-risk candidates from the cut
#             (graph-rank 150: 0.0048 → 0.0095) without reshuffling much.
#   medium=3.0 promotes them into the middle of the pool (~rank 60).
#   high=6.0   makes the boosted arm dominate the top of the candidate pool.
#
# Stage 2 (additive, flat bump to the post-rerank weight in [0, 1]). The local
# cross-encoder is sharply bimodal: strong direct matches score 0.5–0.999, while
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
    "low": BoostWeights(rrf=1.0, additive=0.05),
    "medium": BoostWeights(rrf=3.0, additive=0.2),
    "high": BoostWeights(rrf=6.0, additive=0.5),
}


def boosted_rrf_score(candidate: MergedCandidate, boosts: dict[str, str], k: int = 60) -> float:
    """Return ``candidate``'s RRF score plus a weighted-RRF boost delta.

    For each boosted arm the candidate appeared in, adds ``level.rrf * 1/(k+rank)``
    — i.e. scales that arm's RRF contribution by the level's multiplier. Staying
    in RRF units keeps the boost comparable to the base score and rank-aware.

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
            delta += BOOST_LEVELS[level].rrf * (1.0 / (k + rank))
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
