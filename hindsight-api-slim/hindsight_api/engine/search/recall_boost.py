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

2. **After a cross-encoder rerank** — :func:`additive_strategy_boost` adds a
   bump that is the level's full ``additive`` at rank 1 and shrinks as the
   candidate's rank in that arm gets worse. A passthrough reranker skips this
   bump (:func:`apply_post_rerank_boost`). The bump is still an absolute add, so
   it does not fix cross-encoder score calibration and does not guarantee that a
   strong direct match stays ahead.

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
rank ``r`` beats another arm's rank ``s`` iff ``r < w*s``. That comparison is one
arm's contribution against another's. It is not a property of the fused list once
several arms have been summed. Displacement no longer depends on the pool size.
"""

from dataclasses import dataclass

from .types import MergedCandidate, ScoredResult


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
# Stage 2 (additive ceiling for rank 1 of a boosted arm, then decay). The local
# cross-encoder is sharply bimodal: strong direct matches score 0.5-0.999, while
# everything else — including graph hits the CE undervalues — collapses near 0.
# ``additive`` is what rank 1 receives. A worse rank gets
# ``additive * rank_divisor / (rank_divisor + rank - 1)``, so ``high`` is half
# by rank 9. Reusing ``rank_divisor`` here is an initial parameter choice: in
# stage 1 that number is a rank divisor (``r < divisor * s``), and here it is
# only the decay scale. The two do not have to stay equal; a later change can
# split them with an internal field and no new user setting.
#
# This is still an absolute add. It does not fix cross-encoder calibration
# (a clearly relevant match can score ~0.001), it does not cap how far the
# final order can move, and summed nudges from two arms have no shared cap.
# Rank 1 ceilings, as a description of the full amount and not a guarantee
# about who finishes first:
#   low=0.05  nudges above the near-0 tail.
#   medium=0.2 competes with weak matches when the arm rank is near 1.
#   high=0.5  can pass a moderate match from rank 1; a deep rank should not.
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
    """Return the post-rerank bump for a candidate given its source ranks.

    Each boosted arm that surfaced the candidate contributes
    ``additive * rank_divisor / (rank_divisor + rank - 1)``: rank 1 keeps the
    level's full ``additive``, and deeper ranks decay toward zero. Arms the
    candidate did not appear in contribute nothing. Matched arms are summed, so
    two rank-1 ``high`` hits add to ``1.0`` — there is no combined cap.

    Args:
        source_ranks: ``{"graph_rank": 3, "semantic_rank": 50, ...}`` from RRF.
        boosts: Map of strategy name -> priority level. Empty means no boost.

    Returns:
        The bump (0.0 when no boosted arm surfaced this candidate).
    """
    if not boosts:
        return 0.0
    total = 0.0
    for strategy, level in boosts.items():
        rank = source_ranks.get(f"{strategy}_rank")
        if rank is None:
            continue
        weights = BOOST_LEVELS[level]
        total += weights.additive * weights.rank_divisor / (weights.rank_divisor + rank - 1)
    return total


@dataclass(frozen=True)
class TrimmedCandidates:
    """Who survived the pre-rerank cap, and how many were dropped."""

    kept: list[MergedCandidate]
    dropped: int


def trim_merged_candidates(
    candidates: list[MergedCandidate],
    max_candidates: int,
    boosts: dict[str, str],
) -> TrimmedCandidates:
    """Keep the pre-rerank budget, promoting boosted arms in rank space.

    When the pool does not exceed ``max_candidates`` the same list is returned
    and nothing is sorted: the boost does not run and does not rewrite
    ``rrf_score``. When it does exceed the cap, candidates are ordered by
    :func:`boosted_rrf_score` and the tail is dropped. ``rrf_score`` itself is
    left as fusion wrote it.
    """
    if len(candidates) <= max_candidates:
        return TrimmedCandidates(kept=candidates, dropped=0)
    candidates.sort(key=lambda mc: boosted_rrf_score(mc, boosts), reverse=True)
    dropped = len(candidates) - max_candidates
    return TrimmedCandidates(kept=candidates[:max_candidates], dropped=dropped)


def stage2_passthrough(reranking: str, provider_name: str | None) -> bool:
    """Whether stage 2 must not add to the post-rerank weight.

    Explicit ``reranking="rrf"`` keeps fusion order. A cross-encoder whose
    ``provider_name`` is ``"rrf"`` is the same path, including a failover chain
    whose ``rrf`` member served this request. ``provider_name`` must be the one
    captured on the rerank result, not a later read of the chain's shared cursor.
    """
    return reranking == "rrf" or provider_name == "rrf"


def apply_post_rerank_boost(
    scored_results: list[ScoredResult],
    boosts: dict[str, str],
    *,
    passthrough: bool,
) -> str | None:
    """Add the stage-2 bump in place, unless this recall is a passthrough.

    Does not sort. Returns the ``stage2=...`` token for the ``[4.7]`` log, or
    ``None`` when ``boosts`` is empty so the caller skips that log.
    """
    if not boosts:
        return None
    if passthrough:
        return "stage2=skipped_passthrough"
    for sr in scored_results:
        sr.weight += additive_strategy_boost(sr.candidate.source_ranks, boosts)
    return "stage2=rank_decay"
