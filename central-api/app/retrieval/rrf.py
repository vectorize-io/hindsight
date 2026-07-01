"""Reciprocal Rank Fusion — rank-based cross-engine result merge.

RToF is score-agnostic: it fuses lists from backends with incomparable raw scores
(OpenMemory cosine vs MemLord hybrid vs CodeRAG RRF) using only their rank order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

RRF_K = 60


def reciprocal_rank_fusion[T](
    ranked_lists: Sequence[Sequence[T]],
    key: Callable[[T], str],
    *,
    k: int = RRF_K,
    limit: int | None = None,
) -> list[tuple[T, float]]:
    """Fuse several ranked lists into one. Returns (item, fused_score) descending.

    Items with the same `key` across lists are merged (their RRF contributions
    sum), implementing cross-engine dedup.
    """
    scores: dict[str, float] = {}
    items: dict[str, T] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst):
            kk = key(item)
            scores[kk] = scores.get(kk, 0.0) + 1.0 / (k + rank + 1)
            items.setdefault(kk, item)
    ordered = sorted(items.values(), key=lambda it: scores[key(it)], reverse=True)
    fused = [(it, scores[key(it)]) for it in ordered]
    return fused[:limit] if limit is not None else fused
