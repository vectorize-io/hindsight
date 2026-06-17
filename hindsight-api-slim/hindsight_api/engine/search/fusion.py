"""
Helper functions for hybrid search (semantic + BM25 + graph).
"""

from .types import MergedCandidate, RetrievalResult


def cap_per_source(results: list[RetrievalResult], cap: int) -> list[RetrievalResult]:
    """Truncate a single retrieval arm to its top-``cap`` results.

    Applied per source (semantic, BM25, graph, temporal) before fusion so that
    one over-expanding backend cannot crowd out the others when the merged pool
    is later trimmed to the reranker's global candidate budget. The caller is
    responsible for sorting ``results`` by relevance first; this only slices.

    Args:
        results: Results for a single source, already sorted best-first.
        cap: Maximum results to keep. ``0`` (or negative) disables the cap.

    Returns:
        The original list when the cap is disabled or not exceeded, otherwise a
        truncated copy of the top ``cap`` results.
    """
    if cap <= 0 or len(results) <= cap:
        return results
    return results[:cap]


def reciprocal_rank_fusion(result_lists: list[list[RetrievalResult]], k: int = 60) -> list[MergedCandidate]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF formula: score(d) = sum_over_lists(1 / (k + rank(d)))

    Args:
        result_lists: List of result lists, each containing RetrievalResult objects
        k: Constant for RRF formula (default: 60)

    Returns:
        Merged list of MergedCandidate objects, sorted by RRF score

    Example:
        semantic_results = [RetrievalResult(...), RetrievalResult(...), ...]
        bm25_results = [RetrievalResult(...), RetrievalResult(...), ...]
        graph_results = [RetrievalResult(...), RetrievalResult(...), ...]

        merged = reciprocal_rank_fusion([semantic_results, bm25_results, graph_results])
        # Returns: [MergedCandidate(...), MergedCandidate(...), ...]
    """
    # Track scores from each list
    rrf_scores = {}
    source_ranks = {}  # Track rank from each source for each doc_id
    all_retrievals = {}  # Store the actual RetrievalResult (use first occurrence)

    source_names = ["semantic", "bm25", "graph", "temporal"]

    for source_idx, results in enumerate(result_lists):
        source_name = source_names[source_idx] if source_idx < len(source_names) else f"source_{source_idx}"

        for rank, retrieval in enumerate(results, start=1):
            # Type check to catch tuple issues
            if isinstance(retrieval, tuple):
                raise TypeError(
                    f"Expected RetrievalResult but got tuple in {source_name} results at rank {rank}. "
                    f"Tuple value: {retrieval[:2] if len(retrieval) >= 2 else retrieval}. "
                    f"This suggests the retrieval function returned tuples instead of RetrievalResult objects."
                )
            if not isinstance(retrieval, RetrievalResult):
                raise TypeError(
                    f"Expected RetrievalResult but got {type(retrieval).__name__} in {source_name} results at rank {rank}"
                )
            doc_id = retrieval.id

            # Store retrieval result (use first occurrence)
            if doc_id not in all_retrievals:
                all_retrievals[doc_id] = retrieval

            # Calculate RRF score contribution
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                source_ranks[doc_id] = {}

            rrf_scores[doc_id] += 1.0 / (k + rank)
            source_ranks[doc_id][f"{source_name}_rank"] = rank

    # Combine into final results with metadata
    merged_results = []
    for rrf_rank, (doc_id, rrf_score) in enumerate(
        sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True), start=1
    ):
        merged_candidate = MergedCandidate(
            retrieval=all_retrievals[doc_id], rrf_score=rrf_score, rrf_rank=rrf_rank, source_ranks=source_ranks[doc_id]
        )
        merged_results.append(merged_candidate)

    return merged_results


def interleave_fusion(result_lists: list[list[RetrievalResult]]) -> list[MergedCandidate]:
    """Round-robin (interleaved) fusion — an alternative to RRF for dedup-style recall.

    RRF scores a doc by the *sum* of its reciprocal ranks across arms, so a result
    that is #1 in one arm but absent/low in the others gets averaged down. That is
    exactly the consolidation-dedup failure mode: the near-identical existing
    observation (the "twin" to merge into) is semantic rank #1, yet shares no
    source-fact graph link and little lexical overlap, so RRF drops it below the
    recall budget cutoff and the LLM never sees it → creates a duplicate.

    Interleave instead *guarantees every arm's top hits a slot*: take each arm's
    #1, then each arm's #2, … in arm-priority order, de-duplicating, until all
    results are placed. The arm priority is the order of ``result_lists``
    (semantic, bm25, graph, temporal), so semantic #1 is always first.

    ``rrf_score`` is assigned strictly decreasing by final interleave position so
    downstream order-by-score sorts preserve the interleave order; ``source_ranks``
    mirrors the RRF bookkeeping (each doc's rank within every arm it appears in).
    """
    source_names = ["semantic", "bm25", "graph", "temporal"]
    source_ranks: dict[str, dict[str, int]] = {}
    all_retrievals: dict[str, RetrievalResult] = {}

    for source_idx, results in enumerate(result_lists):
        source_name = source_names[source_idx] if source_idx < len(source_names) else f"source_{source_idx}"
        for rank, retrieval in enumerate(results, start=1):
            if not isinstance(retrieval, RetrievalResult):
                raise TypeError(
                    f"Expected RetrievalResult but got {type(retrieval).__name__} in {source_name} results at rank {rank}"
                )
            doc_id = retrieval.id
            all_retrievals.setdefault(doc_id, retrieval)
            source_ranks.setdefault(doc_id, {})[f"{source_name}_rank"] = rank

    # Round-robin pick across arms in priority order: all #1s, then all #2s, ...
    ordered_ids: list[str] = []
    seen: set[str] = set()
    max_len = max((len(r) for r in result_lists), default=0)
    for r in range(max_len):
        for results in result_lists:
            if r < len(results):
                doc_id = results[r].id
                if doc_id not in seen:
                    seen.add(doc_id)
                    ordered_ids.append(doc_id)

    n = len(ordered_ids)
    return [
        MergedCandidate(
            retrieval=all_retrievals[doc_id],
            # Strictly decreasing by interleave position → sorting desc by rrf_score
            # reproduces the interleave order downstream.
            rrf_score=float(n - pos),
            rrf_rank=pos + 1,
            source_ranks=source_ranks[doc_id],
        )
        for pos, doc_id in enumerate(ordered_ids)
    ]
