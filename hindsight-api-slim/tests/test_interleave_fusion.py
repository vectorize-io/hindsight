"""Unit tests for round-robin interleave fusion (consolidation dedup recall)."""

from hindsight_api.engine.search.fusion import interleave_fusion
from hindsight_api.engine.search.types import RetrievalResult


def _r(doc_id: str) -> RetrievalResult:
    return RetrievalResult(id=doc_id, text=f"text-{doc_id}", fact_type="observation")


def _ids(merged) -> list[str]:
    return [mc.id for mc in merged]


def test_round_robin_order_takes_each_arm_in_turn():
    semantic = [_r("s1"), _r("s2"), _r("s3")]
    bm25 = [_r("b1"), _r("b2")]
    graph = [_r("g1")]

    merged = interleave_fusion([semantic, bm25, graph])

    # Round 0: s1, b1, g1 ; round 1: s2, b2 ; round 2: s3
    assert _ids(merged) == ["s1", "b1", "g1", "s2", "b2", "s3"]


def test_semantic_top_hit_is_always_first():
    # The dedup twin: semantic #1 but absent from every other arm. Must still lead.
    semantic = [_r("twin"), _r("s2")]
    bm25 = [_r("b1"), _r("b2"), _r("b3")]
    graph = [_r("g1")]

    merged = interleave_fusion([semantic, bm25, graph])

    assert merged[0].id == "twin"


def test_dedup_keeps_first_occurrence_and_records_all_arm_ranks():
    # "x" is semantic #1 and bm25 #2; it should appear once, at its first (semantic) slot,
    # but carry ranks from every arm it appears in.
    semantic = [_r("x"), _r("s2")]
    bm25 = [_r("b1"), _r("x")]

    merged = interleave_fusion([semantic, bm25])

    assert _ids(merged) == ["x", "b1", "s2"]
    x = next(mc for mc in merged if mc.id == "x")
    assert x.source_ranks == {"semantic_rank": 1, "bm25_rank": 2}


def test_rrf_score_strictly_decreasing_preserves_order_on_sort():
    merged = interleave_fusion([[_r("a"), _r("b")], [_r("c")]])
    scores = [mc.rrf_score for mc in merged]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores)  # strictly decreasing, no ties
    # rrf_rank reflects the interleave position
    assert [mc.rrf_rank for mc in merged] == [1, 2, 3]


def test_empty_inputs():
    assert interleave_fusion([]) == []
    assert interleave_fusion([[], []]) == []
