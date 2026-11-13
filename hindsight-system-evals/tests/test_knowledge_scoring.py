"""The knowledge suites' scoring, without a server."""

from __future__ import annotations

from hindsight_system_evals.knowledge import SearchHit, ranked_documents, score_ranked
from hindsight_system_evals.knowledge_data import RankedQuery


def test_score_ranked_breaks_down_by_group():
    queries = [RankedQuery("q1", "", "english", {"a": 1}), RankedQuery("q2", "", "french", {"a": 1})]
    metrics = score_ranked(queries, {"q1": ["a", "b"], "q2": ["b", "a"]})
    assert metrics["recall@1"] == 50.0 and metrics["english:recall@1"] == 100.0
    assert metrics["french:recall@1"] == 0.0


def test_a_hit_on_a_multi_page_pdf_counts_as_that_page():
    hit = SearchHit
    page_ids = {"report": ("c-7", "c-9")}
    results = [hit("report", 2), hit("report", 1), hit("report", 2), hit("note", None)]
    assert ranked_documents(results, page_ids) == ["c-9", "c-7", "note"]
    assert ranked_documents(results) == ["report", "note"]
