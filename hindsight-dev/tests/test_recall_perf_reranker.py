"""The in-process recall benchmark must follow the engine's rerank contract."""

import asyncio

from hindsight_api.engine.search.reranking import RerankResult
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult

from benchmarks.perf.recall_perf import _RRFReranker


def test_rrf_benchmark_returns_scores_with_serving_provider() -> None:
    candidates = [
        MergedCandidate(
            retrieval=RetrievalResult(id=str(i), text="fact", fact_type="world"),
            rrf_score=score,
        )
        for i, score in enumerate([0.1, 0.3, 0.2])
    ]
    result = asyncio.run(_RRFReranker().rerank("query", candidates))
    assert isinstance(result, RerankResult)
    assert result.provider_name == "rrf"
    assert [row.weight for row in result.results] == [0.3, 0.2, 0.1]
    assert [row.id for row in result.results] == ["1", "2", "0"]


def test_rrf_benchmark_empty_results_keep_provider() -> None:
    result = asyncio.run(_RRFReranker().rerank("query", []))
    assert isinstance(result, RerankResult)
    assert result.results == []
    assert result.provider_name == "rrf"
