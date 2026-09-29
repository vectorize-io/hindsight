"""Cross-fact-type ordering of the temporal arm before fusion.

Recall concatenates the per-fact-type temporal results into one list, sorts it,
and hands it to RRF as the temporal arm. RRF consumes the list's ORDER, not the
raw scores, so that sort decides which memory holds the temporal arm's best rank.

The sort used to read ``combined_score`` — a field of :class:`ScoredResult`, which
only exists after fusion. The list at that point holds :class:`RetrievalResult`
objects, so every sort key fell back to ``0`` and the sort was a stable no-op:
concatenation order (fact-type bucket order) survived into fusion. These tests
drive ``recall_async`` with deliberately mis-ordered temporal results and assert
on the list fusion actually receives.
"""

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine import memory_engine
from hindsight_api.engine.memory_engine import Budget
from hindsight_api.engine.search import fusion as fusion_module
from hindsight_api.engine.search.reranking import RerankResult
from hindsight_api.engine.search.retrieval import MultiFactTypeRetrievalResult, ParallelRetrievalResult
from hindsight_api.engine.search.types import RetrievalResult, ScoredResult
from hindsight_api.models import RequestContext


class _ConfigResolver:
    async def get_bank_config(self, _bank_id: str, _request_context: RequestContext) -> dict[str, object]:
        return {}


class _Reranker:
    def __init__(self) -> None:
        self.cross_encoder = None

    async def ensure_initialized(self) -> None:
        return None

    async def rerank(self, _query: str, candidates: list[Any]) -> RerankResult:
        # The reranker output is irrelevant here — the assertion happens on the
        # fusion input — so pass the candidates through untouched.
        return RerankResult(
            results=[
                ScoredResult(candidate=c, cross_encoder_score=0.5, cross_encoder_score_normalized=0.5, weight=0.5)
                for c in candidates
            ],
            provider_name=None,
        )


def _temporal_result(unit_id: str, text: str, temporal_score: float | None) -> RetrievalResult:
    return RetrievalResult(
        id=unit_id,
        text=text,
        fact_type="world",
        occurred_start=datetime(2025, 1, 10, tzinfo=UTC),
        similarity=0.5,
        temporal_score=temporal_score,
    )


async def _run_recall(
    monkeypatch, temporal_by_fact_type: dict[str, list[RetrievalResult]]
) -> list[list[RetrievalResult]]:
    """Run recall_async with stubbed retrieval; return the result_lists fusion received."""
    captured: list[list[RetrievalResult]] = []

    engine = memory_engine.MemoryEngine.__new__(memory_engine.MemoryEngine)
    engine._operation_validator = None
    engine._config_resolver = _ConfigResolver()
    engine._search_semaphore = asyncio.Semaphore(1)
    engine._initialized = True
    engine._read_backend = object()
    engine.embeddings = object()
    engine.query_analyzer = object()
    engine._cross_encoder_reranker = _Reranker()
    engine._authenticate_tenant = AsyncMock()
    # Recall 404s for a bank that was never created (#4442); this engine has no
    # database behind it, so the existence check is stubbed along with auth.
    engine._require_bank_exists = AsyncMock()

    async def generate_embeddings_batch(*_args: object, **_kwargs: object) -> list[list[float]]:
        return [[0.1, 0.2, 0.3]]

    async def retrieve_all_fact_types_parallel(*_args: object, **_kwargs: object) -> MultiFactTypeRetrievalResult:
        return MultiFactTypeRetrievalResult(
            results_by_fact_type={
                ft: ParallelRetrievalResult(
                    semantic=[],
                    bm25=[],
                    graph=[],
                    temporal=results,
                    timings={"semantic": 0.0, "bm25": 0.0, "graph": 0.0, "temporal": 0.0},
                )
                for ft, results in temporal_by_fact_type.items()
            }
        )

    real_fusion = fusion_module.reciprocal_rank_fusion

    def spy_fusion(result_lists: list[list[RetrievalResult]]) -> list[Any]:
        captured.extend(result_lists)
        return real_fusion(result_lists)

    monkeypatch.setattr(memory_engine.embedding_utils, "generate_embeddings_batch", generate_embeddings_batch)
    monkeypatch.setattr(
        "hindsight_api.engine.search.retrieval.retrieve_all_fact_types_parallel",
        retrieve_all_fact_types_parallel,
    )
    monkeypatch.setattr(fusion_module, "reciprocal_rank_fusion", spy_fusion)

    await engine.recall_async(
        bank_id="test-bank",
        query="What happened in January?",
        budget=Budget.LOW,
        fact_type=list(temporal_by_fact_type),
        question_date=datetime(2025, 2, 1, tzinfo=UTC),
        request_context=RequestContext(),
        _quiet=True,
    )
    return captured


@pytest.mark.asyncio
async def test_temporal_arm_is_ordered_by_temporal_score(monkeypatch):
    """A higher temporal_score must outrank a lower one regardless of which fact-type
    bucket it came from — fusion sees the score-ordered list, not concatenation order."""
    low = _temporal_result("00000000-0000-0000-0000-000000000001", "low score", 0.2)
    high = _temporal_result("00000000-0000-0000-0000-000000000002", "high score", 0.95)

    # 'experience' results are concatenated AFTER 'world' results, so concatenation
    # order is [low, high] — the wrong order. The sort must invert it.
    result_lists = await _run_recall(
        monkeypatch,
        {"world": [low], "experience": [high]},
    )

    temporal_arm = result_lists[-1]
    assert [r.id for r in temporal_arm] == [high.id, low.id]


@pytest.mark.asyncio
async def test_temporal_arm_orders_none_score_last(monkeypatch):
    """A result with no temporal_score (None) sorts below every scored result."""
    scored = _temporal_result("00000000-0000-0000-0000-000000000003", "scored", 0.4)
    unscored = _temporal_result("00000000-0000-0000-0000-000000000004", "unscored", None)

    result_lists = await _run_recall(
        monkeypatch,
        {"world": [unscored], "experience": [scored]},
    )

    temporal_arm = result_lists[-1]
    assert [r.id for r in temporal_arm] == [scored.id, unscored.id]
