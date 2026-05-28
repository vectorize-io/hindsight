import asyncio
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine import memory_engine
from hindsight_api.engine.memory_engine import Budget
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

    async def rerank(self, _query: str, candidates: list[Any]) -> list[ScoredResult]:
        return [
            ScoredResult(
                candidate=candidates[0],
                cross_encoder_score=0.8,
                cross_encoder_score_normalized=0.8,
                weight=0.8,
            )
        ]


def test_recall_scoring_now_uses_question_date():
    question_date = datetime(2023, 5, 30, 13, 53, tzinfo=UTC)

    assert memory_engine._recall_scoring_now(question_date) == question_date


def test_recall_scoring_now_normalizes_naive_question_date_to_utc():
    question_date = datetime(2023, 5, 30, 13, 53)

    assert memory_engine._recall_scoring_now(question_date) == datetime(2023, 5, 30, 13, 53, tzinfo=UTC)


def test_recall_scoring_now_converts_aware_question_date_to_utc():
    question_date = datetime(2023, 5, 30, 21, 53, tzinfo=timezone(timedelta(hours=8)))

    assert memory_engine._recall_scoring_now(question_date) == datetime(2023, 5, 30, 13, 53, tzinfo=UTC)


def test_recall_scoring_now_falls_back_to_current_time(monkeypatch):
    now = datetime(2026, 5, 27, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(memory_engine, "utcnow", lambda: now)

    assert memory_engine._recall_scoring_now(None) == now


@pytest.mark.asyncio
async def test_recall_async_passes_question_date_to_combined_scoring(monkeypatch):
    question_date = datetime(2023, 5, 30, 21, 53, tzinfo=timezone(timedelta(hours=8)))
    captured_now: datetime | None = None

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

    async def generate_embeddings_batch(*_args: object, **_kwargs: object) -> list[list[float]]:
        return [[0.1, 0.2, 0.3]]

    async def retrieve_all_fact_types_parallel(*_args: object, **_kwargs: object) -> MultiFactTypeRetrievalResult:
        retrieval = RetrievalResult(
            id="00000000-0000-0000-0000-000000000001",
            text="Alice likes machine learning.",
            fact_type="world",
            occurred_start=datetime(2023, 5, 29, tzinfo=UTC),
            similarity=0.9,
        )
        return MultiFactTypeRetrievalResult(
            results_by_fact_type={
                "world": ParallelRetrievalResult(
                    semantic=[retrieval],
                    bm25=[],
                    graph=[],
                    temporal=None,
                    timings={"semantic": 0.0, "bm25": 0.0, "graph": 0.0, "temporal_extraction": 0.0},
                )
            }
        )

    def apply_combined_scoring(
        scored_results: list[ScoredResult], *, now: datetime, is_passthrough_reranker: bool
    ) -> None:
        nonlocal captured_now
        assert is_passthrough_reranker is False
        captured_now = now
        for result in scored_results:
            result.combined_score = result.cross_encoder_score_normalized
            result.weight = result.combined_score

    monkeypatch.setattr(memory_engine.embedding_utils, "generate_embeddings_batch", generate_embeddings_batch)
    monkeypatch.setattr(
        "hindsight_api.engine.search.retrieval.retrieve_all_fact_types_parallel",
        retrieve_all_fact_types_parallel,
    )
    monkeypatch.setattr(memory_engine, "apply_combined_scoring", apply_combined_scoring)

    result = await engine.recall_async(
        bank_id="test-bank",
        query="What does Alice like?",
        budget=Budget.LOW,
        fact_type=["world"],
        question_date=question_date,
        request_context=RequestContext(),
        _quiet=True,
    )

    assert [fact.text for fact in result.results] == ["Alice likes machine learning."]
    assert captured_now == datetime(2023, 5, 30, 13, 53, tzinfo=UTC)
