"""Regression tests for reranker initialization/prediction fallback behavior."""

import asyncio

import pytest

from hindsight_api.engine.search.reranking import CrossEncoderReranker
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult


class FailingInitCrossEncoder:
    provider_name = "remote-test"

    async def initialize(self):
        raise RuntimeError("init failed")

    async def predict(self, pairs):
        raise AssertionError("predict should not be called when init failed")


class FailingPredictCrossEncoder:
    provider_name = "remote-test"

    async def initialize(self):
        return None

    async def predict(self, pairs):
        raise RuntimeError("predict failed")


class SlowInitCrossEncoder:
    provider_name = "remote-test"

    async def initialize(self):
        await asyncio.sleep(1)

    async def predict(self, pairs):
        raise AssertionError("predict should not be called when init timed out")


def _candidate(memory_id: str, text: str, rrf_score: float) -> MergedCandidate:
    return MergedCandidate(
        retrieval=RetrievalResult(
            id=memory_id,
            text=text,
            fact_type="observation",
        ),
        rrf_score=rrf_score,
    )


def _candidates() -> list[MergedCandidate]:
    return [
        _candidate("low", "less relevant", 0.1),
        _candidate("high", "more relevant", 0.9),
        _candidate("mid", "middle relevant", 0.5),
    ]


@pytest.mark.asyncio
async def test_lazy_init_failure_falls_back_to_rrf_order():
    reranker = CrossEncoderReranker(cross_encoder=FailingInitCrossEncoder())

    await reranker.ensure_initialized()
    assert reranker._initialized is False

    results = await reranker.rerank("query", _candidates())

    assert [result.id for result in results] == ["high", "mid", "low"]
    assert [result.cross_encoder_score_normalized for result in results] == [1.0, 0.55, 0.09999999999999998]


@pytest.mark.asyncio
async def test_prediction_failure_falls_back_to_rrf_order_and_marks_uninitialized():
    reranker = CrossEncoderReranker(cross_encoder=FailingPredictCrossEncoder())

    await reranker.ensure_initialized()
    assert reranker._initialized is True

    results = await reranker.rerank("query", _candidates())

    assert reranker._initialized is False
    assert [result.id for result in results] == ["high", "mid", "low"]


@pytest.mark.asyncio
async def test_lazy_init_timeout_falls_back_to_rrf_order(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_MODEL_INIT_TIMEOUT", "0.01")
    reranker = CrossEncoderReranker(cross_encoder=SlowInitCrossEncoder())

    await reranker.ensure_initialized()
    assert reranker._initialized is False

    results = await reranker.rerank("query", _candidates())

    assert [result.id for result in results] == ["high", "mid", "low"]
