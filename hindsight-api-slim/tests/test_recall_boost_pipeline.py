"""Exercise stage-2 scoring through recall_async without a database or model server."""

import asyncio
from dataclasses import dataclass
from typing import Literal
from unittest.mock import AsyncMock

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine import memory_engine
from hindsight_api.engine.cross_encoder import CrossEncoderModel, MultiCrossEncoder, RRFPassthroughCrossEncoder
from hindsight_api.engine.memory_engine import Budget
from hindsight_api.engine.response_models import MinScores, RecallResult
from hindsight_api.engine.search.reranking import CrossEncoderReranker, RerankResult
from hindsight_api.engine.search.retrieval import MultiFactTypeRetrievalResult, ParallelRetrievalResult
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult
from hindsight_api.models import RequestContext


class _Primary(CrossEncoderModel):
    @property
    def provider_name(self) -> str:
        return "tei"

    async def initialize(self) -> None:
        pass

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if pairs[0][0] == "fallback":
            raise RuntimeError("primary unavailable for this request")
        return [0.2] * len(pairs)


class _ConfigResolver:
    async def get_bank_config(self, _bank_id: str, _request_context: RequestContext) -> dict[str, object]:
        return {}


@dataclass
class _RecallHarness:
    engine: memory_engine.MemoryEngine

    async def recall(
        self,
        query: str = "primary",
        *,
        reranking: Literal["cross_encoder", "rrf", "interleave"] = "cross_encoder",
        min_scores: MinScores | None = None,
    ) -> RecallResult:
        return await self.engine.recall_async(
            bank_id="test-bank",
            query=query,
            budget=Budget.LOW,
            fact_type=["world"],
            request_context=RequestContext(),
            reranking=reranking,
            min_scores=min_scores,
            _quiet=True,
        )


@pytest.fixture
def recall_harness(monkeypatch: pytest.MonkeyPatch) -> _RecallHarness:
    config = _get_raw_config()
    monkeypatch.setattr(config, "recall_strategy_boosts", {"graph": "high"})
    monkeypatch.setattr(config, "recency_decay_function", "none")
    monkeypatch.setattr(config, "reranker_max_candidates", 10)
    monkeypatch.setattr(config, "reranker_max_candidates_low", 0)
    monkeypatch.setattr(memory_engine, "get_config", lambda: config)
    engine = memory_engine.MemoryEngine.__new__(memory_engine.MemoryEngine)
    engine._operation_validator = None
    engine._config_resolver = _ConfigResolver()
    engine._search_semaphore = asyncio.Semaphore(2)
    engine._initialized = True
    engine._read_backend = object()
    engine.embeddings = object()
    engine.query_analyzer = object()
    engine._cross_encoder_reranker = CrossEncoderReranker(cross_encoder=_Primary())
    engine._authenticate_tenant = AsyncMock()
    engine._require_bank_exists = AsyncMock()

    async def embeddings(*_args: object, **_kwargs: object) -> list[list[float]]:
        return [[0.1, 0.2, 0.3]]

    async def retrieve(*_args: object, **_kwargs: object) -> MultiFactTypeRetrievalResult:
        return MultiFactTypeRetrievalResult(
            results_by_fact_type={
                "world": ParallelRetrievalResult(
                    semantic=[],
                    bm25=[],
                    graph=[
                        RetrievalResult(
                            id=f"00000000-0000-0000-0000-{i:012d}",
                            text=f"graph fact {i}",
                            fact_type="world",
                            activation=1 / i,
                        )
                        for i in range(1, 4)
                    ],
                    temporal=None,
                    timings={"semantic": 0.0, "bm25": 0.0, "graph": 0.0, "temporal_extraction": 0.0},
                )
            }
        )

    monkeypatch.setattr(memory_engine.embedding_utils, "generate_embeddings_batch", embeddings)
    monkeypatch.setattr("hindsight_api.engine.search.retrieval.retrieve_all_fact_types_parallel", retrieve)
    return _RecallHarness(engine=engine)


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [2, 10], ids=["over-cap", "under-cap"])
@pytest.mark.parametrize("mode", ["explicit", "provider", "failover"])
async def test_passthrough_skips_stage2_through_recall(
    recall_harness: _RecallHarness, monkeypatch: pytest.MonkeyPatch, cap: int, mode: str
) -> None:
    monkeypatch.setattr(_get_raw_config(), "reranker_max_candidates", cap)
    encoder: CrossEncoderModel = RRFPassthroughCrossEncoder()
    if mode == "failover":
        encoder = MultiCrossEncoder([_Primary(), encoder])
    recall_harness.engine._cross_encoder_reranker = CrossEncoderReranker(cross_encoder=encoder)
    reranking = "rrf" if mode == "explicit" else "cross_encoder"
    boosted = await recall_harness.recall("fallback", reranking=reranking)
    monkeypatch.setattr(_get_raw_config(), "recall_strategy_boosts", {})
    plain = await recall_harness.recall("fallback", reranking=reranking)
    assert len(boosted.results) == min(cap, 3)
    assert [r.id for r in boosted.results] == [r.id for r in plain.results]
    assert [r.scores.final for r in boosted.results] == pytest.approx([r.scores.final for r in plain.results])
    assert boosted.results[0].scores.final == pytest.approx(1.0)
    assert boosted.results[-1].scores.final == pytest.approx(0.1)
    assert all(r.scores.reranker is None for r in boosted.results)


@pytest.mark.asyncio
async def test_min_final_filters_after_rank_decay(recall_harness: _RecallHarness) -> None:
    unfiltered = await recall_harness.recall()
    assert len(unfiltered.results) == 3
    assert unfiltered.results[1].scores.final == pytest.approx(0.2 + 4 / 9)
    # A flat +0.5 would retain all three. The decayed rank-2/3 bumps do not.
    assert 0.2 + 0.5 > 0.68
    filtered = await recall_harness.recall(min_scores=MinScores(final=0.68))
    assert [r.id for r in filtered.results] == [unfiltered.results[0].id]
    # The floor is inclusive and must be applied after the bump, not to CE=0.2.
    at_boundary = await recall_harness.recall(min_scores=MinScores(final=unfiltered.results[1].scores.final))
    assert [r.id for r in at_boundary.results] == [r.id for r in unfiltered.results[:2]]


@pytest.mark.asyncio
@pytest.mark.parametrize("first_query", ["primary", "fallback"])
async def test_concurrent_recalls_use_their_own_rerank_provider(
    recall_harness: _RecallHarness, first_query: str
) -> None:
    first_scored = asyncio.Event()
    second_scored = asyncio.Event()
    chain = MultiCrossEncoder([_Primary(), RRFPassthroughCrossEncoder()])

    class _InterleavedReranker(CrossEncoderReranker):
        async def rerank(self, query: str, candidates: list[MergedCandidate]) -> RerankResult:
            result = await super().rerank(query, candidates)
            # Let the other request move the shared cursor before recall consumes
            # this result. Both requests use the real rerank/normalization path.
            if query == first_query:
                first_scored.set()
                await second_scored.wait()
            else:
                second_scored.set()
            return result

    recall_harness.engine._cross_encoder_reranker = _InterleavedReranker(cross_encoder=chain)
    second_query = "fallback" if first_query == "primary" else "primary"

    async def second_recall() -> RecallResult:
        await first_scored.wait()
        return await recall_harness.recall(second_query)

    results = await asyncio.wait_for(asyncio.gather(recall_harness.recall(first_query), second_recall()), timeout=5)
    by_query = dict(zip([first_query, second_query], results))
    assert chain.provider_name == ("rrf" if second_query == "fallback" else "tei")
    assert [r.scores.reranker for r in by_query["primary"].results] == pytest.approx([0.2] * 3)
    assert [r.scores.final for r in by_query["primary"].results] == pytest.approx([0.7, 0.2 + 4 / 9, 0.6])
    assert all(r.scores.reranker is None for r in by_query["fallback"].results)
    assert [r.scores.final for r in by_query["fallback"].results] == pytest.approx([1.0, 0.55, 0.1])
