"""Exercise stage-2 scoring through recall_async without a database or model server."""

import asyncio
from dataclasses import dataclass
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine import memory_engine
from hindsight_api.engine.cross_encoder import (
    CrossEncoderModel,
    MultiCrossEncoder,
    RRFPassthroughCrossEncoder,
    ScoreSemantics,
)
from hindsight_api.engine.memories import FullRecallRequest
from hindsight_api.engine.memory_engine import Budget
from hindsight_api.engine.response_models import MinScores, RecallResult
from hindsight_api.engine.search.reranking import CrossEncoderReranker, RerankResult
from hindsight_api.engine.search.retrieval import MultiFactTypeRetrievalResult, ParallelRetrievalResult
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult
from hindsight_api.extensions.operation_validator import OperationValidationError
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


class _Ordinal(CrossEncoderModel):
    score_semantics = ScoreSemantics.ORDINAL

    def __init__(self, *, prunes_candidates: bool = False) -> None:
        self.prunes_candidates = prunes_candidates

    @property
    def provider_name(self) -> str:
        return "typesafe"

    async def initialize(self) -> None:
        pass

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = [1.0, 2 / 3, 1 / 3][: len(pairs)]
        if self.prunes_candidates:
            return [scores[0], *([0.0] * (len(scores) - 1))]
        return scores


class _Listwise(_Primary):
    score_semantics = ScoreSemantics.LISTWISE

    @property
    def provider_name(self) -> str:
        return "jina-mlx"


class _ConfigResolver:
    async def get_bank_config(self, _bank_id: str, _request_context: RequestContext) -> dict[str, object]:
        return {}


class _ClaimingEmptyStore:
    def __init__(self) -> None:
        self.calls = 0

    async def full_recall(self, _request: FullRecallRequest) -> RecallResult:
        self.calls += 1
        return RecallResult(results=[])


@dataclass
class _RecallHarness:
    engine: memory_engine.MemoryEngine

    async def recall(
        self,
        query: str = "primary",
        *,
        reranking: Literal["cross_encoder", "rrf", "interleave"] = "cross_encoder",
        min_scores: MinScores | None = None,
        enable_trace: bool = False,
    ) -> RecallResult:
        return await self.engine.recall_async(
            bank_id="test-bank",
            query=query,
            budget=Budget.LOW,
            fact_type=["world"],
            request_context=RequestContext(),
            reranking=reranking,
            min_scores=min_scores,
            enable_trace=enable_trace,
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
@pytest.mark.parametrize("floor", [0.0, 0.5, 1.0])
async def test_ordinal_reranker_floor_is_rejected(recall_harness: _RecallHarness, floor: float) -> None:
    recall_harness.engine._cross_encoder_reranker = CrossEncoderReranker(cross_encoder=_Ordinal())

    with pytest.raises(OperationValidationError) as exc_info:
        await recall_harness.recall(min_scores=MinScores(reranker=floor))

    assert exc_info.value.status_code == 400
    assert "min_scores.reranker" in exc_info.value.reason
    assert "ordinal" in exc_info.value.reason
    assert "typesafe" in exc_info.value.reason


@pytest.mark.asyncio
async def test_ordinal_scores_remain_published_without_a_floor(recall_harness: _RecallHarness) -> None:
    recall_harness.engine._cross_encoder_reranker = CrossEncoderReranker(cross_encoder=_Ordinal())
    result = await recall_harness.recall()

    assert sorted((item.scores.reranker for item in result.results), reverse=True) == pytest.approx([1.0, 2 / 3, 1 / 3])


@pytest.mark.asyncio
async def test_ordinal_reranker_still_allows_min_final(recall_harness: _RecallHarness) -> None:
    recall_harness.engine._cross_encoder_reranker = CrossEncoderReranker(cross_encoder=_Ordinal())
    result = await recall_harness.recall(min_scores=MinScores(final=0.5))
    assert result.results


@pytest.mark.asyncio
async def test_provider_pruning_keeps_published_ordinal_score(recall_harness: _RecallHarness) -> None:
    recall_harness.engine._cross_encoder_reranker = CrossEncoderReranker(cross_encoder=_Ordinal(prunes_candidates=True))
    result = await recall_harness.recall()
    assert len(result.results) == 1
    assert result.results[0].scores.reranker == pytest.approx(1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["explicit", "interleave", "provider", "failover"])
async def test_rrf_ordinal_reranker_floor_is_rejected(recall_harness: _RecallHarness, mode: str) -> None:
    encoder: CrossEncoderModel = RRFPassthroughCrossEncoder()
    if mode == "failover":
        encoder = MultiCrossEncoder([_Primary(), encoder])
    recall_harness.engine._cross_encoder_reranker = CrossEncoderReranker(cross_encoder=encoder)
    reranking = {"explicit": "rrf", "interleave": "interleave"}.get(mode, "cross_encoder")

    with pytest.raises(OperationValidationError) as exc_info:
        await recall_harness.recall("fallback", reranking=reranking, min_scores=MinScores(reranker=0.5))

    assert exc_info.value.status_code == 400
    assert "min_scores.reranker" in exc_info.value.reason
    assert "ordinal" in exc_info.value.reason


@pytest.mark.asyncio
@pytest.mark.parametrize("reranking", ["rrf", "interleave"])
async def test_explicit_ordinal_floor_is_rejected_before_store_full_recall(
    recall_harness: _RecallHarness,
    monkeypatch: pytest.MonkeyPatch,
    reranking: Literal["rrf", "interleave"],
) -> None:
    store = _ClaimingEmptyStore()
    generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    monkeypatch.setattr("hindsight_api.engine.memories.get_memories", lambda: store)
    monkeypatch.setattr(memory_engine.embedding_utils, "generate_embeddings_batch", generate_embeddings)

    with pytest.raises(OperationValidationError) as exc_info:
        await recall_harness.recall(reranking=reranking, min_scores=MinScores(reranker=0.5))

    assert exc_info.value.status_code == 400
    assert "min_scores.reranker" in exc_info.value.reason
    assert "ordinal" in exc_info.value.reason
    assert store.calls == 0
    generate_embeddings.assert_not_awaited()


@pytest.mark.asyncio
async def test_listwise_reranker_floor_is_rejected(recall_harness: _RecallHarness) -> None:
    recall_harness.engine._cross_encoder_reranker = CrossEncoderReranker(cross_encoder=_Listwise())

    with pytest.raises(OperationValidationError) as exc_info:
        await recall_harness.recall(min_scores=MinScores(reranker=0.5))

    assert exc_info.value.status_code == 400
    assert "listwise" in exc_info.value.reason
    assert "jina-mlx" in exc_info.value.reason


@pytest.mark.asyncio
async def test_served_provider_reaches_span_and_recall_trace(
    recall_harness: _RecallHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    otel_tracer = MagicMock()
    recall_span = MagicMock()
    span_context = MagicMock()
    span_context.__enter__.return_value = recall_span
    span_context.__exit__.return_value = False
    otel_tracer.start_as_current_span.return_value = span_context
    spans: dict[str, MagicMock] = {}

    def start_span(name: str) -> MagicMock:
        span = MagicMock()
        spans[name] = span
        return span

    otel_tracer.start_span.side_effect = start_span
    monkeypatch.setattr("hindsight_api.tracing.get_tracer", lambda: otel_tracer)

    result = await recall_harness.recall(enable_trace=True)

    rerank_attributes = {
        call.args[0]: call.args[1] for call in spans["hindsight.recall_rerank"].set_attribute.call_args_list
    }
    assert rerank_attributes["hindsight.reranker_provider"] == "tei"
    assert rerank_attributes["hindsight.score_semantics"] == "pointwise"
    assert rerank_attributes["hindsight.reranker_prunes_candidates"] is False

    assert result.trace is not None
    rerank_phase = next(
        phase for phase in result.trace["summary"]["phase_metrics"] if phase["phase_name"] == "reranking"
    )
    assert rerank_phase["details"]["reranker_provider"] == "tei"
    assert rerank_phase["details"]["score_semantics"] == "pointwise"
    assert rerank_phase["details"]["prunes_candidates"] is False


async def _empty_retrieval(*_args: object, **_kwargs: object) -> MultiFactTypeRetrievalResult:
    return MultiFactTypeRetrievalResult(
        results_by_fact_type={
            "world": ParallelRetrievalResult(
                semantic=[],
                bm25=[],
                graph=[],
                temporal=None,
                timings={"semantic": 0.0, "bm25": 0.0, "graph": 0.0, "temporal_extraction": 0.0},
            )
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    ["typesafe", "jina", "rrf", "interleave", "rrf_provider", "ordinal_chain", "pool_dependent_chain"],
)
async def test_known_pool_dependent_empty_retrieval_rejects_reranker_floor(
    recall_harness: _RecallHarness, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setattr("hindsight_api.engine.search.retrieval.retrieve_all_fact_types_parallel", _empty_retrieval)
    encoder: CrossEncoderModel = _Ordinal()
    if mode == "jina":
        encoder = _Listwise()
    elif mode == "rrf_provider":
        encoder = RRFPassthroughCrossEncoder()
    elif mode == "ordinal_chain":
        encoder = MultiCrossEncoder([_Ordinal(), RRFPassthroughCrossEncoder()])
    elif mode == "pool_dependent_chain":
        encoder = MultiCrossEncoder([_Ordinal(), _Listwise()])
    recall_harness.engine._cross_encoder_reranker = CrossEncoderReranker(cross_encoder=encoder)
    reranking = {"rrf": "rrf", "interleave": "interleave"}.get(mode, "cross_encoder")

    with pytest.raises(OperationValidationError) as exc_info:
        await recall_harness.recall(reranking=reranking, min_scores=MinScores(reranker=0.5))

    assert exc_info.value.status_code == 400
    assert "min_scores.reranker" in exc_info.value.reason
    expected_semantics = "listwise" if mode == "jina" else "ordinal"
    assert expected_semantics in exc_info.value.reason


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["listwise", "pool_dependent_chain"])
async def test_empty_pool_dependent_floor_is_rejected_before_reranker_initialization(
    recall_harness: _RecallHarness, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setattr("hindsight_api.engine.search.retrieval.retrieve_all_fact_types_parallel", _empty_retrieval)
    encoder: CrossEncoderModel = _Listwise()
    if mode == "pool_dependent_chain":
        encoder = MultiCrossEncoder([_Ordinal(), _Listwise()])
    reranker = CrossEncoderReranker(cross_encoder=encoder)
    reranker.ensure_initialized = AsyncMock(side_effect=RuntimeError("model load failed"))
    recall_harness.engine._cross_encoder_reranker = reranker

    with pytest.raises(OperationValidationError) as exc_info:
        await recall_harness.recall(min_scores=MinScores(reranker=0.5))

    assert exc_info.value.status_code == 400
    assert "pool" in exc_info.value.reason or "listwise" in exc_info.value.reason
    reranker.ensure_initialized.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["listwise", "pool_dependent_chain"])
async def test_nonempty_pool_dependent_floor_is_rejected_before_reranker_initialization(
    recall_harness: _RecallHarness, mode: str
) -> None:
    encoder: CrossEncoderModel = _Listwise()
    if mode == "pool_dependent_chain":
        encoder = MultiCrossEncoder([_Ordinal(), _Listwise()])
    reranker = CrossEncoderReranker(cross_encoder=encoder)
    reranker.ensure_initialized = AsyncMock(side_effect=RuntimeError("model load failed"))
    recall_harness.engine._cross_encoder_reranker = reranker

    with pytest.raises(OperationValidationError) as exc_info:
        await recall_harness.recall(min_scores=MinScores(reranker=0.5))

    assert exc_info.value.status_code == 400
    assert "pool" in exc_info.value.reason or "listwise" in exc_info.value.reason
    reranker.ensure_initialized.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["pointwise", "mixed_chain"])
async def test_empty_retrieval_with_possible_pointwise_member_accepts_floor(
    recall_harness: _RecallHarness, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setattr("hindsight_api.engine.search.retrieval.retrieve_all_fact_types_parallel", _empty_retrieval)
    encoder: CrossEncoderModel = _Primary()
    if mode == "mixed_chain":
        encoder = MultiCrossEncoder([_Ordinal(), _Primary()])
    recall_harness.engine._cross_encoder_reranker = CrossEncoderReranker(cross_encoder=encoder)

    result = await recall_harness.recall(min_scores=MinScores(reranker=0.5))
    assert result.results == []


@pytest.mark.asyncio
@pytest.mark.parametrize("first_query", ["primary", "fallback"])
@pytest.mark.parametrize("with_floor", [False, True])
async def test_concurrent_recalls_use_their_own_rerank_provider(
    recall_harness: _RecallHarness, first_query: str, with_floor: bool
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
        return await recall_harness.recall(second_query, min_scores=MinScores(reranker=0.1) if with_floor else None)

    results = await asyncio.wait_for(
        asyncio.gather(
            recall_harness.recall(first_query, min_scores=MinScores(reranker=0.1) if with_floor else None),
            second_recall(),
            return_exceptions=with_floor,
        ),
        timeout=5,
    )
    by_query = dict(zip([first_query, second_query], results))
    assert chain.provider_name == ("rrf" if second_query == "fallback" else "tei")
    if with_floor:
        assert isinstance(by_query["primary"], RecallResult)
        assert isinstance(by_query["fallback"], OperationValidationError)
        assert by_query["fallback"].status_code == 400
        return
    assert [r.scores.reranker for r in by_query["primary"].results] == pytest.approx([0.2] * 3)
    assert [r.scores.final for r in by_query["primary"].results] == pytest.approx([0.7, 0.2 + 4 / 9, 0.6])
    assert all(r.scores.reranker is None for r in by_query["fallback"].results)
    assert [r.scores.final for r in by_query["fallback"].results] == pytest.approx([1.0, 0.55, 0.1])
