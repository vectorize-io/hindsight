"""Unit tests for SearchOrchestrator, SearchRequest, and hydration dataclasses."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.cancellation import OperationCancelledError
from hindsight_api.engine.response_models import ChunkInfo, EntityState
from hindsight_api.engine.search.orchestrator import (
    ChunkOrderItem,
    ChunkRow,
    EntityReference,
    HydratedChunks,
    HydratedEntities,
    HydratedSourceFacts,
    ObservationSourceRef,
    RawSourceFact,
    RetrievalTraceEntry,
    SearchOrchestrator,
    SearchRequest,
    _entity_map_from_results,
    _recall_scoring_now,
    to_trace_entries,
)
from hindsight_api.engine.search.reranking import RerankResult
from hindsight_api.engine.search.retrieval import (
    MultiFactTypeRetrievalResult,
    ParallelRetrievalResult,
)
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult, ScoredResult


def test_search_request_defaults():
    req = SearchRequest(
        bank_id="bank-1",
        query="test query",
        fact_type=["world"],
        thinking_budget=10,
        max_tokens=1000,
    )
    assert req.bank_id == "bank-1"
    assert req.query == "test query"
    assert req.fact_type == ["world"]
    assert req.thinking_budget == 10
    assert req.max_tokens == 1000
    assert req.enable_trace is False
    assert req.include_entities is False
    assert req.include_chunks is False
    assert req.reranking == "cross_encoder"
    assert req.enable_text_search is True
    assert req.enable_graph_retrieval is True


def test_dataclasses_instantiation():
    ref = EntityReference(entity_id="e1", canonical_name="Name")
    assert ref.entity_id == "e1"
    assert ref.canonical_name == "Name"

    chunks = HydratedChunks(chunks={"c1": ChunkInfo(chunk_text="text", chunk_index=0)}, total_tokens=10)
    assert chunks.chunks is not None
    assert chunks.total_tokens == 10

    sf = HydratedSourceFacts(source_facts=None, source_fact_ids_by_obs={"obs1": ["sf1"]}, truncated=False)
    assert sf.source_facts is None
    assert sf.source_fact_ids_by_obs == {"obs1": ["sf1"]}
    assert sf.truncated is False

    ent = HydratedEntities(
        fact_entity_map={"u1": [ref]},
        entities={"Name": EntityState(entity_id="e1", canonical_name="Name", observations=[])},
    )
    assert ent.fact_entity_map["u1"][0].canonical_name == "Name"
    assert ent.entities is not None and "Name" in ent.entities

    raw = RawSourceFact(
        id="sf1",
        text="some fact",
        fact_type="world",
        context=None,
        occurred_start=None,
        occurred_end=None,
        mentioned_at=None,
        document_id="doc1",
        chunk_id="c1",
        tags=["t1"],
        metadata=None,
    )
    assert raw.id == "sf1"
    assert raw.text == "some fact"
    assert raw.tags == ["t1"]

    obs_ref = ObservationSourceRef(id="obs-1", source_memory_ids=["sf1", "sf2"])
    assert obs_ref.id == "obs-1"
    assert obs_ref.source_memory_ids == ["sf1", "sf2"]

    chunk_row = ChunkRow(chunk_id="c1", chunk_text="hello", chunk_index=0, document_id="d1")
    assert chunk_row.chunk_id == "c1"
    assert chunk_row.chunk_text == "hello"
    assert chunk_row.chunk_index == 0
    assert chunk_row.document_id == "d1"

    order_item = ChunkOrderItem(item_type="chunk", id="c1")
    assert order_item.item_type == "chunk"
    assert order_item.id == "c1"


def test_retrieval_trace_entry_unpacking():
    entry = RetrievalTraceEntry(id="node-1", data={"text": "sample", "score": 0.8})
    assert entry.id == "node-1"
    doc_id, data = entry
    assert doc_id == "node-1"
    assert data["text"] == "sample"
    assert data["score"] == 0.8

    rr = RetrievalResult(id="r1", text="text1", fact_type="world")
    entries = to_trace_entries([rr])
    assert len(entries) == 1
    assert entries[0].id == "r1"
    assert entries[0].data["text"] == "text1"


def test_recall_scoring_now():
    now_dt = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    assert _recall_scoring_now(now_dt) == now_dt

    naive = datetime(2026, 6, 1, 12, 0)
    assert _recall_scoring_now(naive) == datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    aware = datetime(2026, 6, 1, 20, 0, tzinfo=timezone(timedelta(hours=8)))
    assert _recall_scoring_now(aware) == datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    fallback = _recall_scoring_now(None)
    assert fallback.tzinfo == UTC


def test_entity_map_from_results():
    ids_by_unit = {
        "u1": ["e1", "e2", "e1"],  # duplicate e1, unresolved e3 not present
        "u2": ["e_missing"],  # missing name
        "u3": [],  # empty
    }
    names = {"e1": "Entity 1", "e2": "Entity 2"}

    result = _entity_map_from_results(ids_by_unit, names)

    assert "u1" in result
    assert len(result["u1"]) == 2
    assert result["u1"][0] == EntityReference(entity_id="e1", canonical_name="Entity 1")
    assert result["u1"][1] == EntityReference(entity_id="e2", canonical_name="Entity 2")
    assert "u2" not in result
    assert "u3" not in result


@pytest.mark.asyncio
async def test_orchestrator_initialization_and_backend_resolution():
    mock_backend = cast(Any, object())
    mock_embeddings = cast(Any, object())
    mock_reranker = cast(Any, object())
    mock_analyzer = cast(Any, object())

    # 1. Direct backend attribute
    orchestrator = SearchOrchestrator(
        backend=mock_backend,
        embeddings=mock_embeddings,
        cross_encoder_reranker=mock_reranker,
        query_analyzer=mock_analyzer,
    )
    assert await orchestrator.get_read_backend() is mock_backend
    assert orchestrator.embeddings is mock_embeddings
    assert orchestrator.cross_encoder_reranker is mock_reranker
    assert orchestrator.query_analyzer is mock_analyzer

    # 2. get_read_backend callback
    custom_backend = object()

    async def custom_get_backend():
        return custom_backend

    orchestrator_cb = SearchOrchestrator(get_read_backend=custom_get_backend)
    assert await orchestrator_cb.get_read_backend() is custom_backend

    # 3. Engine delegation
    engine_backend = object()
    mock_engine = MagicMock()
    mock_engine._get_read_backend = AsyncMock(return_value=engine_backend)
    orchestrator_eng = SearchOrchestrator(engine=mock_engine)
    assert await orchestrator_eng.get_read_backend() is engine_backend

    # 4. Unconfigured raises RuntimeError
    orchestrator_none = SearchOrchestrator()
    with pytest.raises(RuntimeError, match="No backend provider configured for SearchOrchestrator"):
        await orchestrator_none.get_read_backend()


@pytest.mark.asyncio
async def test_search_orchestrator_cancellation_propagation(monkeypatch):
    monkeypatch.setattr(
        "hindsight_api.engine.retain.embedding_utils.generate_embeddings_batch",
        AsyncMock(return_value=[[0.1, 0.2, 0.3]]),
    )
    orchestrator = SearchOrchestrator(backend=MagicMock())
    request_context = MagicMock()

    def raise_cancel():
        raise OperationCancelledError("Operation cancelled")

    request_context.raise_if_cancelled.side_effect = raise_cancel

    request = SearchRequest(
        bank_id="bank-1",
        query="cancelled query",
        fact_type=["world"],
        thinking_budget=10,
        max_tokens=1000,
        request_context=request_context,
    )

    with pytest.raises(OperationCancelledError):
        await orchestrator.search(request)


@pytest.mark.asyncio
async def test_search_orchestrator_search_basic(monkeypatch):
    mock_backend = MagicMock()
    mock_reranker = MagicMock()
    mock_reranker.ensure_initialized = AsyncMock()

    retrieval = RetrievalResult(
        id="00000000-0000-0000-0000-000000000001",
        text="A test memory about python.",
        fact_type="world",
        similarity=0.9,
    )
    merged = MergedCandidate(
        retrieval=retrieval,
        rrf_score=0.9,
    )
    mock_reranker.rerank = AsyncMock(
        return_value=RerankResult(
            results=[
                ScoredResult(
                    candidate=merged,
                    cross_encoder_score=0.95,
                    cross_encoder_score_normalized=0.95,
                    weight=0.95,
                )
            ],
            provider_name="test_provider",
        )
    )

    async def fake_retrieve_all(*_args, **_kwargs):
        return MultiFactTypeRetrievalResult(
            results_by_fact_type={
                "world": ParallelRetrievalResult(
                    semantic=[retrieval],
                    bm25=[],
                    graph=[],
                    temporal=None,
                    timings={"semantic": 0.01, "bm25": 0.0, "graph": 0.0, "temporal_extraction": 0.0},
                )
            }
        )

    monkeypatch.setattr(
        "hindsight_api.engine.search.retrieval.retrieve_all_fact_types_parallel",
        fake_retrieve_all,
    )
    monkeypatch.setattr(
        "hindsight_api.engine.retain.embedding_utils.generate_embeddings_batch",
        AsyncMock(return_value=[[0.1, 0.2, 0.3]]),
    )

    orchestrator = SearchOrchestrator(
        backend=mock_backend,
        cross_encoder_reranker=mock_reranker,
    )

    request = SearchRequest(
        bank_id="test-bank",
        query="python",
        fact_type=["world"],
        thinking_budget=10,
        max_tokens=1000,
        enable_trace=True,
    )

    result = await orchestrator.search(request)

    assert len(result.results) == 1
    assert result.results[0].text == "A test memory about python."
    assert result.results[0].fact_type == "world"
    assert result.trace is not None


@pytest.mark.asyncio
async def test_search_orchestrator_store_read_conn():
    mock_conn = object()

    @asynccontextmanager
    async def custom_store_read_conn(bank_id: str):
        assert bank_id == "bank-custom"
        yield mock_conn

    orchestrator_custom = SearchOrchestrator(store_read_conn=custom_store_read_conn)
    async with orchestrator_custom.store_read_conn("bank-custom") as conn:
        assert conn is mock_conn

    engine_conn = object()
    mock_engine = MagicMock()

    @asynccontextmanager
    async def engine_store_read_conn(bank_id: str):
        assert bank_id == "bank-engine"
        yield engine_conn

    mock_engine._store_read_conn = engine_store_read_conn
    orchestrator_engine = SearchOrchestrator(engine=mock_engine)
    async with orchestrator_engine.store_read_conn("bank-engine") as conn:
        assert conn is engine_conn

    orchestrator_none = SearchOrchestrator()
    with pytest.raises(RuntimeError, match="No store_read_conn configured for SearchOrchestrator"):
        orchestrator_none.store_read_conn("bank-err")


@pytest.mark.asyncio
async def test_search_orchestrator_missing_cross_encoder_reranker_raises(monkeypatch):
    monkeypatch.setattr(
        "hindsight_api.engine.retain.embedding_utils.generate_embeddings_batch",
        AsyncMock(return_value=[[0.1, 0.2, 0.3]]),
    )
    retrieval = RetrievalResult(
        id="00000000-0000-0000-0000-000000000001",
        text="A test memory.",
        fact_type="world",
        similarity=0.9,
    )

    async def fake_retrieve_all(*_args, **_kwargs):
        return MultiFactTypeRetrievalResult(
            results_by_fact_type={
                "world": ParallelRetrievalResult(
                    semantic=[retrieval],
                    bm25=[],
                    graph=[],
                    temporal=None,
                    timings={},
                )
            }
        )

    monkeypatch.setattr(
        "hindsight_api.engine.search.retrieval.retrieve_all_fact_types_parallel",
        fake_retrieve_all,
    )

    orchestrator = SearchOrchestrator(backend=MagicMock(), cross_encoder_reranker=None)
    request = SearchRequest(
        bank_id="test-bank",
        query="test",
        fact_type=["world"],
        thinking_budget=10,
        max_tokens=1000,
        reranking="cross_encoder",
    )
    with pytest.raises(RuntimeError, match="No CrossEncoderReranker configured for SearchOrchestrator"):
        await orchestrator.search(request)
