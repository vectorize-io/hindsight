"""Per-bank recall pipeline stage toggles.

Recall runs four arms (semantic, BM25, graph, temporal) and then a cross-encoder
rerank. A bank whose content has no relational or temporal structure pays latency
for arms it cannot use, so each stage can be switched off per bank:

    enable_temporal_retrieval / enable_graph_retrieval / enable_reranking

All three default to True, so recall behaviour is unchanged unless a bank opts out.
"""

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from hindsight_api.engine import memory_engine as memory_engine_module
from hindsight_api.engine.search import retrieval as retrieval_module

_QUERY = "[0.1,0.2,0.3]"


@pytest.fixture
def stub_retrieval(monkeypatch):
    """Stub the DB-backed arms and record which ones actually ran.

    The per-arm orchestration now lives in ``PostgresMemories.recall_unified``, so the stubs sit at
    the seams IT uses: the shared connection acquire, the dense+BM25 SQL, the temporal SQL, and the
    graph retriever (installed into the module-global cache, restored automatically at teardown).
    """
    calls = {"temporal_extraction": 0, "temporal_combined": 0, "graph": 0}

    @asynccontextmanager
    async def fake_acquire_with_retry(pool, *args, **kwargs):
        yield object()

    async def fake_semantic_bm25_combined_sql(*args, **kwargs):
        return {"world": retrieval_module.SemanticBm25Result(semantic=[], bm25=[], graph_seeds=None)}

    async def fake_temporal_combined_sql(*args, **kwargs):
        calls["temporal_combined"] += 1
        return {"world": []}

    def fake_extract_temporal_constraint(*args, **kwargs):
        calls["temporal_extraction"] += 1
        return (datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 2, 1, tzinfo=UTC))

    class FakeGraphRetriever:
        async def retrieve(self, **kwargs):
            calls["graph"] += 1
            return [], None

    fake_config = SimpleNamespace(graph_seed_min_similarity=0.3, temporal_semantic_min_similarity=0.24)

    from hindsight_api.engine.memories.postgres import PostgresMemories

    monkeypatch.setattr("hindsight_api.engine.memories._memories", PostgresMemories({}))
    monkeypatch.setattr("hindsight_api.engine.db_utils.acquire_with_retry", fake_acquire_with_retry)
    monkeypatch.setattr(retrieval_module, "retrieve_semantic_bm25_combined_sql", fake_semantic_bm25_combined_sql)
    monkeypatch.setattr(retrieval_module, "retrieve_temporal_combined_sql", fake_temporal_combined_sql)
    monkeypatch.setattr(retrieval_module, "_default_graph_retriever", FakeGraphRetriever())
    monkeypatch.setattr(
        "hindsight_api.engine.search.temporal_extraction.extract_temporal_constraint",
        fake_extract_temporal_constraint,
    )
    monkeypatch.setattr(retrieval_module, "get_config", lambda: fake_config)
    monkeypatch.setattr("hindsight_api.config.get_config", lambda: fake_config)
    return calls, FakeGraphRetriever


async def _run(**kwargs):
    return await retrieval_module.retrieve_all_fact_types_parallel(
        object(),
        query_text="what happened in January?",
        query_embedding_str=_QUERY,
        bank_id="test_recall_pipeline_toggles",
        fact_types=["world"],
        thinking_budget=10,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_all_stages_run_by_default(stub_retrieval):
    """Defaults preserve current behaviour: every arm runs."""
    calls, _FakeGraphRetriever = stub_retrieval

    await _run()

    assert calls["temporal_extraction"] == 1
    assert calls["temporal_combined"] == 1
    assert calls["graph"] == 1


@pytest.mark.asyncio
async def test_disabling_temporal_retrieval_skips_the_temporal_arm(stub_retrieval):
    """No constraint extraction means nothing to filter on, so the temporal query is skipped too."""
    calls, _FakeGraphRetriever = stub_retrieval

    result = await _run(enable_temporal_retrieval=False)

    assert calls["temporal_extraction"] == 0
    assert calls["temporal_combined"] == 0
    # The other arms are unaffected.
    assert calls["graph"] == 1
    assert result.results_by_fact_type["world"].temporal is None


@pytest.mark.asyncio
async def test_disabling_graph_retrieval_skips_the_graph_arm(stub_retrieval):
    """The graph retriever is never invoked and the arm returns nothing."""
    calls, _FakeGraphRetriever = stub_retrieval

    result = await _run(enable_graph_retrieval=False)

    assert calls["graph"] == 0
    # The other arms are unaffected.
    assert calls["temporal_extraction"] == 1
    assert calls["temporal_combined"] == 1
    assert result.results_by_fact_type["world"].graph == []


@pytest.mark.asyncio
async def test_disabling_graph_retrieval_does_not_resolve_a_retriever(stub_retrieval, monkeypatch):
    """With the arm off, the default retriever is not resolved either — resolving one
    can lazily build a retriever, which is wasted work when the arm is off."""
    calls, _ = stub_retrieval

    def boom():
        raise AssertionError("default graph retriever must not be resolved when the arm is disabled")

    monkeypatch.setattr(retrieval_module, "get_default_graph_retriever", boom)

    await _run(enable_graph_retrieval=False)

    assert calls["graph"] == 0


@pytest.mark.parametrize(
    ("requested", "enable_reranking", "expected"),
    [
        # Default: unchanged.
        ("cross_encoder", True, "cross_encoder"),
        # Disabled: fall back to the RRF-fused ordering.
        ("cross_encoder", False, "rrf"),
        # Already rerank-free — nothing to downgrade.
        ("rrf", False, "rrf"),
        # An explicit caller choice (consolidation dedup) is never overridden: interleave
        # guarantees each arm a slot, which RRF does not, so clobbering it would change
        # which candidates consolidation sees.
        ("interleave", False, "interleave"),
        ("interleave", True, "interleave"),
    ],
)
def test_reranking_resolution(requested, enable_reranking, expected):
    resolved = memory_engine_module._resolve_reranking({"enable_reranking": enable_reranking}, requested)
    assert resolved == expected


def test_reranking_defaults_to_enabled_when_unset():
    """A bank with no override keeps the cross-encoder."""
    assert memory_engine_module._resolve_reranking({}, "cross_encoder") == "cross_encoder"


# ── config → pipeline wiring ─────────────────────────────────────────────────
#
# Everything above drives retrieve_all_fact_types_parallel and _resolve_reranking
# directly, which leaves the layer that *reads* the bank's config untested:
# recall_async looks the stages up by plain string key, so a typo would resolve to
# the default and every test above would still pass. These go through a real bank
# override to cover that seam.


def _bank(prefix: str) -> str:
    return f"{prefix}_{datetime.now().timestamp()}"


@pytest.mark.asyncio
async def test_bank_override_reaches_the_retrieval_arms(memory, request_context, monkeypatch):
    """The bank's temporal/graph settings arrive at the retrieval call itself."""
    bank_id = _bank("recall_toggle_arms")
    await memory._ensure_bank_exists(bank_id, request_context)

    seen: list[dict[str, bool]] = []
    real_retrieve = retrieval_module.retrieve_all_fact_types_parallel

    async def spy(*args, **kwargs):
        seen.append(
            {
                "temporal": kwargs["enable_temporal_retrieval"],
                "graph": kwargs["enable_graph_retrieval"],
            }
        )
        return await real_retrieve(*args, **kwargs)

    monkeypatch.setattr(retrieval_module, "retrieve_all_fact_types_parallel", spy)

    await memory.recall_async(bank_id=bank_id, query="anything", request_context=request_context)
    assert seen[-1] == {"temporal": True, "graph": True}, "a bank with no overrides runs every arm"

    await memory._config_resolver.update_bank_config(
        bank_id, {"enable_temporal_retrieval": False, "enable_graph_retrieval": False}
    )
    await memory.recall_async(bank_id=bank_id, query="anything", request_context=request_context)
    assert seen[-1] == {"temporal": False, "graph": False}


@pytest.mark.asyncio
async def test_bank_override_disables_the_cross_encoder(memory, request_context, monkeypatch):
    """enable_reranking=False stops recall reaching the cross-encoder at all.

    Asserted on the reranker rather than on the resolved strategy string: skipping
    that call is the whole point of the toggle, and it is what a caller feels.
    """
    bank_id = _bank("recall_toggle_rerank")
    await memory._ensure_bank_exists(bank_id, request_context)

    rerank_calls: list[str] = []
    real_rerank = memory._cross_encoder_reranker.rerank

    async def spy(query, candidates):
        rerank_calls.append(query)
        return await real_rerank(query, candidates)

    monkeypatch.setattr(memory._cross_encoder_reranker, "rerank", spy)

    await memory.recall_async(bank_id=bank_id, query="anything", request_context=request_context)
    assert len(rerank_calls) == 1, "a bank with no override still reranks"

    await memory._config_resolver.update_bank_config(bank_id, {"enable_reranking": False})
    await memory.recall_async(bank_id=bank_id, query="anything", request_context=request_context)
    assert len(rerank_calls) == 1, "the cross-encoder must not be reached once reranking is off"
