"""Configuration wiring tests for Link Expansion retrieval."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from hindsight_api.engine.search import link_expansion_retrieval
from hindsight_api.engine.search.link_expansion_retrieval import LinkExpansionRetriever
from hindsight_api.engine.search.types import RetrievalResult


@pytest.mark.asyncio
async def test_retrieve_passes_configured_graph_seed_threshold(monkeypatch):
    """Graph seed selection must use its dedicated configured threshold."""
    retriever = LinkExpansionRetriever()
    seed_thresholds: list[float] = []

    @asynccontextmanager
    async def fake_acquire_with_retry(_pool):
        yield object()

    async def fake_find_semantic_seeds(_conn, _embedding, _bank_id, fact_type, **kwargs):
        seed_thresholds.append(kwargs["threshold"])
        return [RetrievalResult(id="seed", text="seed", fact_type=fact_type)]

    async def fake_expand_combined(_conn, _seed_ids, _fact_type, _budget, *, ops):
        return [], [], []

    monkeypatch.setattr(link_expansion_retrieval, "acquire_with_retry", fake_acquire_with_retry)
    monkeypatch.setattr(link_expansion_retrieval, "_find_semantic_seeds", fake_find_semantic_seeds)
    monkeypatch.setattr(
        link_expansion_retrieval,
        "get_config",
        lambda: SimpleNamespace(graph_seed_min_similarity=0.47),
    )
    monkeypatch.setattr(retriever, "_expand_combined", fake_expand_combined)

    await retriever.retrieve(
        SimpleNamespace(ops=object()),
        query_embedding_str="unused",
        bank_id="bank",
        fact_type="world",
        budget=2,
    )

    assert seed_thresholds == [0.47]
