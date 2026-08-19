"""Regression tests for Link Expansion's final graph score."""

import math
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from hindsight_api.engine.db.ops import LinkExpansionRows
from hindsight_api.engine.search import link_expansion_retrieval
from hindsight_api.engine.search.link_expansion_retrieval import LinkExpansionRetriever
from hindsight_api.engine.search.types import RetrievalResult


def _row(fact_id: str, score: float, fact_type: str) -> dict[str, str | float]:
    """Create the subset of an expansion query row needed by RetrievalResult."""
    return {"id": fact_id, "text": fact_id, "fact_type": fact_type, "score": score}


@pytest.mark.asyncio
async def test_activation_preserves_additive_score_across_fact_types(monkeypatch):
    """Graph merge order must match Link Expansion's additive per-type score."""
    retriever = LinkExpansionRetriever()

    @asynccontextmanager
    async def fake_acquire_with_retry(_pool):
        yield object()

    async def fake_expand_combined(_conn, _seed_ids, fact_type, _budget, *, ops, created_after, created_before):
        if fact_type == "world":
            # Convergent semantic and causal signals make this fact's total
            # score higher, despite its raw entity count being only 1.
            return LinkExpansionRows(
                entity=[_row("a", 1.0, fact_type)],
                semantic=[_row("a", 0.9, fact_type)],
                causal=[_row("a", 0.3, fact_type)],
            )
        return LinkExpansionRows(entity=[_row("b", 2.0, fact_type)], semantic=[_row("b", 0.7, fact_type)], causal=[])

    async def fake_find_semantic_seeds(_conn, _embedding, _bank_id, fact_type, **_kwargs):
        # Link Expansion chooses its own seeds internally (#2683), so stub the
        # lookup rather than injecting seeds through retrieve().
        return [RetrievalResult(id=f"seed-{fact_type}", text="seed", fact_type=fact_type)]

    monkeypatch.setattr(link_expansion_retrieval, "acquire_with_retry", fake_acquire_with_retry)
    monkeypatch.setattr(link_expansion_retrieval, "_find_semantic_seeds", fake_find_semantic_seeds)
    monkeypatch.setattr(retriever, "_expand_combined", fake_expand_combined)
    pool = SimpleNamespace(ops=object())

    world_results, _ = await retriever.retrieve(
        pool,
        query_embedding_str="unused",
        bank_id="bank",
        fact_type="world",
        budget=2,
    )
    experience_results, _ = await retriever.retrieve(
        pool,
        query_embedding_str="unused",
        bank_id="bank",
        fact_type="experience",
        budget=2,
    )

    combined = world_results + experience_results
    combined.sort(key=lambda result: result.activation or 0.0, reverse=True)

    assert [result.id for result in combined] == ["a", "b"]
    assert world_results[0].activation == pytest.approx(math.tanh(0.5) + 0.9 + 0.3)
    assert experience_results[0].activation == pytest.approx(math.tanh(1.0) + 0.7)


@pytest.mark.asyncio
async def test_preselected_semantic_seeds_skip_seed_query(monkeypatch):
    retriever = LinkExpansionRetriever()

    @asynccontextmanager
    async def fake_acquire_with_retry(_pool):
        yield object()

    async def fail_find_semantic_seeds(*args, **kwargs):
        raise AssertionError("preselected seeds must bypass the graph seed query")

    async def fake_expand_combined(_conn, seed_ids, fact_type, _budget, *, ops, created_after, created_before):
        assert set(seed_ids) == {"seed-a", "seed-b"}
        return LinkExpansionRows(entity=[_row("result", 1.0, fact_type)], semantic=[], causal=[])

    monkeypatch.setattr(link_expansion_retrieval, "acquire_with_retry", fake_acquire_with_retry)
    monkeypatch.setattr(link_expansion_retrieval, "_find_semantic_seeds", fail_find_semantic_seeds)
    monkeypatch.setattr(retriever, "_expand_combined", fake_expand_combined)

    results, timings = await retriever.retrieve(
        SimpleNamespace(ops=object()),
        query_embedding_str="unused",
        bank_id="bank",
        fact_type="world",
        budget=2,
        preselected_semantic_seeds=[
            RetrievalResult(id="seed-a", text="seed", fact_type="world"),
            RetrievalResult(id="seed-b", text="seed", fact_type="world"),
        ],
    )

    assert [result.id for result in results] == ["result"]
    assert timings is not None
    assert timings.seeds_time == 0.0


@pytest.mark.asyncio
async def test_empty_preselected_semantic_seeds_do_not_fall_back(monkeypatch):
    retriever = LinkExpansionRetriever()

    @asynccontextmanager
    async def fake_acquire_with_retry(_pool):
        yield object()

    async def fail_find_semantic_seeds(*args, **kwargs):
        raise AssertionError("an empty shared pool must not trigger a second seed query")

    monkeypatch.setattr(link_expansion_retrieval, "acquire_with_retry", fake_acquire_with_retry)
    monkeypatch.setattr(link_expansion_retrieval, "_find_semantic_seeds", fail_find_semantic_seeds)

    results, timings = await retriever.retrieve(
        SimpleNamespace(ops=object()),
        query_embedding_str="unused",
        bank_id="bank",
        fact_type="world",
        budget=2,
        preselected_semantic_seeds=[],
    )

    assert results == []
    assert timings is not None
    assert timings.seeds_time == 0.0
    assert timings.db_queries == 0
