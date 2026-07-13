"""Regression tests for Link Expansion's final graph score."""

from contextlib import asynccontextmanager
import math
from types import SimpleNamespace

import pytest

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

    async def fake_expand_combined(_conn, _seed_ids, fact_type, _budget, *, ops):
        if fact_type == "world":
            # Convergent semantic and causal signals make this fact's total
            # score higher, despite its raw entity count being only 1.
            return [_row("a", 1.0, fact_type)], [_row("a", 0.9, fact_type)], [_row("a", 0.3, fact_type)]
        return [_row("b", 2.0, fact_type)], [_row("b", 0.7, fact_type)], []

    monkeypatch.setattr(link_expansion_retrieval, "acquire_with_retry", fake_acquire_with_retry)
    monkeypatch.setattr(retriever, "_expand_combined", fake_expand_combined)
    pool = SimpleNamespace(ops=object())

    world_results, _ = await retriever.retrieve(
        pool,
        query_embedding_str="unused",
        bank_id="bank",
        fact_type="world",
        budget=2,
        semantic_seeds=[RetrievalResult(id="seed-world", text="seed", fact_type="world")],
    )
    experience_results, _ = await retriever.retrieve(
        pool,
        query_embedding_str="unused",
        bank_id="bank",
        fact_type="experience",
        budget=2,
        semantic_seeds=[RetrievalResult(id="seed-experience", text="seed", fact_type="experience")],
    )

    combined = world_results + experience_results
    combined.sort(key=lambda result: result.activation or 0.0, reverse=True)

    assert [result.id for result in combined] == ["a", "b"]
    assert world_results[0].activation == pytest.approx(math.tanh(0.5) + 0.9 + 0.3)
    assert experience_results[0].activation == pytest.approx(math.tanh(1.0) + 0.7)
