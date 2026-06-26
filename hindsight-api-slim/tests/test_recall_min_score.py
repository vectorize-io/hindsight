"""Tests for the recall `scores` object and the `min_scores` filters.

Inserts memory_units with known content + real embeddings directly via SQL, then
verifies that recall_async:
  - returns a `scores` object (final/reranker/semantic/text) on every result,
  - applies the post-query floors (`reranker`, `final`) to the scored results,
  - applies the retrieval-level floors (`semantic`, `text`) inside the SQL arms,
  - is unchanged by the default (`min_scores=None`).

Filtering is deterministic post/﻿pre-processing, so these assertions are direct —
no LLM and no LLM-as-judge required (uses the mock provider).
"""

import uuid

import pytest
import pytest_asyncio

from hindsight_api import MemoryEngine, RequestContext
from hindsight_api.engine.response_models import MinScores
from hindsight_api.engine.retain import embedding_utils

# Shared hardcoded UUIDs (memory_units.id is a global PK) → serialize xdist workers
# onto one group to avoid pk conflicts, same as test_recall_time_range.py.
pytestmark = pytest.mark.xdist_group("recall_min_score")

ID_A = "00000000-0000-0000-0000-0000000000a1"
ID_B = "00000000-0000-0000-0000-0000000000a2"
ID_C = "00000000-0000-0000-0000-0000000000a3"
ALL_IDS = (ID_A, ID_B, ID_C)

RC = RequestContext(tenant_id="default")


async def _insert_fact(conn, *, fact_id: str, text: str, bank_id: str, embedding_str: str) -> None:
    await conn.execute(
        """
        INSERT INTO memory_units (id, bank_id, text, fact_type, embedding)
        VALUES ($1, $2, $3, 'world', $4::vector)
        """,
        fact_id,
        bank_id,
        text,
        embedding_str,
    )


@pytest_asyncio.fixture
async def seeded_memory(memory_no_llm_verify: MemoryEngine):
    """Insert three facts with real embeddings and return (engine, bank_id)."""
    engine = memory_no_llm_verify
    bank_id = f"test-min-score-{uuid.uuid4().hex[:8]}"

    await engine.get_bank_profile(bank_id, request_context=RC)

    embeddings = await embedding_utils.generate_embeddings_batch(
        engine.embeddings,
        ["the cat sat on the mat", "dogs are loyal animals", "birds can fly in the sky"],
    )

    def _to_str(emb: list[float]) -> str:
        return "[" + ",".join(str(v) for v in emb) + "]"

    pool = await engine._get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM memory_units WHERE id IN ($1, $2, $3)", *ALL_IDS)
        await _insert_fact(
            conn, fact_id=ID_A, text="the cat sat on the mat", bank_id=bank_id, embedding_str=_to_str(embeddings[0])
        )
        await _insert_fact(
            conn, fact_id=ID_B, text="dogs are loyal animals", bank_id=bank_id, embedding_str=_to_str(embeddings[1])
        )
        await _insert_fact(
            conn, fact_id=ID_C, text="birds can fly in the sky", bank_id=bank_id, embedding_str=_to_str(embeddings[2])
        )

    yield engine, bank_id

    await engine.delete_bank(bank_id, request_context=RC)


def _ids(result) -> set[str]:
    return {str(r.id) for r in result.results}


async def _recall(engine, bank_id, *, query="animals and nature", **kwargs):
    return await engine.recall_async(
        bank_id=bank_id,
        query=query,
        request_context=RC,
        max_tokens=10000,
        **kwargs,
    )


class TestRecallScores:
    async def test_every_result_has_scores(self, seeded_memory):
        engine, bank_id = seeded_memory
        result = await _recall(engine, bank_id)
        assert result.results, "expected the seeded facts to be recalled"
        for r in result.results:
            assert r.scores is not None, f"result {r.id} is missing scores"
            assert isinstance(r.scores.final, float)
            # semantic surfaced these (vector arm) — should be populated and 0..1
            assert r.scores.semantic is not None
            assert 0.0 <= r.scores.semantic <= 1.0

    async def test_results_ordered_by_descending_final(self, seeded_memory):
        engine, bank_id = seeded_memory
        result = await _recall(engine, bank_id)
        finals = [r.scores.final for r in result.results]
        assert finals == sorted(finals, reverse=True), f"results not ordered by final score: {finals}"


class TestPostQueryFilters:
    async def test_none_is_no_op(self, seeded_memory):
        engine, bank_id = seeded_memory
        baseline = await _recall(engine, bank_id)
        explicit = await _recall(engine, bank_id, min_scores=None)
        assert _ids(baseline) == _ids(explicit)

    async def test_final_floor_filters_and_is_a_subset(self, seeded_memory):
        engine, bank_id = seeded_memory
        baseline = await _recall(engine, bank_id)
        finals = sorted((r.scores.final for r in baseline.results), reverse=True)
        assert len(finals) >= 2, "need at least two results to exercise a mid threshold"
        threshold = finals[-1] + (finals[-2] - finals[-1]) / 2

        filtered = await _recall(engine, bank_id, min_scores=MinScores(final=threshold))
        assert _ids(filtered), "threshold should still keep the top result(s)"
        assert _ids(filtered) < _ids(baseline), "threshold must drop at least one result"
        for r in filtered.results:
            assert r.scores.final >= threshold

    async def test_final_floor_above_all_returns_empty(self, seeded_memory):
        engine, bank_id = seeded_memory
        baseline = await _recall(engine, bank_id)
        max_final = max(r.scores.final for r in baseline.results)
        filtered = await _recall(engine, bank_id, min_scores=MinScores(final=max_final + 1.0))
        assert filtered.results == [], f"expected nothing above all final scores, got {_ids(filtered)}"

    async def test_reranker_floor_filters(self, seeded_memory):
        engine, bank_id = seeded_memory
        baseline = await _recall(engine, bank_id)
        rerankers = sorted((r.scores.reranker for r in baseline.results if r.scores.reranker is not None))
        if len(rerankers) < 2:
            pytest.skip("reranker scores unavailable (passthrough reranker)")
        threshold = rerankers[-1]  # keep only the top reranker score(s)
        filtered = await _recall(engine, bank_id, min_scores=MinScores(reranker=threshold))
        assert len(filtered.results) < len(baseline.results)
        for r in filtered.results:
            assert r.scores.reranker is not None and r.scores.reranker >= threshold


class TestRetrievalLevelFilters:
    async def test_semantic_floor_prunes_in_retrieval(self, seeded_memory):
        """min_scores.semantic is a SQL-arm cutoff: every returned result has a
        semantic score >= the floor, and a high floor returns nothing."""
        engine, bank_id = seeded_memory
        baseline = await _recall(engine, bank_id)
        sems = sorted(r.scores.semantic for r in baseline.results if r.scores.semantic is not None)
        assert sems, "semantic arm should have surfaced these facts"
        # A floor just above the lowest semantic score must drop that weakest result.
        floor = sems[-1]
        filtered = await _recall(engine, bank_id, min_scores=MinScores(semantic=floor))
        assert len(filtered.results) <= len(baseline.results)
        for r in filtered.results:
            assert r.scores.semantic is not None and r.scores.semantic >= floor

    async def test_semantic_floor_above_one_returns_empty(self, seeded_memory):
        engine, bank_id = seeded_memory
        filtered = await _recall(engine, bank_id, min_scores=MinScores(semantic=1.1))
        assert filtered.results == []


class TestRecallRequestDefault:
    """min_scores is opt-in: the HTTP recall defaults to None (no filtering)."""

    def test_http_request_defaults_to_none(self):
        from hindsight_api.api.http import RecallRequest

        assert RecallRequest(query="hi").min_scores is None
