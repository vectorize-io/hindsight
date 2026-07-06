from types import SimpleNamespace
from uuid import uuid4

import pytest

from hindsight_api.engine import memory_engine as memory_engine_module
from hindsight_api.engine.memory_engine import MemoryEngine, _resolve_recall_dedup_threshold
from hindsight_api.engine.search import dedup as dedup_module
from hindsight_api.engine.search.dedup import (
    collapse_near_duplicate_raw_facts,
    normalize_recall_dedup_text,
    recall_text_similarity,
)
from hindsight_api.engine.search.retrieval import MultiFactTypeRetrievalResult, ParallelRetrievalResult
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult, ScoredResult


def _scored_result(result_id: str, text: str, fact_type: str = "world") -> ScoredResult:
    return ScoredResult(
        candidate=MergedCandidate(
            retrieval=RetrievalResult(id=result_id, text=text, fact_type=fact_type),
            rrf_score=1.0,
        ),
        weight=1.0,
    )


def test_normalize_recall_dedup_text_ignores_case_and_punctuation() -> None:
    assert normalize_recall_dedup_text("RTX 3090, GPU.") == "rtx 3090 gpu"


def test_recall_text_similarity_returns_one_for_normalized_exact_match() -> None:
    assert recall_text_similarity("Alice owns an RTX 3090.", "alice owns an rtx 3090") == 1.0


def test_exact_raw_duplicate_is_collapsed_at_default_threshold() -> None:
    results = [
        _scored_result("1", "Alice owns an RTX 3090."),
        _scored_result("2", "alice owns an rtx 3090"),
        _scored_result("3", "Alice owns an RTX 4090."),
    ]

    collapsed = collapse_near_duplicate_raw_facts(results, threshold=1.0)

    assert [result.id for result in collapsed] == ["1", "3"]


def test_default_threshold_uses_exact_membership_without_fuzzy_match(monkeypatch) -> None:
    def fail_sequence_matcher(*_args, **_kwargs):
        raise AssertionError("SequenceMatcher should not run for threshold >= 1.0")

    monkeypatch.setattr(dedup_module, "SequenceMatcher", fail_sequence_matcher)
    results = [
        _scored_result("1", "Alice owns an RTX 3090."),
        _scored_result("2", "alice owns an rtx 3090"),
        _scored_result("3", "Alice owns an RTX 4090."),
    ]

    collapsed = collapse_near_duplicate_raw_facts(results, threshold=1.0)

    assert [result.id for result in collapsed] == ["1", "3"]


def test_near_duplicate_requires_lower_threshold() -> None:
    results = [
        _scored_result("1", "Alice owns an RTX 3090 GPU."),
        _scored_result("2", "Alice has an RTX 3090 GPU."),
    ]

    exact_only = collapse_near_duplicate_raw_facts(results, threshold=1.0)
    fuzzy = collapse_near_duplicate_raw_facts(results, threshold=0.85)

    assert [result.id for result in exact_only] == ["1", "2"]
    assert [result.id for result in fuzzy] == ["1"]


def test_observations_are_not_collapsed_by_text() -> None:
    results = [
        _scored_result("1", "Alice owns an RTX 3090.", "observation"),
        _scored_result("2", "Alice owns an RTX 3090.", "observation"),
    ]

    collapsed = collapse_near_duplicate_raw_facts(results, threshold=1.0)

    assert [result.id for result in collapsed] == ["1", "2"]


def test_resolve_recall_dedup_threshold_tolerates_bad_bank_overrides() -> None:
    assert _resolve_recall_dedup_threshold("not-a-number") == 1.0
    assert _resolve_recall_dedup_threshold(None) == 1.0
    assert _resolve_recall_dedup_threshold(-0.25) == 0.0
    assert _resolve_recall_dedup_threshold(1.25) == 1.0
    assert _resolve_recall_dedup_threshold("0.75") == 0.75


@pytest.mark.asyncio
async def test_memory_engine_recall_dedups_before_truncation(monkeypatch) -> None:
    ids = [str(uuid4()) for _ in range(3)]
    retrievals = [
        RetrievalResult(id=ids[0], text="Alice owns an RTX 3090.", fact_type="world", similarity=0.99),
        RetrievalResult(id=ids[1], text="alice owns an rtx 3090", fact_type="world", similarity=0.98),
        RetrievalResult(id=ids[2], text="Alice owns an RTX 4090.", fact_type="world", similarity=0.97),
    ]

    async def fake_generate_embeddings_batch(*_args, **_kwargs):
        return [[0.1, 0.2, 0.3]]

    async def fake_retrieve_all_fact_types_parallel(*_args, **_kwargs):
        return MultiFactTypeRetrievalResult(
            results_by_fact_type={
                "world": ParallelRetrievalResult(
                    semantic=retrievals,
                    bm25=[],
                    graph=[],
                    temporal=None,
                    timings={"semantic": 0.0, "bm25": 0.0, "graph": 0.0, "temporal": 0.0},
                )
            },
            max_conn_wait=0.0,
        )

    class FakeBudgetedOperation:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def wrap_pool(self, backend):
            return backend

    def fake_budgeted_operation(*_args, **_kwargs):
        return FakeBudgetedOperation()

    class FakeSpan:
        def set_attribute(self, *_args, **_kwargs):
            return None

        def end(self):
            return None

    class FakeTracer:
        def start_span(self, *_args, **_kwargs):
            return FakeSpan()

    def preserve_rrf_order(scored_results, **_kwargs):
        for scored_result in scored_results:
            scored_result.weight = scored_result.candidate.rrf_score

    monkeypatch.setattr(
        memory_engine_module.embedding_utils,
        "generate_embeddings_batch",
        fake_generate_embeddings_batch,
    )
    monkeypatch.setattr(
        "hindsight_api.engine.search.retrieval.retrieve_all_fact_types_parallel",
        fake_retrieve_all_fact_types_parallel,
    )
    monkeypatch.setattr(memory_engine_module, "budgeted_operation", fake_budgeted_operation)
    monkeypatch.setattr("hindsight_api.tracing.get_tracer", lambda: FakeTracer())
    monkeypatch.setattr(memory_engine_module, "apply_combined_scoring", preserve_rrf_order)

    engine = MemoryEngine.__new__(MemoryEngine)
    engine.embeddings = object()
    engine.query_analyzer = object()
    engine._initialized = True
    engine._read_backend = object()
    engine._backend = SimpleNamespace(ops=SimpleNamespace(uses_observation_sources_table=False))
    engine._cross_encoder_reranker = SimpleNamespace(cross_encoder=None)
    engine._filter_by_token_budget = MemoryEngine._filter_by_token_budget.__get__(engine, MemoryEngine)

    result = await MemoryEngine._search_with_retries(
        engine,
        bank_id="test-bank",
        query="gpu",
        fact_type=["world"],
        thinking_budget=1,
        max_tokens=1000,
        enable_trace=False,
        reranking="rrf",
        recall_dedup_threshold=1.0,
    )

    assert [fact.id for fact in result.results] == [ids[0], ids[2]]


def test_world_and_experience_duplicates_are_kept_separate() -> None:
    results = [
        _scored_result("1", "Alice owns an RTX 3090.", "world"),
        _scored_result("2", "Alice owns an RTX 3090.", "experience"),
    ]

    collapsed = collapse_near_duplicate_raw_facts(results, threshold=1.0)

    assert [result.id for result in collapsed] == ["1", "2"]


def test_threshold_zero_disables_deduplication() -> None:
    results = [
        _scored_result("1", "Alice owns an RTX 3090."),
        _scored_result("2", "Alice owns an RTX 3090."),
    ]

    collapsed = collapse_near_duplicate_raw_facts(results, threshold=0.0)

    assert [result.id for result in collapsed] == ["1", "2"]
