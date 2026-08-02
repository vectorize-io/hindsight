"""Tests for the temporal-recall entry-point selection (Option A: similarity-gated +
window coverage).

`retrieve_temporal_combined` selects in-window entry points by *embedding similarity*
(not recency) and then narrows them to span the window's time range via
`_select_with_temporal_coverage`. This replaced an earlier recency-ranked selection that
biased toward the end of the window and, on banks with dense/near-uniform dates, degraded
to a full scan + disk-spilling sort while dropping the most relevant in-window memory.

These are pure mechanics (no LLM), so they assert directly.
"""

import asyncio
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

import hindsight_api.engine.query_analyzer as query_analyzer_module
import hindsight_api.engine.search.retrieval as retrieval_module
from hindsight_api.engine.search.retrieval import _select_with_temporal_coverage, retrieve_temporal_combined
from hindsight_api.engine.task_backend import fq_table

EMBED_DIM = 384


def _vec(*leading: float) -> str:
    values = list(leading) + [0.0] * (EMBED_DIM - len(leading))
    return "[" + ",".join(str(v) for v in values) + "]"


# Query vector + vectors at known cosine similarities to it.
_QUERY = _vec(1.0)
_SIM_100 = _vec(1.0)  # cosine 1.0
_SIM_090 = _vec(0.9, 0.4358898943540674)  # cosine 0.9
_SIM_050 = _vec(0.5, 0.8660254037844386)  # cosine 0.5


def _row(sim: float, day: datetime) -> dict:
    """A minimal pool row for the pure selector test."""
    return {
        "id": f"{sim}-{day.isoformat()}",
        "similarity": sim,
        "occurred_start": None,
        "mentioned_at": day,
        "occurred_end": None,
    }


# ---------------------------------------------------------------------------
# Pure selector: coverage round-robin
# ---------------------------------------------------------------------------


def test_coverage_round_robin_spreads_across_buckets():
    """One item per populated bucket is taken before any bucket gets a second."""
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 12, 31, tzinfo=UTC)
    jan, jul = datetime(2025, 1, 15, tzinfo=UTC), datetime(2025, 7, 15, tzinfo=UTC)
    # Two time-buckets; January is denser and slightly more similar.
    pool = [_row(1.0, jan), _row(0.99, jan), _row(0.98, jan), _row(0.95, jul)]

    selected = _select_with_temporal_coverage(pool, start, end, limit=2, n_buckets=8)

    # Coverage: the single July item beats the 2nd/3rd January items despite lower similarity.
    months = {r["mentioned_at"].month for r in selected}
    assert months == {1, 7}


def test_coverage_degenerate_dates_fall_back_to_similarity():
    """When all dates land in one bucket, selection is plain top-by-similarity."""
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 12, 31, tzinfo=UTC)
    same_day = datetime(2025, 1, 15, tzinfo=UTC)
    pool = [_row(0.4, same_day), _row(0.9, same_day), _row(0.7, same_day), _row(0.95, same_day)]

    selected = _select_with_temporal_coverage(pool, start, end, limit=2, n_buckets=8)

    assert [r["similarity"] for r in selected] == [0.95, 0.9]


@pytest.mark.asyncio
async def test_temporal_analysis_is_bounded_and_cancellation_safe(monkeypatch):
    """Saturation must fail open without releasing cancelled parser work."""
    analysis_started = threading.Event()
    queued_analysis_started = threading.Event()
    release_analysis = threading.Event()
    analysis_calls = 0

    def blocking_extract(*_args, **_kwargs):
        nonlocal analysis_calls
        analysis_calls += 1
        if analysis_calls == 1:
            analysis_started.set()
        elif analysis_calls == 2:
            queued_analysis_started.set()
        release_analysis.wait(timeout=2)
        return None

    @asynccontextmanager
    async def fake_acquire_with_retry(pool):
        yield object()

    async def fake_semantic_bm25_combined(*args, **kwargs):
        return {"world": retrieval_module.SemanticBm25Result(semantic=[], bm25=[], graph_seeds=None)}

    async def fake_temporal_combined(*args, **kwargs):
        return {"world": []}

    class FakeGraphRetriever:
        async def retrieve(self, **kwargs):
            return [], None

    monkeypatch.setattr(retrieval_module, "acquire_with_retry", fake_acquire_with_retry)
    monkeypatch.setattr(retrieval_module, "retrieve_semantic_bm25_combined", fake_semantic_bm25_combined)
    monkeypatch.setattr(retrieval_module, "retrieve_temporal_combined", fake_temporal_combined)
    monkeypatch.setattr(
        retrieval_module,
        "get_config",
        lambda: SimpleNamespace(
            graph_seed_min_similarity=0.3,
            temporal_semantic_min_similarity=0.24,
        ),
    )
    monkeypatch.setattr(
        "hindsight_api.engine.search.temporal_extraction.extract_temporal_constraint",
        blocking_extract,
    )

    class CountingExecutor:
        def __init__(self, executor):
            self.executor = executor
            self.calls = 0

        def submit(self, *args, **kwargs):
            self.calls += 1
            return self.executor.submit(*args, **kwargs)

    executor = retrieval_module.ThreadPoolExecutor(max_workers=1)
    counting_executor = CountingExecutor(executor)
    analysis_disabled = threading.Event()
    monkeypatch.setattr(retrieval_module, "_TEMPORAL_ANALYSIS_EXECUTOR", counting_executor)
    monkeypatch.setattr(retrieval_module, "_TEMPORAL_ANALYSIS_SLOTS", threading.BoundedSemaphore(2))
    monkeypatch.setattr(retrieval_module, "_TEMPORAL_ANALYSIS_DISABLED", analysis_disabled)
    monkeypatch.setattr(retrieval_module, "_TEMPORAL_ANALYSIS_TIMEOUT_SECONDS", 1.0)

    oversized_result = await asyncio.wait_for(
        retrieval_module.retrieve_all_fact_types_parallel(
            object(),
            query_text="x" * (query_analyzer_module._MAX_TEMPORAL_ANALYSIS_CHARS + 1),
            query_embedding_str=_QUERY,
            bank_id="test_temporal_oversized",
            fact_types=["world"],
            thinking_budget=10,
            graph_retriever=FakeGraphRetriever(),
        ),
        timeout=0.5,
    )
    assert oversized_result.results_by_fact_type["world"].temporal_constraint is None
    assert analysis_calls == 0
    assert counting_executor.calls == 0

    custom_calls = []

    def custom_extract(query, *_args, **_kwargs):
        custom_calls.append((query, threading.get_ident()))
        return None

    monkeypatch.setattr(
        "hindsight_api.engine.search.temporal_extraction.extract_temporal_constraint",
        custom_extract,
    )
    custom_query = "x" * (query_analyzer_module._MAX_TEMPORAL_ANALYSIS_CHARS + 1)
    custom_result = await asyncio.wait_for(
        retrieval_module.retrieve_all_fact_types_parallel(
            object(),
            query_text=custom_query,
            query_embedding_str=_QUERY,
            bank_id="test_custom_temporal_analyzer",
            fact_types=["world"],
            thinking_budget=10,
            query_analyzer=object(),
            graph_retriever=FakeGraphRetriever(),
        ),
        timeout=0.5,
    )
    assert custom_result.results_by_fact_type["world"].temporal_constraint is None
    assert custom_calls == [(custom_query, threading.get_ident())]
    assert counting_executor.calls == 0

    valid_start = datetime(2025, 1, 1, tzinfo=UTC)
    valid_end = datetime(2025, 1, 31, 23, 59, 59, tzinfo=UTC)
    monkeypatch.setattr(
        "hindsight_api.engine.search.temporal_extraction.extract_temporal_constraint",
        lambda *_args, **_kwargs: (valid_start, valid_end),
    )
    boundary_result = await asyncio.wait_for(
        retrieval_module.retrieve_all_fact_types_parallel(
            object(),
            query_text="x" * query_analyzer_module._MAX_TEMPORAL_ANALYSIS_CHARS,
            query_embedding_str=_QUERY,
            bank_id="test_temporal_boundary",
            fact_types=["world"],
            thinking_budget=10,
            graph_retriever=FakeGraphRetriever(),
        ),
        timeout=0.5,
    )
    assert boundary_result.results_by_fact_type["world"].temporal_constraint == (valid_start, valid_end)
    assert counting_executor.calls == 1

    monkeypatch.setattr(
        "hindsight_api.engine.search.temporal_extraction.extract_temporal_constraint",
        blocking_extract,
    )

    loop = asyncio.get_running_loop()
    started_at = loop.time()
    blocked_retrieval = asyncio.create_task(
        retrieval_module.retrieve_all_fact_types_parallel(
            object(),
            query_text="what happened?",
            query_embedding_str=_QUERY,
            bank_id="test_temporal_off_loop",
            fact_types=["world"],
            thinking_budget=10,
            graph_retriever=FakeGraphRetriever(),
        )
    )

    parser_started = await asyncio.wait_for(asyncio.to_thread(analysis_started.wait, 1), timeout=1.5)
    event_loop_delay = loop.time() - started_at
    assert parser_started
    assert event_loop_delay < 0.5
    assert await asyncio.wait_for(asyncio.to_thread(lambda: "responsive"), timeout=0.5) == "responsive"

    blocked_retrieval.cancel()
    with pytest.raises(asyncio.CancelledError):
        await blocked_retrieval

    monkeypatch.setattr(retrieval_module, "_TEMPORAL_ANALYSIS_TIMEOUT_SECONDS", 0.05)
    saturated_result = await asyncio.wait_for(
        retrieval_module.retrieve_all_fact_types_parallel(
            object(),
            query_text="another query",
            query_embedding_str=_QUERY,
            bank_id="test_temporal_saturated",
            fact_types=["world"],
            thinking_budget=10,
            graph_retriever=FakeGraphRetriever(),
        ),
        timeout=0.5,
    )
    assert saturated_result.results_by_fact_type["world"].temporal_constraint is None
    assert analysis_calls == 1
    assert counting_executor.calls == 3

    cancelled_result = await asyncio.wait_for(
        retrieval_module.retrieve_all_fact_types_parallel(
            object(),
            query_text="query after cancellation",
            query_embedding_str=_QUERY,
            bank_id="test_temporal_cancelled",
            fact_types=["world"],
            thinking_budget=10,
            graph_retriever=FakeGraphRetriever(),
        ),
        timeout=0.5,
    )
    assert cancelled_result.results_by_fact_type["world"].temporal_constraint is None
    assert analysis_calls == 1
    assert counting_executor.calls == 3

    release_analysis.set()
    queued_parser_started = await asyncio.wait_for(
        asyncio.to_thread(queued_analysis_started.wait, 1),
        timeout=1.5,
    )
    assert queued_parser_started
    await asyncio.wait_for(asyncio.wrap_future(executor.submit(lambda: None)), timeout=1.0)

    recovered_result = await asyncio.wait_for(
        retrieval_module.retrieve_all_fact_types_parallel(
            object(),
            query_text="query after recovery",
            query_embedding_str=_QUERY,
            bank_id="test_temporal_recovered",
            fact_types=["world"],
            thinking_budget=10,
            graph_retriever=FakeGraphRetriever(),
        ),
        timeout=0.5,
    )
    assert recovered_result.results_by_fact_type["world"].temporal_constraint is None
    assert analysis_calls == 3
    assert counting_executor.calls == 4

    def failed_extract(*_args, **_kwargs):
        raise ValueError("parser failure")

    monkeypatch.setattr(
        "hindsight_api.engine.search.temporal_extraction.extract_temporal_constraint",
        failed_extract,
    )
    failed_result = await asyncio.wait_for(
        retrieval_module.retrieve_all_fact_types_parallel(
            object(),
            query_text="query that triggers a parser failure",
            query_embedding_str=_QUERY,
            bank_id="test_temporal_failure",
            fact_types=["world"],
            thinking_budget=10,
            graph_retriever=FakeGraphRetriever(),
        ),
        timeout=0.5,
    )
    assert failed_result.results_by_fact_type["world"].temporal_constraint is None
    assert counting_executor.calls == 5
    executor.shutdown(wait=True)

    class RejectingExecutor:
        calls = 0

        def submit(self, *_args, **_kwargs):
            self.calls += 1
            raise RuntimeError("executor unavailable")

    rejecting_executor = RejectingExecutor()
    monkeypatch.setattr(retrieval_module, "_TEMPORAL_ANALYSIS_EXECUTOR", rejecting_executor)
    monkeypatch.setattr(retrieval_module, "_TEMPORAL_ANALYSIS_SLOTS", threading.BoundedSemaphore(2))

    for bank_id in ("test_temporal_submit_failure", "test_temporal_circuit_open"):
        submit_failure_result = await asyncio.wait_for(
            retrieval_module.retrieve_all_fact_types_parallel(
                object(),
                query_text="query after executor failure",
                query_embedding_str=_QUERY,
                bank_id=bank_id,
                fact_types=["world"],
                thinking_budget=10,
                graph_retriever=FakeGraphRetriever(),
            ),
            timeout=0.5,
        )
        assert submit_failure_result.results_by_fact_type["world"].temporal_constraint is None

    assert analysis_disabled.is_set()
    assert rejecting_executor.calls == 1


# ---------------------------------------------------------------------------
# DB-backed: similarity gating + window filter + coverage
# ---------------------------------------------------------------------------


async def _insert_unit(conn, bank_id: str, text: str, fact_type: str, when: datetime, embedding: str) -> str:
    table = fq_table("memory_units")
    row = await conn.fetchrow(
        f"""
        INSERT INTO {table} (bank_id, text, fact_type, embedding, event_date, mentioned_at)
        VALUES ($1, $2, $3, $4::vector, $5, $5)
        RETURNING id
        """,
        bank_id,
        text,
        fact_type,
        embedding,
        when,
    )
    return str(row["id"])


@pytest.mark.asyncio
async def test_temporal_recall_selects_by_similarity_not_recency(memory):
    """The most-similar in-window unit is returned even when it is the OLDEST — the inverse
    of the old recency-ranked behavior, where it would have been dropped."""
    bank_id = "test_temporal_similarity_gate"
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 2, 1, tzinfo=UTC)

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(f"DELETE FROM {fq_table('memory_units')} WHERE bank_id = $1", bank_id)

        # Oldest in-window unit, perfect similarity.
        oldest_relevant = await _insert_unit(
            conn, bank_id, "oldest relevant", "world", datetime(2025, 1, 2, tzinfo=UTC), _SIM_100
        )
        # Newer, less-similar units.
        for i in range(8):
            await _insert_unit(
                conn,
                bank_id,
                f"recent less-relevant {i}",
                "world",
                datetime(2025, 1, 20, tzinfo=UTC) + timedelta(hours=i),
                _SIM_050,
            )
        # Out-of-window, perfect similarity → must be excluded by the window.
        before = await _insert_unit(conn, bank_id, "before", "world", datetime(2024, 12, 1, tzinfo=UTC), _SIM_100)
        after = await _insert_unit(conn, bank_id, "after", "world", datetime(2025, 3, 1, tzinfo=UTC), _SIM_100)

        results = await retrieve_temporal_combined(conn, _QUERY, bank_id, ["world"], start, end, budget=100)

    by_id = {r.id: r for r in results.get("world", [])}
    # Similarity wins over recency: the oldest, most-relevant unit is selected.
    assert oldest_relevant in by_id
    # Window filter still excludes out-of-window units, even at perfect similarity.
    assert before not in by_id
    assert after not in by_id


@pytest.mark.asyncio
async def test_temporal_recall_covers_window_range(memory):
    """Entry points span the window: relevant units in distinct time-slices are all
    represented, instead of clustering in the densest slice."""
    bank_id = "test_temporal_coverage"
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 12, 31, tzinfo=UTC)

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(f"DELETE FROM {fq_table('memory_units')} WHERE bank_id = $1", bank_id)

        # A dense January cluster at the highest similarity.
        for i in range(20):
            await _insert_unit(
                conn, bank_id, f"jan {i}", "world", datetime(2025, 1, 10, tzinfo=UTC) + timedelta(hours=i), _SIM_100
            )
        # One slightly-less-similar unit in three other quarters.
        apr = await _insert_unit(conn, bank_id, "apr", "world", datetime(2025, 4, 15, tzinfo=UTC), _SIM_090)
        jul = await _insert_unit(conn, bank_id, "jul", "world", datetime(2025, 7, 15, tzinfo=UTC), _SIM_090)
        octo = await _insert_unit(conn, bank_id, "oct", "world", datetime(2025, 10, 15, tzinfo=UTC), _SIM_090)

        results = await retrieve_temporal_combined(conn, _QUERY, bank_id, ["world"], start, end, budget=100)

    ids = {r.id for r in results.get("world", [])}
    # Without coverage, the top-10 by similarity would be 10 January units and the Apr/Jul/Oct
    # units (lower similarity) would be crowded out. Coverage surfaces every populated slice.
    assert {apr, jul, octo} <= ids
    selected_months = {r.mentioned_at.month for r in results["world"] if r.mentioned_at}
    assert len(selected_months) >= 3


@pytest.mark.asyncio
async def test_min_semantic_does_not_tighten_temporal_seed_threshold(monkeypatch):
    """min_scores.semantic filters the semantic arm, not temporal entry-point seeds."""
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 2, 1, tzinfo=UTC)
    temporal_thresholds: list[float] = []
    graph_call_kwargs: list[set[str]] = []

    @asynccontextmanager
    async def fake_acquire_with_retry(pool):
        yield object()

    async def fake_semantic_bm25_combined(*args, **kwargs):
        return {"world": retrieval_module.SemanticBm25Result(semantic=[], bm25=[], graph_seeds=None)}

    async def fake_temporal_combined(*args, **kwargs):
        temporal_thresholds.append(kwargs["semantic_threshold"])
        return {"world": []}

    class FakeGraphRetriever:
        async def retrieve(self, **kwargs):
            graph_call_kwargs.append(set(kwargs))
            return [], None

    monkeypatch.setattr(retrieval_module, "acquire_with_retry", fake_acquire_with_retry)
    monkeypatch.setattr(retrieval_module, "retrieve_semantic_bm25_combined", fake_semantic_bm25_combined)
    monkeypatch.setattr(retrieval_module, "retrieve_temporal_combined", fake_temporal_combined)
    monkeypatch.setattr(
        retrieval_module,
        "get_config",
        lambda: SimpleNamespace(
            graph_seed_min_similarity=0.3,
            temporal_semantic_min_similarity=0.24,
        ),
    )
    monkeypatch.setattr(
        "hindsight_api.engine.search.temporal_extraction.extract_temporal_constraint",
        lambda *args, **kwargs: (start, end),
    )

    await retrieval_module.retrieve_all_fact_types_parallel(
        object(),
        query_text="what happened in January?",
        query_embedding_str=_QUERY,
        bank_id="test_temporal_min_semantic_decoupling",
        fact_types=["world"],
        thinking_budget=10,
        graph_retriever=FakeGraphRetriever(),
        min_semantic=0.5,
    )

    assert temporal_thresholds == [0.24]
    assert graph_call_kwargs == [
        {
            "pool",
            "query_embedding_str",
            "bank_id",
            "fact_type",
            "budget",
            "query_text",
            "preselected_semantic_seeds",
            "tags",
            "tags_match",
            "tag_groups",
            "created_after",
            "created_before",
        }
    ]
