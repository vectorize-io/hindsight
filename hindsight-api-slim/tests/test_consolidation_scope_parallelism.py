"""End-to-end tests for parallel consolidation under each ``observation_scopes`` mode.

These tests pin two invariants the per-scope lock design has to enforce:

1. **Scope correctness**: regardless of ``consolidation_llm_parallelism``, each
   ``observation_scopes`` mode produces observations whose tags match the
   scope spec — combined writes to the memory's exact tag set, per_tag writes
   one per tag, all_combinations writes one per nonempty subset, and an
   explicit list writes one per declared scope. This is what the recall path
   was given, and what the write path persisted.
2. **No concurrent in-flight LLM call on a shared scope**: the per-scope lock
   guarantee. We wrap ``_find_related_observations`` to record entry/exit
   per scope tag set and assert max concurrency == 1 per scope, even under
   parallelism > 1 with deliberately overlapping write scopes.

All tests use the mock-LLM ``memory`` fixture (pool_max_size=5; well above the
parallelism levels used here).
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections import defaultdict
from unittest.mock import MagicMock, patch

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.consolidation.consolidator import (
    _ConsolidationBatchResponse,
    _CreateAction,
    run_consolidation_job,
)
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.providers.mock_llm import MockLLM


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def enable_observations():
    config = _get_raw_config()
    original = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original


def _override_config(memory: MemoryEngine, **overrides):
    """Patch the resolver to return a fake config for this test only.

    Returns a context-manager-friendly patcher. Usage::

        with _override_config(memory, consolidation_llm_parallelism=4, ...):
            await run_consolidation_job(...)
    """
    raw = _get_raw_config()
    fake = type(raw)(
        **{
            **{f: getattr(raw, f) for f in raw.__dataclass_fields__},
            **overrides,
        }
    )
    return patch.object(memory._config_resolver, "resolve_full_config", return_value=fake)


async def _insert_memory(
    conn,
    bank_id: str,
    text: str,
    tags: list[str],
    observation_scopes,
) -> uuid.UUID:
    """Insert a single experience memory with an explicit observation_scopes column.

    The JSONB column is written as a JSON-encoded string to match how the API
    write path stores per-memory scope overrides.
    """
    mem_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO memory_units (id, bank_id, text, fact_type, tags, observation_scopes, created_at)
        VALUES ($1, $2, $3, 'experience', $4, $5::jsonb, now())
        """,
        mem_id,
        bank_id,
        text,
        tags,
        json.dumps(observation_scopes) if observation_scopes is not None else None,
    )
    return mem_id


def _mock_llm_one_obs_per_fact():
    """A MockLLM wrapper that emits one CREATE per fact in the prompt.

    Returned as (config_wrapper, mock_llm). The config wrapper short-circuits
    ``.with_config(...)`` to return the underlying MockLLM unchanged so we
    don't have to mock the whole per-bank config plumbing.
    """
    mock_llm = MockLLM(provider="mock", api_key="", base_url="", model="mock-model")

    def callback(messages, scope):
        if scope != "consolidation":
            return _ConsolidationBatchResponse()
        prompt = messages[0]["content"] if messages else ""
        fact_ids = re.findall(r"\[([0-9a-f-]{36})\]", prompt)
        creates = [
            _CreateAction(text=f"Observation about fact {fid[:8]}", source_fact_ids=[fid])
            for fid in fact_ids
        ]
        return _ConsolidationBatchResponse(creates=creates)

    mock_llm.set_response_callback(callback)
    wrapper = MagicMock()
    wrapper.with_config.return_value = mock_llm
    return wrapper, mock_llm


async def _fetch_observation_tag_sets(memory: MemoryEngine, bank_id: str) -> list[frozenset[str]]:
    """Return the tag set (as a frozenset) of every observation in the bank."""
    async with memory._pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT tags FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation'",
            bank_id,
        )
    return [frozenset(r["tags"] or []) for r in rows]


# ---------------------------------------------------------------------------
# Scope-correctness tests: observations land at the right scopes under
# parallelism > 1, for each observation_scopes mode.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_combined_mode_parallel_writes_to_memory_tag_set(memory: MemoryEngine, request_context):
    """combined (default) → each memory yields exactly one observation tagged
    with the memory's full tag set. With three disjoint tag sets, dispatch
    runs all three groups concurrently and each writes its own scope."""
    bank_id = f"test-combined-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice likes tea", ["user:alice"], None)
            await _insert_memory(conn, bank_id, "Bob bikes daily", ["user:bob"], None)
            await _insert_memory(conn, bank_id, "Carol reads books", ["user:carol"], None)

        wrapper, _ = _mock_llm_one_obs_per_fact()
        original_llm = memory._consolidation_llm_config
        memory._consolidation_llm_config = wrapper
        try:
            with (
                _override_config(memory, consolidation_llm_parallelism=3, consolidation_llm_batch_size=1),
                patch.object(memory, "submit_async_consolidation"),
            ):
                result = await run_consolidation_job(
                    memory_engine=memory, bank_id=bank_id, request_context=request_context
                )
        finally:
            memory._consolidation_llm_config = original_llm

        assert result["status"] == "completed"
        tag_sets = _ag_sorted(await _fetch_observation_tag_sets(memory, bank_id))
        assert tag_sets == _ag_sorted([
            frozenset({"user:alice"}),
            frozenset({"user:bob"}),
            frozenset({"user:carol"}),
        ])
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_per_tag_mode_parallel_writes_one_observation_per_tag(memory: MemoryEngine, request_context):
    """per_tag with tags [a, b] → two observations, tagged [a] and [b] respectively.

    Two memories with overlapping single-tag scopes ensure the parallel
    dispatcher must serialise on the shared scope and yields the same
    observation set as the sequential path."""
    bank_id = f"test-pertag-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            # M1: per_tag on [alice] -> writes obs at [alice]
            await _insert_memory(conn, bank_id, "Alice likes tea", ["alice"], "per_tag")
            # M2: per_tag on [alice, session] -> writes obs at [alice] AND [session]
            await _insert_memory(conn, bank_id, "Alice session detail", ["alice", "session"], "per_tag")

        wrapper, _ = _mock_llm_one_obs_per_fact()
        original_llm = memory._consolidation_llm_config
        memory._consolidation_llm_config = wrapper
        try:
            with (
                _override_config(memory, consolidation_llm_parallelism=4, consolidation_llm_batch_size=1),
                patch.object(memory, "submit_async_consolidation"),
            ):
                result = await run_consolidation_job(
                    memory_engine=memory, bank_id=bank_id, request_context=request_context
                )
        finally:
            memory._consolidation_llm_config = original_llm

        assert result["status"] == "completed"

        tag_sets = await _fetch_observation_tag_sets(memory, bank_id)
        # M1 writes to [alice]; M2 writes to [alice] and [session]. The mock LLM
        # creates one observation per fact per pass, so we expect:
        #   - one [alice] obs from M1
        #   - one [alice] obs from M2 (per_tag pass on alice)
        #   - one [session] obs from M2 (per_tag pass on session)
        alice_count = sum(1 for t in tag_sets if t == frozenset({"alice"}))
        session_count = sum(1 for t in tag_sets if t == frozenset({"session"}))
        assert alice_count == 2, f"expected 2 [alice] observations, got tag_sets={tag_sets}"
        assert session_count == 1, f"expected 1 [session] observation, got tag_sets={tag_sets}"
        # No observation should leak a tag set other than the per_tag scopes.
        assert all(t in (frozenset({"alice"}), frozenset({"session"})) for t in tag_sets), tag_sets
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_all_combinations_mode_parallel_writes_every_subset(memory: MemoryEngine, request_context):
    """all_combinations with tags [a, b] → three observations at [a], [b], [a, b]."""
    bank_id = f"test-allcombo-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice session detail", ["alice", "session"], "all_combinations")

        wrapper, _ = _mock_llm_one_obs_per_fact()
        original_llm = memory._consolidation_llm_config
        memory._consolidation_llm_config = wrapper
        try:
            with (
                _override_config(memory, consolidation_llm_parallelism=4, consolidation_llm_batch_size=1),
                patch.object(memory, "submit_async_consolidation"),
            ):
                result = await run_consolidation_job(
                    memory_engine=memory, bank_id=bank_id, request_context=request_context
                )
        finally:
            memory._consolidation_llm_config = original_llm

        assert result["status"] == "completed"
        tag_sets = set(await _fetch_observation_tag_sets(memory, bank_id))
        assert tag_sets == {
            frozenset({"alice"}),
            frozenset({"session"}),
            frozenset({"alice", "session"}),
        }
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_explicit_scope_list_parallel_writes_declared_scopes(memory: MemoryEngine, request_context):
    """Explicit list[list[str]] → observations land at exactly those scopes,
    regardless of the memory's own tag set."""
    bank_id = f"test-explicit-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(
                conn,
                bank_id,
                "Memory with explicit scopes",
                ["tag_ignored"],
                [["scope_a"], ["scope_b", "scope_c"]],
            )

        wrapper, _ = _mock_llm_one_obs_per_fact()
        original_llm = memory._consolidation_llm_config
        memory._consolidation_llm_config = wrapper
        try:
            with (
                _override_config(memory, consolidation_llm_parallelism=4, consolidation_llm_batch_size=1),
                patch.object(memory, "submit_async_consolidation"),
            ):
                result = await run_consolidation_job(
                    memory_engine=memory, bank_id=bank_id, request_context=request_context
                )
        finally:
            memory._consolidation_llm_config = original_llm

        assert result["status"] == "completed"
        tag_sets = set(await _fetch_observation_tag_sets(memory, bank_id))
        assert tag_sets == {frozenset({"scope_a"}), frozenset({"scope_b", "scope_c"})}
        # And NOT the memory's own tag.
        assert frozenset({"tag_ignored"}) not in tag_sets
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Lock-serialisation test: groups whose write scopes share a scope must not
# have concurrent in-flight recalls / writes on the shared scope.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_overlapping_scopes_serialise_under_parallelism(memory: MemoryEngine, request_context):
    """Two groups whose write-scope sets intersect on scope S must not have
    overlapping in-flight LLM-recall windows for S.

    Setup: M1 tagged [a] (per_tag → writes [a]) and M2 tagged [a, b] (per_tag
    → writes [a] and [b]). M1's lock set = {[a]}; M2's lock set = {[a], [b]}.
    They share scope [a], so under parallelism>1 they must serialise on [a].
    We wrap ``_find_related_observations`` to record per-scope entry/exit and
    assert max concurrency per scope == 1.
    """
    bank_id = f"test-locks-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

    from hindsight_api.engine.consolidation import consolidator as consolidator_mod

    in_flight: dict[frozenset[str], int] = defaultdict(int)
    max_concurrent: dict[frozenset[str], int] = defaultdict(int)
    tracker_lock = asyncio.Lock()
    orig_find = consolidator_mod._find_related_observations

    async def tracked_find(*, memory_engine, bank_id, query, request_context, tags=None):
        scope = frozenset(tags or [])
        async with tracker_lock:
            in_flight[scope] += 1
            if in_flight[scope] > max_concurrent[scope]:
                max_concurrent[scope] = in_flight[scope]
        try:
            # Sleep to widen the window for races, so a missing lock would be visible.
            await asyncio.sleep(0.05)
            return await orig_find(
                memory_engine=memory_engine,
                bank_id=bank_id,
                query=query,
                request_context=request_context,
                tags=tags,
            )
        finally:
            async with tracker_lock:
                in_flight[scope] -= 1

    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Memory one alice only", ["a"], "per_tag")
            await _insert_memory(conn, bank_id, "Memory two alice and beta", ["a", "b"], "per_tag")

        wrapper, _ = _mock_llm_one_obs_per_fact()
        original_llm = memory._consolidation_llm_config
        memory._consolidation_llm_config = wrapper
        try:
            with (
                _override_config(memory, consolidation_llm_parallelism=4, consolidation_llm_batch_size=1),
                patch.object(memory, "submit_async_consolidation"),
                patch.object(consolidator_mod, "_find_related_observations", tracked_find),
            ):
                result = await run_consolidation_job(
                    memory_engine=memory, bank_id=bank_id, request_context=request_context
                )
        finally:
            memory._consolidation_llm_config = original_llm

        assert result["status"] == "completed"
        assert max_concurrent, "tracker saw no recalls — test did not exercise the dispatch path"

        # The whole point: lock invariant per scope.
        for scope, peak in max_concurrent.items():
            assert peak <= 1, (
                f"scope {set(scope) or '<untagged>'} had {peak} concurrent in-flight recalls; "
                "lock invariant violated"
            )
        # Sanity: we DID see recalls for the shared scope, so the test wasn't trivial.
        assert frozenset({"a"}) in max_concurrent
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Disjoint scopes actually run concurrently (the throughput justification for
# the whole feature). Without this, the "lock-on-everything" implementation
# would still pass the safety tests.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_batch_log_line_attributes_only_own_work(memory: MemoryEngine, request_context, caplog):
    """Per-batch log timings / llm_calls / tokens / processed must reflect only
    that batch's own work — not totals leaking in from other in-flight batches
    under parallelism. Pins the per-batch perf isolation that ``batch_perf``
    + ``perf.merge_from`` provide.

    Setup: 3 disjoint memories under combined mode at parallelism=3 so all
    three batches run concurrently. The mock LLM is deterministic (1 obs per
    fact, 1 LLM call per batch). After the job we parse each emitted log line
    and assert per-batch attributes against that single batch's known work,
    plus check the cumulative ``processed=N/total`` field is monotonic.
    """
    import logging

    bank_id = f"test-perbatch-log-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice fact", ["alice"], None)
            await _insert_memory(conn, bank_id, "Bob fact", ["bob"], None)
            await _insert_memory(conn, bank_id, "Carol fact", ["carol"], None)

        wrapper, _ = _mock_llm_one_obs_per_fact()
        original_llm = memory._consolidation_llm_config
        memory._consolidation_llm_config = wrapper
        try:
            with (
                _override_config(memory, consolidation_llm_parallelism=3, consolidation_llm_batch_size=1),
                patch.object(memory, "submit_async_consolidation"),
                caplog.at_level(logging.INFO, logger="hindsight_api.engine.consolidation.consolidator"),
            ):
                await run_consolidation_job(
                    memory_engine=memory, bank_id=bank_id, request_context=request_context
                )
        finally:
            memory._consolidation_llm_config = original_llm

        # Parse the per-batch log lines.
        per_batch = [r.message for r in caplog.records if "llm_batch #" in r.message]
        assert len(per_batch) == 3, f"expected 3 per-batch log lines, got {len(per_batch)}:\n{per_batch}"

        # Every batch processed exactly one memory (llm_batch_size=1) and made
        # exactly one LLM call. If a stale-snapshot bug let counters from
        # other batches leak in, ``llm calls`` would be > 1 for some batches.
        processed_values: list[int] = []
        for line in per_batch:
            m_calls = re.search(r"(\d+) llm calls", line)
            assert m_calls and int(m_calls.group(1)) == 1, f"expected 1 llm call per batch, got: {line}"
            m_mems = re.search(r"\((\d+) memories,", line)
            assert m_mems and int(m_mems.group(1)) == 1, f"expected 1 memory per batch, got: {line}"
            # Per-batch created count must be 1 (mock LLM creates one obs per fact).
            m_created = re.search(r"created=(\d+)", line)
            assert m_created and int(m_created.group(1)) == 1, f"expected created=1, got: {line}"
            # processed=N/3 — cumulative; collect for monotonicity check.
            m_proc = re.search(r"processed=(\d+)/3", line)
            assert m_proc, f"expected processed=N/3 cumulative indicator, got: {line}"
            processed_values.append(int(m_proc.group(1)))

        # Cumulative counter must be monotonically increasing and end at 3.
        assert processed_values == sorted(processed_values), (
            f"processed counter must be monotonic, got {processed_values}"
        )
        assert max(processed_values) == 3, (
            f"final cumulative processed should be 3, got {max(processed_values)}"
        )
        assert set(processed_values) == {1, 2, 3}, (
            f"each batch should bump the counter by exactly 1, got {processed_values}"
        )

        # Per-batch llm timing must be > 0 (every batch made an LLM call) and
        # finite (not bleeding from concurrent batches into an inflated delta).
        for line in per_batch:
            m_llm_time = re.search(r"llm=(\d+\.\d+)s", line)
            assert m_llm_time, f"expected llm=Xs timing, got: {line}"
            # Sanity: a single mock-LLM call is fast — under a second easily.
            # If snapshot leaked, this would catch concurrent batches' LLM time too.
            assert float(m_llm_time.group(1)) < 5.0, (
                f"llm timing implausibly large for a single mock-LLM call: {line}"
            )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_disjoint_scopes_run_concurrently(memory: MemoryEngine, request_context):
    """When write-scope sets are pairwise disjoint, the dispatcher must let
    groups run in parallel — we should observe simultaneous in-flight recalls
    on *different* scopes."""
    bank_id = f"test-disjoint-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

    from hindsight_api.engine.consolidation import consolidator as consolidator_mod

    distinct_concurrent_scopes_seen = 0
    in_flight_scopes: set[frozenset[str]] = set()
    sample_lock = asyncio.Lock()
    orig_find = consolidator_mod._find_related_observations

    async def tracked_find(*, memory_engine, bank_id, query, request_context, tags=None):
        nonlocal distinct_concurrent_scopes_seen
        scope = frozenset(tags or [])
        async with sample_lock:
            in_flight_scopes.add(scope)
            if len(in_flight_scopes) > distinct_concurrent_scopes_seen:
                distinct_concurrent_scopes_seen = len(in_flight_scopes)
        try:
            await asyncio.sleep(0.1)  # widen the window so concurrency is observable
            return await orig_find(
                memory_engine=memory_engine,
                bank_id=bank_id,
                query=query,
                request_context=request_context,
                tags=tags,
            )
        finally:
            async with sample_lock:
                in_flight_scopes.discard(scope)

    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice memory", ["alice"], None)
            await _insert_memory(conn, bank_id, "Bob memory", ["bob"], None)
            await _insert_memory(conn, bank_id, "Carol memory", ["carol"], None)

        wrapper, _ = _mock_llm_one_obs_per_fact()
        original_llm = memory._consolidation_llm_config
        memory._consolidation_llm_config = wrapper
        try:
            with (
                _override_config(memory, consolidation_llm_parallelism=3, consolidation_llm_batch_size=1),
                patch.object(memory, "submit_async_consolidation"),
                patch.object(consolidator_mod, "_find_related_observations", tracked_find),
            ):
                result = await run_consolidation_job(
                    memory_engine=memory, bank_id=bank_id, request_context=request_context
                )
        finally:
            memory._consolidation_llm_config = original_llm

        assert result["status"] == "completed"
        # Three disjoint scopes + parallelism=3 → at some moment we should
        # see at least 2 distinct in-flight scopes.
        assert distinct_concurrent_scopes_seen >= 2, (
            f"expected concurrent in-flight recalls across disjoint scopes, "
            f"max observed = {distinct_concurrent_scopes_seen}"
        )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Internal helper — sort lists of frozensets by their key for stable equality
# ---------------------------------------------------------------------------


def _ag_sorted(scopes):
    return sorted(scopes, key=lambda s: tuple(sorted(s)))
