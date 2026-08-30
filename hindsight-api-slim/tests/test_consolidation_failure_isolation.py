"""Failure isolation for the consolidation dispatcher.

A DB failure inside one tag group's recall aborts the consolidation operation,
which the worker retries with a 5s base backoff. Plain ``asyncio.gather`` does
not cancel the sibling groups when it re-raises, so before this was fixed those
groups kept running detached — still calling the LLM, still stamping
``mark_consolidated`` and committing write-groups — while the retry was already
under way. The per-scope ``scope_locks`` are local to one dispatch, so nothing
serialised an orphan against the retry and two consolidators could write the
same observation scope concurrently.

These tests pin:

1. ``_gather_or_cancel`` cancels and awaits its siblings, and re-raises the
   ORIGINAL exception (not an ``ExceptionGroup``) so the worker's
   ``_is_non_retryable_task_error`` classification still works.
2. End to end: a failing recall in one tag group cancels the other groups
   before they write, and the job does not wait for them.
3. A batch that fails between ``mint_txn`` and the witness commit discards its
   write-group instead of leaving it pending for the recovery sweep.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.consolidation import consolidator as consolidator_module
from hindsight_api.engine.consolidation.consolidator import (
    _BatchLLMResult,
    _ConsolidationBatchResponse,
    _CreateAction,
    _DeleteAction,
    _gather_or_cancel,
    _process_memory_batch,
    run_consolidation_job,
)
from hindsight_api.engine.memory_engine import MemoryEngine, _is_non_retryable_task_error
from hindsight_api.engine.providers.mock_llm import MockLLM
from hindsight_api.engine.response_models import MemoryFact, RecallResult


class _RecallTimeout(Exception):
    """Stands in for a database command timeout raised inside a recall."""


# ---------------------------------------------------------------------------
# _gather_or_cancel unit tests (no database)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gather_or_cancel_returns_results_in_order():
    async def one(value: int) -> int:
        await asyncio.sleep(0)
        return value

    assert await _gather_or_cancel([one(1), one(2), one(3)]) == [1, 2, 3]


@pytest.mark.asyncio
async def test_gather_or_cancel_cancels_siblings_before_returning():
    """The sibling must be cancelled AND awaited before the exception surfaces.

    ``sibling_finished`` would be True under plain ``asyncio.gather``, which
    leaves the sibling running past the raise.
    """
    started = asyncio.Event()
    sibling_finished = False
    sibling_cancelled = False

    async def sibling():
        nonlocal sibling_finished, sibling_cancelled
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            sibling_cancelled = True
            raise
        sibling_finished = True

    async def boom():
        await started.wait()
        raise _RecallTimeout("simulated DB command timeout")

    with pytest.raises(_RecallTimeout):
        await _gather_or_cancel([boom(), sibling()])

    assert sibling_cancelled is True
    assert sibling_finished is False


@pytest.mark.asyncio
async def test_gather_or_cancel_reraises_original_exception_unwrapped():
    """An ExceptionGroup here (i.e. asyncio.TaskGroup) would break retry
    classification: the worker isinstance-checks the raised exception, so a
    wrapped integrity violation would be retried forever instead of dropped."""

    async def boom():
        raise asyncpg.exceptions.UniqueViolationError("duplicate key")

    async def idle():
        await asyncio.sleep(30)

    with pytest.raises(asyncpg.exceptions.UniqueViolationError) as exc_info:
        await _gather_or_cancel([boom(), idle()])

    assert not isinstance(exc_info.value, BaseExceptionGroup)
    assert _is_non_retryable_task_error(exc_info.value) is True


@pytest.mark.asyncio
async def test_create_failure_does_not_execute_requested_delete():
    """A replacement CREATE must succeed before its destructive DELETE runs."""
    source_id = uuid.uuid4()
    observation_id = uuid.uuid4()
    recall = RecallResult(
        results=[MemoryFact(id=str(observation_id), text="Old durable observation", fact_type="observation")]
    )
    llm_result = _BatchLLMResult(
        creates=[_CreateAction(text="Replacement observation", source_fact_ids=[str(source_id)])],
        deletes=[_DeleteAction(observation_id=str(observation_id))],
    )
    config = SimpleNamespace(
        consolidation_dedup_threshold=1.0,
        max_observations_per_scope=-1,
        observation_scope_limits=[],
    )
    create_error = RuntimeError("simulated embedding failure")

    with (
        patch.object(consolidator_module, "_find_related_observations", AsyncMock(return_value=recall)),
        patch.object(consolidator_module, "_consolidate_batch_with_llm", AsyncMock(return_value=llm_result)),
        patch.object(consolidator_module, "_execute_create_action", AsyncMock(side_effect=create_error)),
        patch.object(consolidator_module, "_execute_delete_action", AsyncMock()) as execute_delete,
    ):
        with pytest.raises(RuntimeError, match="simulated embedding failure"):
            await _process_memory_batch(
                pool=MagicMock(),
                memory_engine=MagicMock(),
                llm_config=MagicMock(),
                bank_id="bank",
                memories=[{"id": source_id, "text": "New evidence", "tags": []}],
                request_context=MagicMock(),
                config=config,
            )

    execute_delete.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_delete_failure_rolls_back_the_entire_delete_set(memory: MemoryEngine, request_context):
    """The final delete transaction must not commit only a prefix."""
    bank_id = f"test-delete-rollback-{uuid.uuid4().hex[:8]}"
    source_id = uuid.uuid4()
    first_observation_id = uuid.uuid4()
    second_observation_id = uuid.uuid4()
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

    try:
        # Seed history directly because the public API only creates history through an
        # observation update; this test needs a pre-existing snapshot before deletion.
        async with memory._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO memory_units (id, bank_id, text, fact_type, tags, observation_scopes, created_at)
                VALUES ($1, $2, $3, 'observation', $4, $5::jsonb, now())
                """,
                [
                    (first_observation_id, bank_id, "First durable observation", [], json.dumps(None)),
                    (second_observation_id, bank_id, "Second durable observation", [], json.dumps(None)),
                ],
            )
            await conn.executemany(
                """
                INSERT INTO observation_history (observation_id, bank_id, content, changed_at)
                VALUES ($1, $2, $3::jsonb, now())
                """,
                [
                    (
                        first_observation_id,
                        bank_id,
                        json.dumps({"previous_text": "First observation before update"}),
                    ),
                    (
                        second_observation_id,
                        bank_id,
                        json.dumps({"previous_text": "Second observation before update"}),
                    ),
                ],
            )

        recall = RecallResult(
            results=[
                MemoryFact(id=str(first_observation_id), text="First durable observation", fact_type="observation"),
                MemoryFact(id=str(second_observation_id), text="Second durable observation", fact_type="observation"),
            ]
        )
        llm_result = _BatchLLMResult(
            deletes=[
                _DeleteAction(observation_id=str(first_observation_id)),
                _DeleteAction(observation_id=str(second_observation_id)),
            ]
        )
        config = SimpleNamespace(
            consolidation_dedup_threshold=1.0,
            max_observations_per_scope=-1,
            observation_scope_limits=[],
        )
        original_execute_delete = consolidator_module._execute_delete_action
        successful_delete_count = 0

        async def fail_on_second_delete(*, conn: Any, bank_id: str, observation_id: str, txn: Any = None) -> None:
            nonlocal successful_delete_count
            if successful_delete_count == 1:
                raise RuntimeError("simulated second delete failure")
            await original_execute_delete(
                conn=conn,
                bank_id=bank_id,
                observation_id=observation_id,
                txn=txn,
            )
            # The engine API uses another pooled connection and cannot observe this
            # transaction's uncommitted state. Verify on the supplied connection that
            # the first delete really happened before the second delete raises.
            assert not await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM memory_units WHERE id = $1 AND bank_id = $2)",
                uuid.UUID(observation_id),
                bank_id,
            )
            assert not await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM observation_history WHERE observation_id = $1 AND bank_id = $2)",
                uuid.UUID(observation_id),
                bank_id,
            )
            successful_delete_count += 1

        with (
            patch.object(consolidator_module, "_find_related_observations", AsyncMock(return_value=recall)),
            patch.object(consolidator_module, "_consolidate_batch_with_llm", AsyncMock(return_value=llm_result)),
            patch.object(consolidator_module, "_execute_delete_action", fail_on_second_delete),
        ):
            with pytest.raises(RuntimeError, match="simulated second delete failure"):
                await _process_memory_batch(
                    pool=memory._pool,
                    memory_engine=memory,
                    llm_config=MagicMock(),
                    bank_id=bank_id,
                    memories=[{"id": source_id, "text": "New evidence", "tags": []}],
                    request_context=request_context,
                    config=config,
                )

        assert successful_delete_count == 1
        observations = await memory.list_memory_units(
            bank_id,
            fact_type="observation",
            limit=1000,
            request_context=request_context,
        )
        assert {item["id"] for item in observations["items"]} == {
            str(first_observation_id),
            str(second_observation_id),
        }
        for observation_id in (first_observation_id, second_observation_id):
            history = await memory.get_observation_history(
                bank_id,
                str(observation_id),
                request_context=request_context,
            )
            assert history is not None
            assert len(history) == 1
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# End-to-end fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def enable_observations():
    config = _get_raw_config()
    original = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original


def _override_config(memory: MemoryEngine, **overrides):
    raw = _get_raw_config()
    fake = type(raw)(**{**{f: getattr(raw, f) for f in raw.__dataclass_fields__}, **overrides})
    return patch.object(memory._config_resolver, "resolve_full_config", return_value=fake)


def _mock_llm_one_obs_per_fact():
    """MockLLM wrapper emitting one CREATE per fact id found in the prompt."""
    mock_llm = MockLLM(provider="mock", api_key="", base_url="", model="mock-model")

    def callback(messages, scope):
        if scope != "consolidation":
            return _ConsolidationBatchResponse()
        prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
        fact_ids = re.findall(r"\[([0-9a-f-]{36})\]", prompt)
        creates = [_CreateAction(text=f"Observation about fact {fid[:8]}", source_fact_ids=[fid]) for fid in fact_ids]
        return _ConsolidationBatchResponse(creates=creates)

    mock_llm.set_response_callback(callback)
    wrapper = MagicMock()
    wrapper.with_config.return_value = mock_llm
    return wrapper


async def _insert_memory(conn, bank_id: str, text: str, tags: list[str]) -> uuid.UUID:
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
        json.dumps(None),
    )
    return mem_id


async def _count_observations(memory: MemoryEngine, bank_id: str, request_context) -> int:
    return (
        await memory.list_memory_units(bank_id, fact_type="observation", limit=1000, request_context=request_context)
    )["total"]


class _TxnSpy:
    """Delegates to the real memories store while recording txn decisions."""

    def __init__(self, inner):
        self._inner = inner
        self.decisions: list[bool] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def decide_txn(self, txn, *, commit: bool):
        self.decisions.append(commit)
        return await self._inner.decide_txn(txn, commit=commit)


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_recall_failure_cancels_sibling_tag_groups(memory: MemoryEngine, request_context):
    """One group's recall times out → the other two groups are cancelled before
    they write, and the job propagates the original error without waiting for
    them."""
    bank_id = f"test-cancel-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice likes tea", ["boom"])
            await _insert_memory(conn, bank_id, "Bob bikes daily", ["user:bob"])
            await _insert_memory(conn, bank_id, "Carol reads books", ["user:carol"])

        siblings_parked = asyncio.Event()
        parked_count = 0
        cancelled_scopes: list[frozenset[str]] = []

        async def fake_recall(*, tags=None, **_kwargs):
            nonlocal parked_count
            tag_set = frozenset(tags or [])
            if "boom" in tag_set:
                # Fail only once both siblings are parked, so the assertion below
                # is about cancellation rather than about which group won a race.
                await siblings_parked.wait()
                raise _RecallTimeout("simulated DB command timeout")
            parked_count += 1
            if parked_count >= 2:
                siblings_parked.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled_scopes.append(tag_set)
                raise
            return RecallResult(results=[])

        original_llm = memory._consolidation_llm_config
        memory._consolidation_llm_config = _mock_llm_one_obs_per_fact()
        try:
            with (
                _override_config(memory, consolidation_llm_parallelism=3, consolidation_llm_batch_size=1),
                patch.object(memory, "submit_async_consolidation"),
                patch.object(consolidator_module, "_find_related_observations", fake_recall),
            ):
                started = asyncio.get_running_loop().time()
                with pytest.raises(_RecallTimeout):
                    await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)
                elapsed = asyncio.get_running_loop().time() - started
        finally:
            memory._consolidation_llm_config = original_llm

        # Both sibling groups were cancelled, not left running detached.
        assert sorted(cancelled_scopes, key=sorted) == [frozenset({"user:bob"}), frozenset({"user:carol"})]
        # And the job did not block on their 30s sleep.
        assert elapsed < 10

        # Nothing was written: the cancelled groups never reached their commit,
        # and no orphan lands a write after the operation has already failed.
        await asyncio.sleep(0.2)
        assert await _count_observations(memory, bank_id, request_context) == 0
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_failed_batch_aborts_its_write_group(memory: MemoryEngine, request_context):
    """A batch that raises before its witness commit decides its write-group
    abort, rather than leaving it pending for the recovery sweep."""
    bank_id = f"test-abort-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice likes tea", ["user:alice"])

        async def fake_recall(**_kwargs):
            raise _RecallTimeout("simulated DB command timeout")

        spy = _TxnSpy(consolidator_module.get_memories())
        original_llm = memory._consolidation_llm_config
        memory._consolidation_llm_config = _mock_llm_one_obs_per_fact()
        try:
            with (
                _override_config(memory, consolidation_llm_parallelism=1, consolidation_llm_batch_size=1),
                patch.object(memory, "submit_async_consolidation"),
                patch.object(consolidator_module, "get_memories", return_value=spy),
                patch.object(consolidator_module, "_find_related_observations", fake_recall),
            ):
                with pytest.raises(_RecallTimeout):
                    await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)
        finally:
            memory._consolidation_llm_config = original_llm

        assert spy.decisions == [False]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_successful_batch_still_commits_its_write_group(memory: MemoryEngine, request_context):
    """Guard on the abort path: the happy path must still decide commit=True."""
    bank_id = f"test-commit-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice likes tea", ["user:alice"])

        spy = _TxnSpy(consolidator_module.get_memories())
        original_llm = memory._consolidation_llm_config
        memory._consolidation_llm_config = _mock_llm_one_obs_per_fact()
        try:
            with (
                _override_config(memory, consolidation_llm_parallelism=1, consolidation_llm_batch_size=1),
                patch.object(memory, "submit_async_consolidation"),
                patch.object(consolidator_module, "get_memories", return_value=spy),
            ):
                result = await run_consolidation_job(
                    memory_engine=memory, bank_id=bank_id, request_context=request_context
                )
        finally:
            memory._consolidation_llm_config = original_llm

        assert result["status"] == "completed"
        assert spy.decisions == [True]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
