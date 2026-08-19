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
from unittest.mock import MagicMock, patch

import asyncpg
import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.consolidation import consolidator as consolidator_module
from hindsight_api.engine.consolidation.consolidator import (
    _ConsolidationBatchResponse,
    _CreateAction,
    _gather_or_cancel,
    run_consolidation_job,
)
from hindsight_api.engine.memory_engine import MemoryEngine, _is_non_retryable_task_error
from hindsight_api.engine.providers.mock_llm import MockLLM
from hindsight_api.engine.response_models import RecallResult


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
