"""Failed consolidation batch calls are visible at run level (#4151).

``failed_consolidation`` is a gauge over rows still carrying
``consolidation_failed_at``, so it counts facts left STUCK. It cannot count a
batch call that failed and whose facts the adaptive bisection then rescued —
and that is the common case for a schema-invalid response, which is classified
FAIL_FAST, fails the batch, and gets bisected into calls small enough to
validate. A run can therefore burn dozens of failed calls, discard every action
those responses carried, and end with ``failed_consolidation`` at 0 and
``observations_deleted`` at 0: indistinguishable from a clean run.

Two counters close that: ``llm_batch_failures`` in the job's own stats, and the
``hindsight.consolidation.batch_failures`` metric, labelled by failure class so
transport-shaped retries stay separable from schema rejections.
"""

from __future__ import annotations

import json
import re
import uuid
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.consolidation import consolidator as consolidator_module
from hindsight_api.engine.consolidation.consolidator import (
    _BatchFailureClass,
    _classify_batch_failure,
    _ConsolidationBatchResponse,
    _CreateAction,
    run_consolidation_job,
)
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.providers.mock_llm import MockLLM
from hindsight_api.metrics import NoOpMetricsCollector


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


def _schema_validation_error() -> ValidationError:
    """A real pydantic ValidationError, the shape a model emitting non-conforming
    JSON produces on the way out of the LLM layer."""

    class _Probe(BaseModel):
        creates: list[_CreateAction]

    try:
        _Probe.model_validate({"creates": "not-a-list"})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


class _RecordingCollector(NoOpMetricsCollector):
    """Captures batch-failure records; every other metric stays a no-op."""

    def __init__(self):
        self.batch_failures: list[tuple[str, str]] = []

    def record_consolidation_batch_failure(self, failure_class: str, error_type: str):
        self.batch_failures.append((failure_class, error_type))


def test_schema_validation_failure_is_fail_fast():
    """A re-send of the identical payload cannot fix a schema rejection, so the
    batch fails immediately and the caller bisects instead."""
    assert _classify_batch_failure(_schema_validation_error()) is _BatchFailureClass.FAIL_FAST


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


def _install_llm(memory: MemoryEngine, callback):
    mock_llm = MockLLM(provider="mock", api_key="", base_url="", model="mock-model")
    mock_llm.set_response_callback(callback)
    wrapper = MagicMock()
    wrapper.with_config.return_value = mock_llm
    memory._consolidation_llm_config = wrapper


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_failures_rescued_by_bisection_are_still_counted(memory: MemoryEngine, request_context):
    """The reported symptom: N calls fail schema validation, bisection rescues every
    fact, ``failed_consolidation`` reads 0 — and the run now says so anyway."""
    bank_id = f"test-4151-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    collector = _RecordingCollector()
    original_llm = memory._consolidation_llm_config
    try:
        async with memory._pool.acquire() as conn:
            for text in ("Alice likes tea", "Alice bikes daily", "Alice reads books", "Alice runs races"):
                await _insert_memory(conn, bank_id, text, ["user:alice"])

        validation_failures = 0

        def callback(messages, scope):
            nonlocal validation_failures
            if scope != "consolidation":
                return _ConsolidationBatchResponse()
            prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
            fact_ids = re.findall(r"\[([0-9a-f-]{36})\]", prompt)
            # Stands in for a model that only emits valid strict JSON for the
            # simplest prompts: anything with more than one fact comes back
            # schema-invalid, and bisection is what eventually gets through.
            if len(fact_ids) > 1:
                validation_failures += 1
                raise _schema_validation_error()
            return _ConsolidationBatchResponse(
                creates=[
                    _CreateAction(text=f"Observation about fact {fid[:8]}", source_fact_ids=[fid]) for fid in fact_ids
                ]
            )

        _install_llm(memory, callback)
        with (
            _override_config(memory, consolidation_llm_batch_size=4, consolidation_llm_parallelism=1),
            patch.object(memory, "submit_async_consolidation"),
            patch.object(consolidator_module, "get_metrics_collector", return_value=collector),
        ):
            result = await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)

        # 4 -> 2+2 -> 1x4: three calls failed before anything validated.
        assert validation_failures >= 3

        # The stuck-fact gauge legitimately reads 0 — bisection rescued them all.
        stats = await memory.get_bank_stats(bank_id, request_context=request_context)
        assert stats["failed_consolidation"] == 0
        assert stats["pending_consolidation"] == 0

        # The run-level counter reports what the gauge structurally cannot.
        assert result["llm_batch_failures"] == validation_failures

        # And so does the metric, labelled so schema rejections stay separable
        # from transport-shaped retries.
        assert len(collector.batch_failures) == validation_failures
        assert set(collector.batch_failures) == {(str(_BatchFailureClass.FAIL_FAST), "ValidationError")}
    finally:
        memory._consolidation_llm_config = original_llm
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_a_clean_run_reports_no_batch_failures(memory: MemoryEngine, request_context):
    """The counter must stay 0 on a healthy run, or it is noise rather than signal."""
    bank_id = f"test-4151-clean-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    collector = _RecordingCollector()
    original_llm = memory._consolidation_llm_config
    try:
        async with memory._pool.acquire() as conn:
            for text in ("Dave likes tea", "Dave bikes daily"):
                await _insert_memory(conn, bank_id, text, ["user:dave"])

        def callback(messages, scope):
            if scope != "consolidation":
                return _ConsolidationBatchResponse()
            prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
            fact_ids = re.findall(r"\[([0-9a-f-]{36})\]", prompt)
            return _ConsolidationBatchResponse(
                creates=[
                    _CreateAction(text=f"Observation about fact {fid[:8]}", source_fact_ids=[fid]) for fid in fact_ids
                ]
            )

        _install_llm(memory, callback)
        with (
            _override_config(memory, consolidation_llm_batch_size=2, consolidation_llm_parallelism=1),
            patch.object(memory, "submit_async_consolidation"),
            patch.object(consolidator_module, "get_metrics_collector", return_value=collector),
        ):
            result = await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)

        assert result["llm_batch_failures"] == 0
        assert collector.batch_failures == []
    finally:
        memory._consolidation_llm_config = original_llm
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_retried_attempts_each_count(memory: MemoryEngine, request_context):
    """One batch call can burn several attempts, and each is a discarded response.

    A transport-shaped failure is RETRY, so the outer ladder re-sends up to
    ``consolidation_max_attempts`` times. The counter must total attempts rather
    than batch calls — a call retried three times cost three generations.
    """
    bank_id = f"test-4151-retry-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    collector = _RecordingCollector()
    original_llm = memory._consolidation_llm_config
    try:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Erin likes tea", ["user:erin"])

        def callback(messages, scope):
            if scope != "consolidation":
                return _ConsolidationBatchResponse()
            # Not JSONDecodeError/ValidationError/OutputTooLong, so it classifies RETRY
            # and the outer ladder re-sends instead of failing the batch immediately.
            raise RuntimeError("connection reset by peer")

        _install_llm(memory, callback)
        with (
            _override_config(
                memory,
                consolidation_llm_batch_size=1,
                consolidation_llm_parallelism=1,
                consolidation_max_attempts=3,
            ),
            patch.object(memory, "submit_async_consolidation"),
            patch.object(consolidator_module, "get_metrics_collector", return_value=collector),
            patch.object(consolidator_module, "_OUTER_RETRY_INITIAL_BACKOFF", 0),
        ):
            result = await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)

        # One fact, one batch call, three attempts — all three counted.
        assert result["llm_batch_failures"] == 3
        assert collector.batch_failures == [(str(_BatchFailureClass.RETRY), "RuntimeError")] * 3
    finally:
        memory._consolidation_llm_config = original_llm
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_facts_bisection_cannot_rescue_are_counted_in_both_places(memory: MemoryEngine, request_context):
    """Contrast: when even a single-fact call fails, the fact IS stuck and both the
    gauge and the new counter report it — they measure different things, not the
    same thing twice."""
    bank_id = f"test-4151-stuck-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    collector = _RecordingCollector()
    original_llm = memory._consolidation_llm_config
    try:
        async with memory._pool.acquire() as conn:
            for text in ("Bob likes tea", "Bob bikes daily"):
                await _insert_memory(conn, bank_id, text, ["user:bob"])

        def callback(messages, scope):
            if scope != "consolidation":
                return _ConsolidationBatchResponse()
            raise _schema_validation_error()

        _install_llm(memory, callback)
        with (
            _override_config(
                memory,
                consolidation_llm_batch_size=2,
                consolidation_llm_parallelism=1,
                consolidation_max_attempts=1,
            ),
            patch.object(memory, "submit_async_consolidation"),
            patch.object(consolidator_module, "get_metrics_collector", return_value=collector),
        ):
            result = await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)

        stats = await memory.get_bank_stats(bank_id, request_context=request_context)
        assert stats["failed_consolidation"] == 2
        # One call for the pair plus one per bisected half.
        assert result["llm_batch_failures"] == 3
        assert len(collector.batch_failures) == 3
    finally:
        memory._consolidation_llm_config = original_llm
        await memory.delete_bank(bank_id, request_context=request_context)
