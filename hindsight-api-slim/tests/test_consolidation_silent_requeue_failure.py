"""Regression guard for the silent re-queue failure path (issue #1842).

The consolidator used to wrap its round-limit re-queue in a permissive
try/except (consolidator.py:606-613):

    try:
        await memory_engine.submit_async_consolidation(...)
    except Exception as e:
        logger.warning("... failed to re-queue consolidation: %s", e)

If ``submit_async_consolidation`` raised for any reason (DB hiccup, custom
operation-validator rejection, etc.) the failure was logged but otherwise
swallowed. The consolidator returned "completed", ``execute_task`` marked
the op completed normally, and **no follow-up pending op was created**.
The bank ended up with unconsolidated backlog and zero pending consolidation
work — exactly the "16 of 17 backlogged banks have no running or pending op"
symptom from issue #1842.

The fix removes the try/except and lets the exception propagate to
``execute_task``'s retry handler. The op is retried with backoff; on retry
the consolidator skips already-consolidated rows via the ``consolidated_at``
filter and picks up the remainder. Eventually the work drains and the op is
marked completed.

This test asserts the new contract: a failing re-queue must NOT silently
complete the op. Either a follow-up pending row exists, or the op is in a
retryable state (raises RetryTaskAt → worker reschedules it).
"""

import json
import uuid
from unittest.mock import patch

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.task_backend import WorkerTaskBackend
from hindsight_api.worker.exceptions import RetryTaskAt


def _make_config(**overrides):
    raw = _get_raw_config()
    return type(raw)(
        **{
            **{f: getattr(raw, f) for f in raw.__dataclass_fields__},
            **overrides,
        }
    )


@pytest.fixture(autouse=True)
def enable_observations():
    config = _get_raw_config()
    original = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original


async def _count_unconsolidated(memory, bank_id: str) -> int:
    async with memory._pool.acquire() as conn:
        return await conn.fetchval(
            """
            SELECT COUNT(*) FROM memory_units
            WHERE bank_id = $1 AND consolidated_at IS NULL
              AND consolidation_failed_at IS NULL AND fact_type IN ('experience', 'world')
            """,
            bank_id,
        )


@pytest.mark.asyncio
async def test_requeue_failure_propagates_to_worker_retry(
    memory: MemoryEngine, request_context
):
    """When the in-task ``submit_async_consolidation`` call raises, the op
    must NOT be silently completed. The consolidator's work for this round
    is durably committed (memories marked consolidated_at in their own
    transaction), so re-running the op picks up where it left off. The
    failure path is: exception bubbles → execute_task's retry handler
    raises RetryTaskAt → poller schedules a retry → next attempt drains
    the remainder.
    """
    bank_id = f"test-requeue-prop-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

    round_limit = 5
    backlog_size = 12

    fake_config_no_obs = _make_config(enable_observations=False)
    with patch.object(memory._config_resolver, "resolve_full_config", return_value=fake_config_no_obs):
        for i in range(backlog_size):
            await memory.retain_async(
                bank_id=bank_id,
                content=f"Fact {i}: the user did activity number {i} on day {i}.",
                request_context=request_context,
            )

    unconsolidated_before = await _count_unconsolidated(memory, bank_id)
    assert unconsolidated_before >= backlog_size

    op_id = uuid.uuid4()
    payload = {
        "type": "consolidation",
        "operation_id": str(op_id),
        "bank_id": bank_id,
    }
    async with memory._pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
            VALUES ($1, $2, 'consolidation', 'processing', $3::jsonb)
            """,
            op_id,
            bank_id,
            json.dumps(payload),
        )

    original_backend = memory._task_backend
    memory._task_backend = WorkerTaskBackend()
    await memory._task_backend.initialize()

    fake_config = _make_config(consolidation_max_memories_per_round=round_limit)

    # Simulate ANY plausible re-queue failure: DB hiccup, validator
    # rejection (e.g. credit-check extension), tenant-ext blip during auth.
    original_submit = memory.submit_async_consolidation
    call_count = {"n": 0}

    async def failing_submit(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated: re-queue failed")
        return await original_submit(*args, **kwargs)

    try:
        with patch.object(memory._config_resolver, "resolve_full_config", return_value=fake_config):
            with patch.object(memory, "submit_async_consolidation", side_effect=failing_submit):
                # execute_task's exception handler converts the propagated
                # RuntimeError into a RetryTaskAt — the worker poller will
                # reset status to 'pending' with next_retry_at.
                with pytest.raises(RetryTaskAt):
                    await memory.execute_task(payload)
    finally:
        memory._task_backend = original_backend

    # The work this round actually did is durable. Memories that were
    # consolidated stay consolidated; the consolidator's per-batch
    # `UPDATE ... SET consolidated_at = NOW()` commits in its own
    # transaction (consolidator.py:524-534), not inside the op-level state.
    unconsolidated_after = await _count_unconsolidated(memory, bank_id)
    assert unconsolidated_after < unconsolidated_before, (
        f"completed-round work must be durable across the re-queue failure; "
        f"before={unconsolidated_before}, after={unconsolidated_after}"
    )

    # The op MUST NOT be silently marked completed. It either:
    #   (a) stays 'processing' here because execute_task raised RetryTaskAt
    #       (the poller would later transition it to 'pending' with
    #       next_retry_at via _schedule_retry — that path isn't exercised
    #       when we drive execute_task directly), or
    #   (b) is already in some non-completed state.
    # The contract we're guarding against: status='completed' with no
    # pending follow-up — the silent-stuck failure mode.
    async with memory._pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status FROM async_operations WHERE operation_id = $1",
            op_id,
        )
        pending = await conn.fetch(
            """
            SELECT operation_id::text AS oid FROM async_operations
            WHERE bank_id = $1 AND operation_type = 'consolidation' AND status = 'pending'
            """,
            bank_id,
        )

    is_silent_stuck = row["status"] == "completed" and len(pending) == 0
    assert not is_silent_stuck, (
        f"silent re-queue failure must not silently complete the op; "
        f"status={row['status']}, pending_followups={len(pending)}, "
        f"unconsolidated_remaining={unconsolidated_after}"
    )

    assert call_count["n"] == 1, (
        f"only one in-task submit_async_consolidation call expected, got {call_count['n']}"
    )

    await memory.delete_bank(bank_id, request_context=request_context)
