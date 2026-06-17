"""Round-limit re-queue regression guard (related to issue #1842).

The consolidator calls ``submit_async_consolidation`` at the end of a
round-limited run (consolidator.py:600). The existing
``test_consolidation_round_limit.py`` patches that call away and only checks
it was invoked, so it does not exercise the actual DB-row creation.

This test exercises the full end-to-end path:

  1. Build a backlog > max_memories_per_round with auto-consolidation
     disabled at retain time.
  2. Insert a pending consolidation op row (as the API would on
     ``POST /consolidate``) and execute it through ``MemoryEngine.execute_task``
     exactly as the worker poller does.
  3. After ``execute_task`` returns, assert:
       - the first op is marked ``completed``
       - a *new* pending consolidation row exists for the same bank
         (the round-limit re-queue actually landed in the DB)
       - unconsolidated rows remain (the backlog isn't drained in one round)

The engine's task_backend is swapped to ``WorkerTaskBackend`` for the
duration of the run so the in-task ``submit_async_consolidation`` call does
NOT recursively execute (which is what ``SyncTaskBackend`` would do). That
matches production worker semantics, where ``submit_task`` is a no-op and
the worker poller picks up the new row on its next cycle.
"""

import json
import uuid
from unittest.mock import patch

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.task_backend import WorkerTaskBackend


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


async def _pending_consolidation_ops(memory, bank_id: str) -> list[str]:
    async with memory._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT operation_id::text AS oid FROM async_operations
            WHERE bank_id = $1 AND operation_type = 'consolidation' AND status = 'pending'
            """,
            bank_id,
        )
    return [r["oid"] for r in rows]


@pytest.mark.asyncio
async def test_round_limited_consolidation_leaves_followup_pending_op(memory: MemoryEngine, request_context):
    """A round-limited consolidation must leave a new ``pending`` consolidation
    op in ``async_operations`` for the same bank so the worker poller can
    drain the backlog without external intervention."""
    bank_id = f"test-reschedule-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

    round_limit = 5
    backlog_size = 12  # > 2x round_limit so we expect multiple re-queues

    # 1. Build the backlog with consolidation disabled during retain.
    fake_config_no_obs = _make_config(enable_observations=False)
    with patch.object(memory._config_resolver, "resolve_full_config", return_value=fake_config_no_obs):
        for i in range(backlog_size):
            await memory.retain_async(
                bank_id=bank_id,
                content=f"Fact {i}: the user did activity number {i} on day {i}.",
                request_context=request_context,
            )

    unconsolidated_before = await _count_unconsolidated(memory, bank_id)
    assert unconsolidated_before >= backlog_size, (
        f"Expected at least {backlog_size} unconsolidated memories, got {unconsolidated_before}"
    )

    # 2. Insert a pending consolidation op (simulating POST /consolidate)
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

    # Swap to WorkerTaskBackend so the in-task submit_async_consolidation does
    # NOT recursively execute (matches production worker semantics — the row
    # is left for the poller's next cycle).
    original_backend = memory._task_backend
    memory._task_backend = WorkerTaskBackend()
    await memory._task_backend.initialize()

    fake_config = _make_config(consolidation_max_memories_per_round=round_limit)

    try:
        with patch.object(memory._config_resolver, "resolve_full_config", return_value=fake_config):
            await memory.execute_task(payload)
    finally:
        memory._task_backend = original_backend

    # 3. Verify the first op completed
    async with memory._pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status FROM async_operations WHERE operation_id = $1",
            op_id,
        )
    assert row is not None
    assert row["status"] == "completed", f"first consolidation op should be marked completed, got {row['status']}"

    # 4. Backlog must remain (round limit kept one round under the total)
    unconsolidated_after = await _count_unconsolidated(memory, bank_id)
    assert 0 < unconsolidated_after < unconsolidated_before, (
        f"expected backlog to shrink but still remain after one round; "
        f"before={unconsolidated_before}, after={unconsolidated_after}"
    )

    # 5. KEY ASSERTION — a new pending consolidation op must exist for this bank.
    #    The worker poller will pick this up on its next cycle. If this assertion
    #    fails, the bank is silently stuck: it has unconsolidated rows but no
    #    queued work.
    pending_ops = await _pending_consolidation_ops(memory, bank_id)
    assert len(pending_ops) == 1, (
        f"After a round-limited consolidation finishes, exactly one pending "
        f"consolidation op must remain so the worker can continue draining the "
        f"backlog. Found {len(pending_ops)} pending ops; backlog still has "
        f"{unconsolidated_after} unconsolidated memory_units."
    )
    assert pending_ops[0] != str(op_id), "The pending op must be a NEW row, not the original op we just executed."

    await memory.delete_bank(bank_id, request_context=request_context)
