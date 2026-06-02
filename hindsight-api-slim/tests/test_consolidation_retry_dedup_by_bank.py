"""
Regression tests for the consolidation retry path.

When a consolidation task hits a transient error, the worker's generic retry
path raises ``RetryTaskAt`` to re-queue the operation (memory_engine.py, see
``execute_task``'s exception branch). Two pieces of behaviour live there:

1. **Dedup guard.** During a long upstream outage, every retain on the same
   bank enqueues a fresh consolidation op via ``submit_async_consolidation``.
   Without dedup, each op independently consumes its own retry budget,
   multiplying load on the broken dependency. The guard: if another
   consolidation op is already ``pending`` for the same bank when the retry
   decision is made, skip the retry and let the other op cover the work.

2. **Indefinite retry with capped exponential backoff** (60s, 120, 240, 480,
   960, 1800-cap). Distinct from the generic 60s × 3 used by other task
   types. Transient consolidation failures (LLM/DB outage) must eventually
   recover; capping after a fixed number of attempts silently dead-letters
   the bank's backlog. The dedup-by-bank guard above prevents a retry
   storm when multiple ops exist for the same bank.
"""

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from hindsight_api.engine.memory_engine import (
    _CONSOLIDATION_RETRY_BACKOFF_MAX_SECONDS,
    _consolidation_retry_backoff_seconds,
)
from hindsight_api.worker.exceptions import RetryTaskAt


async def _ensure_bank(pool, bank_id: str) -> None:
    await pool.execute(
        "INSERT INTO banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        bank_id,
        bank_id,
    )


async def _insert_consolidation_op(pool, bank_id: str, operation_id: uuid.UUID, status: str) -> None:
    """Insert a consolidation row with the given ``status``."""
    payload = json.dumps(
        {
            "type": "consolidation",
            "operation_id": str(operation_id),
            "bank_id": bank_id,
        }
    )
    await pool.execute(
        """
        INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
        VALUES ($1, $2, 'consolidation', $3, $4::jsonb)
        """,
        operation_id,
        bank_id,
        status,
        payload,
    )


async def _cleanup(pool, bank_id: str, *operation_ids: uuid.UUID) -> None:
    if operation_ids:
        await pool.execute(
            "DELETE FROM async_operations WHERE operation_id = ANY($1::uuid[])",
            list(operation_ids),
        )
    await pool.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
async def test_retry_skipped_when_other_pending_consolidation_exists(memory):
    """
    With another pending consolidation op already covering the same bank, a
    transient failure on the currently-executing op must NOT raise
    ``RetryTaskAt`` — the pending op will process the same unconsolidated
    rows when the worker picks it up. Without this guard, a long outage
    causes a retry storm: every retain enqueues a new op, and each op burns
    three retry slots against the same broken dependency.
    """
    bank_id = f"test-dedup-{uuid.uuid4().hex[:8]}"
    running_op_id = uuid.uuid4()
    peer_pending_op_id = uuid.uuid4()

    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    # The op currently executing (drives execute_task below).
    await _insert_consolidation_op(pool, bank_id, running_op_id, status="processing")
    # A peer consolidation already pending for the same bank — the dedup guard
    # should fire because of this row.
    await _insert_consolidation_op(pool, bank_id, peer_pending_op_id, status="pending")

    task_dict = {
        "type": "consolidation",
        "operation_id": str(running_op_id),
        "bank_id": bank_id,
    }

    transient = RuntimeError("upstream LLM 503")
    with patch.object(memory, "_handle_consolidation", side_effect=transient):
        # The original exception must propagate as-is (the poller catches it
        # and calls _mark_failed). The point of the test is that it is NOT
        # converted to RetryTaskAt — pytest.raises(RuntimeError) asserts that
        # implicitly, since RetryTaskAt is not a RuntimeError subclass.
        with pytest.raises(RuntimeError, match="upstream LLM 503"):
            await memory.execute_task(task_dict)

    # Peer pending op must be untouched by the dedup guard.
    peer_row = await pool.fetchrow(
        "SELECT status FROM async_operations WHERE operation_id = $1",
        peer_pending_op_id,
    )
    assert peer_row["status"] == "pending"

    await _cleanup(pool, bank_id, running_op_id, peer_pending_op_id)


@pytest.mark.asyncio
async def test_retry_happens_when_no_other_pending_consolidation(memory):
    """
    Sanity check: with no peer pending op, the normal retry path still fires
    and ``RetryTaskAt`` is raised — the dedup guard must not regress the
    common-case behaviour.
    """
    bank_id = f"test-dedup-{uuid.uuid4().hex[:8]}"
    running_op_id = uuid.uuid4()

    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    await _insert_consolidation_op(pool, bank_id, running_op_id, status="processing")

    task_dict = {
        "type": "consolidation",
        "operation_id": str(running_op_id),
        "bank_id": bank_id,
    }

    transient = RuntimeError("transient blip")
    with patch.object(memory, "_handle_consolidation", side_effect=transient):
        with pytest.raises(RetryTaskAt):
            await memory.execute_task(task_dict)

    await _cleanup(pool, bank_id, running_op_id)


@pytest.mark.asyncio
async def test_peer_only_dedups_when_same_bank(memory):
    """
    A pending consolidation for a DIFFERENT bank must not suppress the retry
    — the dedup guard is per-bank, not global.
    """
    bank_id_a = f"test-dedup-a-{uuid.uuid4().hex[:8]}"
    bank_id_b = f"test-dedup-b-{uuid.uuid4().hex[:8]}"
    running_op_id = uuid.uuid4()
    unrelated_pending_op_id = uuid.uuid4()

    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id_a)
    await _ensure_bank(pool, bank_id_b)
    await _insert_consolidation_op(pool, bank_id_a, running_op_id, status="processing")
    await _insert_consolidation_op(pool, bank_id_b, unrelated_pending_op_id, status="pending")

    task_dict = {
        "type": "consolidation",
        "operation_id": str(running_op_id),
        "bank_id": bank_id_a,
    }

    with patch.object(memory, "_handle_consolidation", side_effect=RuntimeError("blip")):
        with pytest.raises(RetryTaskAt):
            await memory.execute_task(task_dict)

    await _cleanup(pool, bank_id_a, running_op_id)
    await _cleanup(pool, bank_id_b, unrelated_pending_op_id)


def test_backoff_helper_schedule():
    """
    The backoff helper produces capped exponential growth starting at 5s so a
    transient blip (LLM 503 that clears in a few seconds) doesn't park the
    bank for a full minute: 5, 10, 20, 40, 80, 160, 320, 640, 1280, then
    pinned at the 1800s cap.
    """
    assert _consolidation_retry_backoff_seconds(0) == 5
    assert _consolidation_retry_backoff_seconds(1) == 10
    assert _consolidation_retry_backoff_seconds(2) == 20
    assert _consolidation_retry_backoff_seconds(3) == 40
    assert _consolidation_retry_backoff_seconds(4) == 80
    assert _consolidation_retry_backoff_seconds(5) == 160
    assert _consolidation_retry_backoff_seconds(6) == 320
    assert _consolidation_retry_backoff_seconds(7) == 640
    assert _consolidation_retry_backoff_seconds(8) == 1280
    assert _consolidation_retry_backoff_seconds(9) == _CONSOLIDATION_RETRY_BACKOFF_MAX_SECONDS
    # Cap holds at arbitrarily high attempt counts (we never give up).
    assert _consolidation_retry_backoff_seconds(50) == _CONSOLIDATION_RETRY_BACKOFF_MAX_SECONDS
    assert _consolidation_retry_backoff_seconds(1000) == _CONSOLIDATION_RETRY_BACKOFF_MAX_SECONDS


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_count", [0, 1, 2, 9])
async def test_backoff_matches_schedule_by_retry_count(memory, retry_count):
    """
    Each retry schedules `retry_at = now + backoff(retry_count)`. The tolerance
    covers the few seconds between ``now()`` capture in the test and inside
    the retry handler. Includes retry_count=9 to verify the cap branch.
    """
    bank_id = f"test-backoff-{uuid.uuid4().hex[:8]}"
    op_id = uuid.uuid4()

    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    await _insert_consolidation_op(pool, bank_id, op_id, status="processing")

    task_dict = {
        "type": "consolidation",
        "operation_id": str(op_id),
        "bank_id": bank_id,
        "_retry_count": retry_count,
    }

    expected_backoff = _consolidation_retry_backoff_seconds(retry_count)
    before = datetime.now(UTC)
    with patch.object(memory, "_handle_consolidation", side_effect=RuntimeError("transient")):
        with pytest.raises(RetryTaskAt) as excinfo:
            await memory.execute_task(task_dict)

    delta = (excinfo.value.retry_at - before).total_seconds()
    assert expected_backoff <= delta <= expected_backoff + 10, (
        f"retry_count={retry_count}: expected backoff ~{expected_backoff}s, "
        f"got delta={delta:.2f}s"
    )

    await _cleanup(pool, bank_id, op_id)


@pytest.mark.asyncio
async def test_retry_is_indefinite(memory):
    """
    Consolidation has no attempt cap — even at very high retry_count, a
    transient failure still raises RetryTaskAt (capped at the max backoff).
    Without this property, a long outage silently dead-letters the bank's
    unconsolidated rows once the budget is exhausted.
    """
    bank_id = f"test-indefinite-{uuid.uuid4().hex[:8]}"
    op_id = uuid.uuid4()

    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    await _insert_consolidation_op(pool, bank_id, op_id, status="processing")

    task_dict = {
        "type": "consolidation",
        "operation_id": str(op_id),
        "bank_id": bank_id,
        "_retry_count": 100,  # well past any plausible attempt cap
    }

    before = datetime.now(UTC)
    with patch.object(memory, "_handle_consolidation", side_effect=RuntimeError("still failing")):
        with pytest.raises(RetryTaskAt) as excinfo:
            await memory.execute_task(task_dict)

    delta = (excinfo.value.retry_at - before).total_seconds()
    cap = _CONSOLIDATION_RETRY_BACKOFF_MAX_SECONDS
    assert cap <= delta <= cap + 10, (
        f"At retry_count=100 expected backoff at cap (~{cap}s), got {delta:.2f}s"
    )

    await _cleanup(pool, bank_id, op_id)
