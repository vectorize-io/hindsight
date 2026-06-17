"""Regression tests for issue #1842: submitting consolidation is atomic per bank.

`submit_async_consolidation` promises "at most one pending full-bank consolidation
per bank", but the dedup was a check-then-INSERT that raced under READ COMMITTED —
concurrent submits (a manual `/consolidate` loop racing retain-driven submits and
the round-limit re-queue) each saw no pending row and each inserted, leaking
duplicate pending ops that then piled up as `retry_blocked` and starved the bank.

The submit now serializes per bank (SELECT ... FOR UPDATE on the bank row) so the
check-and-insert is atomic. Scoped runs (`observation_scopes` set) are exempt and
may have multiple pending ops.
"""

import asyncio
import uuid

import pytest


@pytest.fixture
def no_inline_execution(memory):
    """Stop the SyncTaskBackend from executing submitted ops inline so the pending
    rows survive for inspection (we're testing the submit/dedup path, not execution)."""

    async def _noop(_payload):
        return None

    original = memory._task_backend.submit_task
    memory._task_backend.submit_task = _noop
    yield
    memory._task_backend.submit_task = original


async def _ensure_bank(pool, bank_id: str) -> None:
    await pool.execute(
        "INSERT INTO banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        bank_id,
        bank_id,
    )


async def _count_pending(pool, bank_id: str) -> int:
    return await pool.fetchval(
        """
        SELECT COUNT(*) FROM async_operations
        WHERE bank_id = $1 AND operation_type = 'consolidation' AND status = 'pending'
        """,
        bank_id,
    )


async def _cleanup(pool, bank_id: str) -> None:
    await pool.execute("DELETE FROM async_operations WHERE bank_id = $1", bank_id)
    await pool.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
async def test_concurrent_submits_leave_one_pending(memory, request_context, no_inline_execution):
    """Five concurrent unscoped submits on a bank with no pending op must end with
    exactly one pending row, and all calls must resolve to that one operation_id."""
    bank_id = f"test-atomic-{uuid.uuid4().hex[:8]}"
    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    try:
        results = await asyncio.gather(
            *(memory.submit_async_consolidation(bank_id=bank_id, request_context=request_context) for _ in range(5))
        )
        assert await _count_pending(pool, bank_id) == 1
        op_ids = {r["operation_id"] for r in results}
        assert len(op_ids) == 1, f"all submits should share one op, got {op_ids}"
        # Exactly one call created the op; the other four were deduplicated.
        assert sum(1 for r in results if r.get("deduplicated")) == 4
    finally:
        await _cleanup(pool, bank_id)


@pytest.mark.asyncio
async def test_sequential_submit_dedups(memory, request_context, no_inline_execution):
    """A second submit while one is already pending returns the existing op."""
    bank_id = f"test-atomic-{uuid.uuid4().hex[:8]}"
    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    try:
        first = await memory.submit_async_consolidation(bank_id=bank_id, request_context=request_context)
        second = await memory.submit_async_consolidation(bank_id=bank_id, request_context=request_context)
        assert second["operation_id"] == first["operation_id"]
        assert second.get("deduplicated") is True
        assert await _count_pending(pool, bank_id) == 1
    finally:
        await _cleanup(pool, bank_id)


@pytest.mark.asyncio
async def test_unscoped_not_deduped_against_scoped_pending(memory, request_context, no_inline_execution):
    """A full-bank (unscoped) submit must still run when only a scoped op is pending —
    a scoped run covers a tag subset and must not swallow the full-bank sweep."""
    bank_id = f"test-atomic-{uuid.uuid4().hex[:8]}"
    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    try:
        scoped = await memory.submit_async_consolidation(
            bank_id=bank_id, request_context=request_context, observation_scopes=[["proj-a"]]
        )
        unscoped = await memory.submit_async_consolidation(bank_id=bank_id, request_context=request_context)
        assert not unscoped.get("deduplicated"), "full-bank submit must not dedup against a scoped pending op"
        assert unscoped["operation_id"] != scoped["operation_id"]
        assert await _count_pending(pool, bank_id) == 2

        # A second unscoped submit, however, dedups against the now-pending unscoped op.
        unscoped2 = await memory.submit_async_consolidation(bank_id=bank_id, request_context=request_context)
        assert unscoped2["operation_id"] == unscoped["operation_id"]
        assert unscoped2.get("deduplicated") is True
        assert await _count_pending(pool, bank_id) == 2
    finally:
        await _cleanup(pool, bank_id)


@pytest.mark.asyncio
async def test_scoped_submits_are_not_deduped(memory, request_context, no_inline_execution):
    """Scoped runs are targeted and intentionally exempt — they may pile up pending."""
    bank_id = f"test-atomic-{uuid.uuid4().hex[:8]}"
    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    try:
        r1 = await memory.submit_async_consolidation(
            bank_id=bank_id, request_context=request_context, observation_scopes=[["proj-a"]]
        )
        r2 = await memory.submit_async_consolidation(
            bank_id=bank_id, request_context=request_context, observation_scopes=[["proj-b"]]
        )
        assert r1["operation_id"] != r2["operation_id"]
        assert not r2.get("deduplicated")
        assert await _count_pending(pool, bank_id) == 2
    finally:
        await _cleanup(pool, bank_id)
