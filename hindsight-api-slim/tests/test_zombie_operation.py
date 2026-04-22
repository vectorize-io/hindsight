"""Test zombie operation detection and self-healing in deduplication.

Verifies that:
1. A zombie row (status=pending, task_payload=NULL) does NOT block new submissions
2. The zombie is cleaned up (marked failed) rather than accumulated
3. A new operation is created when the only pending match is a zombie
"""

import uuid

import pytest

from hindsight_api.extensions import RequestContext


async def _ensure_bank(pool, bank_id: str) -> None:
    """Upsert a minimal bank row so FK on async_operations passes."""
    await pool.execute(
        "INSERT INTO banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        bank_id,
        bank_id,
    )


def _insert_zombie_operation(pool, bank_id: str, operation_type: str = "consolidation") -> str:
    """Insert a zombie operation row: status=pending but task_payload=NULL."""
    op_id = str(uuid.uuid4())
    pool.execute(
        """
        INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload, result_metadata)
        VALUES ($1, $2, $3, 'pending', NULL, '{}')
        """,
        op_id,
        bank_id,
        operation_type,
    )
    return op_id


@pytest.mark.asyncio
async def test_zombie_does_not_block_new_submission(memory, request_context):
    """When only a zombie exists, a new operation is created (not deduplicated to the zombie)."""
    bank_id = f"test_zombie_block_{uuid.uuid4().hex[:8]}"

    pool = memory._pool
    await _ensure_bank(pool, bank_id)

    # Create a zombie consolidation operation (null payload)
    zombie_id = _insert_zombie_operation(pool, bank_id, operation_type="consolidation")

    # Attempt a new consolidation — should NOT be deduplicated to the zombie
    result = await memory._submit_async_operation(
        bank_id=bank_id,
        operation_type="consolidation",
        task_type="consolidation",
        task_payload={"dummy": True},
        dedupe_by_bank=True,
    )

    # Must get a new operation_id back (not the zombie's)
    assert result["operation_id"] != zombie_id
    assert result.get("deduplicated") is None  # not deduplicated

    # Verify zombie was marked failed
    row = await pool.fetchrow(
        "SELECT status, result_metadata FROM async_operations WHERE operation_id = $1",
        zombie_id,
    )
    assert row["status"] == "failed"
    assert row["result_metadata"].get("zombie_cleanup") is True


@pytest.mark.asyncio
async def test_valid_pending_blocks_new_submission(memory, request_context):
    """When a valid pending operation exists, deduplication returns it (not a new one)."""
    bank_id = f"test_valid_dedup_{uuid.uuid4().hex[:8]}"

    pool = memory._pool
    await _ensure_bank(pool, bank_id)

    # Submit a legitimate operation first
    first = await memory._submit_async_operation(
        bank_id=bank_id,
        operation_type="consolidation",
        task_type="consolidation",
        task_payload={"dummy": True},
        dedupe_by_bank=True,
    )

    # Submit again — should deduplicate to the first
    second = await memory._submit_async_operation(
        bank_id=bank_id,
        operation_type="consolidation",
        task_type="consolidation",
        task_payload={"different": True},
        dedupe_by_bank=True,
    )

    assert second["operation_id"] == first["operation_id"]
    assert second.get("deduplicated") is True


@pytest.mark.asyncio
async def test_mixed_zombie_and_valid_pending_uses_valid(memory, request_context):
    """When both zombie and valid pending exist, deduplication returns the valid one."""
    bank_id = f"test_mixed_{uuid.uuid4().hex[:8]}"

    pool = memory._pool
    await _ensure_bank(pool, bank_id)

    # Zombie first
    zombie_id = _insert_zombie_operation(pool, bank_id, operation_type="consolidation")

    # Then a valid operation
    valid = await memory._submit_async_operation(
        bank_id=bank_id,
        operation_type="consolidation",
        task_type="consolidation",
        task_payload={"dummy": True},
        dedupe_by_bank=True,
    )

    # Another submission should deduplicate to the valid one
    result = await memory._submit_async_operation(
        bank_id=bank_id,
        operation_type="consolidation",
        task_type="consolidation",
        task_payload={"another": True},
        dedupe_by_bank=True,
    )

    assert result["operation_id"] == valid["operation_id"]
    assert result.get("deduplicated") is True

    # Zombie should still be failed (cleanup runs on dedup check, not on valid hit)
    zombie_row = await pool.fetchrow(
        "SELECT status FROM async_operations WHERE operation_id = $1", zombie_id
    )
    assert zombie_row["status"] == "failed"
