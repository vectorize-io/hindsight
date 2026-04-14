"""
Regression tests for vectorize-io/hindsight#980.

Deterministic Postgres integrity-constraint violations (UniqueViolationError,
ForeignKeyViolationError, CheckViolationError, NotNullViolationError,
ExclusionViolationError) must NOT be retried by the worker — they will never
succeed on retry, and retrying just burns worker capacity for ~3 minutes
(3 retries × 60s) before finally giving up.

These tests verify that ``MemoryEngine.execute_task`` classifies
``asyncpg.exceptions.IntegrityConstraintViolationError`` as non-retryable
and marks the operation as failed on the first occurrence.
"""

import json
import uuid
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

from hindsight_api.worker.exceptions import RetryTaskAt


async def _ensure_bank(pool, bank_id: str) -> None:
    """Upsert a minimal bank row so FK on async_operations passes."""
    await pool.execute(
        "INSERT INTO banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        bank_id,
        bank_id,
    )


async def _create_pending_operation(pool, bank_id: str, operation_id: uuid.UUID) -> None:
    """Insert a pending batch_retain operation row for the test."""
    payload = json.dumps(
        {
            "type": "batch_retain",
            "operation_id": str(operation_id),
            "bank_id": bank_id,
            "contents": [{"content": "test", "document_id": "doc-1"}],
        }
    )
    await pool.execute(
        """
        INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
        VALUES ($1, $2, 'retain', 'pending', $3::jsonb)
        """,
        operation_id,
        bank_id,
        payload,
    )


@pytest.mark.asyncio
async def test_unique_violation_marks_failed_without_retry(memory):
    """
    UniqueViolationError must mark the operation as failed immediately, not
    raise RetryTaskAt. This is the primary symptom from #977: re-submitting
    retain caused PK collisions that the poller retried ~3 times before
    giving up. With #980's fix, the first collision fails the task.
    """
    bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
    operation_id = uuid.uuid4()

    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    await _create_pending_operation(pool, bank_id, operation_id)

    # Synthesize a real asyncpg UniqueViolationError the way the server would
    # raise it (matches the error observed in the bug report's logs).
    unique_violation = asyncpg.exceptions.UniqueViolationError(
        'duplicate key value violates unique constraint "pk_chunks"'
    )

    task_dict = {
        "type": "batch_retain",
        "operation_id": str(operation_id),
        "bank_id": bank_id,
        "contents": [{"content": "test", "document_id": "doc-1"}],
    }

    # Force _handle_batch_retain to raise the integrity error, isolating the
    # execute_task exception-classification path.
    with patch.object(memory, "_handle_batch_retain", side_effect=unique_violation):
        # Must not raise RetryTaskAt — the whole point of the fix.
        try:
            await memory.execute_task(task_dict)
        except RetryTaskAt as exc:
            pytest.fail(
                f"IntegrityConstraintViolationError must not be retried, but execute_task raised {exc!r}"
            )

    # The operation must be marked 'failed' (not left pending / retrying).
    row = await pool.fetchrow(
        "SELECT status, error_message FROM async_operations WHERE operation_id = $1",
        operation_id,
    )
    assert row is not None, "Operation row disappeared"
    assert row["status"] == "failed", (
        f"Expected status='failed' after integrity violation, got {row['status']!r}"
    )
    assert row["error_message"] is not None
    assert "pk_chunks" in row["error_message"]

    # Cleanup
    await pool.execute("DELETE FROM async_operations WHERE operation_id = $1", operation_id)
    await pool.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
async def test_foreign_key_violation_also_not_retried(memory):
    """
    All subclasses of IntegrityConstraintViolationError are non-retryable —
    verify ForeignKeyViolationError is classified the same way as
    UniqueViolationError.
    """
    bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
    operation_id = uuid.uuid4()

    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    await _create_pending_operation(pool, bank_id, operation_id)

    fk_violation = asyncpg.exceptions.ForeignKeyViolationError(
        "insert or update on table \"memory_units\" violates foreign key constraint \"fk_bank\""
    )

    task_dict = {
        "type": "batch_retain",
        "operation_id": str(operation_id),
        "bank_id": bank_id,
        "contents": [{"content": "test", "document_id": "doc-1"}],
    }

    with patch.object(memory, "_handle_batch_retain", side_effect=fk_violation):
        try:
            await memory.execute_task(task_dict)
        except RetryTaskAt as exc:
            pytest.fail(
                f"ForeignKeyViolationError must not be retried, but execute_task raised {exc!r}"
            )

    row = await pool.fetchrow(
        "SELECT status FROM async_operations WHERE operation_id = $1",
        operation_id,
    )
    assert row["status"] == "failed"

    await pool.execute("DELETE FROM async_operations WHERE operation_id = $1", operation_id)
    await pool.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
async def test_non_integrity_error_still_retried(memory):
    """
    Sanity check: non-integrity errors (network errors, timeouts, value errors)
    should STILL use the existing retry path — i.e., raise RetryTaskAt when
    ``_retry_count < 3``. Only integrity violations are the new non-retryable
    class.
    """
    bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
    operation_id = uuid.uuid4()

    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    await _create_pending_operation(pool, bank_id, operation_id)

    task_dict = {
        "type": "batch_retain",
        "operation_id": str(operation_id),
        "bank_id": bank_id,
        "contents": [{"content": "test", "document_id": "doc-1"}],
        # _retry_count = 0 (first attempt), so the existing retry path should fire.
    }

    transient_error = RuntimeError("transient connection blip")

    with patch.object(memory, "_handle_batch_retain", side_effect=transient_error):
        with pytest.raises(RetryTaskAt):
            await memory.execute_task(task_dict)

    await pool.execute("DELETE FROM async_operations WHERE operation_id = $1", operation_id)
    await pool.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)
