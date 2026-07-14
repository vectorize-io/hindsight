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
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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
            pytest.fail(f"IntegrityConstraintViolationError must not be retried, but execute_task raised {exc!r}")

    # The operation must be marked 'failed' (not left pending / retrying).
    row = await pool.fetchrow(
        "SELECT status, error_message FROM async_operations WHERE operation_id = $1",
        operation_id,
    )
    assert row is not None, "Operation row disappeared"
    assert row["status"] == "failed", f"Expected status='failed' after integrity violation, got {row['status']!r}"
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
        'insert or update on table "memory_units" violates foreign key constraint "fk_bank"'
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
            pytest.fail(f"ForeignKeyViolationError must not be retried, but execute_task raised {exc!r}")

    row = await pool.fetchrow(
        "SELECT status FROM async_operations WHERE operation_id = $1",
        operation_id,
    )
    assert row["status"] == "failed"

    await pool.execute("DELETE FROM async_operations WHERE operation_id = $1", operation_id)
    await pool.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)


@pytest.mark.parametrize(
    "message",
    [
        "embedding 0 has dimension 0; expected 384",
        "different vector dimensions 384 and 0",
    ],
)
def test_invalid_embedding_dimension_error_is_non_retryable(message):
    """Embedding dimension mismatches are deterministic and must not be retried.

    PR #1670 validates empty/mismatched embedding vectors before pgvector writes.
    pgvector may also raise its own dimension-mismatch error if an invalid vector
    reaches the database layer. In both cases, rerunning the same poisoned
    embedding response only burns worker slots; a fresh retain request or fixed
    embedding backend is required.
    """
    from hindsight_api.engine.memory_engine import _is_non_retryable_task_error

    assert _is_non_retryable_task_error(RuntimeError(message)) is True


def _fake_config() -> SimpleNamespace:
    """Minimal config for _execute_update_action: no native search_vector clause,
    observation-history enabled so the guard is the only thing preventing the
    (FK-violating) history INSERT."""
    return SimpleNamespace(
        text_search_extension="none",
        text_search_extension_native_language="english",
        enable_observation_history=True,
        observation_history_max_entries=10,
    )


def _observation_fact(observation_id: str):
    from hindsight_api.engine.response_models import MemoryFact

    return MemoryFact(
        id=observation_id,
        text="old observation text",
        fact_type="observation",
        tags=["scope_a"],
        source_fact_ids=[],
    )


@pytest.mark.asyncio
async def test_update_action_bails_when_observation_row_missing():
    """
    Regression: when the observation's memory_units row was concurrently
    deleted/invalidated, the ``UPDATE memory_units`` matches 0 rows. The prior
    code ignored the rowcount and still called ``_append_observation_history``,
    whose INSERT carries an ``observation_id`` FK onto memory_units — raising a
    ForeignKeyViolationError (orphan history / integrity failure).

    The fix returns None on rowcount==0 BEFORE the history append. This test
    asserts the guard fires and no history is written.
    """
    from hindsight_api.engine.consolidation import consolidator

    observation_id = str(uuid.uuid4())
    source_ids = [uuid.uuid4()]

    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="UPDATE 0")  # 0 rows matched

    memory_engine = MagicMock()
    memory_engine.embeddings = MagicMock()

    with (
        patch("hindsight_api.config.get_config", _fake_config),
        patch.object(
            consolidator, "_filter_live_source_memories", AsyncMock(return_value=source_ids)
        ),
        patch.object(
            consolidator.embedding_utils,
            "generate_embeddings_batch",
            AsyncMock(return_value=[[0.1, 0.2, 0.3]]),
        ),
        patch.object(consolidator, "_append_observation_history", AsyncMock()) as append_mock,
    ):
        result = await consolidator._execute_update_action(
            conn=conn,
            memory_engine=memory_engine,
            bank_id="bank-x",
            source_memory_ids=source_ids,
            observation_id=observation_id,
            new_text="new observation text",
            observations=[_observation_fact(observation_id)],
            source_fact_tags=["scope_b"],
        )

    assert result is None, "Expected None when the observation row no longer exists"
    append_mock.assert_not_called()  # the FK-violating INSERT must be skipped


@pytest.mark.asyncio
async def test_update_action_writes_history_when_row_present():
    """Positive control: when the UPDATE matches a row (rowcount==1), the guard
    must NOT interfere — observation_history is appended as before."""
    from hindsight_api.engine.consolidation import consolidator

    observation_id = str(uuid.uuid4())
    source_ids = [uuid.uuid4()]

    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="UPDATE 1")  # 1 row matched

    memory_engine = MagicMock()
    memory_engine.embeddings = MagicMock()
    memory_engine._backend.ops.uses_observation_sources_table = False

    with (
        patch("hindsight_api.config.get_config", _fake_config),
        patch.object(
            consolidator, "_filter_live_source_memories", AsyncMock(return_value=source_ids)
        ),
        patch.object(
            consolidator.embedding_utils,
            "generate_embeddings_batch",
            AsyncMock(return_value=[[0.1, 0.2, 0.3]]),
        ),
        patch.object(consolidator, "_append_observation_history", AsyncMock()) as append_mock,
    ):
        result = await consolidator._execute_update_action(
            conn=conn,
            memory_engine=memory_engine,
            bank_id="bank-x",
            source_memory_ids=source_ids,
            observation_id=observation_id,
            new_text="new observation text",
            observations=[_observation_fact(observation_id)],
            source_fact_tags=["scope_b"],
        )

    assert result is not None, "Expected the embedding string back on a successful update"
    append_mock.assert_called_once()


@pytest.mark.asyncio
async def test_non_integrity_error_still_retried(memory):
    """
    Sanity check: non-integrity errors (network errors, timeouts, value errors)
    should STILL use the existing retry path — i.e., raise RetryTaskAt when
    ``_retry_count < 3``. Only deterministic task errors are non-retryable.
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
