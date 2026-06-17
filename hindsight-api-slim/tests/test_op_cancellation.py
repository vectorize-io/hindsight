"""Tests for operation cancellation when a bank is deleted.

Covers:
- CASCADE DELETE: deleting a bank removes async_operations and webhooks rows
- _check_op_alive: returns True when op exists, False when deleted
- _mark_operation_completed / _mark_operation_failed: graceful no-op when row is gone
- Consolidation checkpoint: stops early after a batch commit if op was deleted
- Retain checkpoint: stops between sub-batches if op was deleted
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from hindsight_api.engine.memory_engine import MemoryEngine


pytestmark = pytest.mark.xdist_group("op_cancellation_tests")

_BANK_PREFIX = "test-op-cancel"


@pytest_asyncio.fixture
async def pool(pg0_db_url):
    import asyncpg
    from hindsight_api.pg0 import resolve_database_url

    resolved_url = await resolve_database_url(pg0_db_url)
    p = await asyncpg.create_pool(resolved_url, min_size=1, max_size=5, command_timeout=30)
    yield p
    await p.close()


@pytest_asyncio.fixture(autouse=True)
async def cleanup(pool):
    """Remove test rows before and after each test."""
    await pool.execute(f"DELETE FROM banks WHERE bank_id LIKE '{_BANK_PREFIX}%'")
    yield
    await pool.execute(f"DELETE FROM banks WHERE bank_id LIKE '{_BANK_PREFIX}%'")


async def _insert_bank(pool, bank_id: str):
    await pool.execute(
        "INSERT INTO banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        bank_id,
        bank_id,
    )


async def _insert_op(pool, bank_id: str, op_id: uuid.UUID | None = None) -> uuid.UUID:
    op_id = op_id or uuid.uuid4()
    await pool.execute(
        """
        INSERT INTO async_operations (operation_id, bank_id, operation_type, status)
        VALUES ($1, $2, 'consolidation', 'processing')
        """,
        op_id,
        bank_id,
    )
    return op_id


# ---------------------------------------------------------------------------
# CASCADE DELETE tests
# ---------------------------------------------------------------------------


class TestCascadeDeleteOnBankDeletion:
    @pytest.mark.asyncio
    async def test_bank_deletion_cascades_to_async_operations(self, pool):
        bank_id = f"{_BANK_PREFIX}-{uuid.uuid4().hex[:8]}"
        await _insert_bank(pool, bank_id)
        op_id = await _insert_op(pool, bank_id)

        # Verify op exists
        row = await pool.fetchrow("SELECT operation_id FROM async_operations WHERE operation_id = $1", op_id)
        assert row is not None

        # Delete the bank — should cascade to async_operations
        await pool.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)

        row = await pool.fetchrow("SELECT operation_id FROM async_operations WHERE operation_id = $1", op_id)
        assert row is None, "async_operations row should be deleted by CASCADE"

    @pytest.mark.asyncio
    async def test_bank_deletion_cascades_to_webhooks(self, pool):
        bank_id = f"{_BANK_PREFIX}-{uuid.uuid4().hex[:8]}"
        await _insert_bank(pool, bank_id)
        webhook_id = uuid.uuid4()
        await pool.execute(
            """
            INSERT INTO webhooks (id, bank_id, url, event_types)
            VALUES ($1, $2, 'https://example.com/hook', '{}')
            """,
            webhook_id,
            bank_id,
        )

        row = await pool.fetchrow("SELECT id FROM webhooks WHERE id = $1", webhook_id)
        assert row is not None

        await pool.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)

        row = await pool.fetchrow("SELECT id FROM webhooks WHERE id = $1", webhook_id)
        assert row is None, "webhooks row should be deleted by CASCADE"


# ---------------------------------------------------------------------------
# _check_op_alive tests
# ---------------------------------------------------------------------------


class TestCheckOpAlive:
    @pytest.mark.asyncio
    async def test_returns_true_when_op_exists(self, memory: MemoryEngine, request_context):
        bank_id = f"{_BANK_PREFIX}-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        op_id = uuid.uuid4()
        async with memory._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status)
                VALUES ($1, $2, 'consolidation', 'processing')
                """,
                op_id,
                bank_id,
            )

        assert await memory._check_op_alive(str(op_id)) is True

    @pytest.mark.asyncio
    async def test_returns_false_when_op_deleted(self, memory: MemoryEngine, request_context):
        bank_id = f"{_BANK_PREFIX}-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        op_id = uuid.uuid4()
        async with memory._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status)
                VALUES ($1, $2, 'consolidation', 'processing')
                """,
                op_id,
                bank_id,
            )
            await conn.execute("DELETE FROM async_operations WHERE operation_id = $1", op_id)

        assert await memory._check_op_alive(str(op_id)) is False

    @pytest.mark.asyncio
    async def test_returns_false_after_bank_cascade_delete(self, memory: MemoryEngine, request_context):
        bank_id = f"{_BANK_PREFIX}-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        op_id = uuid.uuid4()
        async with memory._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status)
                VALUES ($1, $2, 'consolidation', 'processing')
                """,
                op_id,
                bank_id,
            )

        # Delete the bank — cascades to the op row
        await memory.delete_bank(bank_id=bank_id, request_context=request_context)

        assert await memory._check_op_alive(str(op_id)) is False


# ---------------------------------------------------------------------------
# _mark_operation_completed / _mark_operation_failed graceful no-op
# ---------------------------------------------------------------------------


class TestMarkOperationGracefulOnMissingRow:
    @pytest.mark.asyncio
    async def test_mark_completed_does_not_raise_when_row_missing(self, memory: MemoryEngine):
        # Row never existed — should log and return cleanly
        missing_id = str(uuid.uuid4())
        await memory._mark_operation_completed(missing_id)  # no exception

    @pytest.mark.asyncio
    async def test_mark_failed_does_not_raise_when_row_missing(self, memory: MemoryEngine):
        missing_id = str(uuid.uuid4())
        await memory._mark_operation_failed(missing_id, "some error", "traceback here")  # no exception

    @pytest.mark.asyncio
    async def test_mark_completed_and_fire_webhook_does_not_raise_when_row_missing(self, memory: MemoryEngine):
        missing_id = str(uuid.uuid4())
        await memory._mark_operation_completed_and_fire_webhook(
            operation_id=missing_id,
            bank_id="nonexistent-bank",
            status="completed",
            result=None,
        )  # no exception


# ---------------------------------------------------------------------------
# Consolidation checkpoint
# ---------------------------------------------------------------------------


class TestConsolidationCheckpoint:
    @pytest.mark.asyncio
    async def test_consolidation_stops_early_when_op_cancelled(self, memory: MemoryEngine, request_context):
        """Consolidation returns 'cancelled' status after the first batch if _check_op_alive is False."""
        from hindsight_api.config import _get_raw_config
        from hindsight_api.engine.consolidation.consolidator import run_consolidation_job

        config = _get_raw_config()
        original = config.enable_observations
        config.enable_observations = True

        try:
            bank_id = f"{_BANK_PREFIX}-{uuid.uuid4().hex[:8]}"
            await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

            # Insert a few unconsolidated memories directly so we control the batch without LLM
            async with memory._pool.acquire() as conn:
                for i in range(3):
                    await conn.execute(
                        """
                        INSERT INTO memory_units
                            (id, bank_id, text, fact_type, created_at, updated_at)
                        VALUES (gen_random_uuid(), $1, $2, 'experience', NOW(), NOW())
                        """,
                        bank_id,
                        f"Test memory {i} for cancellation test",
                    )

            op_id = str(uuid.uuid4())
            call_count = 0

            async def _fake_check(operation_id: str) -> bool:
                nonlocal call_count
                call_count += 1
                # Return False on the very first checkpoint call
                return False

            with patch.object(memory, "_check_op_alive", side_effect=_fake_check):
                result = await run_consolidation_job(
                    memory_engine=memory,
                    bank_id=bank_id,
                    request_context=request_context,
                    operation_id=op_id,
                )

            assert result["status"] == "cancelled"
            assert call_count >= 1
        finally:
            config.enable_observations = original


# ---------------------------------------------------------------------------
# Retain checkpoint
# ---------------------------------------------------------------------------


class TestRetainCheckpoint:
    @pytest.mark.asyncio
    async def test_retain_stops_between_sub_batches_when_cancelled(self, memory: MemoryEngine, request_context):
        """retain_batch_async returns partial results if _check_op_alive is False between sub-batches."""
        from hindsight_api.config import _get_raw_config

        bank_id = f"{_BANK_PREFIX}-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Force sub-batch splitting by temporarily lowering the token threshold
        config = _get_raw_config()
        original_tokens = config.retain_batch_tokens
        # Set threshold very low so each item becomes its own sub-batch
        config.retain_batch_tokens = 1

        try:
            op_id = str(uuid.uuid4())
            check_calls = 0

            async def _fake_check(operation_id: str) -> bool:
                nonlocal check_calls
                check_calls += 1
                # Cancel after the first sub-batch completes
                return check_calls <= 1

            contents = [{"content": f"Memory item {i} about something interesting."} for i in range(4)]

            with patch.object(memory, "_check_op_alive", side_effect=_fake_check):
                result = await memory.retain_batch_async(
                    bank_id=bank_id,
                    contents=contents,
                    request_context=request_context,
                    operation_id=op_id,
                )

            # Public contract change in #1571: ``retain_batch_async`` now
            # always returns one slot per input content. Un-processed
            # inputs (because of cancellation between sub-batches) come
            # back as empty lists instead of being omitted from the
            # result. The cancellation check still has to short-circuit
            # — assert that fewer than all inputs produced unit_ids.
            assert len(result) == len(contents), (
                f"Expected per-input result list (len={len(contents)}), got {len(result)}"
            )
            non_empty = [r for r in result if r]
            assert len(non_empty) < len(contents), (
                f"Expected early stop (fewer non-empty results than inputs), got {non_empty}"
            )
            assert check_calls >= 1
        finally:
            config.retain_batch_tokens = original_tokens
