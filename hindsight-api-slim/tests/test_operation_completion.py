"""Unit tests for MemoryEngine operation-completion side-effect isolation.

These are fast, DB-free tests (fake asyncpg-style connections) that pin the
critical property fixed for issue #2601: a failure in the best-effort side
effects of completing an operation (webhook outbox insert, parent aggregation)
must never roll back — or silently swallow — the completion itself, and the
consolidation webhook must still be delivered on the fallback path.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.worker.claim import OperationClaim, bind_operation_claim


class _FakeTx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeConn:
    def __init__(self, fetchrow_result):
        self._fetchrow_result = fetchrow_result
        self.fetchrow_calls: list[tuple[str, tuple]] = []
        self.execute_calls: list[tuple[str, tuple]] = []

    def transaction(self):
        return _FakeTx()

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append((query, args))
        if isinstance(self._fetchrow_result, list):
            return self._fetchrow_result.pop(0)
        return self._fetchrow_result

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))
        return "DELETE 0"


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeBackend:
    # Route acquire_with_retry down its DatabaseBackend branch without importing one.
    _wraps_backend = True

    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _FakeAcquire(self._conn)


def _make_engine(fetchrow_result, *, webhook_raises: bool):
    """Build a MemoryEngine with only the attributes the completion path touches."""
    conn = _FakeConn(fetchrow_result)
    engine = MemoryEngine.__new__(MemoryEngine)
    engine._get_backend = AsyncMock(return_value=_FakeBackend(conn))
    engine._maybe_update_parent_operation = AsyncMock()
    engine._fire_consolidation_webhook = AsyncMock()

    webhook_manager = MagicMock()
    if webhook_raises:
        webhook_manager.fire_event_with_conn = AsyncMock(side_effect=RuntimeError("outbox insert failed"))
    else:
        webhook_manager.fire_event_with_conn = AsyncMock()
    webhook_manager.fire_event = AsyncMock()
    engine._webhook_manager = webhook_manager
    return engine, conn


class TestMarkOperationCompletedAndFireWebhook:
    async def test_happy_path_uses_outbox_and_does_not_double_fire(self):
        """When the atomic outbox transaction succeeds, the best-effort
        (non-transactional) webhook fire must NOT run — that would duplicate delivery."""
        op_id = str(uuid.uuid4())
        engine, conn = _make_engine({"operation_id": op_id}, webhook_raises=False)

        await engine._mark_operation_completed_and_fire_webhook(
            operation_id=op_id, bank_id="bank-1", status="completed", result={"observations_created": 2}
        )

        # Completion committed exactly once, guarded on 'processing'.
        assert len(conn.fetchrow_calls) == 1
        assert "NOT IN ('completed', 'failed', 'cancelled')" in conn.fetchrow_calls[0][0]
        engine._webhook_manager.fire_event_with_conn.assert_awaited_once()
        engine._fire_consolidation_webhook.assert_not_awaited()

    async def test_webhook_failure_falls_back_to_completion_and_best_effort_webhook(self):
        """If the outbox transaction fails, the operation must still be completed and
        the webhook delivered best-effort — not left stuck in 'processing' (issue #2601)."""
        op_id = str(uuid.uuid4())
        engine, conn = _make_engine({"operation_id": op_id}, webhook_raises=True)

        await engine._mark_operation_completed_and_fire_webhook(
            operation_id=op_id, bank_id="bank-1", status="completed", result=None
        )

        # Two completion UPDATEs: the rolled-back happy path + the fallback commit.
        assert len(conn.fetchrow_calls) == 2
        for query, _args in conn.fetchrow_calls:
            assert "NOT IN ('completed', 'failed', 'cancelled')" in query
        # Best-effort webhook fired exactly once on the fallback path.
        engine._fire_consolidation_webhook.assert_awaited_once()
        _, kwargs = engine._fire_consolidation_webhook.await_args
        assert kwargs["operation_id"] == op_id
        assert kwargs["status"] == "completed"

    async def test_already_terminal_row_is_a_noop(self):
        """Idempotency with the poller backstop (PR #2608): if the row is no longer
        'processing', do nothing — no parent aggregation, no webhook."""
        op_id = str(uuid.uuid4())
        engine, conn = _make_engine(None, webhook_raises=False)

        await engine._mark_operation_completed_and_fire_webhook(
            operation_id=op_id, bank_id="bank-1", status="completed", result=None
        )

        assert len(conn.fetchrow_calls) == 1
        engine._maybe_update_parent_operation.assert_not_awaited()
        engine._webhook_manager.fire_event_with_conn.assert_not_awaited()
        engine._fire_consolidation_webhook.assert_not_awaited()


class TestOperationTerminalClaimFences:
    async def test_unknown_task_delete_fences_on_exact_claim(self):
        op_id = str(uuid.uuid4())
        engine, conn = _make_engine(None, webhook_raises=False)
        claim = OperationClaim(worker_id="worker-a", claimed_at=datetime(2026, 7, 21, 8, 0, tzinfo=UTC))

        with bind_operation_claim(claim):
            await engine._delete_operation_record(op_id)

        query, args = conn.fetchrow_calls[0]
        assert "status = 'processing'" in query
        assert "worker_id = $2" in query
        assert "claimed_at = $3" in query
        assert args == (uuid.UUID(op_id), claim.worker_id, claim.claimed_at)

    async def test_mark_failed_fences_on_exact_claim(self):
        op_id = str(uuid.uuid4())
        engine, conn = _make_engine(None, webhook_raises=False)
        claim = OperationClaim(worker_id="worker-a", claimed_at=datetime(2026, 7, 21, 8, 0, tzinfo=UTC))

        with bind_operation_claim(claim):
            await engine._mark_operation_failed(op_id, "late failure", "traceback")

        query, args = conn.fetchrow_calls[0]
        assert "status = 'processing'" in query
        assert "worker_id = $3" in query
        assert "claimed_at = $4" in query
        assert args[0] == uuid.UUID(op_id)
        assert args[2:] == (claim.worker_id, claim.claimed_at)
        engine._maybe_update_parent_operation.assert_not_awaited()

    async def test_mark_completed_fences_on_exact_claim(self):
        op_id = str(uuid.uuid4())
        engine, conn = _make_engine(None, webhook_raises=False)
        claim = OperationClaim(worker_id="worker-a", claimed_at=datetime(2026, 7, 21, 8, 0, tzinfo=UTC))

        with bind_operation_claim(claim):
            await engine._mark_operation_completed(op_id)

        query, args = conn.fetchrow_calls[1]
        assert "status = 'processing'" in query
        assert "worker_id = $2" in query
        assert "claimed_at = $3" in query
        assert args == (uuid.UUID(op_id), claim.worker_id, claim.claimed_at)
        engine._maybe_update_parent_operation.assert_not_awaited()

    async def test_lost_non_retryable_failure_does_not_fire_consolidation_webhook(self):
        op_id = str(uuid.uuid4())
        engine, _conn = _make_engine({"status": "processing"}, webhook_raises=False)
        engine._audit_logger = None
        engine._ext_ctx = MagicMock()
        engine._handle_consolidation = AsyncMock(side_effect=ValueError("embedding 0 has dimension 0; expected 384"))
        engine._mark_operation_failed = AsyncMock(return_value=False)
        claim = OperationClaim(worker_id="worker-a", claimed_at=datetime(2026, 7, 21, 8, 0, tzinfo=UTC))

        with bind_operation_claim(claim):
            await engine.execute_task(
                {
                    "type": "consolidation",
                    "operation_id": op_id,
                    "bank_id": "bank-1",
                }
            )

        engine._mark_operation_failed.assert_awaited_once()
        engine._fire_consolidation_webhook.assert_not_awaited()

    async def test_lost_transient_failure_does_not_queue_consolidation_webhook(self):
        from hindsight_api.worker.exceptions import RetryTaskAt

        op_id = str(uuid.uuid4())
        engine, conn = _make_engine([{"status": "processing"}, None], webhook_raises=False)
        engine._audit_logger = None
        engine._ext_ctx = MagicMock()
        engine._handle_consolidation = AsyncMock(side_effect=RuntimeError("transient upstream failure"))
        engine._has_other_pending_consolidation = AsyncMock(return_value=False)
        engine._fire_consolidation_webhook = MemoryEngine._fire_consolidation_webhook.__get__(engine)
        claim = OperationClaim(worker_id="worker-a", claimed_at=datetime(2026, 7, 21, 8, 0, tzinfo=UTC))

        with bind_operation_claim(claim), pytest.raises(RetryTaskAt):
            await engine.execute_task(
                {
                    "type": "consolidation",
                    "operation_id": op_id,
                    "bank_id": "bank-1",
                }
            )

        claim_query, claim_args = conn.fetchrow_calls[1]
        assert "status = $2" in claim_query
        assert "worker_id = $3" in claim_query
        assert "claimed_at = $4" in claim_query
        assert claim_args == (uuid.UUID(op_id), "processing", claim.worker_id, claim.claimed_at)
        engine._webhook_manager.fire_event_with_conn.assert_not_awaited()
        engine._webhook_manager.fire_event.assert_not_awaited()

    async def test_current_claim_queues_consolidation_webhook_in_fenced_transaction(self):
        op_id = str(uuid.uuid4())
        engine, conn = _make_engine({"operation_id": op_id}, webhook_raises=False)
        engine._fire_consolidation_webhook = MemoryEngine._fire_consolidation_webhook.__get__(engine)
        claim = OperationClaim(worker_id="worker-a", claimed_at=datetime(2026, 7, 21, 8, 0, tzinfo=UTC))

        with bind_operation_claim(claim):
            await engine._fire_consolidation_webhook(
                bank_id="bank-1",
                operation_id=op_id,
                status="failed",
                operation_status="processing",
                result=None,
                error_message="transient upstream failure",
            )

        query, args = conn.fetchrow_calls[0]
        assert "status = $2" in query
        assert "worker_id = $3" in query
        assert "claimed_at = $4" in query
        assert args == (uuid.UUID(op_id), "processing", claim.worker_id, claim.claimed_at)
        event, event_conn = engine._webhook_manager.fire_event_with_conn.await_args.args
        assert event.operation_id == op_id
        assert event.status == "failed"
        assert event_conn is conn
        engine._webhook_manager.fire_event.assert_not_awaited()

    async def test_lost_claim_does_not_queue_webhook_or_update_parent(self):
        op_id = str(uuid.uuid4())
        engine, conn = _make_engine(None, webhook_raises=False)
        claim = OperationClaim(worker_id="worker-a", claimed_at=datetime(2026, 7, 21, 8, 0, tzinfo=UTC))

        with bind_operation_claim(claim):
            await engine._mark_operation_completed_and_fire_webhook(
                operation_id=op_id,
                bank_id="bank-1",
                status="completed",
                result=None,
            )

        query, args = conn.fetchrow_calls[0]
        assert "status = 'processing'" in query
        assert "worker_id = $2" in query
        assert "claimed_at = $3" in query
        assert args == (uuid.UUID(op_id), claim.worker_id, claim.claimed_at)
        engine._maybe_update_parent_operation.assert_not_awaited()
        engine._webhook_manager.fire_event_with_conn.assert_not_awaited()
        engine._fire_consolidation_webhook.assert_not_awaited()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
