"""Tests for the webhook system.

Covers:
- Unit tests for HMAC signing and retry constants (no DB required)
- Integration tests for fire_event() using a real DB (inserts into async_operations)
- Integration tests for _handle_webhook_delivery() on the memory engine
- HTTP API integration tests for CRUD and delivery listing endpoints
"""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from hindsight_api import LLMConfig
from hindsight_api.api import create_app
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.webhooks.manager import MAX_ATTEMPTS, RETRY_DELAYS, WebhookManager
from hindsight_api.webhooks.models import (
    ConsolidationEventData,
    MemoryDefenseEventData,
    MemoryDefenseHit,
    RetainEventData,
    WebhookConfig,
    WebhookEvent,
    WebhookEventType,
)
from hindsight_api.worker.exceptions import RetryTaskAt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(bank_id: str = "bank-1") -> WebhookEvent:
    return WebhookEvent(
        event=WebhookEventType.CONSOLIDATION_COMPLETED,
        bank_id=bank_id,
        operation_id=uuid.uuid4().hex,
        status="completed",
        timestamp=datetime.now(timezone.utc),
        data=ConsolidationEventData(observations_created=1),
    )


def _make_delivery_task(
    bank_id: str = "bank-1",
    url: str = "https://example.com/hook",
    retry_count: int = 0,
    webhook_id: str | None = None,
) -> dict:
    return {
        "type": "webhook_delivery",
        "bank_id": bank_id,
        "url": url,
        "secret": None,
        "event_type": "consolidation.completed",
        "payload": '{"event":"consolidation.completed"}',
        "webhook_id": webhook_id,
        "_retry_count": retry_count,
    }


# ---------------------------------------------------------------------------
# Unit tests (no DB)
# ---------------------------------------------------------------------------


class TestHmacSigning:
    """Unit tests for WebhookManager._sign_payload()."""

    def _make_manager(self) -> WebhookManager:
        """Create a WebhookManager with a dummy pool (not used for signing)."""
        pool = MagicMock()
        return WebhookManager(backend=pool, global_webhooks=[])

    def test_hmac_signing_format(self):
        """_sign_payload should return a string starting with 'sha256='."""
        manager = self._make_manager()
        sig = manager._sign_payload("my-secret", b"hello world")
        assert sig.startswith("sha256="), f"Expected 'sha256=' prefix, got: {sig!r}"
        hex_part = sig[len("sha256=") :]
        # SHA-256 hex digest is always 64 characters
        assert len(hex_part) == 64
        # Hex characters only
        assert all(c in "0123456789abcdef" for c in hex_part)

    def test_hmac_signing_is_deterministic(self):
        """Same secret + payload always produces the same signature."""
        manager = self._make_manager()
        payload = b'{"event":"consolidation.completed"}'
        sig1 = manager._sign_payload("secret-key", payload)
        sig2 = manager._sign_payload("secret-key", payload)
        assert sig1 == sig2

    def test_hmac_signing_differs_with_different_secret(self):
        """Different secrets must produce different signatures."""
        manager = self._make_manager()
        payload = b"payload"
        sig1 = manager._sign_payload("secret-a", payload)
        sig2 = manager._sign_payload("secret-b", payload)
        assert sig1 != sig2

    def test_hmac_signing_differs_with_different_payload(self):
        """Different payloads must produce different signatures."""
        manager = self._make_manager()
        sig1 = manager._sign_payload("secret", b"payload-one")
        sig2 = manager._sign_payload("secret", b"payload-two")
        assert sig1 != sig2


class TestRetryConstants:
    """Unit tests to verify retry schedule constants."""

    def test_retry_delays_values(self):
        """RETRY_DELAYS must match the documented schedule."""
        assert RETRY_DELAYS == [5, 300, 1800, 7200, 18000]

    def test_max_attempts(self):
        """MAX_ATTEMPTS should be len(RETRY_DELAYS) + 1."""
        assert MAX_ATTEMPTS == 6
        assert MAX_ATTEMPTS == len(RETRY_DELAYS) + 1


# ---------------------------------------------------------------------------
# DB integration tests
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def webhook_manager(memory: MemoryEngine) -> WebhookManager:
    """Return a WebhookManager backed by the test pool with no global webhooks."""
    return WebhookManager(backend=memory._backend, global_webhooks=[])


async def _ensure_bank(pool, bank_id: str) -> None:
    """Upsert a minimal bank row so FK constraints on async_operations/webhooks pass."""
    await pool.execute(
        "INSERT INTO banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        bank_id,
        bank_id,
    )


class TestFireEvent:
    """Integration tests for WebhookManager.fire_event()."""

    @pytest.mark.asyncio
    async def test_fire_event_creates_delivery(self, memory: MemoryEngine, webhook_manager: WebhookManager):
        """fire_event() inserts a pending webhook_delivery task in async_operations."""
        bank_id = f"wh-test-{uuid.uuid4().hex[:8]}"
        webhook_id = uuid.uuid4()

        async with memory._pool.acquire() as conn:
            await _ensure_bank(memory._pool, bank_id)
            await conn.execute(
                """
                INSERT INTO webhooks (id, bank_id, url, secret, event_types, enabled, created_at, updated_at)
                VALUES ($1, $2, $3, NULL, $4, true, NOW(), NOW())
                """,
                webhook_id,
                bank_id,
                "https://example.com/hook",
                ["consolidation.completed"],
            )

        try:
            event = _make_event(bank_id)
            await webhook_manager.fire_event(event)

            async with memory._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT status, task_payload
                    FROM async_operations
                    WHERE operation_type = 'webhook_delivery'
                      AND bank_id = $1
                      AND task_payload->>'webhook_id' = $2
                    """,
                    bank_id,
                    str(webhook_id),
                )

            assert len(rows) == 1
            assert rows[0]["status"] == "pending"
            payload = rows[0]["task_payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            assert payload["event_type"] == "consolidation.completed"
        finally:
            async with memory._pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM async_operations WHERE operation_type = 'webhook_delivery' AND bank_id = $1",
                    bank_id,
                )
                await conn.execute("DELETE FROM webhooks WHERE id = $1", webhook_id)

    @pytest.mark.asyncio
    async def test_fire_event_global_webhook(self, memory: MemoryEngine):
        """fire_event() also queues delivery tasks for global webhooks (not stored in DB)."""
        bank_id = f"wh-global-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory._pool, bank_id)
        global_webhook = WebhookConfig(
            id="",  # No DB row
            bank_id=None,
            url="https://global.example.com/hook",
            secret=None,
            event_types=["consolidation.completed"],
            enabled=True,
        )
        manager = WebhookManager(backend=memory._backend, global_webhooks=[global_webhook])

        event = _make_event(bank_id)
        await manager.fire_event(event)

        async with memory._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT status, task_payload
                FROM async_operations
                WHERE operation_type = 'webhook_delivery'
                  AND bank_id = $1
                  AND task_payload->>'url' = 'https://global.example.com/hook'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                bank_id,
            )

        assert len(rows) == 1
        assert rows[0]["status"] == "pending"
        payload = rows[0]["task_payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        assert payload["webhook_id"] is None  # global webhook has no DB row

        # Cleanup
        async with memory._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM async_operations WHERE operation_type = 'webhook_delivery' AND bank_id = $1",
                bank_id,
            )

    @pytest.mark.asyncio
    async def test_fire_event_no_match_if_event_type_mismatch(
        self, memory: MemoryEngine, webhook_manager: WebhookManager
    ):
        """Webhooks registered for a different event type receive no delivery task."""
        bank_id = f"wh-mismatch-{uuid.uuid4().hex[:8]}"
        webhook_id = uuid.uuid4()

        await _ensure_bank(memory._pool, bank_id)
        async with memory._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO webhooks (id, bank_id, url, secret, event_types, enabled, created_at, updated_at)
                VALUES ($1, $2, $3, NULL, $4, true, NOW(), NOW())
                """,
                webhook_id,
                bank_id,
                "https://example.com/other-hook",
                ["other.event"],
            )

        try:
            event = _make_event(bank_id)
            await webhook_manager.fire_event(event)

            async with memory._pool.acquire() as conn:
                count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM async_operations
                    WHERE operation_type = 'webhook_delivery' AND bank_id = $1
                    """,
                    bank_id,
                )

            assert count == 0
        finally:
            async with memory._pool.acquire() as conn:
                await conn.execute("DELETE FROM webhooks WHERE id = $1", webhook_id)


class TestFireEventWithConn:
    """Integration tests for WebhookManager.fire_event_with_conn()."""

    @pytest.mark.asyncio
    async def test_fire_event_with_conn_queues_delivery(self, memory: MemoryEngine, webhook_manager: WebhookManager):
        """fire_event_with_conn() inserts a delivery task using the provided connection."""
        bank_id = f"wh-conn-{uuid.uuid4().hex[:8]}"
        webhook_id = uuid.uuid4()

        await _ensure_bank(memory._pool, bank_id)
        async with memory._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO webhooks (id, bank_id, url, secret, event_types, enabled, created_at, updated_at)
                VALUES ($1, $2, $3, NULL, $4, true, NOW(), NOW())
                """,
                webhook_id,
                bank_id,
                "https://example.com/conn-hook",
                ["consolidation.completed"],
            )

        try:
            event = _make_event(bank_id)
            # Use fire_event_with_conn inside a transaction
            async with memory._backend.acquire() as conn:
                async with conn.transaction():
                    await webhook_manager.fire_event_with_conn(event, conn)

            async with memory._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT status, task_payload
                    FROM async_operations
                    WHERE operation_type = 'webhook_delivery'
                      AND bank_id = $1
                      AND task_payload->>'webhook_id' = $2
                    """,
                    bank_id,
                    str(webhook_id),
                )

            assert len(rows) == 1
            assert rows[0]["status"] == "pending"
        finally:
            async with memory._pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM async_operations WHERE operation_type = 'webhook_delivery' AND bank_id = $1",
                    bank_id,
                )
                await conn.execute("DELETE FROM webhooks WHERE id = $1", webhook_id)

    @pytest.mark.asyncio
    async def test_fire_event_with_conn_rolls_back_on_transaction_abort(
        self, memory: MemoryEngine, webhook_manager: WebhookManager
    ):
        """When the enclosing transaction rolls back, the delivery row is also rolled back.

        This is the key property of fire_event_with_conn vs fire_event: using the
        caller's connection means the delivery insert is atomic with the caller's work.
        """
        bank_id = f"wh-rollback-{uuid.uuid4().hex[:8]}"
        webhook_id = uuid.uuid4()

        await _ensure_bank(memory._pool, bank_id)
        async with memory._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO webhooks (id, bank_id, url, secret, event_types, enabled, created_at, updated_at)
                VALUES ($1, $2, $3, NULL, $4, true, NOW(), NOW())
                """,
                webhook_id,
                bank_id,
                "https://example.com/rollback-hook",
                ["consolidation.completed"],
            )

        try:
            event = _make_event(bank_id)

            # Fire inside a transaction that we explicitly roll back.
            # Use the raw asyncpg pool to get manual transaction control.
            async with memory._pool.acquire() as raw_conn:
                tx = raw_conn.transaction()
                await tx.start()
                await webhook_manager.fire_event_with_conn(event, raw_conn)
                await tx.rollback()

            # The delivery row should NOT exist because the transaction was rolled back
            async with memory._pool.acquire() as conn:
                count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM async_operations
                    WHERE operation_type = 'webhook_delivery' AND bank_id = $1
                    """,
                    bank_id,
                )
            assert count == 0, f"Expected 0 delivery rows after rollback, got {count}"
        finally:
            async with memory._pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM async_operations WHERE operation_type = 'webhook_delivery' AND bank_id = $1",
                    bank_id,
                )
                await conn.execute("DELETE FROM webhooks WHERE id = $1", webhook_id)


class TestHandleWebhookDelivery:
    """Integration tests for MemoryEngine._handle_webhook_delivery()."""

    @pytest.mark.asyncio
    async def test_deliver_success(self, memory: MemoryEngine):
        """A successful HTTP POST completes without raising."""
        task_dict = _make_delivery_task(retry_count=0)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch.object(memory._http_client, "post", new=AsyncMock(return_value=mock_response)):
            # Should not raise
            await memory._handle_webhook_delivery(task_dict)

    @pytest.mark.asyncio
    async def test_deliver_failure_raises_retry_task_at(self, memory: MemoryEngine):
        """A failed HTTP POST raises RetryTaskAt when retries remain."""
        task_dict = _make_delivery_task(retry_count=0)

        with patch.object(memory._http_client, "post", new=AsyncMock(side_effect=Exception("connection refused"))):
            with pytest.raises(RetryTaskAt):
                await memory._handle_webhook_delivery(task_dict)

    @pytest.mark.asyncio
    async def test_deliver_exhausted_retries_raises(self, memory: MemoryEngine):
        """When retry_count reaches MAX_ATTEMPTS-1, a failure raises the original exception."""
        task_dict = _make_delivery_task(retry_count=MAX_ATTEMPTS - 1)

        with patch.object(memory._http_client, "post", new=AsyncMock(side_effect=Exception("server error"))):
            with pytest.raises(Exception, match="server error"):
                await memory._handle_webhook_delivery(task_dict)

    @pytest.mark.asyncio
    async def test_deliver_retry_at_uses_delay_schedule(self, memory: MemoryEngine):
        """RetryTaskAt.retry_at is approximately now + RETRY_DELAYS[retry_count]."""
        from datetime import timedelta

        task_dict = _make_delivery_task(retry_count=1)

        with patch.object(memory._http_client, "post", new=AsyncMock(side_effect=Exception("fail"))):
            before = datetime.now(timezone.utc)
            with pytest.raises(RetryTaskAt) as exc_info:
                await memory._handle_webhook_delivery(task_dict)
            after = datetime.now(timezone.utc)

        retry_at = exc_info.value.retry_at
        expected_delay = RETRY_DELAYS[1]  # retry_count=1
        assert retry_at >= before + timedelta(seconds=expected_delay - 2)
        assert retry_at <= after + timedelta(seconds=expected_delay + 2)

    @pytest.mark.asyncio
    async def test_execute_task_marks_operation_completed(self, memory: MemoryEngine):
        """After a successful delivery, execute_task marks the async_operations row as completed."""
        operation_id = str(uuid.uuid4())
        bank_id = f"wh-exec-{uuid.uuid4().hex[:8]}"

        await _ensure_bank(memory._pool, bank_id)
        # Insert a real async_operations row so _mark_operation_completed has something to update
        async with memory._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO async_operations
                  (operation_id, bank_id, operation_type, status, task_payload, result_metadata, created_at, updated_at)
                VALUES ($1, $2, 'webhook_delivery', 'processing', '{}'::jsonb, '{}'::jsonb, NOW(), NOW())
                """,
                uuid.UUID(operation_id),
                bank_id,
            )

        task_dict = {
            **_make_delivery_task(bank_id=bank_id, retry_count=0),
            "operation_id": operation_id,
        }

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch.object(memory._http_client, "post", new=AsyncMock(return_value=mock_response)):
            await memory.execute_task(task_dict)

        async with memory._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT status FROM async_operations WHERE operation_id = $1",
                uuid.UUID(operation_id),
            )

        assert row is not None
        assert row["status"] == "completed", f"Expected 'completed', got '{row['status']}'"

        # Cleanup
        async with memory._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM async_operations WHERE operation_id = $1",
                uuid.UUID(operation_id),
            )


# ---------------------------------------------------------------------------
# HTTP API integration tests
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def api_client(memory: MemoryEngine):
    """Async HTTP test client wired to the FastAPI app."""
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestWebhookHttpApi:
    """HTTP API integration tests for webhook CRUD endpoints."""

    @pytest.mark.asyncio
    async def test_http_create_webhook(self, api_client: httpx.AsyncClient):
        """POST /webhooks returns 201 and an id."""
        bank_id = f"http-wh-{uuid.uuid4().hex[:8]}"
        response = await api_client.post(
            f"/v1/default/banks/{bank_id}/webhooks",
            json={
                "url": "https://example.com/create",
                "event_types": ["consolidation.completed"],
            },
        )
        assert response.status_code == 201, response.text
        data = response.json()
        assert "id" in data
        assert data["url"] == "https://example.com/create"
        assert data["bank_id"] == bank_id
        assert data["secret"] is None  # secrets are never echoed back

        # Cleanup
        await api_client.delete(f"/v1/default/banks/{bank_id}/webhooks/{data['id']}")

    @pytest.mark.asyncio
    async def test_http_create_webhook_creates_missing_bank(self, api_client: httpx.AsyncClient):
        """Creating the first webhook for a bank should lazily create the bank
        (webhooks.bank_id has an FK), not raise a constraint error."""
        bank_id = f"http-wh-newbank-{uuid.uuid4().hex[:8]}"

        response = await api_client.post(
            f"/v1/default/banks/{bank_id}/webhooks",
            json={
                "url": "https://example.com/new-bank",
                "event_types": ["consolidation.completed"],
            },
        )
        assert response.status_code == 201, response.text
        webhook_id = response.json()["id"]

        banks_resp = await api_client.get("/v1/default/banks")
        assert banks_resp.status_code == 200
        bank_ids = {bank["bank_id"] for bank in banks_resp.json()["banks"]}
        assert bank_id in bank_ids

        # Cleanup
        await api_client.delete(f"/v1/default/banks/{bank_id}/webhooks/{webhook_id}")
        await api_client.delete(f"/v1/default/banks/{bank_id}")

    @pytest.mark.asyncio
    async def test_http_list_webhooks(self, api_client: httpx.AsyncClient):
        """GET /webhooks returns the webhooks registered for a bank."""
        bank_id = f"http-wh-{uuid.uuid4().hex[:8]}"

        create_resp = await api_client.post(
            f"/v1/default/banks/{bank_id}/webhooks",
            json={"url": "https://example.com/list", "event_types": ["consolidation.completed"]},
        )
        assert create_resp.status_code == 201
        webhook_id = create_resp.json()["id"]

        list_resp = await api_client.get(f"/v1/default/banks/{bank_id}/webhooks")
        assert list_resp.status_code == 200
        items = list_resp.json()["items"]
        assert any(item["id"] == webhook_id for item in items)

        # Cleanup
        await api_client.delete(f"/v1/default/banks/{bank_id}/webhooks/{webhook_id}")

    @pytest.mark.asyncio
    async def test_http_delete_webhook(self, api_client: httpx.AsyncClient):
        """DELETE /webhooks/{id} removes the webhook; subsequent list returns empty for that bank."""
        bank_id = f"http-wh-{uuid.uuid4().hex[:8]}"

        create_resp = await api_client.post(
            f"/v1/default/banks/{bank_id}/webhooks",
            json={"url": "https://example.com/delete", "event_types": ["consolidation.completed"]},
        )
        assert create_resp.status_code == 201
        webhook_id = create_resp.json()["id"]

        delete_resp = await api_client.delete(f"/v1/default/banks/{bank_id}/webhooks/{webhook_id}")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["success"] is True

        list_resp = await api_client.get(f"/v1/default/banks/{bank_id}/webhooks")
        assert list_resp.status_code == 200
        ids = [item["id"] for item in list_resp.json()["items"]]
        assert webhook_id not in ids

    @pytest.mark.asyncio
    async def test_http_delete_webhook_not_found(self, api_client: httpx.AsyncClient):
        """DELETE with a non-existent webhook id returns 404."""
        bank_id = f"http-wh-{uuid.uuid4().hex[:8]}"
        missing_id = str(uuid.uuid4())
        response = await api_client.delete(f"/v1/default/banks/{bank_id}/webhooks/{missing_id}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_http_list_deliveries(self, memory: MemoryEngine, api_client: httpx.AsyncClient):
        """GET /webhooks/{id}/deliveries returns delivery records for a webhook."""
        bank_id = f"http-wh-{uuid.uuid4().hex[:8]}"

        # Create webhook via HTTP API
        create_resp = await api_client.post(
            f"/v1/default/banks/{bank_id}/webhooks",
            json={
                "url": "https://example.com/deliveries",
                "event_types": ["consolidation.completed"],
            },
        )
        assert create_resp.status_code == 201
        webhook_id = create_resp.json()["id"]

        # Insert a delivery row directly into async_operations
        delivery_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        task_payload = json.dumps(
            {
                "type": "webhook_delivery",
                "bank_id": bank_id,
                "url": "https://example.com/deliveries",
                "secret": None,
                "event_type": "consolidation.completed",
                "payload": '{"event":"consolidation.completed"}',
                "webhook_id": webhook_id,
            }
        )
        async with memory._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO async_operations
                  (operation_id, bank_id, operation_type, status, retry_count, task_payload, result_metadata, created_at, updated_at)
                VALUES ($1, $2, 'webhook_delivery', 'completed', 0, $3::jsonb, '{}'::jsonb, $4, $4)
                """,
                delivery_id,
                bank_id,
                task_payload,
                now,
            )

        try:
            deliveries_resp = await api_client.get(f"/v1/default/banks/{bank_id}/webhooks/{webhook_id}/deliveries")
            assert deliveries_resp.status_code == 200
            items = deliveries_resp.json()["items"]
            ids = [item["id"] for item in items]
            assert str(delivery_id) in ids

            # Verify shape of a delivery item
            delivery = next(item for item in items if item["id"] == str(delivery_id))
            assert delivery["status"] == "completed"
            assert delivery["event_type"] == "consolidation.completed"
            assert delivery["attempts"] == 1
        finally:
            async with memory._pool.acquire() as conn:
                await conn.execute("DELETE FROM async_operations WHERE operation_id = $1", delivery_id)
            await api_client.delete(f"/v1/default/banks/{bank_id}/webhooks/{webhook_id}")

    @pytest.mark.asyncio
    async def test_http_list_deliveries_webhook_not_found(self, api_client: httpx.AsyncClient):
        """GET /webhooks/{id}/deliveries for a non-existent webhook returns 404."""
        bank_id = f"http-wh-{uuid.uuid4().hex[:8]}"
        missing_id = str(uuid.uuid4())
        response = await api_client.get(f"/v1/default/banks/{bank_id}/webhooks/{missing_id}/deliveries")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_http_update_webhook_url(self, api_client: httpx.AsyncClient):
        """PATCH /webhooks/{id} updates only the provided fields."""
        bank_id = f"http-wh-{uuid.uuid4().hex[:8]}"

        create_resp = await api_client.post(
            f"/v1/default/banks/{bank_id}/webhooks",
            json={"url": "https://example.com/original", "event_types": ["consolidation.completed"]},
        )
        assert create_resp.status_code == 201
        webhook_id = create_resp.json()["id"]

        patch_resp = await api_client.patch(
            f"/v1/default/banks/{bank_id}/webhooks/{webhook_id}",
            json={"url": "https://example.com/updated"},
        )
        assert patch_resp.status_code == 200
        data = patch_resp.json()
        assert data["url"] == "https://example.com/updated"
        # event_types should be unchanged
        assert "consolidation.completed" in data["event_types"]

        # Cleanup
        await api_client.delete(f"/v1/default/banks/{bank_id}/webhooks/{webhook_id}")

    @pytest.mark.asyncio
    async def test_http_update_webhook_event_types(self, api_client: httpx.AsyncClient):
        """PATCH /webhooks/{id} can update event_types."""
        bank_id = f"http-wh-{uuid.uuid4().hex[:8]}"

        create_resp = await api_client.post(
            f"/v1/default/banks/{bank_id}/webhooks",
            json={"url": "https://example.com/hook", "event_types": ["consolidation.completed"]},
        )
        assert create_resp.status_code == 201
        webhook_id = create_resp.json()["id"]

        patch_resp = await api_client.patch(
            f"/v1/default/banks/{bank_id}/webhooks/{webhook_id}",
            json={"event_types": ["retain.completed"]},
        )
        assert patch_resp.status_code == 200
        data = patch_resp.json()
        assert data["event_types"] == ["retain.completed"]

        # Cleanup
        await api_client.delete(f"/v1/default/banks/{bank_id}/webhooks/{webhook_id}")

    @pytest.mark.asyncio
    async def test_http_update_webhook_enabled(self, api_client: httpx.AsyncClient):
        """PATCH /webhooks/{id} can toggle enabled."""
        bank_id = f"http-wh-{uuid.uuid4().hex[:8]}"

        create_resp = await api_client.post(
            f"/v1/default/banks/{bank_id}/webhooks",
            json={"url": "https://example.com/hook", "event_types": ["consolidation.completed"]},
        )
        assert create_resp.status_code == 201
        webhook_id = create_resp.json()["id"]
        assert create_resp.json()["enabled"] is True

        patch_resp = await api_client.patch(
            f"/v1/default/banks/{bank_id}/webhooks/{webhook_id}",
            json={"enabled": False},
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["enabled"] is False

        # Cleanup
        await api_client.delete(f"/v1/default/banks/{bank_id}/webhooks/{webhook_id}")

    @pytest.mark.asyncio
    async def test_http_update_webhook_http_config(self, api_client: httpx.AsyncClient):
        """PATCH /webhooks/{id} can update http_config."""
        bank_id = f"http-wh-{uuid.uuid4().hex[:8]}"

        create_resp = await api_client.post(
            f"/v1/default/banks/{bank_id}/webhooks",
            json={"url": "https://example.com/hook", "event_types": ["consolidation.completed"]},
        )
        assert create_resp.status_code == 201
        webhook_id = create_resp.json()["id"]

        patch_resp = await api_client.patch(
            f"/v1/default/banks/{bank_id}/webhooks/{webhook_id}",
            json={
                "http_config": {
                    "method": "POST",
                    "timeout_seconds": 10,
                    "headers": {"X-Custom": "value"},
                    "params": {},
                }
            },
        )
        assert patch_resp.status_code == 200
        data = patch_resp.json()
        assert data["http_config"]["timeout_seconds"] == 10
        assert data["http_config"]["headers"] == {"X-Custom": "value"}

        # Cleanup
        await api_client.delete(f"/v1/default/banks/{bank_id}/webhooks/{webhook_id}")

    @pytest.mark.asyncio
    async def test_http_update_webhook_not_found(self, api_client: httpx.AsyncClient):
        """PATCH /webhooks/{id} returns 404 for a non-existent webhook."""
        bank_id = f"http-wh-{uuid.uuid4().hex[:8]}"
        missing_id = str(uuid.uuid4())
        response = await api_client.patch(
            f"/v1/default/banks/{bank_id}/webhooks/{missing_id}",
            json={"url": "https://example.com/new"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_http_update_webhook_no_fields(self, api_client: httpx.AsyncClient):
        """PATCH /webhooks/{id} with empty body returns 422."""
        bank_id = f"http-wh-{uuid.uuid4().hex[:8]}"

        create_resp = await api_client.post(
            f"/v1/default/banks/{bank_id}/webhooks",
            json={"url": "https://example.com/hook", "event_types": ["consolidation.completed"]},
        )
        assert create_resp.status_code == 201
        webhook_id = create_resp.json()["id"]

        patch_resp = await api_client.patch(
            f"/v1/default/banks/{bank_id}/webhooks/{webhook_id}",
            json={},
        )
        assert patch_resp.status_code == 422

        # Cleanup
        await api_client.delete(f"/v1/default/banks/{bank_id}/webhooks/{webhook_id}")


# ---------------------------------------------------------------------------
# retain.completed webhook tests
# ---------------------------------------------------------------------------


class TestRetainCompletedWebhook:
    """Tests for the retain.completed webhook event."""

    def test_retain_event_data_model(self):
        """RetainEventData can be constructed with optional fields."""
        data = RetainEventData(document_id="doc-123", tags=["tag1", "tag2"])
        assert data.document_id == "doc-123"
        assert data.tags == ["tag1", "tag2"]

        empty = RetainEventData()
        assert empty.document_id is None
        assert empty.tags is None

    def test_retain_event_type_value(self):
        """WebhookEventType.RETAIN_COMPLETED has the correct string value."""
        assert WebhookEventType.RETAIN_COMPLETED == "retain.completed"

    @pytest.mark.asyncio
    async def test_fire_retain_webhook_queues_per_document(self, memory: MemoryEngine, webhook_manager: WebhookManager):
        """_fire_retain_webhook queues one delivery task per content item."""
        bank_id = f"wh-retain-{uuid.uuid4().hex[:8]}"
        webhook_id = uuid.uuid4()

        await _ensure_bank(memory._pool, bank_id)
        async with memory._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO webhooks (id, bank_id, url, secret, event_types, enabled, created_at, updated_at)
                VALUES ($1, $2, $3, NULL, $4, true, NOW(), NOW())
                """,
                webhook_id,
                bank_id,
                "https://example.com/retain-hook",
                ["retain.completed"],
            )

        try:
            contents = [
                {"content": "Alice works at Google", "document_id": "doc-1"},
                {"content": "Bob loves Python", "document_id": "doc-2"},
            ]
            # Temporarily replace webhook manager on memory engine
            original_manager = memory._webhook_manager
            memory._webhook_manager = webhook_manager
            try:
                callback = memory._build_retain_outbox_callback(
                    bank_id=bank_id,
                    contents=contents,
                    operation_id="test-op-123",
                )
                assert callback is not None
                async with memory._pool.acquire() as conn:
                    await callback(conn)
            finally:
                memory._webhook_manager = original_manager

            async with memory._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT task_payload
                    FROM async_operations
                    WHERE operation_type = 'webhook_delivery'
                      AND bank_id = $1
                      AND task_payload->>'event_type' = 'retain.completed'
                    ORDER BY created_at
                    """,
                    bank_id,
                )

            assert len(rows) == 2
            payloads = []
            for row in rows:
                p = row["task_payload"]
                if isinstance(p, str):
                    p = json.loads(p)
                payloads.append(p)

            doc_ids_in_payloads = [json.loads(p["payload"]).get("data", {}).get("document_id") for p in payloads]
            assert "doc-1" in doc_ids_in_payloads
            assert "doc-2" in doc_ids_in_payloads
        finally:
            async with memory._pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM async_operations WHERE operation_type = 'webhook_delivery' AND bank_id = $1",
                    bank_id,
                )
                await conn.execute("DELETE FROM webhooks WHERE id = $1", webhook_id)

    @pytest.mark.asyncio
    async def test_async_batch_retain_queues_one_webhook_per_document(
        self, memory_no_llm_verify: MemoryEngine, request_context
    ):
        """Distinct document_id groups must not replay the full-batch outbox."""
        bank_id = f"wh-retain-batch-{uuid.uuid4().hex[:8]}"
        webhook_id = uuid.uuid4()
        original_llm_config = memory_no_llm_verify._llm_config
        original_retain_llm_config = memory_no_llm_verify._retain_llm_config
        original_webhook_manager = memory_no_llm_verify._webhook_manager

        try:
            none_config = LLMConfig(provider="none", api_key="", base_url="", model="none")
            memory_no_llm_verify._llm_config = none_config
            memory_no_llm_verify._retain_llm_config = none_config
            memory_no_llm_verify._webhook_manager = WebhookManager(
                backend=memory_no_llm_verify._backend,
                global_webhooks=[],
            )

            await _ensure_bank(memory_no_llm_verify._pool, bank_id)
            async with memory_no_llm_verify._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO webhooks (id, bank_id, url, secret, event_types, enabled, created_at, updated_at)
                    VALUES ($1, $2, $3, NULL, $4, true, NOW(), NOW())
                    """,
                    webhook_id,
                    bank_id,
                    "https://example.com/retain-hook",
                    ["retain.completed"],
                )

            await memory_no_llm_verify.submit_async_retain(
                bank_id=bank_id,
                contents=[
                    {"content": "Alice works at Google", "document_id": "doc-1"},
                    {"content": "Bob loves Python", "document_id": "doc-2"},
                ],
                request_context=request_context,
            )

            async with memory_no_llm_verify._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT task_payload
                    FROM async_operations
                    WHERE operation_type = 'webhook_delivery'
                      AND bank_id = $1
                      AND task_payload->>'event_type' = 'retain.completed'
                    ORDER BY created_at
                    """,
                    bank_id,
                )

            assert len(rows) == 2
            doc_ids = []
            event_operation_ids = []
            for row in rows:
                payload = row["task_payload"]
                if isinstance(payload, str):
                    payload = json.loads(payload)
                event_payload = json.loads(payload["payload"])
                doc_ids.append(event_payload.get("data", {}).get("document_id"))
                event_operation_ids.append(event_payload.get("operation_id"))

            assert len(doc_ids) == len(set(doc_ids))
            assert sorted(doc_ids) == ["doc-1", "doc-2"]
            assert all(event_operation_ids)
            assert len(set(event_operation_ids)) == 1
        finally:
            memory_no_llm_verify._llm_config = original_llm_config
            memory_no_llm_verify._retain_llm_config = original_retain_llm_config
            memory_no_llm_verify._webhook_manager = original_webhook_manager
            async with memory_no_llm_verify._pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM async_operations WHERE operation_type = 'webhook_delivery' AND bank_id = $1",
                    bank_id,
                )
                await conn.execute("DELETE FROM webhooks WHERE id = $1", webhook_id)


# ---------------------------------------------------------------------------
# Schema-isolation tests
#
# These tests verify that webhook CRUD endpoints honour the per-request schema
# context set by the tenant extension, rather than always operating on the
# default (public) schema. This is the regression test for the bug where the
# HTTP handlers built ``fq_table("webhooks")`` before ``_authenticate_tenant``
# had set the schema context, causing webhook rows to land in the wrong schema
# under multi-target-schema deployments. The fire path correctly resolves the
# bank's schema and would never see those rows, producing silent failures.
# ---------------------------------------------------------------------------


class _NonDefaultSchemaTenantExtension:
    """Minimal tenant extension that always returns a fixed non-default schema.

    Doesn't subclass ``TenantExtension`` because we only need ``authenticate``
    for these tests; ``_authenticate_tenant`` calls just that method.
    """

    def __init__(self, schema_name: str):
        self._schema_name = schema_name

    async def authenticate(self, context):
        from hindsight_api.extensions import TenantContext

        return TenantContext(schema_name=self._schema_name)

    async def list_tenants(self):
        from hindsight_api.extensions.tenant import Tenant

        return [Tenant(schema=self._schema_name)]


@pytest_asyncio.fixture
async def isolated_schema(memory: MemoryEngine, pg0_db_url):
    """Provision a fresh non-default schema with the full migration tree, then
    swap the memory engine's tenant extension so all subsequent operations
    resolve to it. Drops the schema on teardown.
    """
    import asyncpg

    from hindsight_api.migrations import run_migrations

    schema_name = f"tenant_wh_iso_{uuid.uuid4().hex[:8]}"

    # Run migrations to provision the schema with all tables (webhooks, banks,
    # async_operations, ...). This is the same path a real multi-tenant
    # extension would take to provision a new tenant schema.
    run_migrations(pg0_db_url, schema=schema_name)

    original_ext = memory._tenant_extension
    memory._tenant_extension = _NonDefaultSchemaTenantExtension(schema_name)

    try:
        yield schema_name
    finally:
        memory._tenant_extension = original_ext
        # Drop the test schema. Use a dedicated connection so we don't depend
        # on the pool's state.
        conn = await asyncpg.connect(pg0_db_url)
        try:
            await conn.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')
        finally:
            await conn.close()


class TestWebhookSchemaIsolation:
    """Verify the webhook HTTP endpoints write to and read from the schema set
    by the tenant extension, not the default (public) schema.
    """

    @pytest.mark.asyncio
    async def test_create_webhook_lands_in_resolved_schema(
        self, memory: MemoryEngine, api_client: httpx.AsyncClient, isolated_schema: str
    ):
        """POST /webhooks should insert into the resolved schema, not public."""
        bank_id = f"http-wh-iso-{uuid.uuid4().hex[:8]}"

        create_resp = await api_client.post(
            f"/v1/default/banks/{bank_id}/webhooks",
            json={"url": "https://example.com/iso", "event_types": ["consolidation.completed"]},
        )
        assert create_resp.status_code == 201, create_resp.text
        webhook_id = create_resp.json()["id"]

        # Row should exist in the resolved schema...
        async with memory._pool.acquire() as conn:
            row_in_target = await conn.fetchrow(
                f'SELECT id, bank_id, url FROM "{isolated_schema}".webhooks WHERE id = $1',
                uuid.UUID(webhook_id),
            )
            # ...and must NOT exist in public.
            row_in_public = await conn.fetchrow(
                "SELECT id FROM public.webhooks WHERE id = $1",
                uuid.UUID(webhook_id),
            )

        assert row_in_target is not None, "Webhook row should be inserted into the resolved schema"
        assert row_in_target["bank_id"] == bank_id
        assert row_in_target["url"] == "https://example.com/iso"
        assert row_in_public is None, (
            "Webhook row must NOT be written to public when a non-default schema is resolved by the tenant extension"
        )

    @pytest.mark.asyncio
    async def test_list_webhooks_reads_from_resolved_schema(
        self, memory: MemoryEngine, api_client: httpx.AsyncClient, isolated_schema: str
    ):
        """GET /webhooks should only return rows from the resolved schema.

        We seed an unrelated row directly into public.webhooks for the same
        bank_id and assert it does NOT appear in the list response.
        """
        bank_id = f"http-wh-iso-{uuid.uuid4().hex[:8]}"

        # Create one webhook through the HTTP API (lands in the isolated schema)
        create_resp = await api_client.post(
            f"/v1/default/banks/{bank_id}/webhooks",
            json={"url": "https://example.com/in-target", "event_types": ["consolidation.completed"]},
        )
        assert create_resp.status_code == 201
        target_webhook_id = create_resp.json()["id"]

        # Seed an unrelated webhook row directly into public.webhooks for the
        # same bank — represents data that belongs to "another tenant".
        public_webhook_id = uuid.uuid4()
        async with memory._pool.acquire() as conn:
            # public.banks may not have this bank; ensure the FK does not blow up.
            await conn.execute(
                "INSERT INTO public.banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                bank_id,
                bank_id,
            )
            await conn.execute(
                """
                INSERT INTO public.webhooks
                  (id, bank_id, url, secret, event_types, enabled, created_at, updated_at)
                VALUES ($1, $2, 'https://example.com/in-public', NULL, $3, true, NOW(), NOW())
                """,
                public_webhook_id,
                bank_id,
                ["consolidation.completed"],
            )

        try:
            list_resp = await api_client.get(f"/v1/default/banks/{bank_id}/webhooks")
            assert list_resp.status_code == 200
            ids = {item["id"] for item in list_resp.json()["items"]}

            assert target_webhook_id in ids, "list_webhooks should return rows from the resolved schema"
            assert str(public_webhook_id) not in ids, (
                "list_webhooks must NOT leak rows from public when a non-default schema is resolved"
            )
        finally:
            async with memory._pool.acquire() as conn:
                await conn.execute("DELETE FROM public.webhooks WHERE id = $1", public_webhook_id)

    @pytest.mark.asyncio
    async def test_update_webhook_targets_resolved_schema(
        self, memory: MemoryEngine, api_client: httpx.AsyncClient, isolated_schema: str
    ):
        """PATCH /webhooks/{id} should update the row in the resolved schema only."""
        bank_id = f"http-wh-iso-{uuid.uuid4().hex[:8]}"

        create_resp = await api_client.post(
            f"/v1/default/banks/{bank_id}/webhooks",
            json={"url": "https://example.com/before", "event_types": ["consolidation.completed"]},
        )
        assert create_resp.status_code == 201
        webhook_id = create_resp.json()["id"]

        # Seed a row with the SAME id in public (impossible in practice, but
        # demonstrates that PATCH does not silently target public).
        async with memory._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO public.banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                bank_id,
                bank_id,
            )
            await conn.execute(
                """
                INSERT INTO public.webhooks
                  (id, bank_id, url, secret, event_types, enabled, created_at, updated_at)
                VALUES ($1, $2, 'https://example.com/public-stale', NULL, $3, true, NOW(), NOW())
                """,
                uuid.UUID(webhook_id),
                bank_id,
                ["consolidation.completed"],
            )

        try:
            patch_resp = await api_client.patch(
                f"/v1/default/banks/{bank_id}/webhooks/{webhook_id}",
                json={"url": "https://example.com/after"},
            )
            assert patch_resp.status_code == 200
            assert patch_resp.json()["url"] == "https://example.com/after"

            async with memory._pool.acquire() as conn:
                target_url = await conn.fetchval(
                    f'SELECT url FROM "{isolated_schema}".webhooks WHERE id = $1',
                    uuid.UUID(webhook_id),
                )
                public_url = await conn.fetchval(
                    "SELECT url FROM public.webhooks WHERE id = $1",
                    uuid.UUID(webhook_id),
                )

            assert target_url == "https://example.com/after"
            # The public row must remain untouched - the update targeted the
            # resolved schema, not public.
            assert public_url == "https://example.com/public-stale"
        finally:
            async with memory._pool.acquire() as conn:
                await conn.execute("DELETE FROM public.webhooks WHERE id = $1", uuid.UUID(webhook_id))

    @pytest.mark.asyncio
    async def test_delete_webhook_targets_resolved_schema(
        self, memory: MemoryEngine, api_client: httpx.AsyncClient, isolated_schema: str
    ):
        """DELETE /webhooks/{id} should remove the row from the resolved schema only."""
        bank_id = f"http-wh-iso-{uuid.uuid4().hex[:8]}"

        create_resp = await api_client.post(
            f"/v1/default/banks/{bank_id}/webhooks",
            json={"url": "https://example.com/del", "event_types": ["consolidation.completed"]},
        )
        assert create_resp.status_code == 201
        webhook_id = create_resp.json()["id"]

        # Seed a row with the same id into public to ensure DELETE doesn't
        # accidentally target it.
        async with memory._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO public.banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                bank_id,
                bank_id,
            )
            await conn.execute(
                """
                INSERT INTO public.webhooks
                  (id, bank_id, url, secret, event_types, enabled, created_at, updated_at)
                VALUES ($1, $2, 'https://example.com/public-survivor', NULL, $3, true, NOW(), NOW())
                """,
                uuid.UUID(webhook_id),
                bank_id,
                ["consolidation.completed"],
            )

        try:
            del_resp = await api_client.delete(f"/v1/default/banks/{bank_id}/webhooks/{webhook_id}")
            assert del_resp.status_code == 200
            assert del_resp.json()["success"] is True

            async with memory._pool.acquire() as conn:
                target_row = await conn.fetchrow(
                    f'SELECT id FROM "{isolated_schema}".webhooks WHERE id = $1',
                    uuid.UUID(webhook_id),
                )
                public_row = await conn.fetchrow(
                    "SELECT id FROM public.webhooks WHERE id = $1",
                    uuid.UUID(webhook_id),
                )

            assert target_row is None, "row in resolved schema should have been deleted"
            assert public_row is not None, "row in public must NOT be deleted when delete targets a non-default schema"
        finally:
            async with memory._pool.acquire() as conn:
                await conn.execute("DELETE FROM public.webhooks WHERE id = $1", uuid.UUID(webhook_id))

    @pytest.mark.asyncio
    async def test_list_deliveries_targets_resolved_schema(
        self, memory: MemoryEngine, api_client: httpx.AsyncClient, isolated_schema: str
    ):
        """GET /webhooks/{id}/deliveries should only see deliveries in the resolved schema.

        Specifically, if a webhook exists in public with the same id but NOT in
        the resolved schema, the endpoint must return 404 — it must look up the
        webhook in the resolved schema, not public.
        """
        bank_id = f"http-wh-iso-{uuid.uuid4().hex[:8]}"
        orphan_webhook_id = uuid.uuid4()

        # Seed a webhook ONLY in public (not in the resolved schema)
        async with memory._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO public.banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                bank_id,
                bank_id,
            )
            await conn.execute(
                """
                INSERT INTO public.webhooks
                  (id, bank_id, url, secret, event_types, enabled, created_at, updated_at)
                VALUES ($1, $2, 'https://example.com/orphan', NULL, $3, true, NOW(), NOW())
                """,
                orphan_webhook_id,
                bank_id,
                ["consolidation.completed"],
            )

        try:
            resp = await api_client.get(f"/v1/default/banks/{bank_id}/webhooks/{orphan_webhook_id}/deliveries")
            # The webhook does not exist in the resolved schema, so this must 404
            # — not silently fall through to public.
            assert resp.status_code == 404, resp.text
        finally:
            async with memory._pool.acquire() as conn:
                await conn.execute("DELETE FROM public.webhooks WHERE id = $1", orphan_webhook_id)

    @pytest.mark.asyncio
    async def test_list_deliveries_returns_rows_from_resolved_schema(
        self, memory: MemoryEngine, api_client: httpx.AsyncClient, isolated_schema: str
    ):
        """GET /webhooks/{id}/deliveries should read async_operations from the resolved schema."""
        bank_id = f"http-wh-iso-{uuid.uuid4().hex[:8]}"

        create_resp = await api_client.post(
            f"/v1/default/banks/{bank_id}/webhooks",
            json={"url": "https://example.com/del-iso", "event_types": ["consolidation.completed"]},
        )
        assert create_resp.status_code == 201
        webhook_id = create_resp.json()["id"]

        # Seed a delivery row in the RESOLVED schema's async_operations table.
        target_delivery_id = uuid.uuid4()
        target_payload = json.dumps(
            {
                "type": "webhook_delivery",
                "bank_id": bank_id,
                "url": "https://example.com/del-iso",
                "secret": None,
                "event_type": "consolidation.completed",
                "payload": '{"event":"consolidation.completed"}',
                "webhook_id": webhook_id,
            }
        )
        # Seed a confounding delivery row with the same payload->webhook_id in
        # public.async_operations to make sure it is NOT returned.
        public_delivery_id = uuid.uuid4()
        public_payload = json.dumps(
            {
                "type": "webhook_delivery",
                "bank_id": bank_id,
                "url": "https://example.com/del-iso-public",
                "secret": None,
                "event_type": "consolidation.completed",
                "payload": '{"event":"consolidation.completed"}',
                "webhook_id": webhook_id,
            }
        )
        now = datetime.now(timezone.utc)
        async with memory._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO "{isolated_schema}".async_operations
                  (operation_id, bank_id, operation_type, status, retry_count,
                   task_payload, result_metadata, created_at, updated_at)
                VALUES ($1, $2, 'webhook_delivery', 'completed', 0,
                        $3::jsonb, '{{}}'::jsonb, $4, $4)
                """,
                target_delivery_id,
                bank_id,
                target_payload,
                now,
            )
            await conn.execute(
                "INSERT INTO public.banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                bank_id,
                bank_id,
            )
            await conn.execute(
                """
                INSERT INTO public.async_operations
                  (operation_id, bank_id, operation_type, status, retry_count,
                   task_payload, result_metadata, created_at, updated_at)
                VALUES ($1, $2, 'webhook_delivery', 'completed', 0,
                        $3::jsonb, '{}'::jsonb, $4, $4)
                """,
                public_delivery_id,
                bank_id,
                public_payload,
                now,
            )

        try:
            resp = await api_client.get(f"/v1/default/banks/{bank_id}/webhooks/{webhook_id}/deliveries")
            assert resp.status_code == 200
            ids = {item["id"] for item in resp.json()["items"]}
            assert str(target_delivery_id) in ids, "deliveries from the resolved schema should be returned"
            assert str(public_delivery_id) not in ids, (
                "deliveries from public must NOT leak when a non-default schema is resolved"
            )
        finally:
            async with memory._pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM public.async_operations WHERE operation_id = $1",
                    public_delivery_id,
                )


# ─── MemoryDefenseEventData SIEM enrichment fields ──────────────────────────────
#
# OSS only populates action / detector / document_id / matched_types / message.
# The remaining fields are optional SIEM enrichment that downstream extensions
# (e.g. hindsight-cloud) populate when they have richer per-decision context.
# These tests pin the wire contract so OSS evolution doesn't break extensions
# that depend on the optional fields being present and JSON-serialisable.


def test_memory_defense_event_data_base_shape() -> None:
    """The five base fields populated by every implementation round-trip cleanly
    and the optional SIEM-enrichment fields default to None when omitted."""
    data = MemoryDefenseEventData(
        action="redact",
        detector="sensitive_data",
        document_id="doc-1",
        matched_types=["github_token"],
        message="Secrets redacted by policy-driven pre-screen",
    )

    # Base fields populated.
    assert data.action == "redact"
    assert data.detector == "sensitive_data"
    assert data.document_id == "doc-1"
    assert data.matched_types == ["github_token"]
    assert data.message == "Secrets redacted by policy-driven pre-screen"

    # Optional enrichment fields default to None — OSS receivers must see no
    # change vs. before this commit.
    assert data.severity is None
    assert data.api_key_name is None
    assert data.hits is None
    assert data.memory_unit_id is None
    assert data.receipt_uri is None

    # JSON shape: explicit None for absent fields, no extra keys.
    dumped = data.model_dump()
    assert dumped["severity"] is None
    assert dumped["hits"] is None
    assert set(dumped.keys()) == {
        "action",
        "detector",
        "document_id",
        "matched_types",
        "message",
        "severity",
        "api_key_name",
        "hits",
        "memory_unit_id",
        "receipt_uri",
    }


def test_memory_defense_event_data_with_siem_enrichment() -> None:
    """When an extension populates the enrichment fields, they round-trip via
    the model and through WebhookEvent JSON serialisation."""
    hit = MemoryDefenseHit(detector="GitHub Token", preview="ghp_AAAA...BBBB")
    data = MemoryDefenseEventData(
        action="redact",
        detector="sensitive_data",
        document_id="doc-42",
        matched_types=["github_token"],
        message="rotate immediately",
        severity="high",
        api_key_name="Connect Key",
        hits=[hit],
        memory_unit_id="mu-123",
        receipt_uri="memdef://bank/abc/receipt/xyz",
    )

    assert data.severity == "high"
    assert data.api_key_name == "Connect Key"
    assert data.hits == [hit]
    assert data.hits[0].detector == "GitHub Token"
    assert data.hits[0].preview == "ghp_AAAA...BBBB"
    assert data.memory_unit_id == "mu-123"
    assert data.receipt_uri == "memdef://bank/abc/receipt/xyz"

    # Nested-event round trip via JSON (this is what the webhook manager
    # serialises before queuing the delivery).
    event = WebhookEvent(
        event=WebhookEventType.MEMORY_DEFENSE_TRIGGERED,
        bank_id="bank-1",
        operation_id="",
        status="redact",
        timestamp=datetime(2026, 6, 12, 0, 0, tzinfo=timezone.utc),
        data=data,
    )
    payload = json.loads(event.model_dump_json())
    assert payload["data"]["severity"] == "high"
    assert payload["data"]["api_key_name"] == "Connect Key"
    assert payload["data"]["hits"] == [{"detector": "GitHub Token", "preview": "ghp_AAAA...BBBB"}]
    assert payload["data"]["memory_unit_id"] == "mu-123"
    assert payload["data"]["receipt_uri"] == "memdef://bank/abc/receipt/xyz"


def test_memory_defense_hit_rejects_missing_preview() -> None:
    """MemoryDefenseHit requires both fields — guards against extensions
    accidentally posting raw secrets as the only payload (preview must be
    explicit) or omitting the inner detector label."""
    with pytest.raises(Exception):  # pydantic ValidationError
        MemoryDefenseHit(detector="GitHub Token")  # type: ignore[call-arg]
    with pytest.raises(Exception):
        MemoryDefenseHit(preview="ghp_AAAA...BBBB")  # type: ignore[call-arg]
