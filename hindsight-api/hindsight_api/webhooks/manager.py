"""Webhook manager for delivering event notifications."""

import hashlib
import hmac
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import asyncpg

from .models import WebhookConfig, WebhookEvent, WebhookHttpConfig

if TYPE_CHECKING:
    from hindsight_api.extensions.tenant import TenantExtension

logger = logging.getLogger(__name__)

# Retry delay schedule in seconds: 5 retries after the first attempt.
# Fast early retries catch transient failures; later retries handle longer outages.
RETRY_DELAYS = [5, 300, 1800, 7200, 18000]
MAX_ATTEMPTS = len(RETRY_DELAYS) + 1  # first attempt + len(RETRY_DELAYS) retries


def _fq_table(table: str, schema: str | None = None) -> str:
    """Get fully-qualified table name with optional schema prefix."""
    if schema:
        return f'"{schema}".{table}'
    return table


def _parse_http_config(value: str | dict | None) -> WebhookHttpConfig:
    """Parse http_config column value (JSONB returned as text or dict) into a model."""
    if value is None:
        return WebhookHttpConfig()
    if isinstance(value, str):
        return WebhookHttpConfig.model_validate_json(value)
    return WebhookHttpConfig.model_validate(value)


class WebhookManager:
    """
    Manages webhook registration and event firing.

    Supports both global webhooks (configured via env vars) and per-bank
    webhooks stored in the database. Deliveries are queued as async_operations
    tasks (operation_type='webhook_delivery') and picked up by the worker poller.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        global_webhooks: list[WebhookConfig],
        tenant_extension: "TenantExtension | None" = None,
    ):
        self._pool = pool
        self._global_webhooks = global_webhooks
        self._tenant_extension = tenant_extension

    def _sign_payload(self, secret: str, payload_bytes: bytes) -> str:
        """Compute HMAC-SHA256 signature for a payload."""
        return "sha256=" + hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()

    async def fire_event(self, event: WebhookEvent, schema: str | None = None) -> None:
        """
        Queue webhook deliveries for an event as async_operations tasks.

        Loads per-bank and global webhooks, inserts pending webhook_delivery tasks for
        any webhook whose event_types list matches the fired event type. The worker
        poller picks these up and calls MemoryEngine._handle_webhook_delivery().

        Args:
            event: The event to deliver.
            schema: Database schema (for multi-tenant). None = default schema.
        """
        webhook_table = _fq_table("webhooks", schema)
        ops_table = _fq_table("async_operations", schema)
        now = datetime.now(timezone.utc)
        payload_str = event.model_dump_json()

        try:
            # Load per-bank webhooks from DB (bank-specific + global NULL rows)
            rows = await self._pool.fetch(
                f"""
                SELECT id, bank_id, url, secret, event_types, enabled, http_config::text
                FROM {webhook_table}
                WHERE (bank_id = $1 OR bank_id IS NULL) AND enabled = true
                """,
                event.bank_id,
            )

            db_webhooks = [
                WebhookConfig(
                    id=str(row["id"]),
                    bank_id=row["bank_id"],
                    url=row["url"],
                    secret=row["secret"],
                    event_types=list(row["event_types"]) if row["event_types"] else [],
                    enabled=row["enabled"],
                    http_config=_parse_http_config(row["http_config"]),
                )
                for row in rows
            ]

            # Merge with global webhooks from env config
            all_webhooks = self._global_webhooks + db_webhooks
            matched = 0

            for webhook in all_webhooks:
                if not webhook.enabled:
                    continue
                if event.event.value not in webhook.event_types:
                    continue

                operation_id = uuid.uuid4()
                webhook_id = webhook.id if webhook.id else None

                task_payload = json.dumps(
                    {
                        "type": "webhook_delivery",
                        "operation_id": str(operation_id),
                        "bank_id": event.bank_id,
                        "url": webhook.url,
                        "secret": webhook.secret,
                        "event_type": event.event.value,
                        "payload": payload_str,
                        "webhook_id": webhook_id,
                        "http_config": webhook.http_config.model_dump(),
                    }
                )

                await self._pool.execute(
                    f"""
                    INSERT INTO {ops_table}
                      (operation_id, bank_id, operation_type, status, task_payload, result_metadata, created_at, updated_at)
                    VALUES ($1, $2, 'webhook_delivery', 'pending', $3::jsonb, '{{}}'::jsonb, $4, $4)
                    """,
                    operation_id,
                    event.bank_id,
                    task_payload,
                    now,
                )
                matched += 1

            logger.debug(f"Fired webhook event {event.event} for bank {event.bank_id}: {matched} delivery(ies) queued")

        except Exception as e:
            logger.error(f"Failed to queue webhook deliveries for event {event.event}: {e}")

    async def fire_event_with_conn(
        self, event: WebhookEvent, conn: asyncpg.Connection, schema: str | None = None
    ) -> None:
        """
        Queue webhook deliveries within an existing database connection/transaction.

        Identical to fire_event() but uses the provided connection instead of acquiring
        one from the pool. Use this to atomically insert delivery tasks in the same
        transaction as the primary operation (transactional outbox pattern).

        Args:
            event: The event to deliver.
            conn: Existing asyncpg connection (may be inside an active transaction).
            schema: Database schema (for multi-tenant). None = default schema.
        """
        webhook_table = _fq_table("webhooks", schema)
        ops_table = _fq_table("async_operations", schema)
        now = datetime.now(timezone.utc)
        payload_str = event.model_dump_json()

        try:
            rows = await conn.fetch(
                f"""
                SELECT id, bank_id, url, secret, event_types, enabled, http_config::text
                FROM {webhook_table}
                WHERE (bank_id = $1 OR bank_id IS NULL) AND enabled = true
                """,
                event.bank_id,
            )

            db_webhooks = [
                WebhookConfig(
                    id=str(row["id"]),
                    bank_id=row["bank_id"],
                    url=row["url"],
                    secret=row["secret"],
                    event_types=list(row["event_types"]) if row["event_types"] else [],
                    enabled=row["enabled"],
                    http_config=_parse_http_config(row["http_config"]),
                )
                for row in rows
            ]

            all_webhooks = self._global_webhooks + db_webhooks
            matched = 0

            for webhook in all_webhooks:
                if not webhook.enabled:
                    continue
                if event.event.value not in webhook.event_types:
                    continue

                operation_id = uuid.uuid4()
                webhook_id = webhook.id if webhook.id else None

                task_payload = json.dumps(
                    {
                        "type": "webhook_delivery",
                        "operation_id": str(operation_id),
                        "bank_id": event.bank_id,
                        "url": webhook.url,
                        "secret": webhook.secret,
                        "event_type": event.event.value,
                        "payload": payload_str,
                        "webhook_id": webhook_id,
                        "http_config": webhook.http_config.model_dump(),
                    }
                )

                await conn.execute(
                    f"""
                    INSERT INTO {ops_table}
                      (operation_id, bank_id, operation_type, status, task_payload, result_metadata, created_at, updated_at)
                    VALUES ($1, $2, 'webhook_delivery', 'pending', $3::jsonb, '{{}}'::jsonb, $4, $4)
                    """,
                    operation_id,
                    event.bank_id,
                    task_payload,
                    now,
                )
                matched += 1

            logger.debug(
                f"Fired webhook event {event.event} for bank {event.bank_id}: {matched} delivery(ies) queued (in-transaction)"
            )

        except Exception as e:
            logger.error(f"Failed to queue webhook deliveries (in-transaction) for event {event.event}: {e}")
