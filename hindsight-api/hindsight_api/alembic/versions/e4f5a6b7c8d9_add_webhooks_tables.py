"""Add webhooks table and next_retry_at to async_operations.

Webhook deliveries are handled as async_operations tasks (operation_type='webhook_delivery')
rather than a dedicated webhook_deliveries table.

Revision ID: e4f5a6b7c8d9
Revises: d2e3f4a5b6c7
Create Date: 2026-03-04
"""

from collections.abc import Sequence

from alembic import context, op

revision: str = "e4f5a6b7c8d9"
down_revision: str | Sequence[str] | None = "d2e3f4a5b6c7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    schema = _get_schema_prefix()

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}webhooks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            bank_id TEXT,
            url TEXT NOT NULL,
            secret TEXT,
            event_types TEXT[] NOT NULL DEFAULT '{{}}',
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )

    # Index for bank-scoped webhook lookup
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_webhooks_bank_id ON {schema}webhooks(bank_id)")

    # Add next_retry_at to async_operations for task-owned retry scheduling
    op.execute(f"ALTER TABLE {schema}async_operations ADD COLUMN IF NOT EXISTS next_retry_at TIMESTAMPTZ NULL")

    # Index for polling: status + next_retry_at
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_async_operations_status_retry "
        f"ON {schema}async_operations(status, next_retry_at)"
    )


def downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_async_operations_status_retry")
    op.execute(f"ALTER TABLE {schema}async_operations DROP COLUMN IF EXISTS next_retry_at")
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_webhooks_bank_id")
    op.execute(f"DROP TABLE IF EXISTS {schema}webhooks")
