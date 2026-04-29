"""Add 'cancelled' to async_operations status check constraint

Revision ID: i4j5k6l7m8n9
Revises: d5y6z7a8b9c0
Create Date: 2026-04-23
"""

from collections.abc import Sequence

from alembic import context, op

revision: str = "i4j5k6l7m8n9"
down_revision: str | Sequence[str] | None = "d5y6z7a8b9c0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"ALTER TABLE {schema}async_operations DROP CONSTRAINT IF EXISTS async_operations_status_check")
    op.execute(
        f"ALTER TABLE {schema}async_operations ADD CONSTRAINT async_operations_status_check "
        f"CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled'))"
    )


def downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"ALTER TABLE {schema}async_operations DROP CONSTRAINT IF EXISTS async_operations_status_check")
    op.execute(
        f"ALTER TABLE {schema}async_operations ADD CONSTRAINT async_operations_status_check "
        f"CHECK (status IN ('pending', 'processing', 'completed', 'failed'))"
    )
