"""add_worker_columns

Revision ID: l7g8h9i0j1k2
Revises: k6f7g8h9i0j1
Create Date: 2026-01-19 00:00:00.000000

This migration adds columns to async_operations for distributed worker support:
- worker_id: ID of the worker that claimed the task
- claimed_at: When the task was claimed
- retry_count: Number of retry attempts
- task_payload: The serialized task dictionary
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

from hindsight_api.alembic._dialect import run_for_dialect

# revision identifiers, used by Alembic.
revision: str = "l7g8h9i0j1k2"
down_revision: str | Sequence[str] | None = "k6f7g8h9i0j1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    """Add worker columns to async_operations."""
    schema = _get_schema_prefix()

    # Add worker_id column (ID of worker that claimed the task)
    op.add_column(
        "async_operations",
        sa.Column("worker_id", sa.Text(), nullable=True),
        schema=context.config.get_main_option("target_schema") or None,
    )

    # Add claimed_at column (when task was claimed by worker)
    op.add_column(
        "async_operations",
        sa.Column("claimed_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        schema=context.config.get_main_option("target_schema") or None,
    )

    # Add retry_count column (number of retry attempts)
    op.add_column(
        "async_operations",
        sa.Column("retry_count", sa.Integer(), server_default="0", nullable=False),
        schema=context.config.get_main_option("target_schema") or None,
    )

    # Add task_payload column (serialized task dictionary)
    op.add_column(
        "async_operations",
        sa.Column(
            "task_payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        schema=context.config.get_main_option("target_schema") or None,
    )

    # Add index for efficient worker polling (pending tasks ordered by creation time)
    op.execute(
        f"CREATE INDEX idx_async_operations_pending_claim ON {schema}async_operations (status, created_at) "
        f"WHERE status = 'pending' AND task_payload IS NOT NULL"
    )

    # Add index for finding tasks by worker_id (for decommissioning)
    op.execute(
        f"CREATE INDEX idx_async_operations_worker_id ON {schema}async_operations (worker_id) WHERE worker_id IS NOT NULL"
    )


def _pg_downgrade() -> None:
    """Remove worker columns from async_operations."""
    schema = _get_schema_prefix()

    # Drop indexes
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_async_operations_pending_claim")
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_async_operations_worker_id")

    # Drop columns
    op.drop_column(
        "async_operations",
        "task_payload",
        schema=context.config.get_main_option("target_schema") or None,
    )
    op.drop_column(
        "async_operations",
        "retry_count",
        schema=context.config.get_main_option("target_schema") or None,
    )
    op.drop_column(
        "async_operations",
        "claimed_at",
        schema=context.config.get_main_option("target_schema") or None,
    )
    op.drop_column(
        "async_operations",
        "worker_id",
        schema=context.config.get_main_option("target_schema") or None,
    )


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
