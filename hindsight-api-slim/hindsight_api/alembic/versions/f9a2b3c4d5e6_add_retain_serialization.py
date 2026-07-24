"""Serialize idempotent async retains for the same document.

Revision ID: f9a2b3c4d5e6
Revises: e8f1a2b3c4d5
Create Date: 2026-07-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "f9a2b3c4d5e6"
down_revision: str | Sequence[str] | None = "e8f1a2b3c4d5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_upgrade() -> None:
    op.add_column("async_operations", sa.Column("serialization_key", sa.String(length=64), nullable=True))
    op.add_column(
        "async_operations",
        sa.Column("blocked_by_operation_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_index(
        "idx_async_operations_retain_serialization",
        "async_operations",
        ["bank_id", "operation_type", "serialization_key"],
    )
    op.create_index(
        "idx_async_operations_blocked_by",
        "async_operations",
        ["blocked_by_operation_id"],
    )


def _pg_downgrade() -> None:
    op.execute(
        """
        UPDATE async_operations
        SET next_retry_at = NULL
        WHERE status = 'pending' AND blocked_by_operation_id IS NOT NULL
        """
    )
    op.drop_index("idx_async_operations_blocked_by", table_name="async_operations")
    op.drop_index("idx_async_operations_retain_serialization", table_name="async_operations")
    op.drop_column("async_operations", "blocked_by_operation_id")
    op.drop_column("async_operations", "serialization_key")


def _oracle_upgrade() -> None:
    op.add_column("async_operations", sa.Column("serialization_key", sa.String(length=64), nullable=True))
    op.execute("ALTER TABLE async_operations ADD blocked_by_operation_id RAW(16)")
    op.create_index(
        "idx_async_operations_retain_serialization",
        "async_operations",
        ["bank_id", "operation_type", "serialization_key"],
    )
    op.create_index(
        "idx_async_operations_blocked_by",
        "async_operations",
        ["blocked_by_operation_id"],
    )


def _oracle_downgrade() -> None:
    op.execute(
        """
        UPDATE async_operations
        SET next_retry_at = NULL
        WHERE status = 'pending' AND blocked_by_operation_id IS NOT NULL
        """
    )
    op.drop_index("idx_async_operations_blocked_by", table_name="async_operations")
    op.drop_index("idx_async_operations_retain_serialization", table_name="async_operations")
    op.drop_column("async_operations", "blocked_by_operation_id")
    op.drop_column("async_operations", "serialization_key")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
