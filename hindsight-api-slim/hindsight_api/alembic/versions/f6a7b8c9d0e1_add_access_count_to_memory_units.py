"""Add access_count column to memory_units for retrieval frequency tracking

Tracks how many times a memory unit has been retrieved via user-facing recall
operations.  System/maintenance operations (reflect, consolidation) do NOT
increment the counter.

Revision ID: f6a7b8c9d0e1
Revises: c2d3e4f5g6h7, c5d6e7f8a9b0, a2b3c4d5e6f8
Create Date: 2026-03-31
"""

from collections.abc import Sequence

from alembic import context, op

revision: str = "f6a7b8c9d0e1"
down_revision: str | Sequence[str] | None = ("c2d3e4f5g6h7", "c5d6e7f8a9b0", "a2b3c4d5e6f8")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    schema = _get_schema_prefix()

    # Add access_count column (default 0, NOT NULL)
    op.execute(
        f"ALTER TABLE {schema}memory_units "
        f"ADD COLUMN IF NOT EXISTS access_count INTEGER NOT NULL DEFAULT 0"
    )

    # Index for sorting/filtering by retrieval frequency
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_memory_units_access_count "
        f"ON {schema}memory_units (access_count DESC)"
    )


def downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_access_count")
    op.execute(f"ALTER TABLE {schema}memory_units DROP COLUMN IF EXISTS access_count")
