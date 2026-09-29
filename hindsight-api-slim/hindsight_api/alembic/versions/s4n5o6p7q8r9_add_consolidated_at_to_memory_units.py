"""Add consolidated_at column to memory_units for incremental consolidation tracking.

This allows consolidation to track progress at the memory level rather than
using a bank-level watermark. If consolidation crashes, already-processed
memories won't be reprocessed.

Revision ID: s4n5o6p7q8r9
Revises: r3m4n5o6p7q8
Create Date: 2025-01-22
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "s4n5o6p7q8r9"
down_revision: str | Sequence[str] | None = "r3m4n5o6p7q8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()

    # Add consolidated_at column to memory_units
    op.execute(
        f"""
        ALTER TABLE {schema}memory_units
        ADD COLUMN IF NOT EXISTS consolidated_at TIMESTAMPTZ DEFAULT NULL
        """
    )

    # Create index for efficient querying of unconsolidated memories
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_memory_units_unconsolidated
        ON {schema}memory_units (bank_id, created_at)
        WHERE consolidated_at IS NULL AND fact_type IN ('experience', 'world')
        """
    )


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()

    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_unconsolidated")
    op.execute(f"ALTER TABLE {schema}memory_units DROP COLUMN IF EXISTS consolidated_at")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
