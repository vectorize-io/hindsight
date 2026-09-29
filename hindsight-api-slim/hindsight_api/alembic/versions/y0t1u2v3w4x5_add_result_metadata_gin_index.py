"""Add GIN index on async_operations.result_metadata for parent_operation_id queries

Revision ID: y0t1u2v3w4x5
Revises: x9s0t1u2v3w4
Create Date: 2026-02-13

This migration adds a GIN index on the result_metadata JSONB column in the
async_operations table to support efficient queries for child operations by
parent_operation_id.

The index enables fast lookups when querying for child operations:
  SELECT * FROM async_operations
  WHERE result_metadata::jsonb @> '{"parent_operation_id": "uuid"}'::jsonb
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "y0t1u2v3w4x5"
down_revision: str | Sequence[str] | None = "x9s0t1u2v3w4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    """Add GIN index on result_metadata for efficient parent_operation_id queries."""
    schema = _get_schema_prefix()

    # Add GIN index for JSONB containment queries (@> operator)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_async_operations_result_metadata
        ON {schema}async_operations
        USING gin(result_metadata)
    """)


def _pg_downgrade() -> None:
    """Remove GIN index on result_metadata."""
    schema = _get_schema_prefix()

    # Drop index
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_async_operations_result_metadata")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
