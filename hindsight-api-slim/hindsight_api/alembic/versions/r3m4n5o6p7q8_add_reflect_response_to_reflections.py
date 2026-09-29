"""Add reflect_response JSONB column to reflections

Revision ID: r3m4n5o6p7q8
Revises: q2l3m4n5o6p7
Create Date: 2026-01-21

This migration adds a reflect_response JSONB column to store the full
reflect API response payload, including based_on facts and trace data.

Note: Table was renamed from pinned_reflections to reflections in p1k2l3m4n5o6.
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "r3m4n5o6p7q8"
down_revision: str | Sequence[str] | None = "q2l3m4n5o6p7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    """Add reflect_response JSONB column to reflections."""
    schema = _get_schema_prefix()

    # Add reflect_response column to store the full reflect API response
    op.execute(f"""
        ALTER TABLE {schema}reflections
        ADD COLUMN IF NOT EXISTS reflect_response JSONB
    """)


def _pg_downgrade() -> None:
    """Remove reflect_response column from reflections."""
    schema = _get_schema_prefix()

    op.execute(f"""
        ALTER TABLE {schema}reflections
        DROP COLUMN IF EXISTS reflect_response
    """)


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
