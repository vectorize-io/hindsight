"""Add max_tokens and trigger columns to mental_models

Revision ID: v7q8r9s0t1u2
Revises: u6p7q8r9s0t1
Create Date: 2026-01-27

This migration adds:
- max_tokens column: token limit for content generation during refresh
- trigger column: JSONB for trigger settings (e.g., refresh_after_consolidation)
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "v7q8r9s0t1u2"
down_revision: str | Sequence[str] | None = "u6p7q8r9s0t1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    """Add max_tokens and trigger columns to mental_models."""
    schema = _get_schema_prefix()

    op.execute(f"""
        ALTER TABLE {schema}mental_models
        ADD COLUMN IF NOT EXISTS max_tokens INT NOT NULL DEFAULT 2048
    """)

    # trigger column stores trigger settings as JSONB
    # Default: refresh_after_consolidation = false (not "real time")
    op.execute(f"""
        ALTER TABLE {schema}mental_models
        ADD COLUMN IF NOT EXISTS trigger JSONB NOT NULL DEFAULT '{{"refresh_after_consolidation": false}}'::jsonb
    """)


def _pg_downgrade() -> None:
    """Remove max_tokens and trigger columns from mental_models."""
    schema = _get_schema_prefix()

    op.execute(f"ALTER TABLE {schema}mental_models DROP COLUMN IF EXISTS max_tokens")
    op.execute(f"ALTER TABLE {schema}mental_models DROP COLUMN IF EXISTS trigger")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
