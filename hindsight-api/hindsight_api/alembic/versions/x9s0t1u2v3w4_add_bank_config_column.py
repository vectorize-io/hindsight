"""Add config JSONB column to banks table for hierarchical configuration

Revision ID: x9s0t1u2v3w4
Revises: w8r9s0t1u2v3
Create Date: 2026-02-09

This migration adds a `config` JSONB column to the banks table to support
per-bank configuration overrides. This enables hierarchical configuration where:
- Global config is loaded from environment variables
- Tenant config is provided via TenantExtension
- Bank config overrides are stored in banks.config JSONB column

The config column stores overrides for hierarchical fields (LLM settings,
retention parameters, retrieval settings, etc.) in Python field name format.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "x9s0t1u2v3w4"
down_revision: str | Sequence[str] | None = "w8r9s0t1u2v3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    """Add config JSONB column to banks table with GIN index."""
    schema = _get_schema_prefix()

    # Add config column to banks table
    op.execute(f"""
        ALTER TABLE {schema}banks
        ADD COLUMN config JSONB NOT NULL DEFAULT '{{}}'::jsonb
    """)

    # Add GIN index for efficient JSONB queries
    op.execute(f"""
        CREATE INDEX idx_banks_config
        ON {schema}banks
        USING gin(config)
    """)


def downgrade() -> None:
    """Remove config column and index from banks table."""
    schema = _get_schema_prefix()

    # Drop index first
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_banks_config")

    # Drop column
    op.execute(f"""
        ALTER TABLE {schema}banks
        DROP COLUMN IF EXISTS config
    """)
