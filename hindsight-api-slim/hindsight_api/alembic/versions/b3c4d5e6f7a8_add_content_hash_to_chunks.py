"""add content_hash to chunks table for delta retain

Revision ID: b3c4d5e6f7a8
Revises: a3b4c5d6e7f8
Create Date: 2026-03-25
"""

from collections.abc import Sequence

from alembic import context, op

revision: str = "b3c4d5e6f7a8"
down_revision: str | Sequence[str] | None = "a3b4c5d6e7f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    schema = _get_schema_prefix()
    # Add content_hash column to chunks table for delta comparison
    op.execute(f"ALTER TABLE {schema}chunks ADD COLUMN IF NOT EXISTS content_hash TEXT")


def downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"ALTER TABLE {schema}chunks DROP COLUMN IF EXISTS content_hash")
