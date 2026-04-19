"""Drop unused metadata column from documents table

Revision ID: d6e7f8a9b0c1
Revises: c2d3e4f5g6h7, c5d6e7f8a9b0
Create Date: 2026-03-30

The metadata column on documents was always stored as an empty dict {}.
Actual document metadata is stored inside retain_params.metadata.

This migration was originally shipped in v0.4.22, then its file was deleted
in v0.5.0 (and its revision ID accidentally reused by 2eee35aa3cfc).
Restoring the file so that databases stamped at this revision can upgrade
cleanly to v0.5.x+.
"""

from collections.abc import Sequence

from alembic import context, op

revision: str = "d6e7f8a9b0c1"
down_revision: str | Sequence[str] | None = ("c2d3e4f5g6h7", "c5d6e7f8a9b0")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"ALTER TABLE {schema}documents DROP COLUMN IF EXISTS metadata")


def downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"ALTER TABLE {schema}documents ADD COLUMN IF NOT EXISTS metadata jsonb DEFAULT '{{}}'")
