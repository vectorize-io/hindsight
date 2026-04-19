"""Add structured_content JSONB column to mental_models

Revision ID: b3w4x5y6z7a8
Revises: a2v3w4x5y6z7
Create Date: 2026-04-16

Stores the structured representation of a mental model document (sections,
blocks). The plain ``content`` column remains the rendered markdown shown to
users. ``structured_content`` is the source of truth for delta-mode refreshes:
each refresh applies a list of typed operations to the structured doc, then
re-renders to markdown — so unchanged sections come through byte-identical
without an LLM round-trip.

Nullable: existing markdown-only mental models continue to work in full mode;
the column is populated lazily the first time a model is refreshed in delta
mode.
"""

from collections.abc import Sequence

from alembic import context, op

revision: str = "b3w4x5y6z7a8"
down_revision: str | Sequence[str] | None = "a2v3w4x5y6z7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"""
        ALTER TABLE {schema}mental_models
        ADD COLUMN IF NOT EXISTS structured_content JSONB
    """)


def downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"ALTER TABLE {schema}mental_models DROP COLUMN IF EXISTS structured_content")
