"""Add last_refreshed_source_query column to mental_models

Revision ID: a2v3w4x5y6z7
Revises: z1u2v3w4x5y6
Create Date: 2026-04-15

Tracks the source_query that was used during the most recent refresh.
Used by delta-mode refresh to detect when the query has changed: if it has,
delta mode falls back to a full regeneration because the surgical-edit
assumption (same topic, new facts) no longer holds.
"""

from collections.abc import Sequence

from alembic import context, op

revision: str = "a2v3w4x5y6z7"
down_revision: str | Sequence[str] | None = "z1u2v3w4x5y6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"""
        ALTER TABLE {schema}mental_models
        ADD COLUMN IF NOT EXISTS last_refreshed_source_query TEXT
    """)


def downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"ALTER TABLE {schema}mental_models DROP COLUMN IF EXISTS last_refreshed_source_query")
