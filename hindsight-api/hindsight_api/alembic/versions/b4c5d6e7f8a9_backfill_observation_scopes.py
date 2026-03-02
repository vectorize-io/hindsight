"""Backfill observation_scopes column if missing.

This migration ensures observation_scopes exists even on databases that had
revision z1u2v3w4x5y6 applied when it referred to the old text_signals migration
(before it was renamed to a2b3c4d5e6f7). The ADD COLUMN IF NOT EXISTS makes this
a no-op on databases that already have the column.

Revision ID: b4c5d6e7f8a9
Revises: a2b3c4d5e6f7
Create Date: 2026-03-02
"""

from collections.abc import Sequence

from alembic import context, op

revision: str = "b4c5d6e7f8a9"
down_revision: str | Sequence[str] | None = "a2b3c4d5e6f7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(
        f"ALTER TABLE {schema}memory_units ADD COLUMN IF NOT EXISTS observation_scopes JSONB"
    )


def downgrade() -> None:
    pass  # intentionally no-op — safe to leave the column in place
