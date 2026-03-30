"""Add bank_id column to memory_links for direct filtering

The stats endpoint JOINs memory_links to memory_units just to filter by
bank_id.  With millions of links this takes 18+ seconds.  Adding bank_id
directly to memory_links lets Postgres push the filter down before the JOIN.

Revision ID: c5d6e7f8a9b0
Revises: b3c4d5e6f7a8
Create Date: 2026-03-26
"""

from collections.abc import Sequence

from alembic import context, op

revision: str = "c5d6e7f8a9b0"
down_revision: str | Sequence[str] | None = "b3c4d5e6f7a8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    schema = _get_schema_prefix()

    # 1. Add nullable column
    op.execute(f"ALTER TABLE {schema}memory_links ADD COLUMN IF NOT EXISTS bank_id TEXT")

    # 2. Backfill from memory_units
    op.execute(f"""
        UPDATE {schema}memory_links ml
        SET bank_id = mu.bank_id
        FROM {schema}memory_units mu
        WHERE ml.from_unit_id = mu.id
          AND ml.bank_id IS NULL
    """)

    # 3. Set NOT NULL
    op.execute(f"ALTER TABLE {schema}memory_links ALTER COLUMN bank_id SET NOT NULL")


def downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"ALTER TABLE {schema}memory_links DROP COLUMN IF EXISTS bank_id")
