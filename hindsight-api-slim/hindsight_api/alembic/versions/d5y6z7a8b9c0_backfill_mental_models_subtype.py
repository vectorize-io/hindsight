"""Backfill mental_models.subtype for databases that ran h3c4d5e6f7g8 before the fix

Migration h3c4d5e6f7g8 used CREATE TABLE IF NOT EXISTS to create the
mental_models table with a subtype column. But on databases where the table
already existed (from the reflections -> mental_models rename chain), the
CREATE was a no-op and subtype was never added. A fix was later added to
h3c4d5e6f7g8 (Step 4b), but databases that had already run the migration
never re-execute it. This migration adds the missing columns idempotently.

Revision ID: d5y6z7a8b9c0
Revises: 8c6fa6f7230b
Create Date: 2026-04-18
"""

from collections.abc import Sequence

from alembic import context, op

revision: str = "d5y6z7a8b9c0"
down_revision: str | Sequence[str] | None = "8c6fa6f7230b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    schema = _get_schema_prefix()

    # Add columns that h3c4d5e6f7g8 intended to create but missed when
    # the table already existed from the reflections rename chain.
    for col_ddl in [
        "subtype VARCHAR(32) NOT NULL DEFAULT 'structural'",
        "description TEXT NOT NULL DEFAULT ''",
        "entity_id UUID",
        "observations JSONB DEFAULT '{\"observations\": []}'::jsonb",
        "links VARCHAR[]",
        "last_updated TIMESTAMP WITH TIME ZONE",
    ]:
        op.execute(f"ALTER TABLE {schema}mental_models ADD COLUMN IF NOT EXISTS {col_ddl}")

    # Ensure the CHECK constraint exists
    op.execute(f"ALTER TABLE {schema}mental_models DROP CONSTRAINT IF EXISTS ck_mental_models_subtype")
    op.execute(f"""
        ALTER TABLE {schema}mental_models
        ADD CONSTRAINT ck_mental_models_subtype CHECK (subtype IN ('structural', 'emergent', 'pinned', 'learned'))
    """)

    op.execute(f"CREATE INDEX IF NOT EXISTS idx_mental_models_subtype ON {schema}mental_models(bank_id, subtype)")


def downgrade() -> None:
    # No-op: these columns are part of the intended schema
    pass
