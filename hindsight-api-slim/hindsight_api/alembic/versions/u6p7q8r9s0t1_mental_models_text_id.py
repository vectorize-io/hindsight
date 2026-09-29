"""Change mental_models.id from UUID to TEXT

Revision ID: u6p7q8r9s0t1
Revises: t5o6p7q8r9s0
Create Date: 2026-01-27

This migration changes the mental_models.id column from UUID to TEXT
to support user-defined text identifiers like 'team-communication' instead of UUIDs.
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "u6p7q8r9s0t1"
down_revision: str | Sequence[str] | None = "t5o6p7q8r9s0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    """Change mental_models.id from UUID to TEXT."""
    schema = _get_schema_prefix()

    # Change the id column type from UUID to TEXT
    # Existing UUIDs will be converted to their string representation
    op.execute(f"ALTER TABLE {schema}mental_models ALTER COLUMN id TYPE TEXT USING id::TEXT")


def _pg_downgrade() -> None:
    """Revert mental_models.id from TEXT to UUID."""
    schema = _get_schema_prefix()

    # Note: This will fail if any id values are not valid UUIDs
    op.execute(f"ALTER TABLE {schema}mental_models ALTER COLUMN id TYPE UUID USING id::UUID")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
