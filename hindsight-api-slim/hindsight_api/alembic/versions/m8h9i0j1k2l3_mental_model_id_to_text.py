"""mental_model_id_to_text

Revision ID: m8h9i0j1k2l3
Revises: l7g8h9i0j1k2
Create Date: 2026-01-19 00:00:00.000000

This migration changes the mental_models.id column from VARCHAR(64) to TEXT
to support longer model IDs (e.g., entity names that exceed 64 characters).
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

# revision identifiers, used by Alembic.
revision: str = "m8h9i0j1k2l3"
down_revision: str | Sequence[str] | None = "l7g8h9i0j1k2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    """Change mental_models.id from VARCHAR(64) to TEXT."""
    schema = _get_schema_prefix()

    # Alter the id column type from VARCHAR(64) to TEXT
    op.execute(f"ALTER TABLE {schema}mental_models ALTER COLUMN id TYPE TEXT")


def _pg_downgrade() -> None:
    """Revert mental_models.id from TEXT to VARCHAR(64)."""
    schema = _get_schema_prefix()

    # Note: This may fail if any id values exceed 64 characters
    op.execute(f"ALTER TABLE {schema}mental_models ALTER COLUMN id TYPE VARCHAR(64)")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
