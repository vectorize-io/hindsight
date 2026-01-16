"""delete_opinions

Revision ID: i4d5e6f7g8h9
Revises: h3c4d5e6f7g8
Create Date: 2026-01-15 00:00:00.000000

This migration removes opinion facts from memory_units.
Opinions are no longer a separate fact type - they are now represented
through mental model observations with confidence scores.
"""

from collections.abc import Sequence

from alembic import context, op

# revision identifiers, used by Alembic.
revision: str = "i4d5e6f7g8h9"
down_revision: str | Sequence[str] | None = "h3c4d5e6f7g8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    """Delete opinion memory_units."""
    schema = _get_schema_prefix()

    # Delete opinion memory_units (cascades to unit_entities links)
    # Opinions are now handled through mental model observations
    op.execute(f"DELETE FROM {schema}memory_units WHERE fact_type = 'opinion'")


def downgrade() -> None:
    """Cannot restore deleted opinions."""
    # Note: Cannot restore deleted opinions - they are lost on downgrade
    pass
