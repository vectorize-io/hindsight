"""add_directive_subtype

Revision ID: k6f7g8h9i0j1
Revises: j5e6f7g8h9i0
Create Date: 2026-01-16 00:00:00.000000

This migration adds 'directive' to the mental_models subtype constraint.
Directives are hard rules with user-provided observations that the reflect agent must follow.
"""

from collections.abc import Sequence

from alembic import context, op

# revision identifiers, used by Alembic.
revision: str = "k6f7g8h9i0j1"
down_revision: str | Sequence[str] | None = "j5e6f7g8h9i0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    """Add 'directive' to mental_models subtype constraint."""
    schema = _get_schema_prefix()

    # Drop existing constraint
    op.execute(f"ALTER TABLE {schema}mental_models DROP CONSTRAINT IF EXISTS ck_mental_models_subtype")

    # Create new constraint with 'directive' added
    op.execute(f"""
        ALTER TABLE {schema}mental_models
        ADD CONSTRAINT ck_mental_models_subtype
        CHECK (subtype IN ('structural', 'emergent', 'pinned', 'learned', 'directive'))
    """)


def downgrade() -> None:
    """Remove 'directive' from mental_models subtype constraint."""
    schema = _get_schema_prefix()

    # First delete any directives (cannot downgrade if they exist)
    op.execute(f"DELETE FROM {schema}mental_models WHERE subtype = 'directive'")

    # Drop constraint with directive
    op.execute(f"ALTER TABLE {schema}mental_models DROP CONSTRAINT IF EXISTS ck_mental_models_subtype")

    # Recreate original constraint without directive
    op.execute(f"""
        ALTER TABLE {schema}mental_models
        ADD CONSTRAINT ck_mental_models_subtype
        CHECK (subtype IN ('structural', 'emergent', 'pinned', 'learned'))
    """)
