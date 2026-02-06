"""Fix mental_models primary key to be scoped per bank

Revision ID: w8r9s0t1u2v3
Revises: v7q8r9s0t1u2
Create Date: 2026-02-05

This migration fixes a critical bank isolation bug where mental_models.id was
globally unique across all banks instead of being scoped per bank. This caused
conflicts when different banks tried to use the same custom ID.

CRITICAL FIX: Changes primary key from (id) to (bank_id, id) to ensure proper isolation.
"""

from collections.abc import Sequence

from alembic import context, op

revision: str = "w8r9s0t1u2v3"
down_revision: str | Sequence[str] | None = "v7q8r9s0t1u2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    """Change mental_models primary key from (id) to (bank_id, id) for proper bank isolation."""
    schema = _get_schema_prefix()

    # Drop the old primary key constraint (just id)
    # Note: The constraint might be named differently on different DBs
    # Try both old names (pinned_reflections_pkey from original, mental_models_pkey from rename)
    op.execute(f"ALTER TABLE {schema}mental_models DROP CONSTRAINT IF EXISTS pinned_reflections_pkey")
    op.execute(f"ALTER TABLE {schema}mental_models DROP CONSTRAINT IF EXISTS mental_models_pkey")

    # Create the new composite primary key (bank_id, id)
    # This ensures IDs are scoped per bank, not globally
    op.execute(f"""
        ALTER TABLE {schema}mental_models
        ADD CONSTRAINT mental_models_pkey PRIMARY KEY (bank_id, id)
    """)


def downgrade() -> None:
    """Revert mental_models primary key from (bank_id, id) to (id)."""
    schema = _get_schema_prefix()

    # Drop the composite primary key
    op.execute(f"ALTER TABLE {schema}mental_models DROP CONSTRAINT IF EXISTS mental_models_pkey")

    # Restore the old primary key (just id)
    # WARNING: This downgrade will fail if there are duplicate IDs across banks
    op.execute(f"""
        ALTER TABLE {schema}mental_models
        ADD CONSTRAINT mental_models_pkey PRIMARY KEY (id)
    """)
