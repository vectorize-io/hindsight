"""add_tags_column

Revision ID: g2a3b4c5d6e7
Revises: f1a2b3c4d5e6
Create Date: 2025-01-13

Add tags column to memory_units and documents tables for visibility scoping.
Tags enable filtering memories by scope (e.g., user IDs, session IDs) during recall/reflect.
"""

from collections.abc import Sequence

from alembic import context, op

# revision identifiers, used by Alembic.
revision: str = "g2a3b4c5d6e7"
down_revision: str | Sequence[str] | None = "f1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (e.g., 'tenant_x.' or '' for public)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    """Add tags column to memory_units and documents tables."""
    schema = _get_schema_prefix()

    # Add tags column to memory_units table
    op.execute(f"ALTER TABLE {schema}memory_units ADD COLUMN IF NOT EXISTS tags VARCHAR[] NOT NULL DEFAULT '{{}}'")

    # Create GIN index for efficient array containment queries (tags && ARRAY['x'])
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_memory_units_tags ON {schema}memory_units USING GIN (tags)")

    # Add tags column to documents table for document-level tags
    op.execute(f"ALTER TABLE {schema}documents ADD COLUMN IF NOT EXISTS tags VARCHAR[] NOT NULL DEFAULT '{{}}'")


def downgrade() -> None:
    """Remove tags columns and index."""
    schema = _get_schema_prefix()

    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_tags")
    op.execute(f"ALTER TABLE {schema}memory_units DROP COLUMN IF EXISTS tags")
    op.execute(f"ALTER TABLE {schema}documents DROP COLUMN IF EXISTS tags")
