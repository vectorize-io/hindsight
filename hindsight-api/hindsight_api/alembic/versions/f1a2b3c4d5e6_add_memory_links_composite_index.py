"""add_memory_links_from_type_weight_index

Revision ID: f1a2b3c4d5e6
Revises: e0a1b2c3d4e5
Create Date: 2025-01-12

Add composite index on memory_links (from_unit_id, link_type, weight DESC)
to optimize MPFP graph traversal queries that need top-k edges per type.
"""

from collections.abc import Sequence

from alembic import context, op

# revision identifiers, used by Alembic.
revision: str = "f1a2b3c4d5e6"
down_revision: str | Sequence[str] | None = "e0a1b2c3d4e5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (e.g., 'tenant_x.' or '' for public)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    """Add composite index for efficient MPFP edge loading."""
    schema = _get_schema_prefix()
    # Create composite index for efficient top-k per (from_node, link_type) queries
    # This enables LATERAL joins to use index-only scans with early termination
    # Note: Not using CONCURRENTLY here as it requires running outside a transaction
    # For production with large tables, consider running this manually with CONCURRENTLY
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_memory_links_from_type_weight "
        f"ON {schema}memory_links(from_unit_id, link_type, weight DESC)"
    )


def downgrade() -> None:
    """Remove the composite index."""
    schema = _get_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_links_from_type_weight")
