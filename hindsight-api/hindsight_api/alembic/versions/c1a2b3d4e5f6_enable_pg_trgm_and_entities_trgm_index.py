"""Enable pg_trgm extension and add GIN trigram index on entities.canonical_name

Revision ID: c1a2b3d4e5f6
Revises: b4c5d6e7f8a9
Create Date: 2026-03-02

Uses CONCURRENTLY for schemas with existing data (avoids blocking writes during
production deployments on large tables). Uses regular CREATE INDEX for empty
schemas (new tenant provisioning) because CONCURRENTLY waits for ALL open
transactions database-wide, which deadlocks against the worker's polling loop.
"""

from collections.abc import Sequence

from alembic import context, op
from sqlalchemy import text

revision: str = "c1a2b3d4e5f6"
down_revision: str | Sequence[str] | None = "b4c5d6e7f8a9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _table_has_data(table: str) -> bool:
    """Check if a table has any rows (used to decide CONCURRENTLY vs regular)."""
    conn = op.get_bind()
    result = conn.execute(text(f"SELECT EXISTS (SELECT 1 FROM {table} LIMIT 1)"))
    return result.scalar()


def upgrade() -> None:
    # pg_trgm ships with every standard PostgreSQL installation as a contrib module.
    # It enables fast similarity lookups via GIN indexes, used for entity name matching.
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    schema = _get_schema_prefix()

    # GIN index on canonical_name enables sub-millisecond trigram similarity queries
    # (% operator, similarity()) instead of full-table scans across all bank entities.
    if _table_has_data(f"{schema}entities"):
        # Existing schema with data: use CONCURRENTLY to avoid blocking writes.
        # Requires running outside a transaction block.
        op.execute("COMMIT")
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS entities_canonical_name_trgm_idx "
            f"ON {schema}entities USING GIN (canonical_name gin_trgm_ops)"
        )
    else:
        # New/empty schema: use regular CREATE INDEX (instant, no deadlock risk).
        op.execute(
            f"CREATE INDEX IF NOT EXISTS entities_canonical_name_trgm_idx "
            f"ON {schema}entities USING GIN (canonical_name gin_trgm_ops)"
        )


def downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute("COMMIT")
    op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}entities_canonical_name_trgm_idx")
    # Note: not dropping pg_trgm extension as other indexes may depend on it
