"""Enable pg_trgm extension and add GIN trigram index on entities.canonical_name

Revision ID: a2b3c4d5e6f7
Revises: z1u2v3w4x5y6
Create Date: 2026-03-02
"""

from collections.abc import Sequence

from alembic import context, op

revision: str = "a2b3c4d5e6f7"
down_revision: str | Sequence[str] | None = "z1u2v3w4x5y6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    # pg_trgm ships with every standard PostgreSQL installation as a contrib module.
    # It enables fast similarity lookups via GIN indexes, used for entity name matching.
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    schema = _get_schema_prefix()
    # GIN index on canonical_name enables sub-millisecond trigram similarity queries
    # (% operator, similarity()) instead of full-table scans across all bank entities.
    # Note: Not using CONCURRENTLY here as it requires running outside a transaction block.
    # For production with large entities tables, consider running this manually:
    #   CREATE INDEX CONCURRENTLY IF NOT EXISTS entities_canonical_name_trgm_idx
    #   ON entities USING GIN (canonical_name gin_trgm_ops);
    op.execute(
        f"CREATE INDEX IF NOT EXISTS entities_canonical_name_trgm_idx "
        f"ON {schema}entities USING GIN (canonical_name gin_trgm_ops)"
    )


def downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}entities_canonical_name_trgm_idx")
    # Note: not dropping pg_trgm extension as other indexes may depend on it
