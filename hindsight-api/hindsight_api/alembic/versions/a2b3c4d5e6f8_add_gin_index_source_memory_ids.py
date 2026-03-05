"""Add GIN index on source_memory_ids for observation lookup performance

Without this index, queries using the array overlap operator (&&) or array
containment (@>) on source_memory_ids require a full sequential scan over all
observation memory_units. At ~77k observations this was measured at 45ms per
query, becoming a bottleneck during consolidation recall (57-64s timeouts) and
user recall (18-27s average).

The GIN index reduces these queries to index scans: 45ms → 0.049ms (927x
speedup). Recall dropped from 18-27s to ~6s, and consolidation recall
stabilised from timeout to ~15s.

Created with CONCURRENTLY so the migration does not block reads or writes.
CONCURRENTLY requires running outside a transaction block, so the migration
emits an explicit COMMIT before the statement and uses IF NOT EXISTS for
idempotency.

Revision ID: a2b3c4d5e6f8
Revises: f7g8h9i0j1k2
Create Date: 2026-03-04
"""

from collections.abc import Sequence

from alembic import context, op

revision: str = "a2b3c4d5e6f8"
down_revision: str | Sequence[str] | None = "f7g8h9i0j1k2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    schema = _get_schema_prefix()

    # CREATE INDEX CONCURRENTLY cannot run inside a transaction block.
    # Commit the current Alembic transaction first.
    op.execute("COMMIT")
    op.execute(
        f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memory_units_source_memory_ids "
        f"ON {schema}memory_units USING GIN (source_memory_ids) "
        f"WHERE source_memory_ids IS NOT NULL"
    )


def downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute("COMMIT")
    op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_memory_units_source_memory_ids")
