"""Add partial indexes on memory_units temporal date fields for fast temporal retrieval

Revision ID: b3c4d5e6f7g8
Revises: c1a2b3d4e5f6
Create Date: 2026-03-02

The temporal retrieval entry-point query filters memory_units by occurred_start,
occurred_end, and mentioned_at using OR conditions. Without dedicated indexes the
planner falls back to a sequential scan of all bank rows after applying the
(bank_id, fact_type) index, then re-checks each date field.

These three partial indexes give the planner bitmap-index scan options for the
three most common date predicates, dramatically reducing the row set before any
embedding computation is required.

Uses CONCURRENTLY for schemas with existing data (avoids blocking writes during
production deployments on large tables). Uses regular CREATE INDEX for empty
schemas (new tenant provisioning) because CONCURRENTLY waits for ALL open
transactions database-wide, which deadlocks against the worker's polling loop.
"""

from collections.abc import Sequence

from alembic import context, op
from sqlalchemy import text

revision: str = "b3c4d5e6f7g8"
down_revision: str | Sequence[str] | None = "c1a2b3d4e5f6"
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


def _create_index(schema: str, definition: str) -> None:
    """Create an index, using CONCURRENTLY only when the table has data."""
    # Extract table name from "ON {schema}table_name(...)"
    table_part = definition.split(" ON ")[1].split("(")[0].strip()
    if _table_has_data(table_part):
        op.execute("COMMIT")
        op.execute(f"CREATE INDEX CONCURRENTLY {definition}")
    else:
        op.execute(f"CREATE INDEX {definition}")


def upgrade() -> None:
    schema = _get_schema_prefix()

    # Partial index on occurred_start (covers "occurred_start BETWEEN $4 AND $5")
    _create_index(
        schema,
        f"IF NOT EXISTS idx_memory_units_bank_occurred_start "
        f"ON {schema}memory_units(bank_id, fact_type, occurred_start) "
        f"WHERE occurred_start IS NOT NULL",
    )
    # Partial index on occurred_end (covers "occurred_end BETWEEN $4 AND $5")
    _create_index(
        schema,
        f"IF NOT EXISTS idx_memory_units_bank_occurred_end "
        f"ON {schema}memory_units(bank_id, fact_type, occurred_end) "
        f"WHERE occurred_end IS NOT NULL",
    )
    # Partial index on mentioned_at (covers "mentioned_at BETWEEN $4 AND $5")
    _create_index(
        schema,
        f"IF NOT EXISTS idx_memory_units_bank_mentioned_at "
        f"ON {schema}memory_units(bank_id, fact_type, mentioned_at) "
        f"WHERE mentioned_at IS NOT NULL",
    )


def downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute("COMMIT")
    op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_memory_units_bank_mentioned_at")
    op.execute("COMMIT")
    op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_memory_units_bank_occurred_end")
    op.execute("COMMIT")
    op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_memory_units_bank_occurred_start")
