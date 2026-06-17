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

All indexes are created CONCURRENTLY so the migration does not block writes on
memory_units during production deployments. CONCURRENTLY requires running outside
a transaction block; see migrations.py for how this is handled safely.
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "b3c4d5e6f7g8"
down_revision: str | Sequence[str] | None = "c1a2b3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()
    # CREATE INDEX CONCURRENTLY cannot run inside a transaction block; an
    # autocommit_block runs each statement outside Alembic's migration transaction.
    with op.get_context().autocommit_block():
        # Partial index on occurred_start (covers "occurred_start BETWEEN $4 AND $5")
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memory_units_bank_occurred_start "
            f"ON {schema}memory_units(bank_id, fact_type, occurred_start) "
            f"WHERE occurred_start IS NOT NULL"
        )
        # Partial index on occurred_end (covers "occurred_end BETWEEN $4 AND $5")
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memory_units_bank_occurred_end "
            f"ON {schema}memory_units(bank_id, fact_type, occurred_end) "
            f"WHERE occurred_end IS NOT NULL"
        )
        # Partial index on mentioned_at (covers "mentioned_at BETWEEN $4 AND $5")
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memory_units_bank_mentioned_at "
            f"ON {schema}memory_units(bank_id, fact_type, mentioned_at) "
            f"WHERE mentioned_at IS NOT NULL"
        )


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_memory_units_bank_mentioned_at")
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_memory_units_bank_occurred_end")
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_memory_units_bank_occurred_start")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
