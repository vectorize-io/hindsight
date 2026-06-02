"""Recreate idx_memory_units_source_memory_ids GIN index with fastupdate=off

GIN indexes use a "fastupdate" pending list by default: small writes are
buffered there and flushed to the main GIN tree in bulk. Flushing requires
AccessExclusiveLock on the index. Under high insert concurrency (e.g. 8
parallel pytest-xdist workers all calling retain_async) two transactions can
each trigger a flush simultaneously and deadlock.

Disabling fastupdate makes every insert write directly to the GIN tree
(slightly slower per insert, but no pending-list lock cycles).

Revision ID: d4e5f6g7h8i9
Revises: d5e6f7a8b9c0
Create Date: 2026-03-11
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "d4e5f6g7h8i9"
down_revision: str | Sequence[str] | None = "d5e6f7a8b9c0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()
    # DROP + CREATE CONCURRENTLY must run outside a transaction block; an
    # autocommit_block runs them outside Alembic's migration transaction.
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_memory_units_source_memory_ids")
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memory_units_source_memory_ids "
            f"ON {schema}memory_units USING GIN (source_memory_ids) "
            f"WITH (fastupdate=off) "
            f"WHERE source_memory_ids IS NOT NULL"
        )


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_memory_units_source_memory_ids")
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memory_units_source_memory_ids "
            f"ON {schema}memory_units USING GIN (source_memory_ids) "
            f"WHERE source_memory_ids IS NOT NULL"
        )


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
