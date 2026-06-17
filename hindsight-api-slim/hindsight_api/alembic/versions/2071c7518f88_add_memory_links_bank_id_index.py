"""Add a composite index on memory_links(bank_id, link_type) (PostgreSQL).

``bank_id`` was added to ``memory_links`` in ``c5d6e7f8a9b0`` precisely so that
bank-scoped reads (e.g. the stats endpoint) could filter on the link table
directly instead of joining ``memory_units`` — that JOIN took 18+ seconds on
banks with millions of links. The column landed without an index, so every
``bank_id = $1`` predicate still falls back to a sequential scan over the whole
table.

This adds the missing btree. It is composite on ``(bank_id, link_type)`` rather
than ``bank_id`` alone because the hot query is the stats endpoint's
``SELECT link_type, COUNT(*) ... WHERE bank_id = $1 GROUP BY link_type``: a
``(bank_id, link_type)`` index serves that filter, grouping and count as an
index-only scan, never touching the heap, whereas a ``bank_id``-only index would
still have to read every matching row to recover ``link_type``. ``link_type`` is
low-cardinality (only ``temporal``/``semantic``/``caused_by`` are written —
entity edges were dropped in ``e9b2c7d1f3a4``), so the trailing column adds
little to the index size while removing the heap fetch.

The Oracle baseline (``o1a2b3c4d5e6``) already creates ``idx_ml_bank_id`` on
``memory_links(bank_id)``; that single-column index already covers Oracle's
bank-scoped filter, so the Oracle slot here is intentionally absent and only the
PostgreSQL dialect gets the composite index.

``memory_links`` can hold tens of millions of rows, so the index is built
CONCURRENTLY to avoid taking a write lock on the table. CONCURRENTLY cannot run
inside a transaction block, so the statement runs in an ``autocommit_block()``;
``IF NOT EXISTS`` keeps it idempotent across retries and re-migrated tenant
schemas. A CONCURRENTLY build interrupted partway (lock conflict, disk
pressure, signal) leaves the index behind as *invalid*; ``IF NOT EXISTS`` would
then skip over it forever, so the upgrade first drops any invalid leftover of
this name before (re)creating it.

Revision ID: 2071c7518f88
Revises: a1d3f5b7c9e2
Create Date: 2026-06-16
"""

from collections.abc import Sequence

from alembic import context, op
from sqlalchemy import text

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "2071c7518f88"
down_revision: str | Sequence[str] | None = "a1d3f5b7c9e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEX_NAME = "idx_memory_links_bank_id_link_type"


def _get_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    bind = op.get_bind()
    # `or None` collapses an unset option and an explicit empty string into NULL
    # so the COALESCE below falls back to current_schema() in both cases.
    target_schema = context.config.get_main_option("target_schema") or None
    schema = _get_schema_prefix()

    # CREATE INDEX CONCURRENTLY cannot run inside a transaction block; the
    # autocommit_block runs each statement outside Alembic's migration
    # transaction.
    with op.get_context().autocommit_block():
        # A CONCURRENTLY build that errored on a previous run leaves an INVALID
        # index of this name behind. `CREATE INDEX ... IF NOT EXISTS` would see
        # that relation and skip, so bank_id queries would keep seq-scanning.
        # Drop only the invalid leftover — never a healthy index — so the retry
        # actually rebuilds a usable one.
        leftover_invalid = bind.execute(
            text(
                "SELECT NOT i.indisvalid "
                "FROM pg_class c "
                "JOIN pg_index i ON c.oid = i.indexrelid "
                "JOIN pg_namespace n ON c.relnamespace = n.oid "
                "WHERE c.relname = :index_name "
                "  AND n.nspname = COALESCE(:target_schema, current_schema())"
            ),
            {"index_name": _INDEX_NAME, "target_schema": target_schema},
        ).scalar()
        if leftover_invalid:
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}{_INDEX_NAME}")

        # IF NOT EXISTS keeps the create idempotent across retries and schemas.
        op.execute(f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {_INDEX_NAME} ON {schema}memory_links(bank_id, link_type)")


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}{_INDEX_NAME}")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
