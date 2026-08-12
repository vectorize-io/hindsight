"""Add async_operations.serialization_key for per-document retain serialization.

``update_mode="append"`` is a read-modify-write over the whole document: the
retain reads ``documents.original_text``, concatenates the new content onto it,
and reprocesses the result. Two appends to one document whose read→write
windows overlap therefore lose an update — the loser's turn is content nobody
else has.

The orchestrator now detects that at write time and fails the loser instead of
committing over it, but detection alone turns lost data into wasted extraction.
This column lets the worker's claim query keep a document to one in-flight
retain at a time, so the conflict is avoided rather than paid for: a second
retain for the same document simply is not claimed until the first finishes,
and the waiting operation holds no worker slot while it waits.

It carries the single document an operation targets (NULL when it targets none
or several), so the claim predicate can compare it without digging into
``task_payload`` — a shape both dialects index cheaply and which the Oracle
rewrite of the claim SQL can handle.

The partial index covers only live rows: claims never look at terminal
operations, and retain queues are dominated by completed history.

Revision ID: d9c1a7b4e2f6
Revises: b3e8d1c6f4a9
Create Date: 2026-08-11
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "d9c1a7b4e2f6"
down_revision: str | Sequence[str] | None = "b3e8d1c6f4a9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEX = "idx_async_operations_serialization_key"


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = context.config.get_main_option("target_schema")
    op.add_column(
        "async_operations",
        sa.Column("serialization_key", sa.Text(), nullable=True),
        schema=schema or None,
    )
    prefix = _pg_schema_prefix()
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {_INDEX} ON {prefix}async_operations "
        f"(bank_id, serialization_key) "
        f"WHERE serialization_key IS NOT NULL AND status IN ('pending', 'processing')"
    )


def _pg_downgrade() -> None:
    schema = context.config.get_main_option("target_schema")
    prefix = _pg_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {prefix}{_INDEX}")
    op.drop_column("async_operations", "serialization_key", schema=schema or None)


def _oracle_upgrade() -> None:
    op.add_column("async_operations", sa.Column("serialization_key", sa.String(4000), nullable=True))
    # Oracle has no partial indexes. A function-based index on the same
    # predicate gets the equivalent selectivity: terminal rows collapse to NULL
    # and Oracle does not store all-NULL entries, so the index only holds the
    # live rows the claim query looks at.
    op.get_bind().exec_driver_sql(
        f"CREATE INDEX {_INDEX} ON async_operations ("
        f"  CASE WHEN status IN ('pending', 'processing') THEN bank_id END,"
        f"  CASE WHEN status IN ('pending', 'processing') THEN serialization_key END)"
    )


def _oracle_downgrade() -> None:
    op.get_bind().exec_driver_sql(f"DROP INDEX {_INDEX}")
    op.drop_column("async_operations", "serialization_key")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
