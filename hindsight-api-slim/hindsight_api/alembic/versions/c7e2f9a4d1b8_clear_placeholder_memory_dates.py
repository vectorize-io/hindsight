"""Clear year-1 / year-9999 placeholder event_date values from memory_units

Revision ID: c7e2f9a4d1b8
Revises: f2a9c4e7b1d3
Create Date: 2026-10-20

A memory could be stored with datetime.min/max as a date, which Postgres keeps as
-infinity/infinity (#5820). Every Python reader now treats a year-1 / 9999 value as an unknown
bound (see ``hindsight_api.engine.stored_dates``), so most stored rows are harmless as they are.
event_date is the exception: retain's temporal-link query subtracts it in SQL, and PostgreSQL 16
and older refuse "cannot subtract infinite timestamps", failing retain into that bank.

This revision first cleared all four date columns with one UPDATE filtered on all of them. No
index leads with occurred_start/occurred_end/mentioned_at (their indexes start with bank_id), so
it read the whole table in every schema at startup to fix a handful of rows. It now touches only
event_date, which idx_memory_units_event_date serves as two index range probes.

PG only: Oracle has no infinite timestamps — a year-9999 value there is finite, so its SQL
arithmetic works and the Python readers already ignore it.

Not reversible: the placeholder carried no date to restore.
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "c7e2f9a4d1b8"
down_revision: str | Sequence[str] | None = "f2a9c4e7b1d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    # -infinity sorts below and infinity above every finite value, so one range test covers both
    # the infinite and the finite year-1 / year-9999 forms. event_date is nullable on PG.
    op.execute(
        f"UPDATE {schema}memory_units SET event_date = NULL "
        "WHERE event_date < '0002-01-01T00:00:00Z'::timestamptz "
        "OR event_date >= '9999-01-01T00:00:00Z'::timestamptz"
    )


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)  # oracle slot intentionally absent: see the module docstring


def downgrade() -> None:
    # Data-only and lossy by design: both slots intentionally absent, nothing to undo.
    run_for_dialect()
