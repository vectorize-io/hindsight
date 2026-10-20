"""Clear year-1 / year-9999 placeholder dates from memory_units

Revision ID: c7e2f9a4d1b8
Revises: f2a9c4e7b1d3
Create Date: 2026-10-20

A memory whose occurred range was open-ended could be stored with datetime.min/max, which
Postgres keeps as -infinity/infinity. asyncpg reads those back without a timezone while every
finite value is aware, so the row broke temporal recall — and the consolidation that runs it —
with "can't subtract offset-naive and offset-aware datetimes" on every retry (#5820). New writes
now treat year 1 / 9999 as an unknown bound (see ``hindsight_api.engine.stored_dates``); this
clears the rows already written the same way.

Not reversible: the placeholder carried no date to restore.
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "c7e2f9a4d1b8"
down_revision: str | Sequence[str] | None = "f2a9c4e7b1d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DATE_COLUMNS = ("occurred_start", "occurred_end", "mentioned_at", "event_date")


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _placeholder(col: str, low: str, high: str) -> str:
    # -infinity sorts below and infinity above every finite value, so one range test covers
    # both the infinite and the finite year-1 / year-9999 forms.
    return f"({col} < {low} OR {col} >= {high})"


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    low, high = "'0002-01-01T00:00:00Z'::timestamptz", "'9999-01-01T00:00:00Z'::timestamptz"
    sets = ", ".join(f"{c} = CASE WHEN {_placeholder(c, low, high)} THEN NULL ELSE {c} END" for c in _DATE_COLUMNS)
    where = " OR ".join(_placeholder(c, low, high) for c in _DATE_COLUMNS)
    op.execute(f"UPDATE {schema}memory_units SET {sets} WHERE {where}")


def _oracle_upgrade() -> None:
    low, high = "TIMESTAMP '0002-01-01 00:00:00 +00:00'", "TIMESTAMP '9999-01-01 00:00:00 +00:00'"
    # event_date is NOT NULL on Oracle: fall back to when the row was written.
    sets = ", ".join(
        f"{c} = CASE WHEN {_placeholder(c, low, high)} THEN {'created_at' if c == 'event_date' else 'NULL'} ELSE {c} END"
        for c in _DATE_COLUMNS
    )
    where = " OR ".join(_placeholder(c, low, high) for c in _DATE_COLUMNS)
    op.execute(f"UPDATE memory_units SET {sets} WHERE {where}")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    # Data-only and lossy by design: both slots intentionally absent, nothing to undo.
    run_for_dialect()
