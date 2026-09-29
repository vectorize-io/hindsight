"""Make event_date nullable in memory_units to support timestamp-free content

Revision ID: aa2b3c4d5e6f
Revises: z1u2v3w4x5y6
Create Date: 2026-03-02

When callers retain content without a timestamp (e.g. fictional documents, static text),
the event_date column should be allowed to be NULL rather than defaulting to utcnow().
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "aa2b3c4d5e6f"
down_revision: str | Sequence[str] | None = "z1u2v3w4x5y6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"ALTER TABLE {schema}memory_units ALTER COLUMN event_date DROP NOT NULL")


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()
    # Backfill NULLs with now() before restoring the NOT NULL constraint
    op.execute(f"UPDATE {schema}memory_units SET event_date = now() WHERE event_date IS NULL")
    op.execute(f"ALTER TABLE {schema}memory_units ALTER COLUMN event_date SET NOT NULL")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
