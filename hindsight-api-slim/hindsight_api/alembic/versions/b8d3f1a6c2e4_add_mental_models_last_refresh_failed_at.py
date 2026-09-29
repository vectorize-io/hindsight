"""Add last_refresh_failed_at column to mental_models

Revision ID: b8d3f1a6c2e4
Revises: a7c2e9f41b60
Create Date: 2026-09-22

When the most recent refresh of a mental model failed. A failed refresh leaves
``last_refreshed_at`` where it was, so on its own the model looks stale forever
and the automatic triggers re-queued the same doomed refresh every tick (#4532).
While this is later than ``last_refreshed_at`` the automatic triggers leave the
model alone; the next successful refresh (an explicit one) moves
``last_refreshed_at`` past it and resumes them.
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "b8d3f1a6c2e4"
down_revision: str | Sequence[str] | None = "a7c2e9f41b60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(
        f"ALTER TABLE {schema}mental_models ADD COLUMN IF NOT EXISTS last_refresh_failed_at TIMESTAMP WITH TIME ZONE"
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {schema}mental_models DROP COLUMN IF EXISTS last_refresh_failed_at")


def _oracle_upgrade() -> None:
    op.execute("ALTER TABLE mental_models ADD (last_refresh_failed_at TIMESTAMP WITH TIME ZONE)")


def _oracle_downgrade() -> None:
    op.execute("ALTER TABLE mental_models DROP COLUMN last_refresh_failed_at")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
