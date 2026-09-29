"""Track delta all-skipped failures independently of optional audit history.

Revision ID: f6a2b8c4d0e1
Revises: e5b1c7d3a902
Create Date: 2026-09-29
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "f6a2b8c4d0e1"
down_revision: str | Sequence[str] | None = "e5b1c7d3a902"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(
        f"ALTER TABLE {schema}mental_models ADD COLUMN IF NOT EXISTS delta_all_skip_streak INTEGER NOT NULL DEFAULT 0"
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {schema}mental_models DROP COLUMN IF EXISTS delta_all_skip_streak")


def _oracle_upgrade() -> None:
    op.execute("ALTER TABLE mental_models ADD (delta_all_skip_streak NUMBER(10) DEFAULT 0 NOT NULL)")


def _oracle_downgrade() -> None:
    op.execute("ALTER TABLE mental_models DROP COLUMN delta_all_skip_streak")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
