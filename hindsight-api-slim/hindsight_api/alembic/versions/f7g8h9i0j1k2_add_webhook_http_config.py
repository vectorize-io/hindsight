"""Add http_config JSONB column to webhooks table.

Stores HTTP delivery configuration (method, timeout, headers, params) as a
single JSONB column rather than separate columns.

Revision ID: f7g8h9i0j1k2
Revises: e4f5a6b7c8d9
Create Date: 2026-03-04
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "f7g8h9i0j1k2"
down_revision: str | Sequence[str] | None = "e4f5a6b7c8d9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"ALTER TABLE {schema}webhooks ADD COLUMN IF NOT EXISTS http_config JSONB NOT NULL DEFAULT '{{}}'")


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"ALTER TABLE {schema}webhooks DROP COLUMN IF EXISTS http_config")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
