"""Store per-request reasoning tokens alongside visible output tokens.

The tracing table exists only on PostgreSQL, so Oracle intentionally has no
operation. Existing rows remain null because their reasoning usage cannot be
reconstructed from the visible-only counts.

Revision ID: a6c4e8f1b203
Revises: d4f8b1c6e903
Create Date: 2026-09-24
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "a6c4e8f1b203"
down_revision: str | Sequence[str] | None = "d4f8b1c6e903"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    op.execute(f"ALTER TABLE {_pg_schema_prefix()}llm_requests ADD COLUMN IF NOT EXISTS thoughts_tokens INTEGER")


def _pg_downgrade() -> None:
    op.execute(f"ALTER TABLE {_pg_schema_prefix()}llm_requests DROP COLUMN IF EXISTS thoughts_tokens")


def upgrade() -> None:
    # oracle slot intentionally absent: llm_requests is created PG-only
    # (d3e4f5a6b7c8), so there is no Oracle table to alter.
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
