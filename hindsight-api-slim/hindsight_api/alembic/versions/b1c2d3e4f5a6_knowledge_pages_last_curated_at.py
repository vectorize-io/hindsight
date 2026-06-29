"""Add last_curated_at watermark to knowledge_pages folders.

The folder curator looks at the memories created since it last ran (a delta,
like the mental-model refresh's ``created_after`` scope) rather than a semantic
recall. ``last_curated_at`` is that per-folder watermark.

Revision ID: b1c2d3e4f5a6
Revises: a5b6c7d8e9f0
Create Date: 2026-06-26
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "b1c2d3e4f5a6"
down_revision: str | Sequence[str] | None = "a5b6c7d8e9f0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {schema}knowledge_pages ADD COLUMN IF NOT EXISTS last_curated_at TIMESTAMP WITH TIME ZONE")


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {schema}knowledge_pages DROP COLUMN IF EXISTS last_curated_at")


def _oracle_upgrade() -> None:
    op.execute("ALTER TABLE knowledge_pages ADD (last_curated_at TIMESTAMP WITH TIME ZONE)")


def _oracle_downgrade() -> None:
    op.execute("ALTER TABLE knowledge_pages DROP COLUMN last_curated_at")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
