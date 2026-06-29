"""Add mission + managed to knowledge_pages (folder curator).

Folders gain a ``mission`` — the steering prompt a curator uses after each
consolidation to decide which pages should exist under that folder. Pages gain a
``managed`` flag: curator-created pages are ``managed = true`` (the curator may
merge/delete them), human-created pages stay ``managed = false`` (pinned — the
curator never touches them).

Revision ID: a5b6c7d8e9f0
Revises: a9b8c7d6e5f4
Create Date: 2026-06-26
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "a5b6c7d8e9f0"
down_revision: str | Sequence[str] | None = "a9b8c7d6e5f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {schema}knowledge_pages ADD COLUMN IF NOT EXISTS mission TEXT")
    op.execute(f"ALTER TABLE {schema}knowledge_pages ADD COLUMN IF NOT EXISTS managed BOOLEAN NOT NULL DEFAULT false")


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {schema}knowledge_pages DROP COLUMN IF EXISTS managed")
    op.execute(f"ALTER TABLE {schema}knowledge_pages DROP COLUMN IF EXISTS mission")


def _oracle_upgrade() -> None:
    op.execute("ALTER TABLE knowledge_pages ADD (mission CLOB)")
    op.execute("ALTER TABLE knowledge_pages ADD (managed NUMBER(1) DEFAULT 0 NOT NULL)")


def _oracle_downgrade() -> None:
    op.execute("ALTER TABLE knowledge_pages DROP COLUMN managed")
    op.execute("ALTER TABLE knowledge_pages DROP COLUMN mission")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
