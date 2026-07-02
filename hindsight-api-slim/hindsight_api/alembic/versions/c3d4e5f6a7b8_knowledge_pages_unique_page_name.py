"""Unique page name per folder in knowledge_pages.

The folder curator can fire concurrently (folder-create trigger + the
post-consolidation sweep), and an in-process lock can't serialize runs that
execute in different threads/loops. A partial unique index on
(bank_id, parent, lower(name)) for pages makes duplicate-named pages in the same
folder impossible at the DB level — the second concurrent insert fails and the
curator treats it as "already exists".

PostgreSQL only: the Oracle ``name`` column is a CLOB and cannot back a
functional unique index; Oracle relies on the in-process serialization instead.

Revision ID: c3d4e5f6a7b8
Revises: a5b6c7d8e9f0
Create Date: 2026-06-26
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "c3d4e5f6a7b8"
down_revision: str | Sequence[str] | None = "a5b6c7d8e9f0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    # First drop any pre-existing duplicate pages (created by the racy curator
    # before this guard existed), keeping the earliest row of each duplicate set,
    # so the unique index can be built. Their backing mental models are left in
    # place (harmless orphans).
    op.execute(
        f"""
        DELETE FROM {schema}knowledge_pages a
        USING {schema}knowledge_pages b
        WHERE a.kind = 'page' AND b.kind = 'page'
          AND a.bank_id = b.bank_id
          AND COALESCE(a.parent_id, '') = COALESCE(b.parent_id, '')
          AND lower(a.name) = lower(b.name)
          AND a.ctid > b.ctid
        """
    )
    # COALESCE(parent_id, '') so root-level pages (NULL parent) are also unique by
    # name — NULLs would otherwise compare distinct and allow duplicates.
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_kp_folder_pagename "
        f"ON {schema}knowledge_pages (bank_id, COALESCE(parent_id, ''), lower(name)) "
        "WHERE kind = 'page'"
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}uq_kp_folder_pagename")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)  # oracle slot intentionally absent (CLOB name)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
