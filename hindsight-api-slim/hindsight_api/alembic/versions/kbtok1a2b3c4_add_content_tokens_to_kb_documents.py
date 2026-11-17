"""Add content_tokens to kb_documents

A knowledge-bank document's billable content size: the tokens of its text, or — for a binary
media file whose content is bytes, not words — its size at a fixed bytes-per-token rate. Set once
when the document is created and read back for metering (an ingest bills it, a delete decrements
the stored total). Knowledge-bank tables are PostgreSQL-only (see k7b1c2d3e4f5), so Oracle is a
no-op.

Revision ID: kbtok1a2b3c4
Revises: z1u2v3w4x5y6
Create Date: 2026-11-17
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "kbtok1a2b3c4"
down_revision: str | Sequence[str] | None = "z1u2v3w4x5y6"
branch_labels: str | Sequence[str] | None = None
# The KB tables are created by k7b1c2d3e4f5, which is on a different branch from
# this migration's down_revision chain. Without this, a from-scratch multi-head
# upgrade can run this ALTER before that CREATE ("relation kb_documents does not
# exist"). depends_on forces the CREATE to apply first.
depends_on: str | Sequence[str] | None = "k7b1c2d3e4f5"


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(
        f"ALTER TABLE {schema}kb_documents ADD COLUMN IF NOT EXISTS content_tokens BIGINT NOT NULL DEFAULT 0"
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {schema}kb_documents DROP COLUMN IF EXISTS content_tokens")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)  # oracle slot intentionally absent: kb_* is PostgreSQL-only


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
