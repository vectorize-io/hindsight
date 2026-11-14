"""Knowledge banks: keyword search on plain words, in every language.

Passages' keyword index (``kb_passages.search_vector``) was built with their
document's detected language (stemming, stopwords of 6 languages), and keyword
search matched only documents in the query's language. It now uses plain words
('simple': no stemming, no language), so the existing index is rebuilt that way,
and the documents' ``language`` column, read by nothing now, is dropped.

PostgreSQL only: the kb_* tables do not exist on Oracle (see k7b1c2d3e4f5).

Revision ID: 300fe58170e9
Revises: k7b1c2d3e4f5
Create Date: 2026-11-13
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "300fe58170e9"
down_revision: str | Sequence[str] | None = "k7b1c2d3e4f5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    # One statement over every passage: knowledge banks are in no release yet, so only dev and
    # benchmark databases hold passages; batching it would matter once a release ships them.
    schema = _pg_schema_prefix()
    op.execute(f"UPDATE {schema}kb_passages SET search_vector = to_tsvector('simple', coalesce(text, ''))")
    op.execute(f"ALTER TABLE {schema}kb_documents DROP COLUMN IF EXISTS language")


def _pg_downgrade() -> None:
    # The per-language index is not rebuilt: plain words still match, and no language is known any more.
    schema = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {schema}kb_documents ADD COLUMN IF NOT EXISTS language TEXT")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)  # oracle slot intentionally absent: no kb_* tables there


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
