"""Drop the unused memory_units_bm25 materialized view.

The initial schema created ``memory_units_bm25`` as a BM25 helper, but nothing
ever read it: keyword retrieval queries ``memory_units.search_vector`` (kept
current inline by retain and consolidation). The only thing that touched the
view was ``hindsight-admin restore``, which refreshed it — so it sat as a stale
snapshot that looked like a recall bug (#4338) and cost a full rebuild on every
restore.

PostgreSQL only: the Oracle baseline never created the view.

Revision ID: f3a5b7c9d1e2
Revises: e2f4a6c8b0d1
Create Date: 2026-09-14
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "f3a5b7c9d1e2"
down_revision: str | Sequence[str] | None = "e2f4a6c8b0d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    # Dropping the view drops its two indexes with it.
    op.execute(f"DROP MATERIALIZED VIEW IF EXISTS {schema}memory_units_bm25")


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"""
        CREATE MATERIALIZED VIEW IF NOT EXISTS {schema}memory_units_bm25 AS
        SELECT
            id,
            bank_id,
            text,
            to_tsvector('english', text) AS text_vector,
            log(1.0 + length(text)::float / (SELECT avg(length(text)) FROM {schema}memory_units)) AS doc_length_factor
        FROM {schema}memory_units
    """)
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_memory_units_bm25_bank ON {schema}memory_units_bm25 (bank_id)")
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_memory_units_bm25_text_vector ON {schema}memory_units_bm25 USING gin (text_vector)"
    )


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)  # oracle slot intentionally absent: the view never existed there


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
