"""Repair: drop every global memory_units vector index on per-bank backends.

Revision ID: e8c1a4f7b2d9
Revises: d4f7a2c9e1b5
Create Date: 2026-10-19

f2a6d8c4b1e9 dropped the stale global index by one name,
``idx_memory_units_embedding``. The startup embedding-dimension resize built
another one under a different name (``idx_memory_units_embedding_<type>``, e.g.
``_hnsw`` on pgvector) on any fresh schema whose model dimension differed from
the column default, after migrations had already run (#5707). The resize no
longer does that; this repair removes what it left behind.

It looks the indexes up in the catalog instead of by name: every non-partial
vector index on ``memory_units.embedding`` is dead weight on these backends,
since every vector search is served by the per-bank ``idx_mu_emb_*`` partial
indexes. Same lock guard and ScaNN exemption as f2a6d8c4b1e9.
"""

import os
from collections.abc import Sequence

from alembic import context, op
from sqlalchemy import text

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "e8c1a4f7b2d9"
down_revision: str | Sequence[str] | None = "d4f7a2c9e1b5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_upgrade() -> None:
    # ScaNN uses a global vector index by design — nothing stale to repair.
    if os.getenv("HINDSIGHT_API_VECTOR_EXTENSION", "pgvector").lower() == "scann":
        return
    # An unset target_schema arrives as "" or None depending on the alembic version;
    # both fall back to current_schema(), like 2071c7518f88.
    target_schema = context.config.get_main_option("target_schema") or None
    rows = (
        op.get_bind()
        .execute(
            text("""
            SELECT n.nspname, i.relname
            FROM pg_index x
            JOIN pg_class t ON t.oid = x.indrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            JOIN pg_class i ON i.oid = x.indexrelid
            JOIN pg_am am ON am.oid = i.relam
            WHERE n.nspname = COALESCE(:target_schema, current_schema())
              AND t.relname = 'memory_units'
              AND x.indpred IS NULL
              AND am.amname IN ('hnsw', 'vchordrq', 'diskann', 'scann')
        """),
            {"target_schema": target_schema},
        )
        .fetchall()
    )
    # Fail fast on a write-busy table rather than queue traffic behind
    # ACCESS EXCLUSIVE; the next migration pass retries (see f2a6d8c4b1e9).
    op.execute("SET LOCAL lock_timeout = '10s'")
    for schema_name, index_name in rows:
        # DDL identifiers cannot be bound parameters, so escape inline.
        safe_schema = schema_name.replace('"', '""')
        safe_index = index_name.replace('"', '""')
        op.execute(f'DROP INDEX IF EXISTS "{safe_schema}"."{safe_index}"')


def _pg_downgrade() -> None:
    # Intentional no-op: recreating an unused vector index is not a safe downgrade.
    pass


def upgrade() -> None:
    # PG-only, like f2a6d8c4b1e9: Oracle never had the resize that created it.
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
