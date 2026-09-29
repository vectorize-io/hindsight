"""Add covering and composite indexes to speed up link expansion graph retrieval.

Two indexes target the two bottlenecks identified by EXPLAIN ANALYZE on a 17M-row
memory_links table:

1. idx_memory_links_to_type_weight  (to_unit_id, link_type, weight DESC)
   The semantic incoming direction — finding facts that consider seeds as their
   nearest neighbour — currently hits an expensive BitmapAnd of two separate
   bitmap scans (to_unit_id bitmap ∩ link_type bitmap).  A composite index
   on (to_unit_id, link_type) turns this into a single index scan and reduces
   latency from ~36 ms to < 5 ms per query.

2. idx_memory_links_entity_covering  (from_unit_id) INCLUDE (to_unit_id, entity_id)
   WHERE link_type = 'entity'
   The entity co-occurrence expansion uses COUNT(DISTINCT ml.entity_id) and
   joins on ml.to_unit_id.  Without a covering index the planner must read
   ~2 500 heap pages to fetch entity_id and to_unit_id after the bitmap index
   scan, adding ~230 ms of random I/O.  INCLUDE adds those two columns to the
   index leaf pages so the entire query can be served from the index (index-only
   scan), eliminating the heap reads entirely.
   Partial index (WHERE link_type = 'entity') keeps index size ~40 % smaller.

Both indexes are created with CONCURRENTLY so the migration does not block
concurrent reads or writes on memory_links.  CONCURRENTLY requires running
outside a transaction block, so the migration emits an explicit COMMIT before
each statement and uses IF NOT EXISTS for idempotency.

Revision ID: d2e3f4a5b6c7
Revises: b3c4d5e6f7g8
Create Date: 2026-03-02
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "d2e3f4a5b6c7"
down_revision: str | Sequence[str] | None = "b3c4d5e6f7g8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()

    # CREATE INDEX CONCURRENTLY cannot run inside a transaction block; an
    # autocommit_block runs each statement outside Alembic's migration
    # transaction.  IF NOT EXISTS makes each statement idempotent on retry.
    with op.get_context().autocommit_block():
        # Index for the semantic *incoming* direction in link_expansion_retrieval.py.
        # Replaces the BitmapAnd of idx_memory_links_to_unit ∩ idx_memory_links_link_type
        # with a single composite index scan.
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memory_links_to_type_weight "
            f"ON {schema}memory_links(to_unit_id, link_type, weight DESC)"
        )

        # Covering index for entity co-occurrence expansion.
        # Enables an index-only scan: entity_id and to_unit_id are read from the
        # index leaf pages instead of the heap, eliminating ~2 500 random heap-page
        # reads per expansion query.
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memory_links_entity_covering "
            f"ON {schema}memory_links(from_unit_id) "
            f"INCLUDE (to_unit_id, entity_id) "
            f"WHERE link_type = 'entity'"
        )


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_memory_links_entity_covering")
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_memory_links_to_type_weight")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
