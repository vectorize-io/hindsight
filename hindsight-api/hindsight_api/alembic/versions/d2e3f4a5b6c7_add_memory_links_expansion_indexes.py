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

Uses CONCURRENTLY for schemas with existing data (avoids blocking writes during
production deployments on large tables). Uses regular CREATE INDEX for empty
schemas (new tenant provisioning) because CONCURRENTLY waits for ALL open
transactions database-wide, which deadlocks against the worker's polling loop.

Revision ID: d2e3f4a5b6c7
Revises: b3c4d5e6f7g8
Create Date: 2026-03-02
"""

from collections.abc import Sequence

from alembic import context, op
from sqlalchemy import text

revision: str = "d2e3f4a5b6c7"
down_revision: str | Sequence[str] | None = "b3c4d5e6f7g8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _table_has_data(table: str) -> bool:
    """Check if a table has any rows (used to decide CONCURRENTLY vs regular)."""
    conn = op.get_bind()
    result = conn.execute(text(f"SELECT EXISTS (SELECT 1 FROM {table} LIMIT 1)"))
    return result.scalar()


def _create_index(schema: str, definition: str) -> None:
    """Create an index, using CONCURRENTLY only when the table has data."""
    # Extract table name from "ON {schema}table_name(...)"
    table_part = definition.split(" ON ")[1].split("(")[0].strip()
    if _table_has_data(table_part):
        op.execute("COMMIT")
        op.execute(f"CREATE INDEX CONCURRENTLY {definition}")
    else:
        op.execute(f"CREATE INDEX {definition}")


def upgrade() -> None:
    schema = _get_schema_prefix()

    # Index for the semantic *incoming* direction in link_expansion_retrieval.py.
    # Replaces the BitmapAnd of idx_memory_links_to_unit ∩ idx_memory_links_link_type
    # with a single composite index scan.
    _create_index(
        schema,
        f"IF NOT EXISTS idx_memory_links_to_type_weight "
        f"ON {schema}memory_links(to_unit_id, link_type, weight DESC)",
    )

    # Covering index for entity co-occurrence expansion.
    # Enables an index-only scan: entity_id and to_unit_id are read from the
    # index leaf pages instead of the heap, eliminating ~2 500 random heap-page
    # reads per expansion query.
    _create_index(
        schema,
        f"IF NOT EXISTS idx_memory_links_entity_covering "
        f"ON {schema}memory_links(from_unit_id) "
        f"INCLUDE (to_unit_id, entity_id) "
        f"WHERE link_type = 'entity'",
    )


def downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute("COMMIT")
    op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_memory_links_entity_covering")
    op.execute("COMMIT")
    op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_memory_links_to_type_weight")
