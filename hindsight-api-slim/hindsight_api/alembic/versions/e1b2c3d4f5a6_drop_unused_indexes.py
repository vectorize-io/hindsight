"""Drop indexes that are unused or redundant with composite indexes.

Code audit identified the following indexes as either dead (no code path
exercises them) or fully covered by composite indexes the planner already
prefers:

memory_links:
1. idx_memory_links_entity_covering — entity co-occurrence expansion was
   rewritten to traverse unit_entities instead of memory_links, so no code
   path filters memory_links on (link_type = 'entity').
2. idx_memory_links_from_unit — redundant.  idx_memory_links_from_type_weight
   (from_unit_id, link_type, weight DESC) leads with the same column and
   answers every from_unit_id = X query.
3. idx_memory_links_to_unit — redundant.  idx_memory_links_to_type_weight
   (to_unit_id, link_type, weight DESC) leads with the same column.
4. idx_memory_links_link_type — no application query filters on link_type
   alone; the composite indexes above serve every (from/to + link_type)
   predicate.

entities:
5. idx_entities_canonical_name — superseded by
   entities_canonical_name_lower_trgm_idx (case-insensitive lookups).
6. entities_canonical_name_trgm_idx — superseded by the lowercase variant
   in migration 2eee35aa3cfc, but the original was never dropped on schemas
   that ran the prior migration.

documents:
7. idx_documents_retain_params — GIN index on retain_params JSONB; no query
   uses jsonb containment on this column.
8. idx_documents_content_hash — content-hash lookups happen on the chunks
   table (chunks.content_hash, indexed separately).

unit_entities:
9. idx_unit_entities_entity — defensive drop.  Migration h3i4j5k6l7m8 already
   issues DROP INDEX IF EXISTS for this; this re-runs the drop idempotently
   to cover any schema that missed the previous migration.

All drops use CONCURRENTLY + IF EXISTS so they neither block writers nor
fail on schemas where the index is already gone.

Revision ID: e1b2c3d4f5a6
Revises: p4q5r6s7t8u9
Create Date: 2026-05-26
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "e1b2c3d4f5a6"
down_revision: str | Sequence[str] | None = "p4q5r6s7t8u9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_PG_INDEXES_TO_DROP: tuple[str, ...] = (
    "idx_memory_links_entity_covering",
    "idx_memory_links_from_unit",
    "idx_memory_links_to_unit",
    "idx_memory_links_link_type",
    "idx_entities_canonical_name",
    "entities_canonical_name_trgm_idx",
    "idx_documents_retain_params",
    "idx_documents_content_hash",
    "idx_unit_entities_entity",
)


def _schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _schema_prefix()
    # DROP INDEX CONCURRENTLY cannot run inside a transaction block; commit
    # the Alembic transaction and issue each statement in its own implicit
    # autocommit transaction.  IF EXISTS makes each statement idempotent
    # across schemas that already dropped (or never had) the index.
    for index_name in _PG_INDEXES_TO_DROP:
        op.execute("COMMIT")
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}{index_name}")


def _pg_downgrade() -> None:
    schema = _schema_prefix()

    # Recreate the dropped indexes in the same shape the prior migrations used,
    # so a downgrade leaves the schema in the state the previous head expected.
    op.execute("COMMIT")
    op.execute(
        f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memory_links_entity_covering "
        f"ON {schema}memory_links(from_unit_id) "
        f"INCLUDE (to_unit_id, entity_id) "
        f"WHERE link_type = 'entity'"
    )
    op.execute("COMMIT")
    op.execute(
        f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memory_links_from_unit ON {schema}memory_links(from_unit_id)"
    )
    op.execute("COMMIT")
    op.execute(f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memory_links_to_unit ON {schema}memory_links(to_unit_id)")
    op.execute("COMMIT")
    op.execute(f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memory_links_link_type ON {schema}memory_links(link_type)")
    op.execute("COMMIT")
    op.execute(
        f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entities_canonical_name ON {schema}entities(canonical_name)"
    )
    op.execute("COMMIT")
    op.execute(
        f"CREATE INDEX CONCURRENTLY IF NOT EXISTS entities_canonical_name_trgm_idx "
        f"ON {schema}entities USING GIN (canonical_name gin_trgm_ops)"
    )
    op.execute("COMMIT")
    op.execute(
        f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_retain_params "
        f"ON {schema}documents USING GIN (retain_params)"
    )
    op.execute("COMMIT")
    op.execute(f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_content_hash ON {schema}documents(content_hash)")
    op.execute("COMMIT")
    op.execute(f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_unit_entities_entity ON {schema}unit_entities(entity_id)")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
