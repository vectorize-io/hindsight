"""Add internal_id to banks and per-(bank, fact_type) partial vector indexes

Revision ID: d5e6f7a8b9c0
Revises: a3b4c5d6e7f8
Create Date: 2026-03-11

This migration:
1. Adds internal_id UUID column to banks (stable identifier for index naming)
2. For non-ScaNN backends, drops the global vector index (competes with
   per-bank partial indexes)
3. For non-ScaNN backends, creates per-(bank_id, fact_type) partial vector
   indexes for all existing banks using the configured vector extension
   (HNSW for pgvector, DiskANN for pgvectorscale, vchordrq for vchord).
   (new banks get indexes created at bank-creation time via bank_utils.create_bank_vector_indexes)

Why per-(bank, fact_type) indexes:
- fact_type-only partial indexes are never chosen by the planner when bank_id is in the WHERE
  clause, because the idx_memory_units_bank_id B-tree index always wins at planning time.
- Per-(bank, fact_type) partial indexes have both predicates matching → planner selects them.
- The global vector index competes for larger partitions (world, observation) and must be dropped.
- AlloyDB ScaNN uses global vector indexes with filtered vector search instead
  because empty or tiny per-bank indexes cannot be built safely.
"""

import os
from collections.abc import Sequence

from alembic import context, op
from sqlalchemy import text

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "d5e6f7a8b9c0"
down_revision: str | Sequence[str] | None = "c3d4e5f6g7h8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FACT_TYPES: dict[str, str] = {
    "world": "worl",
    "experience": "expr",
    "observation": "obsv",
}


def _configured_vector_extension() -> str:
    ext = os.getenv("HINDSIGHT_API_VECTOR_EXTENSION", "pgvector").lower()
    if ext not in {"pgvector", "pgvectorscale", "vchord", "scann"}:
        raise ValueError(
            f"Invalid HINDSIGHT_API_VECTOR_EXTENSION: {ext}. Must be 'pgvector', 'vchord', 'pgvectorscale', or 'scann'"
        )
    return ext


def _vector_index_using_clause(ext: str) -> str:
    if ext == "pgvectorscale":
        return "USING diskann (embedding vector_cosine_ops) WITH (num_neighbors = 50)"
    if ext == "vchord":
        return "USING vchordrq (embedding vector_l2_ops)"
    if ext == "scann":
        return "USING scann (embedding cosine) WITH (mode = 'AUTO')"
    return "USING hnsw (embedding vector_cosine_ops)"


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()

    # 1. Add internal_id column to banks
    op.execute(
        f"ALTER TABLE {schema}banks ADD COLUMN IF NOT EXISTS internal_id UUID DEFAULT gen_random_uuid() NOT NULL"
    )
    op.execute(f"ALTER TABLE {schema}banks ADD CONSTRAINT banks_internal_id_unique UNIQUE (internal_id)")

    ext = _configured_vector_extension()
    if ext == "scann":
        # ScaNN should keep/use a global vector index. Per-bank partial indexes
        # are created while banks are empty and can fail AlloyDB's ScaNN build
        # requirements, so this migration leaves vector index reconciliation to
        # runtime ensure_vector_extension once enough rows exist.
        return

    # 2. Drop any fact_type-only partial indexes that may exist from prior migrations
    #    (bank_id B-tree always wins over them when bank_id is in the WHERE clause)
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_mu_emb_world")
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_mu_emb_observation")
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_mu_emb_experience")

    # 4. Drop global vector index (competes with per-bank partial indexes)
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_embedding")

    # 5. Create per-(bank, fact_type) partial vector indexes for all existing banks
    #    using the configured extension (HNSW / DiskANN / vchordrq)
    bind = op.get_bind()
    schema_name = context.config.get_main_option("target_schema")
    table_ref = f'"{schema_name}".memory_units' if schema_name else "memory_units"
    banks_ref = f'"{schema_name}".banks' if schema_name else "banks"
    using_clause = _vector_index_using_clause(ext)

    rows = bind.execute(text(f"SELECT bank_id, internal_id FROM {banks_ref}")).fetchall()  # noqa: S608
    for row in rows:
        bank_id = row[0]
        internal_id = str(row[1]).replace("-", "")[:16]
        escaped_bank_id = bank_id.replace("'", "''")
        for ft, ft_short in _FACT_TYPES.items():
            idx_name = f"idx_mu_emb_{ft_short}_{internal_id}"
            # Index name is schema-unqualified (indexes live in the schema of their table)
            bind.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} "
                    f"ON {table_ref} {using_clause} "
                    f"WHERE fact_type = '{ft}' AND bank_id = '{escaped_bank_id}'"
                )
            )


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()

    # Drop per-bank HNSW indexes (iterate existing banks)
    bind = op.get_bind()
    schema_name = context.config.get_main_option("target_schema")
    banks_ref = f'"{schema_name}".banks' if schema_name else "banks"

    rows = bind.execute(text(f"SELECT internal_id FROM {banks_ref}")).fetchall()  # noqa: S608
    for row in rows:
        internal_id = str(row[0]).replace("-", "")[:16]
        for ft_short in _FACT_TYPES.values():
            idx_name = f"idx_mu_emb_{ft_short}_{internal_id}"
            bind.execute(text(f"DROP INDEX IF EXISTS {schema}{idx_name}"))

    # Restore the global HNSW index
    table_ref = f'"{schema_name}".memory_units' if schema_name else "memory_units"
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_memory_units_embedding ON {table_ref} USING hnsw (embedding vector_cosine_ops)"
    )

    # Restore old fact_type-only partial indexes
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_mu_emb_world "
        f"ON {table_ref} USING hnsw (embedding vector_cosine_ops) "
        f"WHERE fact_type = 'world'"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_mu_emb_observation "
        f"ON {table_ref} USING hnsw (embedding vector_cosine_ops) "
        f"WHERE fact_type = 'observation'"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_mu_emb_experience "
        f"ON {table_ref} USING hnsw (embedding vector_cosine_ops) "
        f"WHERE fact_type = 'experience'"
    )

    # Drop internal_id column
    op.execute(f"ALTER TABLE {schema}banks DROP CONSTRAINT IF EXISTS banks_internal_id_unique")
    op.execute(f"ALTER TABLE {schema}banks DROP COLUMN IF EXISTS internal_id")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
