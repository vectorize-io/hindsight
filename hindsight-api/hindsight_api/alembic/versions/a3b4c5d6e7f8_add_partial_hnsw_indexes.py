"""Add partial HNSW indexes for per-fact_type semantic retrieval

Revision ID: a3b4c5d6e7f8
Revises: z1u2v3w4x5y6
Create Date: 2026-03-10

Creates partial HNSW indexes on memory_units.embedding partitioned by fact_type.
These indexes are required by retrieve_semantic_bm25_combined() which uses
per-fact_type queries with ORDER BY embedding <=> $1 LIMIT n to enable HNSW
index scans instead of sequential scans.

Without these partial indexes, a global HNSW index with post-filtering by
fact_type returns near-zero results for minority fact_types (e.g., experience)
because the index returns nearest neighbors regardless of fact_type.

Note: This migration uses CREATE INDEX IF NOT EXISTS (not CONCURRENTLY) because
Alembic migrations run inside a transaction. For large deployments with existing
data, operators may prefer to create the indexes manually with CONCURRENTLY
before upgrading to avoid blocking writes during index build:

    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mu_emb_world
        ON memory_units USING hnsw (embedding vector_cosine_ops)
        WHERE fact_type = 'world';
    -- (repeat for observation and experience)

If the indexes already exist, this migration is a no-op.
"""

from collections.abc import Sequence

from alembic import context, op

# revision identifiers, used by Alembic.
revision: str = "a3b4c5d6e7f8"
down_revision: str | Sequence[str] | None = "z1u2v3w4x5y6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (e.g., 'tenant_x.' or '' for public)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    """Create partial HNSW indexes for per-fact_type semantic retrieval."""
    schema = _get_schema_prefix()

    # Partial index for world facts
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_mu_emb_world "
        f"ON {schema}memory_units USING hnsw (embedding vector_cosine_ops) "
        f"WHERE fact_type = 'world'"
    )

    # Partial index for observation facts
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_mu_emb_observation "
        f"ON {schema}memory_units USING hnsw (embedding vector_cosine_ops) "
        f"WHERE fact_type = 'observation'"
    )

    # Partial index for experience facts
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_mu_emb_experience "
        f"ON {schema}memory_units USING hnsw (embedding vector_cosine_ops) "
        f"WHERE fact_type = 'experience'"
    )


def downgrade() -> None:
    """Remove partial HNSW indexes."""
    schema = _get_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_mu_emb_world")
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_mu_emb_observation")
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_mu_emb_experience")
