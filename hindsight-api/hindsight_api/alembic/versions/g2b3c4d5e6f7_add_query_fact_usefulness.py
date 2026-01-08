"""add_query_fact_usefulness

Revision ID: g2b3c4d5e6f7
Revises: f1a2b3c4d5e6
Create Date: 2025-01-05

Adds query-context aware usefulness scoring:
- query_fact_usefulness: Tracks usefulness per (query_embedding, fact) pair
  allowing feedback signals to be tied to specific query contexts.

This addresses the limitation where a fact marked "helpful" for one query
would be boosted for all queries, even unrelated ones.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "g2b3c4d5e6f7"
down_revision: str | Sequence[str] | None = "f1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add query_fact_usefulness table for query-context aware scoring."""

    # Create query_fact_usefulness table - per (query, fact) usefulness scores
    op.create_table(
        "query_fact_usefulness",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("bank_id", sa.Text(), nullable=False),
        sa.Column("fact_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("query_embedding", Vector(384), nullable=False),
        sa.Column("query_example", sa.Text(), nullable=True),
        sa.Column("usefulness_score", sa.Float(), server_default="0.5", nullable=False),
        sa.Column("signal_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_signal_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "last_decay_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_query_fact_usefulness")),
    )

    # Index for looking up by bank_id
    op.create_index(
        "idx_query_fact_usefulness_bank_id",
        "query_fact_usefulness",
        ["bank_id"],
    )

    # Index for looking up by fact_id (to find all query contexts for a fact)
    op.create_index(
        "idx_query_fact_usefulness_fact_id",
        "query_fact_usefulness",
        ["fact_id"],
    )

    # Composite index for (bank_id, fact_id) lookups
    op.create_index(
        "idx_query_fact_usefulness_bank_fact",
        "query_fact_usefulness",
        ["bank_id", "fact_id"],
    )

    # HNSW index for semantic similarity search on query embeddings
    # This enables finding similar past queries efficiently
    op.create_index(
        "idx_query_fact_usefulness_embedding",
        "query_fact_usefulness",
        ["query_embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"query_embedding": "vector_cosine_ops"},
    )

    # Add query_embedding column to usefulness_signals for audit trail
    op.add_column(
        "usefulness_signals",
        sa.Column("query_embedding", Vector(384), nullable=True),
    )

    # Update query_hash to be required (we'll make query required in API)
    # Note: Existing rows with NULL query_hash are preserved for backwards compatibility


def downgrade() -> None:
    """Remove query_fact_usefulness table."""

    # Remove query_embedding from usefulness_signals
    op.drop_column("usefulness_signals", "query_embedding")

    # Drop indexes
    op.drop_index("idx_query_fact_usefulness_embedding", table_name="query_fact_usefulness")
    op.drop_index("idx_query_fact_usefulness_bank_fact", table_name="query_fact_usefulness")
    op.drop_index("idx_query_fact_usefulness_fact_id", table_name="query_fact_usefulness")
    op.drop_index("idx_query_fact_usefulness_bank_id", table_name="query_fact_usefulness")

    # Drop table
    op.drop_table("query_fact_usefulness")
