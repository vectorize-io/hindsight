"""initial_schema

Revision ID: 5a366d414dce
Revises:
Create Date: 2025-11-27 11:54:19.228030

"""

import os
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy import text
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "5a366d414dce"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _detect_vector_extension() -> str:
    """
    Detect or validate vector extension: 'vchord' or 'pgvector'.
    Respects HINDSIGHT_API_VECTOR_EXTENSION env var if set.
    """
    conn = op.get_bind()
    vector_extension = os.getenv("HINDSIGHT_API_VECTOR_EXTENSION", "pgvector").lower()

    # Validate configured extension is installed
    if vector_extension == "vchord":
        vchord_check = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vchord'")).scalar()
        if not vchord_check:
            raise RuntimeError(
                "Configured vector extension 'vchord' not found. Install it with: CREATE EXTENSION vchord CASCADE;"
            )
        return "vchord"
    elif vector_extension == "pgvector":
        pgvector_check = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).scalar()
        if not pgvector_check:
            raise RuntimeError(
                "Configured vector extension 'pgvector' not found. Install it with: CREATE EXTENSION vector;"
            )
        return "pgvector"
    else:
        raise ValueError(f"Invalid HINDSIGHT_API_VECTOR_EXTENSION: {vector_extension}. Must be 'pgvector' or 'vchord'")


def _detect_text_search_extension() -> str:
    """
    Detect or validate text search extension: 'native', 'vchord', or 'pg_textsearch'.
    Respects HINDSIGHT_API_TEXT_SEARCH_EXTENSION env var.
    Creates the extension if needed.
    """
    text_search_extension = os.getenv("HINDSIGHT_API_TEXT_SEARCH_EXTENSION", "native").lower()

    if text_search_extension == "vchord":
        # Create vchord_bm25 extension if not exists
        try:
            op.execute("CREATE EXTENSION IF NOT EXISTS vchord_bm25 CASCADE")
        except Exception:
            # Extension might already exist or user lacks permissions - verify it exists
            conn = op.get_bind()
            result = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vchord_bm25'")).fetchone()
            if not result:
                # Extension truly doesn't exist - re-raise the error
                raise
        return "vchord"
    elif text_search_extension == "pg_textsearch":
        # Create pg_textsearch extension if not exists
        try:
            op.execute("CREATE EXTENSION IF NOT EXISTS pg_textsearch CASCADE")
        except Exception:
            # Extension might already exist or user lacks permissions - verify it exists
            conn = op.get_bind()
            result = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'pg_textsearch'")).fetchone()
            if not result:
                # Extension truly doesn't exist - re-raise the error
                raise
        return "pg_textsearch"
    elif text_search_extension == "native":
        return "native"
    else:
        raise ValueError(
            f"Invalid HINDSIGHT_API_TEXT_SEARCH_EXTENSION: {text_search_extension}. Must be 'native', 'vchord', or 'pg_textsearch'"
        )


def upgrade() -> None:
    """Upgrade schema - create all tables from scratch."""

    # Note: pgvector extension is installed globally BEFORE migrations run
    # See migrations.py:run_migrations() - this ensures the extension is available
    # to all schemas, not just the one being migrated

    # We keep this here as a fallback for backwards compatibility
    # This may fail if user lacks permissions, which is fine if extension already exists
    try:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except Exception:
        # Extension might already exist or user lacks permissions - verify it exists
        conn = op.get_bind()
        result = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).fetchone()
        if not result:
            # Extension truly doesn't exist - re-raise the error
            raise

    # Create banks table
    op.create_table(
        "banks",
        sa.Column("bank_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column(
            "personality",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("background", sa.Text(), nullable=True),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("bank_id", name=op.f("pk_banks")),
    )

    # Create documents table
    op.create_table(
        "documents",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("bank_id", sa.Text(), nullable=False),
        sa.Column("original_text", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.Text(), nullable=True),
        sa.Column(
            "metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", "bank_id", name=op.f("pk_documents")),
    )
    op.create_index("idx_documents_bank_id", "documents", ["bank_id"])
    op.create_index("idx_documents_content_hash", "documents", ["content_hash"])

    # Create async_operations table
    op.create_table(
        "async_operations",
        sa.Column(
            "operation_id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("bank_id", sa.Text(), nullable=False),
        sa.Column("operation_type", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), server_default="pending", nullable=False),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("completed_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "result_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("operation_id", name=op.f("pk_async_operations")),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'completed', 'failed')", name="async_operations_status_check"
        ),
    )
    op.create_index("idx_async_operations_bank_id", "async_operations", ["bank_id"])
    op.create_index("idx_async_operations_status", "async_operations", ["status"])
    op.create_index("idx_async_operations_bank_status", "async_operations", ["bank_id", "status"])

    # Create entities table
    op.create_table(
        "entities",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("canonical_name", sa.Text(), nullable=False),
        sa.Column("bank_id", sa.Text(), nullable=False),
        sa.Column(
            "metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        sa.Column("first_seen", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("last_seen", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("mention_count", sa.Integer(), server_default="1", nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_entities")),
    )
    op.create_index("idx_entities_bank_id", "entities", ["bank_id"])
    op.create_index("idx_entities_canonical_name", "entities", ["canonical_name"])
    op.create_index("idx_entities_bank_name", "entities", ["bank_id", "canonical_name"])
    # Create unique index on (bank_id, LOWER(canonical_name)) for entity resolution
    op.execute("CREATE UNIQUE INDEX idx_entities_bank_lower_name ON entities (bank_id, LOWER(canonical_name))")

    # Create memory_units table
    # Read embedding dimension from environment, defaulting to 384
    embedding_dimension = int(os.getenv("DEFAULT_EMBEDDING_DIMENSION", "384"))

    op.create_table(
        "memory_units",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("bank_id", sa.Text(), nullable=False),
        sa.Column("document_id", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(embedding_dimension), nullable=True),
        sa.Column("context", sa.Text(), nullable=True),
        sa.Column("event_date", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("occurred_start", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("occurred_end", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("mentioned_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("fact_type", sa.Text(), server_default="world", nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("access_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id", "bank_id"],
            ["documents.id", "documents.bank_id"],
            name="memory_units_document_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_memory_units")),
        sa.CheckConstraint(
            "fact_type IN ('world', 'bank', 'opinion', 'observation')", name="memory_units_fact_type_check"
        ),
        sa.CheckConstraint(
            "confidence_score IS NULL OR (confidence_score >= 0.0 AND confidence_score <= 1.0)",
            name="memory_units_confidence_range_check",
        ),
        sa.CheckConstraint(
            "(fact_type = 'opinion' AND confidence_score IS NOT NULL) OR "
            "(fact_type = 'observation') OR "
            "(fact_type NOT IN ('opinion', 'observation') AND confidence_score IS NULL)",
            name="confidence_score_fact_type_check",
        ),
    )

    # Add search_vector column for full-text search
    # Type depends on configured text search backend
    text_search_ext = _detect_text_search_extension()

    if text_search_ext == "vchord":
        # VectorChord BM25: bm25vector type (no GENERATED - tokenization happens on INSERT)
        # Note: vchord_bm25 extension creates types in bm25_catalog schema
        op.execute("""
            ALTER TABLE memory_units
            ADD COLUMN search_vector bm25_catalog.bm25vector
        """)
    elif text_search_ext == "pg_textsearch":
        # Timescale pg_textsearch: dummy TEXT column for consistency (indexes operate on base columns directly)
        op.execute("""
            ALTER TABLE memory_units
            ADD COLUMN search_vector TEXT
        """)
    else:  # native
        # Native PostgreSQL: tsvector with automatic generation
        op.execute("""
            ALTER TABLE memory_units
            ADD COLUMN search_vector tsvector
            GENERATED ALWAYS AS (to_tsvector('english', COALESCE(text, '') || ' ' || COALESCE(context, ''))) STORED
        """)

    op.create_index("idx_memory_units_bank_id", "memory_units", ["bank_id"])
    op.create_index("idx_memory_units_document_id", "memory_units", ["document_id"])
    op.create_index("idx_memory_units_event_date", "memory_units", [sa.text("event_date DESC")])
    op.create_index("idx_memory_units_bank_date", "memory_units", ["bank_id", sa.text("event_date DESC")])
    op.create_index("idx_memory_units_access_count", "memory_units", [sa.text("access_count DESC")])
    op.create_index("idx_memory_units_fact_type", "memory_units", ["fact_type"])
    op.create_index("idx_memory_units_bank_fact_type", "memory_units", ["bank_id", "fact_type"])
    op.create_index(
        "idx_memory_units_bank_type_date", "memory_units", ["bank_id", "fact_type", sa.text("event_date DESC")]
    )
    op.create_index(
        "idx_memory_units_opinion_confidence",
        "memory_units",
        ["bank_id", sa.text("confidence_score DESC")],
        postgresql_where=sa.text("fact_type = 'opinion'"),
    )
    op.create_index(
        "idx_memory_units_opinion_date",
        "memory_units",
        ["bank_id", sa.text("event_date DESC")],
        postgresql_where=sa.text("fact_type = 'opinion'"),
    )
    op.create_index(
        "idx_memory_units_observation_date",
        "memory_units",
        ["bank_id", sa.text("event_date DESC")],
        postgresql_where=sa.text("fact_type = 'observation'"),
    )
    # Create vector index - conditional based on available extension
    vector_ext = _detect_vector_extension()

    if vector_ext == "vchord":
        # Use vchordrq index for vchord (supports high-dimensional embeddings)
        op.execute("""
            CREATE INDEX idx_memory_units_embedding ON memory_units
            USING vchordrq (embedding vector_l2_ops)
        """)
    else:  # pgvector
        # Use HNSW index for pgvector
        op.create_index(
            "idx_memory_units_embedding",
            "memory_units",
            ["embedding"],
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        )

    # Create full-text search index on search_vector
    # Index type depends on text search backend
    if text_search_ext == "vchord":
        # VectorChord BM25 index
        op.execute("""
            CREATE INDEX idx_memory_units_text_search ON memory_units
            USING bm25 (search_vector bm25_catalog.bm25_ops)
        """)
    elif text_search_ext == "pg_textsearch":
        # Timescale pg_textsearch BM25 index on text column
        # Note: pg_textsearch doesn't support expressions, so we index the main text column
        op.execute("""
            CREATE INDEX idx_memory_units_text_search ON memory_units
            USING bm25(text)
            WITH (text_config='english')
        """)
    else:  # native
        # Native PostgreSQL GIN index
        op.execute("""
            CREATE INDEX idx_memory_units_text_search ON memory_units
            USING gin(search_vector)
        """)

    op.execute("""
        CREATE MATERIALIZED VIEW memory_units_bm25 AS
        SELECT
            id,
            bank_id,
            text,
            to_tsvector('english', text) AS text_vector,
            log(1.0 + length(text)::float / (SELECT avg(length(text)) FROM memory_units)) AS doc_length_factor
        FROM memory_units
    """)

    op.create_index("idx_memory_units_bm25_bank", "memory_units_bm25", ["bank_id"])
    op.create_index("idx_memory_units_bm25_text_vector", "memory_units_bm25", ["text_vector"], postgresql_using="gin")

    # Create entity_cooccurrences table
    op.create_table(
        "entity_cooccurrences",
        sa.Column("entity_id_1", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("entity_id_2", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("cooccurrence_count", sa.Integer(), server_default="1", nullable=False),
        sa.Column(
            "last_cooccurred", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["entity_id_1"],
            ["entities.id"],
            name=op.f("fk_entity_cooccurrences_entity_id_1_entities"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["entity_id_2"],
            ["entities.id"],
            name=op.f("fk_entity_cooccurrences_entity_id_2_entities"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("entity_id_1", "entity_id_2", name=op.f("pk_entity_cooccurrences")),
        sa.CheckConstraint("entity_id_1 < entity_id_2", name="entity_cooccurrence_order_check"),
    )
    op.create_index("idx_entity_cooccurrences_entity1", "entity_cooccurrences", ["entity_id_1"])
    op.create_index("idx_entity_cooccurrences_entity2", "entity_cooccurrences", ["entity_id_2"])
    op.create_index("idx_entity_cooccurrences_count", "entity_cooccurrences", [sa.text("cooccurrence_count DESC")])

    # Create memory_links table
    op.create_table(
        "memory_links",
        sa.Column("from_unit_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("to_unit_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("link_type", sa.Text(), nullable=False),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("weight", sa.Float(), server_default="1.0", nullable=False),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["entity_id"], ["entities.id"], name=op.f("fk_memory_links_entity_id_entities"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["from_unit_id"],
            ["memory_units.id"],
            name=op.f("fk_memory_links_from_unit_id_memory_units"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["to_unit_id"],
            ["memory_units.id"],
            name=op.f("fk_memory_links_to_unit_id_memory_units"),
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "link_type IN ('temporal', 'semantic', 'entity', 'causes', 'caused_by', 'enables', 'prevents')",
            name="memory_links_link_type_check",
        ),
        sa.CheckConstraint("weight >= 0.0 AND weight <= 1.0", name="memory_links_weight_check"),
    )
    # Create unique constraint using COALESCE for nullable entity_id
    op.execute(
        "CREATE UNIQUE INDEX idx_memory_links_unique ON memory_links (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid))"
    )
    op.create_index("idx_memory_links_from_unit", "memory_links", ["from_unit_id"])
    op.create_index("idx_memory_links_to_unit", "memory_links", ["to_unit_id"])
    op.create_index("idx_memory_links_entity", "memory_links", ["entity_id"])
    op.create_index("idx_memory_links_link_type", "memory_links", ["link_type"])

    # Create unit_entities table
    op.create_table(
        "unit_entities",
        sa.Column("unit_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["entity_id"], ["entities.id"], name=op.f("fk_unit_entities_entity_id_entities"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["unit_id"], ["memory_units.id"], name=op.f("fk_unit_entities_unit_id_memory_units"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("unit_id", "entity_id", name=op.f("pk_unit_entities")),
    )
    op.create_index("idx_unit_entities_unit", "unit_entities", ["unit_id"])
    op.create_index("idx_unit_entities_entity", "unit_entities", ["entity_id"])


def downgrade() -> None:
    """Downgrade schema - drop all tables."""

    # Drop tables in reverse dependency order
    op.drop_index("idx_unit_entities_entity", table_name="unit_entities")
    op.drop_index("idx_unit_entities_unit", table_name="unit_entities")
    op.drop_table("unit_entities")

    op.drop_index("idx_memory_links_link_type", table_name="memory_links")
    op.drop_index("idx_memory_links_entity", table_name="memory_links")
    op.drop_index("idx_memory_links_to_unit", table_name="memory_links")
    op.drop_index("idx_memory_links_from_unit", table_name="memory_links")
    op.execute("DROP INDEX IF EXISTS idx_memory_links_unique")
    op.drop_table("memory_links")

    op.drop_index("idx_entity_cooccurrences_count", table_name="entity_cooccurrences")
    op.drop_index("idx_entity_cooccurrences_entity2", table_name="entity_cooccurrences")
    op.drop_index("idx_entity_cooccurrences_entity1", table_name="entity_cooccurrences")
    op.drop_table("entity_cooccurrences")

    # Drop BM25 materialized view and index
    op.drop_index("idx_memory_units_bm25_text_vector", table_name="memory_units_bm25")
    op.drop_index("idx_memory_units_bm25_bank", table_name="memory_units_bm25")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS memory_units_bm25")

    op.drop_index("idx_memory_units_embedding", table_name="memory_units")
    op.drop_index("idx_memory_units_observation_date", table_name="memory_units")
    op.drop_index("idx_memory_units_opinion_date", table_name="memory_units")
    op.drop_index("idx_memory_units_opinion_confidence", table_name="memory_units")
    op.drop_index("idx_memory_units_bank_type_date", table_name="memory_units")
    op.drop_index("idx_memory_units_bank_fact_type", table_name="memory_units")
    op.drop_index("idx_memory_units_fact_type", table_name="memory_units")
    op.drop_index("idx_memory_units_access_count", table_name="memory_units")
    op.drop_index("idx_memory_units_bank_date", table_name="memory_units")
    op.drop_index("idx_memory_units_event_date", table_name="memory_units")
    op.drop_index("idx_memory_units_document_id", table_name="memory_units")
    op.drop_index("idx_memory_units_bank_id", table_name="memory_units")
    op.execute("DROP INDEX IF EXISTS idx_memory_units_text_search")
    op.drop_table("memory_units")

    op.execute("DROP INDEX IF EXISTS idx_entities_bank_lower_name")
    op.drop_index("idx_entities_bank_name", table_name="entities")
    op.drop_index("idx_entities_canonical_name", table_name="entities")
    op.drop_index("idx_entities_bank_id", table_name="entities")
    op.drop_table("entities")

    op.drop_index("idx_async_operations_bank_status", table_name="async_operations")
    op.drop_index("idx_async_operations_status", table_name="async_operations")
    op.drop_index("idx_async_operations_bank_id", table_name="async_operations")
    op.drop_table("async_operations")

    op.drop_index("idx_documents_content_hash", table_name="documents")
    op.drop_index("idx_documents_bank_id", table_name="documents")
    op.drop_table("documents")

    op.drop_table("banks")

    # Drop extensions (optional - comment out if you want to keep them)
    # op.execute('DROP EXTENSION IF EXISTS vector')
    # op.execute('DROP EXTENSION IF EXISTS "uuid-ossp"')
