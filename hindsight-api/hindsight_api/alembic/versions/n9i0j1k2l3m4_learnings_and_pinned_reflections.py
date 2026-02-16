"""learnings_and_pinned_reflections

Revision ID: n9i0j1k2l3m4
Revises: m8h9i0j1k2l3
Create Date: 2026-01-21 00:00:00.000000

This migration:
1. Creates the 'learnings' table for automatic bottom-up consolidation
2. Creates the 'pinned_reflections' table for user-curated living documents
3. Adds consolidation tracking columns to the 'banks' table
"""

import os
from collections.abc import Sequence

from alembic import context, op
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision: str = "n9i0j1k2l3m4"
down_revision: str | Sequence[str] | None = "m8h9i0j1k2l3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


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
    """Create learnings and pinned_reflections tables."""
    schema = _get_schema_prefix()

    # Detect which vector extension is available
    vector_ext = _detect_vector_extension()

    # Detect which text search extension to use
    text_search_ext = _detect_text_search_extension()

    # Read embedding dimension from environment, defaulting to 384
    embedding_dimension = int(os.getenv("DEFAULT_EMBEDDING_DIMENSION", "384"))

    # 1. Create learnings table
    op.execute(f"""
        CREATE TABLE {schema}learnings (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            bank_id VARCHAR(64) NOT NULL,
            text TEXT NOT NULL,
            proof_count INT NOT NULL DEFAULT 1,
            history JSONB DEFAULT '[]'::jsonb,
            mission_context VARCHAR(64),
            pre_mission_change BOOLEAN DEFAULT FALSE,
            embedding vector({embedding_dimension}),
            tags VARCHAR[] DEFAULT ARRAY[]::VARCHAR[],
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
        )
    """)

    # Add foreign key constraint
    op.execute(f"""
        ALTER TABLE {schema}learnings
        ADD CONSTRAINT fk_learnings_bank_id
        FOREIGN KEY (bank_id) REFERENCES {schema}banks(bank_id) ON DELETE CASCADE
    """)

    # Indexes for learnings
    op.execute(f"CREATE INDEX idx_learnings_bank_id ON {schema}learnings(bank_id)")

    # Create vector index based on detected extension
    if vector_ext == "vchord":
        # VectorChord requires dimension type modifier for quantized indexes
        op.execute(f"""
            CREATE INDEX idx_learnings_embedding ON {schema}learnings
            USING vchordrq (embedding vector_l2_ops)
        """)
    else:  # pgvector
        op.execute(f"""
            CREATE INDEX idx_learnings_embedding ON {schema}learnings
            USING hnsw (embedding vector_cosine_ops)
        """)

    op.execute(f"CREATE INDEX idx_learnings_tags ON {schema}learnings USING GIN(tags)")

    # Full-text search for learnings
    if text_search_ext == "vchord":
        # VectorChord BM25: bm25vector type (no GENERATED - tokenization happens on INSERT)
        # Note: vchord_bm25 extension creates types in bm25_catalog schema
        op.execute(f"""
            ALTER TABLE {schema}learnings ADD COLUMN search_vector bm25_catalog.bm25vector
        """)
        op.execute(f"""
            CREATE INDEX idx_learnings_text_search ON {schema}learnings
            USING bm25 (search_vector bm25_catalog.bm25_ops)
        """)
    elif text_search_ext == "pg_textsearch":
        # Timescale pg_textsearch: dummy TEXT column for consistency (indexes operate on base columns directly)
        op.execute(f"""
            ALTER TABLE {schema}learnings ADD COLUMN search_vector TEXT
        """)
        op.execute(f"""
            CREATE INDEX idx_learnings_text_search ON {schema}learnings
            USING bm25(text) WITH (text_config='english')
        """)
    else:  # native
        # Native PostgreSQL: tsvector with automatic generation
        op.execute(f"""
            ALTER TABLE {schema}learnings ADD COLUMN search_vector tsvector
            GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
        """)
        op.execute(f"CREATE INDEX idx_learnings_text_search ON {schema}learnings USING gin(search_vector)")

    # 2. Create pinned_reflections table
    op.execute(f"""
        CREATE TABLE {schema}pinned_reflections (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            bank_id VARCHAR(64) NOT NULL,
            name VARCHAR(256) NOT NULL,
            source_query TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding vector({embedding_dimension}),
            tags VARCHAR[] DEFAULT ARRAY[]::VARCHAR[],
            last_refreshed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
        )
    """)

    # Add foreign key constraint
    op.execute(f"""
        ALTER TABLE {schema}pinned_reflections
        ADD CONSTRAINT fk_pinned_reflections_bank_id
        FOREIGN KEY (bank_id) REFERENCES {schema}banks(bank_id) ON DELETE CASCADE
    """)

    # Indexes for pinned_reflections
    op.execute(f"CREATE INDEX idx_pinned_reflections_bank_id ON {schema}pinned_reflections(bank_id)")

    # Create vector index based on detected extension
    if vector_ext == "vchord":
        # VectorChord requires dimension type modifier for quantized indexes
        op.execute(f"""
            CREATE INDEX idx_pinned_reflections_embedding ON {schema}pinned_reflections
            USING vchordrq (embedding vector_l2_ops)
        """)
    else:  # pgvector
        op.execute(f"""
            CREATE INDEX idx_pinned_reflections_embedding ON {schema}pinned_reflections
            USING hnsw (embedding vector_cosine_ops)
        """)

    op.execute(f"CREATE INDEX idx_pinned_reflections_tags ON {schema}pinned_reflections USING GIN(tags)")

    # Full-text search for pinned_reflections
    if text_search_ext == "vchord":
        # VectorChord BM25: bm25vector type (no GENERATED - tokenization happens on INSERT/UPDATE)
        # Note: vchord_bm25 extension creates types in bm25_catalog schema
        op.execute(f"""
            ALTER TABLE {schema}pinned_reflections ADD COLUMN search_vector bm25_catalog.bm25vector
        """)
        op.execute(f"""
            CREATE INDEX idx_pinned_reflections_text_search ON {schema}pinned_reflections
            USING bm25 (search_vector bm25_catalog.bm25_ops)
        """)
    elif text_search_ext == "pg_textsearch":
        # Timescale pg_textsearch: dummy TEXT column for consistency (indexes operate on base columns directly)
        op.execute(f"""
            ALTER TABLE {schema}pinned_reflections ADD COLUMN search_vector TEXT
        """)
        op.execute(f"""
            CREATE INDEX idx_pinned_reflections_text_search ON {schema}pinned_reflections
            USING bm25(content)
            WITH (text_config='english')
        """)
    else:  # native
        # Native PostgreSQL: tsvector with automatic generation
        op.execute(f"""
            ALTER TABLE {schema}pinned_reflections ADD COLUMN search_vector tsvector
            GENERATED ALWAYS AS (to_tsvector('english', COALESCE(name, '') || ' ' || content)) STORED
        """)
        op.execute(f"""
            CREATE INDEX idx_pinned_reflections_text_search ON {schema}pinned_reflections
            USING gin(search_vector)
        """)

    # 3. Add consolidation tracking columns to banks table
    op.execute(f"""
        ALTER TABLE {schema}banks
        ADD COLUMN IF NOT EXISTS last_consolidated_at TIMESTAMP WITH TIME ZONE
    """)
    op.execute(f"""
        ALTER TABLE {schema}banks
        ADD COLUMN IF NOT EXISTS mission_changed_at TIMESTAMP WITH TIME ZONE
    """)


def downgrade() -> None:
    """Drop learnings and pinned_reflections tables."""
    schema = _get_schema_prefix()

    # Drop tables
    op.execute(f"DROP TABLE IF EXISTS {schema}learnings CASCADE")
    op.execute(f"DROP TABLE IF EXISTS {schema}pinned_reflections CASCADE")

    # Remove columns from banks
    op.execute(f"ALTER TABLE {schema}banks DROP COLUMN IF EXISTS last_consolidated_at")
    op.execute(f"ALTER TABLE {schema}banks DROP COLUMN IF EXISTS mission_changed_at")
