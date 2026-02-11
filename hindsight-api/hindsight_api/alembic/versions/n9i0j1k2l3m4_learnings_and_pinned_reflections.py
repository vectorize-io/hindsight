"""learnings_and_pinned_reflections

Revision ID: n9i0j1k2l3m4
Revises: m8h9i0j1k2l3
Create Date: 2026-01-21 00:00:00.000000

This migration:
1. Creates the 'learnings' table for automatic bottom-up consolidation
2. Creates the 'pinned_reflections' table for user-curated living documents
3. Adds consolidation tracking columns to the 'banks' table
"""

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
    Detect available vector extension: 'vchord' or 'pgvector'.
    Prefers vchord if both available. Raises error if neither found.
    """
    conn = op.get_bind()
    vchord_check = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vchord'")).scalar()
    if vchord_check:
        return "vchord"

    pgvector_check = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).scalar()
    if pgvector_check:
        return "pgvector"

    raise RuntimeError(
        "Neither vchord nor pgvector extension found. Install one: CREATE EXTENSION vchord; or CREATE EXTENSION vector;"
    )


def upgrade() -> None:
    """Create learnings and pinned_reflections tables."""
    schema = _get_schema_prefix()

    # Detect which vector extension is available
    vector_ext = _detect_vector_extension()

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
            embedding vector(384),
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
            embedding vector(384),
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
