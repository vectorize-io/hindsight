"""new_knowledge_architecture

Revision ID: p1k2l3m4n5o6
Revises: o0j1k2l3m4n5
Create Date: 2026-01-21 00:00:00.000000

This migration implements the new knowledge architecture:
1. Drops the 'learnings' table (mental models are now in memory_units)
2. Renames 'pinned_reflections' to 'reflections'
3. Drops the 'mental_models' table completely
4. Creates 'directives' table for hard rules
5. Adds mental model support columns to 'memory_units' (proof_count, source_memory_ids, history)

The new architecture:
- Directives: Hard rules in their own table
- Mental Models: Stored in memory_units with fact_type='mental_model'
- Reflections: User-curated documents (renamed from pinned_reflections)
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

# revision identifiers, used by Alembic.
revision: str = "p1k2l3m4n5o6"
down_revision: str | Sequence[str] | None = "o0j1k2l3m4n5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    """Implement new knowledge architecture."""
    schema = _get_schema_prefix()

    # 1. Drop the learnings table (mental models will be in memory_units)
    op.execute(f"DROP TABLE IF EXISTS {schema}learnings CASCADE")

    # 2. Rename pinned_reflections to reflections
    op.execute(f"ALTER TABLE IF EXISTS {schema}pinned_reflections RENAME TO reflections")

    # Rename indexes for reflections
    op.execute(f"ALTER INDEX IF EXISTS {schema}idx_pinned_reflections_bank_id RENAME TO idx_reflections_bank_id")
    op.execute(f"ALTER INDEX IF EXISTS {schema}idx_pinned_reflections_embedding RENAME TO idx_reflections_embedding")
    op.execute(f"ALTER INDEX IF EXISTS {schema}idx_pinned_reflections_tags RENAME TO idx_reflections_tags")
    op.execute(
        f"ALTER INDEX IF EXISTS {schema}idx_pinned_reflections_text_search RENAME TO idx_reflections_text_search"
    )

    # Rename foreign key constraint
    op.execute(f"""
        ALTER TABLE {schema}reflections
        DROP CONSTRAINT IF EXISTS fk_pinned_reflections_bank_id
    """)
    op.execute(f"""
        ALTER TABLE {schema}reflections
        ADD CONSTRAINT fk_reflections_bank_id
        FOREIGN KEY (bank_id) REFERENCES {schema}banks(bank_id) ON DELETE CASCADE
    """)

    # 3. Drop the mental_models table completely
    op.execute(f"DROP TABLE IF EXISTS {schema}mental_models CASCADE")

    # 4. Create directives table
    op.execute(f"""
        CREATE TABLE {schema}directives (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            bank_id VARCHAR(64) NOT NULL,
            name VARCHAR(256) NOT NULL,
            content TEXT NOT NULL,
            priority INT NOT NULL DEFAULT 0,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            tags VARCHAR[] DEFAULT ARRAY[]::VARCHAR[],
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
        )
    """)

    # Add foreign key and indexes for directives
    op.execute(f"""
        ALTER TABLE {schema}directives
        ADD CONSTRAINT fk_directives_bank_id
        FOREIGN KEY (bank_id) REFERENCES {schema}banks(bank_id) ON DELETE CASCADE
    """)
    op.execute(f"CREATE INDEX idx_directives_bank_id ON {schema}directives(bank_id)")
    op.execute(f"CREATE INDEX idx_directives_bank_active ON {schema}directives(bank_id, is_active)")
    op.execute(f"CREATE INDEX idx_directives_tags ON {schema}directives USING GIN(tags)")

    # 5. Add mental model support columns to memory_units
    # proof_count: Number of memories that support this mental model
    op.execute(f"""
        ALTER TABLE {schema}memory_units
        ADD COLUMN IF NOT EXISTS proof_count INT DEFAULT 1
    """)

    # source_memory_ids: Array of memory IDs that consolidated into this mental model
    op.execute(f"""
        ALTER TABLE {schema}memory_units
        ADD COLUMN IF NOT EXISTS source_memory_ids UUID[] DEFAULT ARRAY[]::UUID[]
    """)

    # history: JSONB array tracking changes to mental models
    op.execute(f"""
        ALTER TABLE {schema}memory_units
        ADD COLUMN IF NOT EXISTS history JSONB DEFAULT '[]'::jsonb
    """)

    # Add index for finding mental models
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_memory_units_mental_models
        ON {schema}memory_units(bank_id, fact_type)
        WHERE fact_type = 'mental_model'
    """)

    # 6. Update fact_type check constraint to include 'mental_model'
    op.execute(f"ALTER TABLE {schema}memory_units DROP CONSTRAINT IF EXISTS memory_units_fact_type_check")
    op.execute(f"""
        ALTER TABLE {schema}memory_units
        ADD CONSTRAINT memory_units_fact_type_check
        CHECK (fact_type IN ('world', 'experience', 'opinion', 'observation', 'mental_model'))
    """)


def _pg_downgrade() -> None:
    """Reverse the migration."""
    schema = _get_schema_prefix()

    # Restore original fact_type check constraint (without 'mental_model')
    op.execute(f"ALTER TABLE {schema}memory_units DROP CONSTRAINT IF EXISTS memory_units_fact_type_check")
    op.execute(f"""
        ALTER TABLE {schema}memory_units
        ADD CONSTRAINT memory_units_fact_type_check
        CHECK (fact_type IN ('world', 'experience', 'opinion', 'observation'))
    """)

    # Drop mental model columns from memory_units
    op.execute(f"ALTER TABLE {schema}memory_units DROP COLUMN IF EXISTS proof_count")
    op.execute(f"ALTER TABLE {schema}memory_units DROP COLUMN IF EXISTS source_memory_ids")
    op.execute(f"ALTER TABLE {schema}memory_units DROP COLUMN IF EXISTS history")
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_mental_models")

    # Drop directives table
    op.execute(f"DROP TABLE IF EXISTS {schema}directives CASCADE")

    # Rename reflections back to pinned_reflections
    op.execute(f"ALTER TABLE IF EXISTS {schema}reflections RENAME TO pinned_reflections")

    # Restore indexes
    op.execute(f"ALTER INDEX IF EXISTS {schema}idx_reflections_bank_id RENAME TO idx_pinned_reflections_bank_id")
    op.execute(f"ALTER INDEX IF EXISTS {schema}idx_reflections_embedding RENAME TO idx_pinned_reflections_embedding")
    op.execute(f"ALTER INDEX IF EXISTS {schema}idx_reflections_tags RENAME TO idx_pinned_reflections_tags")
    op.execute(
        f"ALTER INDEX IF EXISTS {schema}idx_reflections_text_search RENAME TO idx_pinned_reflections_text_search"
    )

    # Restore foreign key
    op.execute(f"""
        ALTER TABLE {schema}pinned_reflections
        DROP CONSTRAINT IF EXISTS fk_reflections_bank_id
    """)
    op.execute(f"""
        ALTER TABLE {schema}pinned_reflections
        ADD CONSTRAINT fk_pinned_reflections_bank_id
        FOREIGN KEY (bank_id) REFERENCES {schema}banks(bank_id) ON DELETE CASCADE
    """)

    # Re-create learnings table
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS {schema}learnings (
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
    op.execute(f"""
        ALTER TABLE {schema}learnings
        ADD CONSTRAINT fk_learnings_bank_id
        FOREIGN KEY (bank_id) REFERENCES {schema}banks(bank_id) ON DELETE CASCADE
    """)

    # Note: mental_models table recreation is complex and would need separate handling


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
