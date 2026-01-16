"""mental_models_v4

Revision ID: h3c4d5e6f7g8
Revises: g2a3b4c5d6e7
Create Date: 2026-01-08 00:00:00.000000

This migration implements the v4 mental models system:
1. Deletes existing observation memory_units (observations now in mental models)
2. Adds mission column to banks (replacing background)
3. Creates mental_models table with final schema

Mental models can reference entities when an entity is "promoted" to a mental model.
Summary content is stored as JSONB observations with per-observation fact attribution.
"""

from collections.abc import Sequence

from alembic import context, op

# revision identifiers, used by Alembic.
revision: str = "h3c4d5e6f7g8"
down_revision: str | Sequence[str] | None = "g2a3b4c5d6e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    """Apply mental models v4 changes."""
    schema = _get_schema_prefix()

    # Step 1: Delete observation memory_units (cascades to unit_entities links)
    # Observations are now handled through mental models, not memory_units
    op.execute(f"DELETE FROM {schema}memory_units WHERE fact_type = 'observation'")

    # Step 2: Drop observation-specific index (if it exists)
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_observation_date")

    # Step 3: Add mission column to banks (replacing background)
    op.execute(f"ALTER TABLE {schema}banks ADD COLUMN IF NOT EXISTS mission TEXT")

    # Migrate: copy background to mission if background column exists
    # Use DO block to check column existence first (idempotent for re-runs)
    schema_name = context.config.get_main_option("target_schema") or "public"
    op.execute(f"""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = '{schema_name}' AND table_name = 'banks' AND column_name = 'background'
            ) THEN
                UPDATE {schema}banks
                SET mission = background
                WHERE mission IS NULL;
            END IF;
        END $$;
    """)

    # Remove background column (replaced by mission)
    op.execute(f"ALTER TABLE {schema}banks DROP COLUMN IF EXISTS background")

    # Step 4: Create mental_models table with final v4 schema (if not exists)
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS {schema}mental_models (
            id VARCHAR(64) NOT NULL,
            bank_id VARCHAR(64) NOT NULL,
            subtype VARCHAR(32) NOT NULL,
            name VARCHAR(256) NOT NULL,
            description TEXT NOT NULL,
            entity_id UUID,
            observations JSONB DEFAULT '{{"observations": []}}'::jsonb,
            links VARCHAR[],
            tags VARCHAR[] DEFAULT '{{}}',
            last_updated TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            PRIMARY KEY (id, bank_id),
            FOREIGN KEY (bank_id) REFERENCES {schema}banks(bank_id) ON DELETE CASCADE,
            FOREIGN KEY (entity_id) REFERENCES {schema}entities(id) ON DELETE SET NULL,
            CONSTRAINT ck_mental_models_subtype CHECK (subtype IN ('structural', 'emergent', 'pinned', 'learned'))
        )
    """)

    # Step 5: Create indexes for efficient queries (if not exist)
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_mental_models_bank_id ON {schema}mental_models(bank_id)")
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_mental_models_subtype ON {schema}mental_models(bank_id, subtype)")
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_mental_models_entity_id ON {schema}mental_models(entity_id)")
    # GIN index for efficient tags array filtering
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_mental_models_tags ON {schema}mental_models USING GIN(tags)")


def downgrade() -> None:
    """Revert mental models v4 changes."""
    schema = _get_schema_prefix()

    # Drop mental_models table (cascades to indexes)
    op.execute(f"DROP TABLE IF EXISTS {schema}mental_models CASCADE")

    # Add back background column to banks
    op.execute(f"ALTER TABLE {schema}banks ADD COLUMN IF NOT EXISTS background TEXT")

    # Migrate mission back to background
    op.execute(f"UPDATE {schema}banks SET background = mission WHERE background IS NULL")

    # Remove mission column
    op.execute(f"ALTER TABLE {schema}banks DROP COLUMN IF EXISTS mission")

    # Note: Cannot restore deleted observations - they are lost on downgrade
