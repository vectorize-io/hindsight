"""mental_models_v4

Revision ID: f1a2b3c4d5e6
Revises: e0a1b2c3d4e5
Create Date: 2026-01-08 00:00:00.000000

This migration implements the v4 mental models system:
1. Deletes existing observation memory_units (observations now in mental models)
2. Adds goal field to banks (for structural model derivation)
3. Creates mental_models table with full v4 schema

Mental models are now the only place summaries exist, and can reference entities
when an entity is "promoted" to a mental model.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "f1a2b3c4d5e6"
down_revision: str | Sequence[str] | None = "e0a1b2c3d4e5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply mental models v4 changes."""

    # Step 1: Delete observation memory_units (cascades to unit_entities links)
    # Observations are now handled through mental models, not memory_units
    op.execute("DELETE FROM memory_units WHERE fact_type = 'observation'")

    # Step 2: Drop observation-specific index (if it exists)
    op.execute("DROP INDEX IF EXISTS idx_memory_units_observation_date")

    # Step 3: Add goal field to banks (if not exists)
    op.execute("ALTER TABLE banks ADD COLUMN IF NOT EXISTS goal TEXT")

    # Step 4: Create mental_models table with v4 schema (if not exists)
    op.execute("""
        CREATE TABLE IF NOT EXISTS mental_models (
            id VARCHAR(64) NOT NULL,
            bank_id VARCHAR(64) NOT NULL,
            type VARCHAR(32) NOT NULL,
            subtype VARCHAR(32) NOT NULL,
            name VARCHAR(256) NOT NULL,
            description TEXT NOT NULL,
            summary TEXT,
            entity_id UUID,
            source_facts VARCHAR[],
            triggers VARCHAR[],
            links VARCHAR[],
            tags VARCHAR[] DEFAULT '{}',
            last_updated TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            PRIMARY KEY (id, bank_id),
            FOREIGN KEY (bank_id) REFERENCES banks(bank_id) ON DELETE CASCADE,
            FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE SET NULL,
            CONSTRAINT ck_mental_models_type CHECK (type IN ('entity', 'concept', 'event')),
            CONSTRAINT ck_mental_models_subtype CHECK (subtype IN ('structural', 'emergent'))
        )
    """)

    # Add tags column if table already exists (for existing databases)
    op.execute("ALTER TABLE mental_models ADD COLUMN IF NOT EXISTS tags VARCHAR[] DEFAULT '{}'")

    # Step 5: Create indexes for efficient queries (if not exist)
    op.execute("CREATE INDEX IF NOT EXISTS idx_mental_models_bank_id ON mental_models(bank_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_mental_models_type ON mental_models(bank_id, type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_mental_models_subtype ON mental_models(bank_id, subtype)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_mental_models_entity_id ON mental_models(entity_id)")
    # GIN index for efficient tags array filtering
    op.execute("CREATE INDEX IF NOT EXISTS idx_mental_models_tags ON mental_models USING GIN(tags)")


def downgrade() -> None:
    """Revert mental models v4 changes."""

    # Drop mental_models table (cascades to indexes)
    op.execute("DROP TABLE IF EXISTS mental_models CASCADE")

    # Remove goal from banks
    op.execute("ALTER TABLE banks DROP COLUMN IF EXISTS goal")

    # Note: Cannot restore deleted observations - they are lost on downgrade
