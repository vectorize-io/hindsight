"""Rename mental_model fact_type to observation and reflections table to mental_models

Revision ID: t5o6p7q8r9s0
Revises: s4n5o6p7q8r9
Create Date: 2026-01-26

This migration implements the terminology rename:
1. mental_model (fact_type in memory_units) -> observation
2. reflections table -> mental_models table

The new terminology:
- Observations: Consolidated knowledge synthesized from facts (was mental_model)
- Mental Models: Stored reflect responses (was reflections)
"""

from collections.abc import Sequence

from alembic import context, op

revision: str = "t5o6p7q8r9s0"
down_revision: str | Sequence[str] | None = "s4n5o6p7q8r9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    """Rename mental_model -> observation and reflections -> mental_models."""
    schema = _get_schema_prefix()

    # 1. Update fact_type values: mental_model -> observation
    op.execute(f"""
        UPDATE {schema}memory_units
        SET fact_type = 'observation'
        WHERE fact_type = 'mental_model'
    """)

    # 2. Update the CHECK constraint - remove mental_model, keep observation
    op.execute(f"ALTER TABLE {schema}memory_units DROP CONSTRAINT IF EXISTS memory_units_fact_type_check")
    op.execute(f"""
        ALTER TABLE {schema}memory_units
        ADD CONSTRAINT memory_units_fact_type_check
        CHECK (fact_type IN ('world', 'experience', 'opinion', 'observation'))
    """)

    # 3. Rename the index for observations (was for mental_models)
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_mental_models")
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_memory_units_observations
        ON {schema}memory_units(bank_id, fact_type)
        WHERE fact_type = 'observation'
    """)

    # 4. Update the unconsolidated index to not filter by fact_type since observations
    # are now the consolidated type
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_unconsolidated")
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_memory_units_unconsolidated
        ON {schema}memory_units (bank_id, created_at)
        WHERE consolidated_at IS NULL AND fact_type IN ('experience', 'world')
    """)

    # 5. Rename reflections table to mental_models
    op.execute(f"ALTER TABLE IF EXISTS {schema}reflections RENAME TO mental_models")

    # 6. Rename indexes for mental_models (was reflections)
    op.execute(f"ALTER INDEX IF EXISTS {schema}idx_reflections_bank_id RENAME TO idx_mental_models_bank_id")
    op.execute(f"ALTER INDEX IF EXISTS {schema}idx_reflections_embedding RENAME TO idx_mental_models_embedding")
    op.execute(f"ALTER INDEX IF EXISTS {schema}idx_reflections_tags RENAME TO idx_mental_models_tags")
    op.execute(f"ALTER INDEX IF EXISTS {schema}idx_reflections_text_search RENAME TO idx_mental_models_text_search")

    # 7. Rename foreign key constraint
    op.execute(f"""
        ALTER TABLE {schema}mental_models
        DROP CONSTRAINT IF EXISTS fk_reflections_bank_id
    """)
    op.execute(f"""
        ALTER TABLE {schema}mental_models
        ADD CONSTRAINT fk_mental_models_bank_id
        FOREIGN KEY (bank_id) REFERENCES {schema}banks(bank_id) ON DELETE CASCADE
    """)


def downgrade() -> None:
    """Reverse: observation -> mental_model and mental_models -> reflections."""
    schema = _get_schema_prefix()

    # 1. Rename mental_models table back to reflections
    op.execute(f"ALTER TABLE IF EXISTS {schema}mental_models RENAME TO reflections")

    # 2. Rename indexes back
    op.execute(f"ALTER INDEX IF EXISTS {schema}idx_mental_models_bank_id RENAME TO idx_reflections_bank_id")
    op.execute(f"ALTER INDEX IF EXISTS {schema}idx_mental_models_embedding RENAME TO idx_reflections_embedding")
    op.execute(f"ALTER INDEX IF EXISTS {schema}idx_mental_models_tags RENAME TO idx_reflections_tags")
    op.execute(f"ALTER INDEX IF EXISTS {schema}idx_mental_models_text_search RENAME TO idx_reflections_text_search")

    # 3. Rename foreign key back
    op.execute(f"""
        ALTER TABLE {schema}reflections
        DROP CONSTRAINT IF EXISTS fk_mental_models_bank_id
    """)
    op.execute(f"""
        ALTER TABLE {schema}reflections
        ADD CONSTRAINT fk_reflections_bank_id
        FOREIGN KEY (bank_id) REFERENCES {schema}banks(bank_id) ON DELETE CASCADE
    """)

    # 4. Update fact_type values: observation -> mental_model
    op.execute(f"""
        UPDATE {schema}memory_units
        SET fact_type = 'mental_model'
        WHERE fact_type = 'observation'
    """)

    # 5. Update the CHECK constraint back
    op.execute(f"ALTER TABLE {schema}memory_units DROP CONSTRAINT IF EXISTS memory_units_fact_type_check")
    op.execute(f"""
        ALTER TABLE {schema}memory_units
        ADD CONSTRAINT memory_units_fact_type_check
        CHECK (fact_type IN ('world', 'experience', 'opinion', 'observation', 'mental_model'))
    """)

    # 6. Rename index back
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_observations")
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_memory_units_mental_models
        ON {schema}memory_units(bank_id, fact_type)
        WHERE fact_type = 'mental_model'
    """)
