"""migrate_mental_models_data

Revision ID: o0j1k2l3m4n5
Revises: n9i0j1k2l3m4
Create Date: 2026-01-21 00:00:00.000000

This migration:
1. Migrates existing 'pinned' mental models to the new 'pinned_reflections' table
2. Migrates existing 'learned' mental models to the new 'learnings' table
3. Deletes non-directive mental models (structural, emergent, pinned, learned)
4. Drops the mental_model_versions table (no longer used)
5. Adds a CHECK constraint that only 'directive' subtype is allowed
"""

from collections.abc import Sequence

from alembic import context, op

# revision identifiers, used by Alembic.
revision: str = "o0j1k2l3m4n5"
down_revision: str | Sequence[str] | None = "n9i0j1k2l3m4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    """Migrate data and clean up old mental models."""
    schema = _get_schema_prefix()

    # 1. Migrate 'pinned' mental models to pinned_reflections
    # For pinned models, the first observation's content becomes the pinned reflection content
    op.execute(f"""
        INSERT INTO {schema}pinned_reflections (bank_id, name, source_query, content, tags, created_at)
        SELECT
            bank_id,
            name,
            description AS source_query,
            COALESCE(
                observations->'observations'->0->>'content',
                description,
                ''
            ) AS content,
            tags,
            created_at
        FROM {schema}mental_models
        WHERE subtype = 'pinned'
        ON CONFLICT DO NOTHING
    """)

    # 2. Migrate 'learned' mental models to learnings
    # Each observation in a learned model becomes a separate learning
    op.execute(f"""
        INSERT INTO {schema}learnings (bank_id, text, proof_count, tags, created_at)
        SELECT
            mm.bank_id,
            obs->>'content' AS text,
            GREATEST(1, COALESCE(jsonb_array_length(obs->'evidence'), 1)) AS proof_count,
            mm.tags,
            mm.created_at
        FROM {schema}mental_models mm,
        LATERAL jsonb_array_elements(mm.observations->'observations') AS obs
        WHERE mm.subtype = 'learned'
          AND obs->>'content' IS NOT NULL
          AND obs->>'content' != ''
        ON CONFLICT DO NOTHING
    """)

    # 3. Delete all non-directive mental models (they've been migrated or are obsolete)
    op.execute(f"""
        DELETE FROM {schema}mental_models
        WHERE subtype != 'directive'
    """)

    # 4. Drop the mental_model_versions table (no longer used)
    op.execute(f"DROP TABLE IF EXISTS {schema}mental_model_versions CASCADE")

    # 5. Drop old constraints and add new one that allows current subtypes.
    # 'pinned' is still used by the code for user-created mental models;
    # 'directive' is used for system directives.
    op.execute(f"ALTER TABLE {schema}mental_models DROP CONSTRAINT IF EXISTS ck_mental_models_subtype")
    op.execute(f"""
        ALTER TABLE {schema}mental_models
        ADD CONSTRAINT ck_mental_models_subtype CHECK (subtype IN ('directive', 'pinned'))
    """)


def downgrade() -> None:
    """Reverse the migration (data migration is one-way, so this just removes constraints)."""
    schema = _get_schema_prefix()

    # Remove the directive-only constraint
    op.execute(f"ALTER TABLE {schema}mental_models DROP CONSTRAINT IF EXISTS ck_mental_models_subtype")

    # Re-create mental_model_versions table
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS {schema}mental_model_versions (
            id SERIAL PRIMARY KEY,
            bank_id VARCHAR(64) NOT NULL,
            model_id VARCHAR(128) NOT NULL,
            version INT NOT NULL,
            observations JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
        )
    """)
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_mm_versions_lookup ON {schema}mental_model_versions(bank_id, model_id, version DESC)"
    )

    # Note: Data migration cannot be reversed - pinned_reflections and learnings data remains
