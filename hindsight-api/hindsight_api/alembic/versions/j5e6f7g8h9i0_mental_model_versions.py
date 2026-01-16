"""mental_model_versions

Revision ID: j5e6f7g8h9i0
Revises: i4d5e6f7g8h9
Create Date: 2026-01-16 00:00:00.000000

This migration adds versioning support for mental models:
1. Creates mental_model_versions table to store observation snapshots
2. Adds version column to mental_models for tracking current version

This enables changelog/diff functionality for mental model observations.
"""

from collections.abc import Sequence

from alembic import context, op

# revision identifiers, used by Alembic.
revision: str = "j5e6f7g8h9i0"
down_revision: str | Sequence[str] | None = "i4d5e6f7g8h9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    """Create mental_model_versions table and add version tracking."""
    schema = _get_schema_prefix()

    # Create mental_model_versions table for storing observation snapshots
    op.execute(f"""
        CREATE TABLE {schema}mental_model_versions (
            id SERIAL PRIMARY KEY,
            mental_model_id VARCHAR(64) NOT NULL,
            bank_id VARCHAR(64) NOT NULL,
            version INT NOT NULL,
            observations JSONB NOT NULL DEFAULT '{{"observations": []}}'::jsonb,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            FOREIGN KEY (mental_model_id, bank_id)
                REFERENCES {schema}mental_models(id, bank_id) ON DELETE CASCADE,
            UNIQUE (mental_model_id, bank_id, version)
        )
    """)

    # Index for efficient version queries (get latest, list versions)
    op.execute(f"""
        CREATE INDEX idx_mental_model_versions_lookup
        ON {schema}mental_model_versions(mental_model_id, bank_id, version DESC)
    """)

    # Add version column to mental_models to track current version
    op.execute(f"""
        ALTER TABLE {schema}mental_models
        ADD COLUMN IF NOT EXISTS version INT NOT NULL DEFAULT 0
    """)

    # Migrate existing mental models: create version 1 for any that have observations
    op.execute(f"""
        INSERT INTO {schema}mental_model_versions (mental_model_id, bank_id, version, observations, created_at)
        SELECT id, bank_id, 1, observations, COALESCE(last_updated, created_at)
        FROM {schema}mental_models
        WHERE observations IS NOT NULL
          AND observations != '{{"observations": []}}'::jsonb
          AND (observations->'observations') IS NOT NULL
          AND jsonb_array_length(observations->'observations') > 0
    """)

    # Update version to 1 for migrated mental models
    op.execute(f"""
        UPDATE {schema}mental_models
        SET version = 1
        WHERE observations IS NOT NULL
          AND observations != '{{"observations": []}}'::jsonb
          AND (observations->'observations') IS NOT NULL
          AND jsonb_array_length(observations->'observations') > 0
    """)


def downgrade() -> None:
    """Remove mental_model_versions table and version column."""
    schema = _get_schema_prefix()

    # Drop index
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_mental_model_versions_lookup")

    # Drop versions table
    op.execute(f"DROP TABLE IF EXISTS {schema}mental_model_versions")

    # Remove version column from mental_models
    op.execute(f"ALTER TABLE {schema}mental_models DROP COLUMN IF EXISTS version")
