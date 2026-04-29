"""Create observation_sources junction table

Replaces the source_memory_ids UUID[] column (PG) / CLOB (Oracle) with a
proper junction table. This eliminates dialect-specific array operators
(&&, unnest, JSON_TABLE) and enables standard SQL joins for all backends.

The old source_memory_ids column is retained for now (dual-write) and will
be dropped in a future migration once all read paths are migrated.

Revision ID: k6l7m8n9o0p1
Revises: i4j5k6l7m8n9
Create Date: 2026-04-24
"""

from collections.abc import Sequence

from alembic import context, op

revision: str = "k6l7m8n9o0p1"
down_revision: str | Sequence[str] | None = "i4j5k6l7m8n9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    schema = _get_schema_prefix()

    # Create junction table.
    # observation_id has ON DELETE CASCADE so deleting an observation cleans up its rows.
    # source_id intentionally has NO FK — when a source memory is deleted, we need
    # observation_sources rows to still exist so delete_stale_observations_for_memories()
    # can find affected observations. Those observations are then deleted, which cascades
    # to observation_sources via the observation_id FK.
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS {schema}observation_sources (
            observation_id UUID NOT NULL,
            source_id UUID NOT NULL,
            PRIMARY KEY (observation_id, source_id),
            FOREIGN KEY (observation_id) REFERENCES {schema}memory_units(id) ON DELETE CASCADE
        )
    """)

    # Index on source_id for reverse lookups (find observations referencing a given source)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_obs_sources_source_id
        ON {schema}observation_sources(source_id, observation_id)
    """)

    # Backfill from existing source_memory_ids array column
    op.execute(f"""
        INSERT INTO {schema}observation_sources (observation_id, source_id)
        SELECT mu.id, unnest(mu.source_memory_ids)
        FROM {schema}memory_units mu
        WHERE mu.fact_type = 'observation'
          AND mu.source_memory_ids IS NOT NULL
          AND array_length(mu.source_memory_ids, 1) > 0
        ON CONFLICT DO NOTHING
    """)


def downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_obs_sources_source_id")
    op.execute(f"DROP TABLE IF EXISTS {schema}observation_sources")
