"""add_entity_summary

Revision ID: f1a2b3c4d5e6
Revises: e0a1b2c3d4e5
Create Date: 2026-01-07 00:00:00.000000

Replaces the observation system (multiple memory_units per entity) with a single
summary TEXT column on the entities table. This makes observations:
- Independent from retain transactions
- A single readme-like text per entity
- Not searchable (only retrieved when entity is matched)
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
    """Add summary columns to entities and migrate existing observations."""

    # Step 1: Add new columns to entities table
    op.add_column("entities", sa.Column("summary", sa.Text(), nullable=True))
    op.add_column(
        "entities",
        sa.Column("summary_updated_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
    )
    op.add_column(
        "entities",
        sa.Column("summary_fact_count", sa.Integer(), server_default="0", nullable=False),
    )

    # Step 2: Migrate existing observations to summaries
    # Aggregate all observation texts for each entity into a single summary
    op.execute("""
        WITH aggregated AS (
            SELECT
                e.id as entity_id,
                e.bank_id,
                string_agg('- ' || mu.text, E'\n' ORDER BY mu.mentioned_at DESC) as combined_summary
            FROM entities e
            JOIN unit_entities ue ON e.id = ue.entity_id
            JOIN memory_units mu ON ue.unit_id = mu.id
            WHERE mu.fact_type = 'observation'
            GROUP BY e.id, e.bank_id
        )
        UPDATE entities e
        SET
            summary = a.combined_summary,
            summary_updated_at = NOW(),
            summary_fact_count = COALESCE((
                SELECT COUNT(*)
                FROM unit_entities ue2
                JOIN memory_units mu2 ON ue2.unit_id = mu2.id
                WHERE ue2.entity_id = e.id
                AND mu2.bank_id = e.bank_id
                AND mu2.fact_type IN ('world', 'experience')
            ), 0)
        FROM aggregated a
        WHERE e.id = a.entity_id AND e.bank_id = a.bank_id
    """)

    # Step 3: Delete observation memory_units (cascades to unit_entities links)
    op.execute("DELETE FROM memory_units WHERE fact_type = 'observation'")

    # Note: We don't modify the fact_type constraint here to keep backwards compatibility
    # The 'observation' type can remain in the constraint, it just won't be used anymore

    # Step 4: Drop observation-specific index (if it exists)
    op.execute("DROP INDEX IF EXISTS idx_memory_units_observation_date")


def downgrade() -> None:
    """Remove summary columns from entities."""

    # Note: Cannot restore deleted observations - they are lost on downgrade
    # Constraint modifications were not made in upgrade, so nothing to restore

    # Drop summary columns from entities
    op.drop_column("entities", "summary_fact_count")
    op.drop_column("entities", "summary_updated_at")
    op.drop_column("entities", "summary")
