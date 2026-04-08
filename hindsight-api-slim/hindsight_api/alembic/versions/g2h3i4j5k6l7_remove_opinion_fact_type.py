"""remove_opinion_fact_type

Revision ID: g2h3i4j5k6l7
Revises: f1a2b3c4d5e6
Create Date: 2026-04-02

Remove the deprecated 'opinion' fact type: drop opinion-specific indexes,
update CHECK constraints, delete any remaining opinion rows, and drop the
confidence_score column (was only used for opinions, always NULL otherwise).
"""

from collections.abc import Sequence

from alembic import context, op

# revision identifiers, used by Alembic.
revision: str = "g2h3i4j5k6l7"
down_revision: str | Sequence[str] | None = "f1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    schema = _get_schema_prefix()

    # 1. Delete any remaining opinion rows
    op.execute(f"DELETE FROM {schema}memory_units WHERE fact_type = 'opinion'")

    # 2. Drop opinion-specific indexes
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_opinion_confidence")
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_opinion_date")

    # 3. Drop confidence_score constraints and column (only used for opinions, always NULL otherwise)
    op.execute(f"ALTER TABLE {schema}memory_units DROP CONSTRAINT IF EXISTS confidence_score_fact_type_check")
    op.execute(f"ALTER TABLE {schema}memory_units DROP CONSTRAINT IF EXISTS memory_units_confidence_score_check")
    op.execute(f"ALTER TABLE {schema}memory_units DROP COLUMN IF EXISTS confidence_score")

    # 4. Replace fact_type CHECK constraint
    op.execute(f"ALTER TABLE {schema}memory_units DROP CONSTRAINT IF EXISTS memory_units_fact_type_check")
    op.execute(
        f"ALTER TABLE {schema}memory_units ADD CONSTRAINT memory_units_fact_type_check "
        f"CHECK (fact_type IN ('world', 'experience', 'observation'))"
    )


def downgrade() -> None:
    schema = _get_schema_prefix()

    # Restore confidence_score column
    op.execute(f"ALTER TABLE {schema}memory_units ADD COLUMN IF NOT EXISTS confidence_score float")
    op.execute(
        f"ALTER TABLE {schema}memory_units ADD CONSTRAINT memory_units_confidence_score_check "
        f"CHECK (confidence_score IS NULL OR (confidence_score >= 0.0 AND confidence_score <= 1.0))"
    )
    op.execute(
        f"ALTER TABLE {schema}memory_units ADD CONSTRAINT confidence_score_fact_type_check "
        f"CHECK ((fact_type = 'opinion' AND confidence_score IS NOT NULL) OR "
        f"(fact_type = 'observation') OR "
        f"(fact_type NOT IN ('opinion', 'observation') AND confidence_score IS NULL))"
    )

    # Restore original fact_type CHECK constraint (with opinion)
    op.execute(f"ALTER TABLE {schema}memory_units DROP CONSTRAINT IF EXISTS memory_units_fact_type_check")
    op.execute(
        f"ALTER TABLE {schema}memory_units ADD CONSTRAINT memory_units_fact_type_check "
        f"CHECK (fact_type IN ('world', 'experience', 'opinion', 'observation'))"
    )

    # Recreate opinion indexes
    op.execute(
        f"CREATE INDEX idx_memory_units_opinion_confidence ON {schema}memory_units "
        f"(bank_id, confidence_score DESC) WHERE fact_type = 'opinion'"
    )
    op.execute(
        f"CREATE INDEX idx_memory_units_opinion_date ON {schema}memory_units "
        f"(bank_id, event_date DESC) WHERE fact_type = 'opinion'"
    )
