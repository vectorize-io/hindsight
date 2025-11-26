"""add_observation_fact_type

Revision ID: 5b2c6d8e9f01
Revises: 4a8b3c5d6e7f
Create Date: 2025-11-26 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5b2c6d8e9f01'
down_revision: Union[str, Sequence[str], None] = '4a8b3c5d6e7f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Drop old constraints
    op.execute("""
        ALTER TABLE memory_units
        DROP CONSTRAINT IF EXISTS confidence_score_fact_type_check
    """)
    op.execute("""
        ALTER TABLE memory_units
        DROP CONSTRAINT IF EXISTS memory_units_fact_type_check
    """)

    # Add new fact_type constraint including 'observation'
    op.execute("""
        ALTER TABLE memory_units
        ADD CONSTRAINT memory_units_fact_type_check
        CHECK (fact_type IN ('world', 'agent', 'opinion', 'observation'))
    """)

    # Add new confidence_score constraint allowing observation to have optional confidence
    op.execute("""
        ALTER TABLE memory_units
        ADD CONSTRAINT confidence_score_fact_type_check
        CHECK (
            (fact_type = 'opinion' AND confidence_score IS NOT NULL) OR
            (fact_type = 'observation') OR
            (fact_type NOT IN ('opinion', 'observation') AND confidence_score IS NULL)
        )
    """)

    # Add index for observation fact_type queries
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_memory_units_observation_date
        ON memory_units (agent_id, event_date DESC)
        WHERE fact_type = 'observation'
    """)


def downgrade() -> None:
    """Downgrade schema."""
    # Drop observation index
    op.execute("DROP INDEX IF EXISTS idx_memory_units_observation_date")

    # Drop new constraints
    op.execute("""
        ALTER TABLE memory_units
        DROP CONSTRAINT IF EXISTS confidence_score_fact_type_check
    """)
    op.execute("""
        ALTER TABLE memory_units
        DROP CONSTRAINT IF EXISTS memory_units_fact_type_check
    """)

    # Restore old fact_type constraint
    op.execute("""
        ALTER TABLE memory_units
        ADD CONSTRAINT memory_units_fact_type_check
        CHECK (fact_type IN ('world', 'agent', 'opinion'))
    """)

    # Restore old confidence_score constraint
    op.execute("""
        ALTER TABLE memory_units
        ADD CONSTRAINT confidence_score_fact_type_check
        CHECK (
            (fact_type = 'opinion' AND confidence_score IS NOT NULL) OR
            (fact_type != 'opinion' AND confidence_score IS NULL)
        )
    """)
