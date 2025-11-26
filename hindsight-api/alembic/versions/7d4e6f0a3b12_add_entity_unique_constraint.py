"""add unique constraint on entities (agent_id, canonical_name)

Revision ID: 7d4e6f0a3b12
Revises: 5b2c6d8e9f01
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7d4e6f0a3b12'
down_revision: Union[str, None] = '5b2c6d8e9f01'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # First, deduplicate existing entities by merging duplicates
    # Keep the one with highest mention_count, update unit_entities to point to it
    op.execute("""
        -- Create temp table with canonical entity per (agent_id, name)
        CREATE TEMP TABLE canonical_entities AS
        SELECT DISTINCT ON (agent_id, LOWER(canonical_name))
            id as keep_id,
            agent_id,
            LOWER(canonical_name) as name_lower
        FROM entities
        ORDER BY agent_id, LOWER(canonical_name), mention_count DESC, first_seen ASC;

        -- Get all entity IDs that will be removed (duplicates)
        CREATE TEMP TABLE duplicate_entities AS
        SELECT e.id as dup_id, ce.keep_id
        FROM entities e
        JOIN canonical_entities ce ON e.agent_id = ce.agent_id AND LOWER(e.canonical_name) = ce.name_lower
        WHERE e.id != ce.keep_id;

        -- Update unit_entities to point to canonical entity
        UPDATE unit_entities ue
        SET entity_id = de.keep_id
        FROM duplicate_entities de
        WHERE ue.entity_id = de.dup_id;

        -- Delete duplicate unit_entities that now exist
        DELETE FROM unit_entities a
        USING unit_entities b
        WHERE a.unit_id = b.unit_id
          AND a.entity_id = b.entity_id
          AND a.ctid < b.ctid;

        -- For entity_cooccurrences, we need to be careful about the check constraint
        -- First, collect all cooccurrences that need updating into a temp table with correct ordering
        CREATE TEMP TABLE new_cooccurrences AS
        SELECT DISTINCT
            LEAST(
                COALESCE(de1.keep_id, ec.entity_id_1),
                COALESCE(de2.keep_id, ec.entity_id_2)
            ) as entity_id_1,
            GREATEST(
                COALESCE(de1.keep_id, ec.entity_id_1),
                COALESCE(de2.keep_id, ec.entity_id_2)
            ) as entity_id_2,
            SUM(ec.cooccurrence_count) as cooccurrence_count,
            MAX(ec.last_cooccurred) as last_cooccurred
        FROM entity_cooccurrences ec
        LEFT JOIN duplicate_entities de1 ON ec.entity_id_1 = de1.dup_id
        LEFT JOIN duplicate_entities de2 ON ec.entity_id_2 = de2.dup_id
        GROUP BY
            LEAST(COALESCE(de1.keep_id, ec.entity_id_1), COALESCE(de2.keep_id, ec.entity_id_2)),
            GREATEST(COALESCE(de1.keep_id, ec.entity_id_1), COALESCE(de2.keep_id, ec.entity_id_2));

        -- Delete rows where entity_id_1 = entity_id_2 (self-references after merge)
        DELETE FROM new_cooccurrences WHERE entity_id_1 = entity_id_2;

        -- Delete all old cooccurrences
        DELETE FROM entity_cooccurrences;

        -- Insert the merged cooccurrences
        INSERT INTO entity_cooccurrences (entity_id_1, entity_id_2, cooccurrence_count, last_cooccurred)
        SELECT entity_id_1, entity_id_2, cooccurrence_count, last_cooccurred
        FROM new_cooccurrences;

        -- Update mention counts on canonical entities
        UPDATE entities e
        SET mention_count = e.mention_count + COALESCE(
            (SELECT SUM(e2.mention_count)
             FROM entities e2
             JOIN duplicate_entities de ON e2.id = de.dup_id
             WHERE de.keep_id = e.id),
            0
        )
        WHERE e.id IN (SELECT keep_id FROM duplicate_entities);

        -- Delete duplicate entities
        DELETE FROM entities
        WHERE id IN (SELECT dup_id FROM duplicate_entities);

        -- Cleanup temp tables
        DROP TABLE new_cooccurrences;
        DROP TABLE duplicate_entities;
        DROP TABLE canonical_entities;
    """)

    # Add unique constraint (case-insensitive)
    op.create_index(
        'idx_entities_agent_canonical_unique',
        'entities',
        [sa.text('agent_id'), sa.text('LOWER(canonical_name)')],
        unique=True
    )


def downgrade() -> None:
    op.drop_index('idx_entities_agent_canonical_unique', table_name='entities')
