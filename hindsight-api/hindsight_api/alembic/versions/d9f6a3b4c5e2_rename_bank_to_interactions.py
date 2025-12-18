"""Rename fact_type 'bank' to 'experience'

Revision ID: d9f6a3b4c5e2
Revises: c8e5f2a3b4d1
Create Date: 2024-12-04 15:00:00.000000

"""

from alembic import context, op

# revision identifiers, used by Alembic.
revision = "d9f6a3b4c5e2"
down_revision = "c8e5f2a3b4d1"
branch_labels = None
depends_on = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (e.g., 'tenant_x.' or '' for public)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade():
    schema = _get_schema_prefix()

    # Drop old check constraint FIRST (before updating data)
    op.drop_constraint("memory_units_fact_type_check", "memory_units", type_="check")

    # Update existing 'bank' values to 'experience'
    op.execute(f"UPDATE {schema}memory_units SET fact_type = 'experience' WHERE fact_type = 'bank'")
    # Also update any 'interactions' values (in case of partial migration)
    op.execute(f"UPDATE {schema}memory_units SET fact_type = 'experience' WHERE fact_type = 'interactions'")

    # Create new check constraint with 'experience' instead of 'bank'
    op.create_check_constraint(
        "memory_units_fact_type_check", "memory_units", "fact_type IN ('world', 'experience', 'opinion', 'observation')"
    )


def downgrade():
    schema = _get_schema_prefix()

    # Drop new check constraint FIRST
    op.drop_constraint("memory_units_fact_type_check", "memory_units", type_="check")

    # Update 'experience' back to 'bank'
    op.execute(f"UPDATE {schema}memory_units SET fact_type = 'bank' WHERE fact_type = 'experience'")

    # Recreate old check constraint
    op.create_check_constraint(
        "memory_units_fact_type_check", "memory_units", "fact_type IN ('world', 'bank', 'opinion', 'observation')"
    )
