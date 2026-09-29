"""fix_mental_model_fact_type

Revision ID: q2l3m4n5o6p7
Revises: p1k2l3m4n5o6
Create Date: 2026-01-21 13:30:00.000000

Fix the fact_type check constraint to include 'mental_model'.
This is a fix for p1k2l3m4n5o6 which should have included this change.
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

# revision identifiers, used by Alembic.
revision: str = "q2l3m4n5o6p7"
down_revision: str | Sequence[str] | None = "p1k2l3m4n5o6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    """Add 'mental_model' to the fact_type check constraint."""
    schema = _get_schema_prefix()

    # Drop the old constraint and add the new one with mental_model included
    op.execute(f"ALTER TABLE {schema}memory_units DROP CONSTRAINT IF EXISTS memory_units_fact_type_check")
    op.execute(f"""
        ALTER TABLE {schema}memory_units
        ADD CONSTRAINT memory_units_fact_type_check
        CHECK (fact_type IN ('world', 'experience', 'opinion', 'observation', 'mental_model'))
    """)


def _pg_downgrade() -> None:
    """Remove 'mental_model' from the fact_type check constraint."""
    schema = _get_schema_prefix()

    op.execute(f"ALTER TABLE {schema}memory_units DROP CONSTRAINT IF EXISTS memory_units_fact_type_check")
    op.execute(f"""
        ALTER TABLE {schema}memory_units
        ADD CONSTRAINT memory_units_fact_type_check
        CHECK (fact_type IN ('world', 'experience', 'opinion', 'observation'))
    """)


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
