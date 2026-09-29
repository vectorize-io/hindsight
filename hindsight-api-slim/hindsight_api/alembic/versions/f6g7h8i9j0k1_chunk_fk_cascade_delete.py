"""chunk_fk_cascade_delete

Revision ID: f6g7h8i9j0k1
Revises: e5f6g7h8i9j0
Create Date: 2026-03-16 00:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

from hindsight_api.alembic._dialect import run_for_dialect

# revision identifiers, used by Alembic.
revision: str = "f6g7h8i9j0k1"
down_revision: str | Sequence[str] | None = "e5f6g7h8i9j0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_upgrade() -> None:
    """Change memory_units.chunk_id FK from SET NULL to CASCADE.

    When a document is deleted the CASCADE reaches chunks first; with SET NULL
    the memory_units rows survived with chunk_id = NULL, leaving ghost records.
    Switching to CASCADE ensures they are removed together with their chunk.
    """
    from alembic import context

    schema = context.config.get_main_option("target_schema")
    schema_prefix = f'"{schema}".' if schema else ""
    # Use raw SQL with IF EXISTS so this is safe on schemas where the FK was
    # already dropped or never existed under this name.
    op.execute(f"ALTER TABLE {schema_prefix}memory_units DROP CONSTRAINT IF EXISTS memory_units_chunk_fkey")
    # Use a DO block so the ADD is also idempotent: if the FK already exists (e.g.
    # the schema was provisioned after the base migration already added it) the
    # duplicate_object exception is swallowed rather than failing the migration.
    op.execute(
        f"""
        DO $$ BEGIN
            ALTER TABLE {schema_prefix}memory_units
                ADD CONSTRAINT memory_units_chunk_fkey
                FOREIGN KEY (chunk_id)
                REFERENCES {schema_prefix}chunks (chunk_id)
                ON DELETE CASCADE;
        EXCEPTION
            WHEN duplicate_object THEN NULL;
        END $$;
        """
    )


def _pg_downgrade() -> None:
    """Revert to SET NULL behaviour."""
    op.drop_constraint("memory_units_chunk_fkey", "memory_units", type_="foreignkey")
    op.create_foreign_key(
        "memory_units_chunk_fkey", "memory_units", "chunks", ["chunk_id"], ["chunk_id"], ondelete="SET NULL"
    )


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
