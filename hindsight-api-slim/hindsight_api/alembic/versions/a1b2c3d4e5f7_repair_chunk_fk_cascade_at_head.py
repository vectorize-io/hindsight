"""Repair memory_units.chunk_id FK to ON DELETE CASCADE

Multiple production deployments report ``ForeignKeyViolationError`` during
delta retain when the orchestrator tries to delete changed/removed chunks.
The ``memory_units.chunk_id`` FK is supposed to be ``ON DELETE CASCADE``
(established by ``f6g7h8i9j0k1_chunk_fk_cascade_delete``) but on databases
whose ``alembic_version`` jumped ahead via a divergent-head path that
bypassed that migration, the FK remains ``NO ACTION`` (the original default
from ``b7c4d8e9f1a2_add_chunks_table``).

The symptom is::

    update or delete on table "chunks" violates foreign key constraint
    "memory_units_chunk_fkey" on table "memory_units"
    DETAIL: Key (chunk_id)=(...) is still referenced from table "memory_units".

``delete_chunks_by_ids`` in ``chunk_storage.py`` deletes ``memory_links``
first, then ``chunks``, expecting the FK CASCADE to remove the
``memory_units`` rows — but without CASCADE the commit fails.

This migration sits at the current head so every affected deployment picks
it up on next container start.  It is fully idempotent (``DROP IF EXISTS``
+ ``DO … WHEN duplicate_object``) so it is a no-op on databases where the
CASCADE FK is already in place.

Revision ID: a1b2c3d4e5f7
Revises: c7d1e9a4b3f2
Create Date: 2026-07-27
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "a1b2c3d4e5f7"
down_revision: str | Sequence[str] | None = "c7d1e9a4b3f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    """Idempotently ensure memory_units.chunk_id FK uses ON DELETE CASCADE.

    Safe to re-apply on databases where ``f6g7h8i9j0k1`` already switched
    the constraint — the ``DROP … IF EXISTS`` followed by a guarded
    ``ADD CONSTRAINT`` is a no-op when the CASCADE FK is already present.
    """
    schema = _pg_schema_prefix()

    # Drop the existing FK regardless of its current ON DELETE action.
    op.execute(
        f"ALTER TABLE {schema}memory_units DROP CONSTRAINT IF EXISTS memory_units_chunk_fkey"
    )
    # Use a DO block so the ADD is idempotent: if a concurrent migration
    # already created the CASCADE FK, the duplicate_object exception is
    # swallowed rather than failing the migration.
    op.execute(
        f"""
        DO $$ BEGIN
            ALTER TABLE {schema}memory_units
                ADD CONSTRAINT memory_units_chunk_fkey
                FOREIGN KEY (chunk_id)
                REFERENCES {schema}chunks (chunk_id)
                ON DELETE CASCADE;
        EXCEPTION
            WHEN duplicate_object THEN NULL;
        END $$;
        """
    )


def _pg_downgrade() -> None:
    """Revert to NO ACTION behaviour (original default)."""
    schema = _pg_schema_prefix()
    op.drop_constraint(
        "memory_units_chunk_fkey", "memory_units", type_="foreignkey"
    )
    op.create_foreign_key(
        "memory_units_chunk_fkey",
        "memory_units",
        "chunks",
        ["chunk_id"],
        ["chunk_id"],
    )


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
