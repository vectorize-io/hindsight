"""Repair memory_units.chunk_id FK to ON DELETE CASCADE

Multiple production deployments report ``ForeignKeyViolationError`` during
delta retain when the orchestrator tries to delete changed/removed chunks.
The ``memory_units.chunk_id`` FK is supposed to be ``ON DELETE CASCADE``
(established by ``f6g7h8i9j0k1_chunk_fk_cascade_delete``) but on databases
whose ``alembic_version`` jumped ahead via a divergent-head path that
bypassed that migration, the FK remains without CASCADE/SET NULL semantics
(e.g. ``NO ACTION``), causing chunk deletes to fail.

The symptom is::

    update or delete on table "chunks" violates foreign key constraint
    "memory_units_chunk_fkey" on table "memory_units"
    DETAIL: Key (chunk_id)=(...) is still referenced from table "memory_units".

``delete_chunks_by_ids`` in ``chunk_storage.py`` deletes ``memory_links``
first, then ``chunks``, expecting the FK CASCADE to remove the
``memory_units`` rows — but without CASCADE the commit fails.

This migration sits at the current head so every affected deployment picks
it up on next container start.  It is fully idempotent (``DROP IF EXISTS``
+ ``DO … WHEN duplicate_object``) so it is safe to re-apply on databases
where the CASCADE FK is already in place.

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

    Checks ``pg_constraint`` first and only drops/recreates the constraint
    when it is missing or uses a non-CASCADE delete action.  This avoids
    an unnecessary ``ACCESS EXCLUSIVE`` lock on healthy databases where the
    FK is already correct.
    """
    schema = _pg_schema_prefix()
    bare_schema = schema.strip(".").strip('"') if schema else ""
    schema_clause = f"AND n.nspname = '{bare_schema}'" if bare_schema else ""

    op.execute(
        f"""DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint c
                JOIN pg_class r ON r.oid = c.conrelid
                JOIN pg_namespace n ON n.oid = r.relnamespace
                WHERE c.conname = 'memory_units_chunk_fkey'
                  AND r.relname = 'memory_units'
                  AND c.contype = 'f'
                  AND c.confdeltype = 'c'  -- 'c' = CASCADE
                  {schema_clause}
            ) THEN
                ALTER TABLE {schema}memory_units
                    DROP CONSTRAINT IF EXISTS memory_units_chunk_fkey;
                ALTER TABLE {schema}memory_units
                    ADD CONSTRAINT memory_units_chunk_fkey
                    FOREIGN KEY (chunk_id)
                    REFERENCES {schema}chunks (chunk_id)
                    ON DELETE CASCADE;
            END IF;
        END$$;
        """
    )


def _pg_downgrade() -> None:
    """No-op: canonical schema already has ON DELETE CASCADE (see f6g7h8i9j0k1)."""
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
