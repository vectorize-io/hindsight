"""backsweep_orphan_memory_units

Two-pass cleanup of memory_units rows that were never removed by earlier bugs:

Pass 1 — any fact_type, bank gone:
  memory_units whose bank_id no longer exists in banks.  These accumulate when
  a bank is deleted without a proper cascade (no FK from memory_units to banks
  exists in the schema).

Pass 2 — observations only, all sources gone:
  observation rows whose bank still exists but every source_memory_id points
  to a deleted memory unit.  These were left behind before PR #580 fixed the
  chunk FK cascade and before delete_document() called
  _delete_stale_observations_for_memories.

Revision ID: g7h8i9j0k1l2
Revises: f6g7h8i9j0k1
Create Date: 2026-03-16
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "g7h8i9j0k1l2"
down_revision: str | Sequence[str] | None = "f6g7h8i9j0k1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()
    mu = f"{schema}memory_units"
    banks = f"{schema}banks"

    # Pass 1: delete all memory_units (any fact_type) whose bank no longer exists.
    # There is no FK from memory_units to banks, so these never cascade away.
    op.execute(
        f"""
        DELETE FROM {mu}
        WHERE NOT EXISTS (
            SELECT 1 FROM {banks} b WHERE b.bank_id = {mu}.bank_id
        )
        """
    )

    # Pass 2: delete orphaned observations whose bank still exists but every
    # source_memory_id refers to a now-deleted memory unit (or the array is
    # empty).  Observations with at least one surviving source are left alone.
    op.execute(
        f"""
        DELETE FROM {mu} orphan
        WHERE orphan.fact_type = 'observation'
          AND NOT EXISTS (
              SELECT 1
              FROM {mu} src
              WHERE src.id = ANY(orphan.source_memory_ids)
                AND src.bank_id = orphan.bank_id
          )
        """
    )


def _pg_downgrade() -> None:
    # Deleted rows cannot be restored.
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
