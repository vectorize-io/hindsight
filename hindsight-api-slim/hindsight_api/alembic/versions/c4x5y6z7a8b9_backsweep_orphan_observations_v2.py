"""backsweep_orphan_observations_v2

Re-run of Pass 2 from migration ``g7h8i9j0k1l2_backsweep_orphan_observations``
to sweep observations that became orphaned between then and now.

Why we need it again:
  ``fact_storage.handle_document_tracking`` (the retain/upsert path) deleted
  the existing document via the FK cascade — which removes the source
  ``memory_units`` — but never invalidated the observations derived from
  them.  Only the explicit ``MemoryEngine.delete_document`` API called
  ``_delete_stale_observations_for_memories``.  Every document re-ingest
  therefore left orphan observations whose ``source_memory_ids`` arrays
  pointed at IDs that no longer existed in ``memory_units``.

  ``handle_document_tracking`` now calls the same cleanup helper before the
  cascade, so no new orphans will accumulate going forward.  This migration
  cleans up the historical residue.

Identical to Pass 2 of g7h8i9j0k1l2.  Pass 1 (memory_units whose bank is
gone) is intentionally not re-run; that scenario has no fresh source.

Revision ID: c4x5y6z7a8b9
Revises: b3w4x5y6z7a8
Create Date: 2026-04-16
"""

from collections.abc import Sequence

from alembic import context, op

revision: str = "c4x5y6z7a8b9"
down_revision: str | Sequence[str] | None = "b3w4x5y6z7a8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    schema = _get_schema_prefix()
    mu = f"{schema}memory_units"

    # Delete observations whose every source_memory_id refers to a now-deleted
    # memory_unit (or the array is empty).  Observations with at least one
    # surviving source are left alone — the consolidation engine will refresh
    # their text on the next pass.
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


def downgrade() -> None:
    # Deleted rows cannot be restored.
    pass
