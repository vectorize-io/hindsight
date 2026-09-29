"""Make memory_links.from_unit_id and memory_links.to_unit_id FKs deferrable.

Revision ID: 9f8e7d6c5b4a
Revises: o1a2b3c4d5e6
Create Date: 2026-05-03

Background
----------
Concurrent retain (which INSERTs into ``memory_links``) and any code path
that DELETEs a row whose deletion cascades into ``memory_links`` (e.g.
delta-retain superseding chunks, which CASCADEs chunks → memory_units →
memory_links) can deadlock under sustained single-tenant write load.

The deadlock cycle:

* Tx A: ``DELETE FROM chunks WHERE chunk_id = ANY(...)``
  → CASCADE acquires row locks on memory_units, then on memory_links rows
    where ``to_unit_id`` matches the deleted units.
* Tx B: ``INSERT INTO memory_links (...)`` referencing one of the same
  memory_units rows.
  → The immediate FK check takes ``FOR KEY SHARE`` on those memory_units
    rows.

The two transactions take row locks on the same memory_units rows in
opposite orders depending on which side started first. PostgreSQL detects
the cycle and aborts one transaction; the loser is killed mid-batch, the
winner continues. Workers then retry, but under sustained write load the
pattern repeats.

Fix
---
Make both ``memory_links → memory_units`` FKs (``from_unit_id`` and
``to_unit_id``) ``DEFERRABLE INITIALLY DEFERRED``. This pushes the FK
check from INSERT time to COMMIT time:

* INSERT no longer takes ``FOR KEY SHARE`` on the memory_units row → no
  contention with the cascading DELETE's row lock.
* At COMMIT the engine validates referential integrity in one shot. If a
  cascade-DELETE has since removed the referenced unit, the INSERT
  transaction commits OR fails with a clean FK violation (sqlstate
  23503) instead of a deadlock (sqlstate 40P01).

The ``WHERE EXISTS`` filter already in ``_bulk_insert_links`` continues to
filter out the typical "stale unit_id" case at INSERT time; the deferred
FK is only the backstop for the narrow race window between the EXISTS
probe and COMMIT. ``ON DELETE CASCADE`` semantics are unchanged — only
the *timing* of the constraint check moves.

The ``entity_id`` FK on ``memory_links`` is not changed; entities are not
involved in the observed deadlock cycle and leaving the constraint
immediate keeps the error message specific when an entity row is missing.
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "9f8e7d6c5b4a"
down_revision: str | Sequence[str] | None = "o1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


# The two FK constraints installed by the initial schema migration
# (5a366d414dce_initial_schema), mapped to the column they constrain.
# They reference memory_units(id) with ON DELETE CASCADE — that
# semantics is preserved; only the deferral attribute changes.
_FK_COLUMNS: dict[str, str] = {
    "fk_memory_links_from_unit_id_memory_units": "from_unit_id",
    "fk_memory_links_to_unit_id_memory_units": "to_unit_id",
}


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()
    # PostgreSQL doesn't allow altering the deferrability of an existing
    # constraint with ALTER CONSTRAINT — the constraint must be dropped
    # and recreated. DROP IF EXISTS makes the migration safe to re-run
    # on schemas where the constraint was already recreated.
    for fk_name, column in _FK_COLUMNS.items():
        op.execute(f"ALTER TABLE {schema}memory_links DROP CONSTRAINT IF EXISTS {fk_name}")
        op.execute(
            f"""
            ALTER TABLE {schema}memory_links
                ADD CONSTRAINT {fk_name}
                FOREIGN KEY ({column})
                REFERENCES {schema}memory_units (id)
                ON DELETE CASCADE
                DEFERRABLE INITIALLY DEFERRED
            """
        )


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()
    # Revert to the default (NOT DEFERRABLE) form so a downgrade actually
    # restores the prior schema state, even though that re-introduces the
    # deadlock window.
    for fk_name, column in _FK_COLUMNS.items():
        op.execute(f"ALTER TABLE {schema}memory_links DROP CONSTRAINT IF EXISTS {fk_name}")
        op.execute(
            f"""
            ALTER TABLE {schema}memory_links
                ADD CONSTRAINT {fk_name}
                FOREIGN KEY ({column})
                REFERENCES {schema}memory_units (id)
                ON DELETE CASCADE
            """
        )


def upgrade() -> None:
    # PG-only: Oracle's deferrable-FK semantics differ and the deadlock
    # cycle was only observed on PostgreSQL. Oracle slot intentionally
    # absent → no-op there.
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
