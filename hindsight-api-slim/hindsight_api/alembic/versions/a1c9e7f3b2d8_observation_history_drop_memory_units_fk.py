"""Drop observation_history's FK to memory_units.

The history table records one snapshot per observation change, keyed by
``(bank_id, observation_id)``. Its foreign key to ``memory_units`` existed only to
cascade-delete history when the observation row went away.

That assumes every observation *is* a ``memory_units`` row, which is true only
while Postgres is the memories store. When another store owns the memories the
observation lives there and Postgres holds no row for it, so every history insert
raises a foreign-key violation — swallowed by the writer as "a race with parallel
consolidation" and logged at warning level. The audit trail goes silently empty.

Dropping the constraint lets history be recorded wherever the observation is
stored. The cleanup the cascade used to do is now explicit, in the paths that
delete observations (``_execute_delete_action``, ``clear_observations``,
``delete_bank``). Rows orphaned by a path that misses — a document delete
cascading through ``memory_units``, for instance — are invisible to readers,
which always filter by ``(bank_id, observation_id)``, and are reclaimed when the
bank is deleted.

Revision ID: a1c9e7f3b2d8
Revises: d7b2f8a1c934
"""

from collections.abc import Sequence

from alembic import op

revision: str = "a1c9e7f3b2d8"
down_revision: str | Sequence[str] | None = "d7b2f8a1c934"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CONSTRAINT = "observation_history_observation_id_fkey"


def upgrade() -> None:
    # Oracle builds this schema through its own DDL runner and never had the
    # constraint; guard so the migration is a no-op there rather than an error.
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute(f"ALTER TABLE observation_history DROP CONSTRAINT IF EXISTS {_CONSTRAINT}")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    # Re-adding the FK requires every row to reference a live memory_unit, so
    # clear any history whose observation is not a Postgres row first — those are
    # exactly the rows this migration made possible.
    op.execute(
        "DELETE FROM observation_history h "
        "WHERE NOT EXISTS (SELECT 1 FROM memory_units m WHERE m.id = h.observation_id)"
    )
    op.execute(
        f"ALTER TABLE observation_history ADD CONSTRAINT {_CONSTRAINT} "
        "FOREIGN KEY (observation_id) REFERENCES memory_units(id) ON DELETE CASCADE"
    )
