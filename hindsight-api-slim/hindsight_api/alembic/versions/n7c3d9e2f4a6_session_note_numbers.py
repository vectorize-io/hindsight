"""Number session notes per bank, so a note the agent was shown can be found again.

A note was known by its ``seq``, its position in its own session's log, so "#123" named a note in
every session of a bank. ``note_number`` numbers a bank's notes in the order they are written: the
agent is shown that number, credits it in its reply, and the control plane can find the session
from it. Only note rows carry one; the unique index is the guard against two sessions numbering at
once. Notes written before this keep no number and are still known by their ``seq``.

Revision ID: n7c3d9e2f4a6
Revises: f3a9c2d7e4b1
Create Date: 2026-11-03
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "n7c3d9e2f4a6"
down_revision: str | Sequence[str] | None = "f3a9c2d7e4b1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {schema}session_events ADD COLUMN IF NOT EXISTS note_number INTEGER")
    op.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS uq_session_events_note_number "
        f"ON {schema}session_events (bank_id, note_number) WHERE note_number IS NOT NULL"
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}uq_session_events_note_number")
    op.execute(f"ALTER TABLE {schema}session_events DROP COLUMN IF EXISTS note_number")


def _oracle_upgrade() -> None:
    op.execute("ALTER TABLE session_events ADD (note_number NUMBER(10))")
    # Oracle has no partial index: an entry whose every key is NULL is left out, so the bank is
    # keyed only on rows that have a number.
    op.execute(
        "CREATE UNIQUE INDEX uq_session_events_note_number ON session_events "
        "(CASE WHEN note_number IS NOT NULL THEN bank_id END, note_number)"
    )


def _oracle_downgrade() -> None:
    op.execute("DROP INDEX uq_session_events_note_number")
    op.execute("ALTER TABLE session_events DROP COLUMN note_number")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
