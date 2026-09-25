"""Let one of a bank's aliases be the id shown in its place.

A bank's ``bank_id`` is its key forever, so after a phased migration the control
plane still shows the id nobody uses any more. Flagging one alias primary lets
the UI present the new id while every row, grant and audit entry keeps naming the
real one.

Optional by design: the bank's own id is not a row in ``bank_aliases``, so "no
primary" is the normal state and the display falls back to ``bank_id``. There is
no implicit primary, and a bank never has to have one.

The flag lives here rather than as ``banks.display_alias`` so it cannot outlive
what it names: this row already goes when the alias is deleted or the bank is
dropped, whereas a column on ``banks`` would keep pointing at an id that stopped
resolving, and would need its own foreign key and cleanup to avoid it.

"At most one per bank" is an index, not application logic — two concurrent
promotions would otherwise both read "no primary yet" and both write one.

Revision ID: d4f8b1c6e903
Revises: c8d1e4f7a20b
Create Date: 2026-09-24
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "d4f8b1c6e903"
down_revision: str | Sequence[str] | None = "c8d1e4f7a20b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {schema}bank_aliases ADD COLUMN IF NOT EXISTS is_primary BOOLEAN NOT NULL DEFAULT FALSE")
    # Partial: only the primaries are indexed, so the many non-primary rows cost
    # nothing and the uniqueness applies exactly where it is meant to.
    op.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS idx_bank_aliases_one_primary "
        f"ON {schema}bank_aliases (bank_id) WHERE is_primary"
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_bank_aliases_one_primary")
    op.execute(f"ALTER TABLE {schema}bank_aliases DROP COLUMN IF EXISTS is_primary")


def _oracle_execute_ignoring(sql: str, *codes: int) -> None:
    """Run a DDL statement and swallow the given ORA- codes, so a re-run is safe."""
    checks = " OR ".join(f"SQLCODE = {code}" for code in codes)
    block = f"BEGIN EXECUTE IMMEDIATE :stmt; EXCEPTION WHEN OTHERS THEN IF {checks} THEN NULL; ELSE RAISE; END IF; END;"
    op.get_bind().exec_driver_sql(block, {"stmt": sql.strip()})


def _oracle_upgrade() -> None:
    # ORA-01430: column already exists (re-run of a partial migration).
    _oracle_execute_ignoring(
        "ALTER TABLE bank_aliases ADD is_primary NUMBER(1) DEFAULT 0 NOT NULL",
        -1430,
    )
    # Oracle has no partial index. A function-based unique index is the
    # equivalent: rows that are not primary yield NULL, and Oracle does not
    # index an all-NULL key, so only the primaries take part in the constraint.
    _oracle_execute_ignoring(
        "CREATE UNIQUE INDEX idx_bank_aliases_one_primary ON bank_aliases (CASE WHEN is_primary = 1 THEN bank_id END)",
        -955,
    )


def _oracle_downgrade() -> None:
    op.execute("DROP INDEX idx_bank_aliases_one_primary")
    op.execute("ALTER TABLE bank_aliases DROP COLUMN is_primary")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
