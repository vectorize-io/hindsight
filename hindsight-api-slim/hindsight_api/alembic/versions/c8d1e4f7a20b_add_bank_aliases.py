"""Add bank_aliases: extra ids a bank also answers to.

``bank_id`` is a TEXT primary key and the foreign key in every bank-scoped
table, so changing it means rewriting every row (see ``hindsight-admin
rename-bank``) and stopping the bank's clients first — anything still calling
the old id gets a 404, or silently auto-creates a fresh empty bank. An alias
sidesteps that: the bank keeps its id and its rows, and additionally answers to
the new one, so callers migrate in phases with the API running.

Many aliases per bank, deliberately: a phased migration may run several ids at
once. There is no unique constraint on ``bank_id`` for that reason.

The alias is only a routing entry. Nothing is keyed on it, and resolution
rewrites it to the canonical ``bank_id`` before any query runs, so no row
anywhere is ever stored under an alias.

``alias`` is the primary key, which is what stops two banks claiming the same
one — the second insert raises a unique violation rather than racing. It cannot
also stop an alias colliding with a real ``bank_id``, since that lives in
another table; the write paths check each other for that (see
``bank_aliases.create_alias`` and ``bank_utils.create_bank_row_on_conn``).

The ``bank_id`` column name is load-bearing beyond the FK: ``rename-bank``
enumerates the tables to move by looking for it in the catalog, so a renamed
bank keeps its aliases with no change there.

Revision ID: c8d1e4f7a20b
Revises: b8d3f1a6c2e4
Create Date: 2026-09-23
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "c8d1e4f7a20b"
down_revision: str | Sequence[str] | None = "b8d3f1a6c2e4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}bank_aliases (
            alias      TEXT NOT NULL,
            bank_id    TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (alias),
            FOREIGN KEY (bank_id) REFERENCES {schema}banks(bank_id) ON DELETE CASCADE
        )
        """
    )
    # Listing or dropping a bank's aliases is the only non-alias lookup, and the
    # PK does not serve it.
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_bank_aliases_bank ON {schema}bank_aliases (bank_id)")


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_bank_aliases_bank")
    op.execute(f"DROP TABLE IF EXISTS {schema}bank_aliases")


def _oracle_execute_ignoring_955(sql: str) -> None:
    """Run a CREATE statement and swallow ORA-00955 (object already exists)."""
    block = (
        "BEGIN "
        "EXECUTE IMMEDIATE :stmt; "
        "EXCEPTION WHEN OTHERS THEN "
        "IF SQLCODE = -955 THEN NULL; ELSE RAISE; END IF; "
        "END;"
    )
    op.get_bind().exec_driver_sql(block, {"stmt": sql.strip()})


def _oracle_upgrade() -> None:
    # VARCHAR2(256) mirrors the bank_id columns elsewhere in the Oracle schema;
    # BANK_ID_MAX_BYTES (192) bounds what either column can actually hold.
    _oracle_execute_ignoring_955(
        """
        CREATE TABLE bank_aliases (
            alias      VARCHAR2(256) NOT NULL,
            bank_id    VARCHAR2(256) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT pk_bank_aliases PRIMARY KEY (alias),
            CONSTRAINT fk_bank_aliases_bank FOREIGN KEY (bank_id)
                REFERENCES banks(bank_id) ON DELETE CASCADE
        )
        """
    )
    _oracle_execute_ignoring_955("CREATE INDEX idx_bank_aliases_bank ON bank_aliases (bank_id)")


def _oracle_downgrade() -> None:
    op.execute("DROP INDEX idx_bank_aliases_bank")
    op.execute("DROP TABLE bank_aliases")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
