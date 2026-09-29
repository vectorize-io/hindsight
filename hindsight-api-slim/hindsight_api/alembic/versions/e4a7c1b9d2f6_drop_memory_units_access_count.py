"""Drop the never-written `access_count` column from memory_units (and its archive).

``memory_units.access_count`` has been dead since the initial schema
(5a366d414dce): no code path anywhere in the repo ever writes it, and — despite
the ``access_count DESC`` index created alongside it — no query ever reads or
orders by it either. It is 0 on every row of every install. The lone remaining
mentions were an index, a stale comment naming an ``access_count_update`` task
type that was never implemented, and the column's name in the Oracle backend's
numeric-RETURNING list; all three go away with this change.

The column is dropped from the curation archive too. ``invalidated_memory_units``
was cloned ``LIKE memory_units`` (c9a1b2d3e4f5), so it inherited the column, and
curation's INSERT…SELECT round-trip builds its column list from the catalog
(``writes.py::_memory_unit_columns``) — the two tables must stay in lockstep or
the round-trip breaks on a column-count mismatch.

Dropping the column implicitly drops its index on both dialects
(``idx_memory_units_access_count`` on PG, ``idx_mu_access_count`` on Oracle), so
PostgreSQL also stops maintaining a btree that nothing ever probed.

Cost: on PostgreSQL ``DROP COLUMN`` is metadata-only (the attribute is marked
dropped, no table rewrite). On Oracle it does delete the column data row by row,
so on a large ``memory_units`` this migration is not free — it is still bounded
work on a single small integer column, and Oracle installs of that size can run
it during a maintenance window ahead of the upgrade if they prefer.

Revision ID: e4a7c1b9d2f6
Revises: a9b8c7d6e5f4
Create Date: 2026-08-03
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "e4a7c1b9d2f6"
down_revision: str | Sequence[str] | None = "a9b8c7d6e5f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ("memory_units", "invalidated_memory_units")


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    for table in _TABLES:
        # Drops idx_memory_units_access_count along with the column.
        op.execute(f"ALTER TABLE {schema}{table} DROP COLUMN IF EXISTS access_count")


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    for table in _TABLES:
        op.execute(f"ALTER TABLE {schema}{table} ADD COLUMN IF NOT EXISTS access_count integer NOT NULL DEFAULT 0")
    # The archive was cloned without indexes; only the live table carried one.
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_memory_units_access_count ON {schema}memory_units (access_count DESC)")


def _oracle_upgrade() -> None:
    # Oracle has no `DROP COLUMN IF EXISTS`; swallow ORA-00904 (column does not
    # exist) so the migration is idempotent and safe on a schema that already
    # lacks the column. Dropping the column also drops idx_mu_access_count.
    for table in _TABLES:
        op.execute(
            f"""
            BEGIN
                EXECUTE IMMEDIATE 'ALTER TABLE {table} DROP COLUMN access_count';
            EXCEPTION WHEN OTHERS THEN
                IF SQLCODE != -904 THEN RAISE; END IF;
            END;
            """
        )


def _oracle_downgrade() -> None:
    # Swallow ORA-01430 (column already exists) for idempotency. Matches the
    # Oracle baseline's declaration: NUMBER(10) DEFAULT 0 NOT NULL.
    for table in _TABLES:
        op.execute(
            f"""
            BEGIN
                EXECUTE IMMEDIATE
                    'ALTER TABLE {table} ADD (access_count NUMBER(10) DEFAULT 0 NOT NULL)';
            EXCEPTION WHEN OTHERS THEN
                IF SQLCODE != -1430 THEN RAISE; END IF;
            END;
            """
        )
    # ORA-00955: index name already in use.
    op.execute(
        """
        BEGIN
            EXECUTE IMMEDIATE 'CREATE INDEX idx_mu_access_count ON memory_units(access_count DESC)';
        EXCEPTION WHEN OTHERS THEN
            IF SQLCODE != -955 THEN RAISE; END IF;
        END;
        """
    )


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
