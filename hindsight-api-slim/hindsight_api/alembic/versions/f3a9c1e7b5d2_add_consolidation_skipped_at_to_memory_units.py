"""Add consolidation_skipped_at to memory_units and its curation archive

Revision ID: f3a9c1e7b5d2
Revises: e5b1c7d3a902
Create Date: 2026-10-01

Consolidation stamps ``consolidated_at`` on every fact it hands to the LLM, including the
ones the model returned no action for. That stamp is what takes a fact out of the pending
set, so a declined fact looked exactly like one folded into an observation: excluded from
rebuild for good, cited by no observation, and nothing recorded that it happened (#5054).

``consolidation_skipped_at`` is written beside the stamp, in the same transaction, for the
facts that ended consolidation without an observation. It changes nothing about what is
pending; it only makes those facts distinguishable and countable.

The column goes on ``invalidated_memory_units`` too: invalidating and reverting a fact copy
every ``memory_units`` column into and out of the archive by name, so an archive without it
would fail the INSERT.
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "f3a9c1e7b5d2"
down_revision: str | Sequence[str] | None = "e5b1c7d3a902"
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
        op.execute(
            f"ALTER TABLE {schema}{table} ADD COLUMN IF NOT EXISTS consolidation_skipped_at TIMESTAMP WITH TIME ZONE"
        )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    for table in _TABLES:
        op.execute(f"ALTER TABLE {schema}{table} DROP COLUMN IF EXISTS consolidation_skipped_at")


def _oracle_upgrade() -> None:
    # Oracle commits each DDL statement on its own, so a failure on the second table leaves the
    # first one altered and a plain re-run would fail on it. Swallow ORA-01430 (column already
    # exists) to keep the migration re-runnable, as c7d1e9a4b3f2 does.
    for table in _TABLES:
        op.execute(
            f"""
            BEGIN
                EXECUTE IMMEDIATE 'ALTER TABLE {table} ADD (consolidation_skipped_at TIMESTAMP WITH TIME ZONE)';
            EXCEPTION WHEN OTHERS THEN
                IF SQLCODE != -1430 THEN RAISE; END IF;
            END;
            """
        )


def _oracle_downgrade() -> None:
    # Swallow ORA-00904 (column does not exist) for the same reason.
    for table in _TABLES:
        op.execute(
            f"""
            BEGIN
                EXECUTE IMMEDIATE 'ALTER TABLE {table} DROP COLUMN consolidation_skipped_at';
            EXCEPTION WHEN OTHERS THEN
                IF SQLCODE != -904 THEN RAISE; END IF;
            END;
            """
        )


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
