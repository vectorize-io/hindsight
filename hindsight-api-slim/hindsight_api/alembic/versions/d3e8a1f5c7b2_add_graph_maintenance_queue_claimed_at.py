"""Add claimed_at to graph_maintenance_queue

A relink worker used to claim queue rows by deleting them and keep those row
locks for the whole relink, so writers re-queuing an overlapping unit waited on
it (#6417). Now the claim stamps ``claimed_at`` in a short transaction and the
relink deletes the rows at its end, only where the stamp is still its own. A
re-queue clears the stamp, so the change it signals survives the delete.

Revision ID: d3e8a1f5c7b2
Revises: d3f8a1c6e9b2
Create Date: 2026-11-09
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "d3e8a1f5c7b2"
down_revision: str | Sequence[str] | None = "d3f8a1c6e9b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {schema}graph_maintenance_queue ADD COLUMN IF NOT EXISTS claimed_at TIMESTAMPTZ")


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {schema}graph_maintenance_queue DROP COLUMN IF EXISTS claimed_at")


def _oracle_upgrade() -> None:
    # ORA-01430 (column already exists) keeps a rerun after a partial upgrade safe.
    op.get_bind().exec_driver_sql(
        "BEGIN "
        "EXECUTE IMMEDIATE 'ALTER TABLE graph_maintenance_queue ADD (claimed_at TIMESTAMP WITH TIME ZONE)'; "
        "EXCEPTION WHEN OTHERS THEN IF SQLCODE = -1430 THEN NULL; ELSE RAISE; END IF; "
        "END;"
    )


def _oracle_downgrade() -> None:
    op.execute("ALTER TABLE graph_maintenance_queue DROP COLUMN claimed_at")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
