"""Add durable causal-link provenance to live and archived memories.

``memory_links`` remains the materialized graph used by retrieval. The new
JSON field carries the small causal descriptor set with both endpoint memories
so curation can restore an edge after both endpoints become live again.

Revision ID: f3a9c1e7b2d4
Revises: d7b2f8a1c934
Create Date: 2026-07-22
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "f3a9c1e7b2d4"
down_revision: str | Sequence[str] | None = "d7b2f8a1c934"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    for table in ("memory_units", "invalidated_memory_units"):
        op.execute(
            f"ALTER TABLE {schema}{table} "
            "ADD COLUMN IF NOT EXISTS suspended_causal_links JSONB NOT NULL DEFAULT '[]'::jsonb"
        )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    for table in ("invalidated_memory_units", "memory_units"):
        op.execute(f"ALTER TABLE {schema}{table} DROP COLUMN IF EXISTS suspended_causal_links")


def _oracle_add_column(table: str, constraint: str) -> None:
    op.execute(
        f"""
        BEGIN
            EXECUTE IMMEDIATE 'ALTER TABLE {table} ADD (
                suspended_causal_links CLOB DEFAULT ''[]'' NOT NULL
                CONSTRAINT {constraint} CHECK (suspended_causal_links IS JSON)
            )';
        EXCEPTION WHEN OTHERS THEN
            IF SQLCODE != -1430 THEN RAISE; END IF;
        END;
        """
    )


def _oracle_drop_column(table: str) -> None:
    op.execute(
        f"""
        BEGIN
            EXECUTE IMMEDIATE 'ALTER TABLE {table} DROP COLUMN suspended_causal_links';
        EXCEPTION WHEN OTHERS THEN
            IF SQLCODE != -904 THEN RAISE; END IF;
        END;
        """
    )


def _oracle_upgrade() -> None:
    _oracle_add_column("memory_units", "ck_mu_suspended_causal_json")
    _oracle_add_column("invalidated_memory_units", "ck_imu_suspended_causal_json")


def _oracle_downgrade() -> None:
    _oracle_drop_column("invalidated_memory_units")
    _oracle_drop_column("memory_units")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
