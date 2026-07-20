"""Add server_settings table

Instance-wide key/value settings persisted in the database so a self-hosted
operator can configure things at runtime from the control plane (e.g. the LLM
provider + credentials) instead of only via environment variables. Values are
JSON so a single row can carry a structured config blob; the table is a small
singleton-style store keyed by ``setting_key``.

This is deliberately dialect-agnostic (a KV/JSON table works on both backends),
so both the PG and Oracle slots are provided.

Revision ID: e2f4a6c8b1d5
Revises: d7b2f8a1c934
Create Date: 2026-07-20
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "e2f4a6c8b1d5"
down_revision: str | Sequence[str] | None = "d7b2f8a1c934"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}server_settings (
            setting_key TEXT PRIMARY KEY,
            value       JSONB NOT NULL,
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"DROP TABLE IF EXISTS {schema}server_settings")


def _oracle_execute_ignoring_955(sql: str) -> None:
    """Run a CREATE statement and swallow ORA-00955 (object already exists).

    Mirrors the helper in the Oracle baseline migration so reruns stay safe on a
    database where the table was created by an earlier partial run.
    """
    block = (
        "BEGIN "
        "EXECUTE IMMEDIATE :stmt; "
        "EXCEPTION WHEN OTHERS THEN "
        "IF SQLCODE = -955 THEN NULL; ELSE RAISE; END IF; "
        "END;"
    )
    op.get_bind().exec_driver_sql(block, {"stmt": sql.strip()})


def _oracle_upgrade() -> None:
    _oracle_execute_ignoring_955(
        """
        CREATE TABLE server_settings (
            setting_key VARCHAR2(128) NOT NULL,
            value       CLOB NOT NULL CONSTRAINT server_settings_value_json CHECK (value IS JSON),
            updated_at  TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT pk_server_settings PRIMARY KEY (setting_key)
        )
        """
    )


def _oracle_downgrade() -> None:
    op.execute("DROP TABLE server_settings")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
