"""Add the ``banks_with_attachments_in_window`` cross-tenant discovery routine.

The daily attachment sweep (#6350) deletes the attachments of retains that
failed after the API stored them. It only has work in the banks that stored an
attachment in its window, which on a multi-tenant install is a handful of banks
among tens of thousands of schemas. Asking each schema from the application is a
round trip per tenant, so, like the sibling routines, discovery is one call that
returns just ``(schema_name, bank_id)`` pairs and the caller works only there.

A schema whose ``attachments`` table has never held a row is skipped on
``pg_relation_size`` — the table file's size, read without opening the table —
so the scan pays a query only where there is something to find. Planner stats
(``reltuples`` / ``relpages``) would be cheaper still but lag a table that holds
a few rows, which is the common case here.

Install policy, lock guard and skip arms mirror ``d7b2f8a1c934`` and
``c8b4e2a71f95``: one copy in the configured schema, called through
``fq_routine``; each per-schema query gets a short ``lock_timeout`` and skips a
schema that vanished or is mid-DDL instead of aborting the scan.

Revision ID: 05867433a58c
Revises: d3e8a1f5c7b2
Create Date: 2026-11-10
"""

from collections.abc import Sequence

from alembic import context

from hindsight_api.alembic._dialect import run_for_dialect
from hindsight_api.alembic._owned import execute_unless_owned
from hindsight_api.config import get_config

revision: str = "05867433a58c"
down_revision: str | Sequence[str] | None = "d3e8a1f5c7b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_LOCK_TIMEOUT = "250ms"


def _configured_schema() -> str:
    """The one schema this deployment's routines live in and are called from."""
    return get_config().database_schema or "public"


def _target_schema() -> str | None:
    return context.config.get_main_option("target_schema")


def _is_install_run() -> bool:
    """True for the single run that owns the routine (mirrors b6d2f8a4c1e7)."""
    target = _target_schema()
    return not target or target == _configured_schema()


def _prefix(schema: str | None) -> str:
    """Qualifier for ``schema``, or ``""`` to fall back to ``search_path``."""
    return f'"{schema}".' if schema else ""


def _drop_routine(schema: str | None) -> None:
    execute_unless_owned(
        "banks_with_attachments_in_window",
        f"DROP FUNCTION IF EXISTS {_prefix(schema)}banks_with_attachments_in_window(timestamptz, timestamptz)",
    )


def _pg_upgrade() -> None:
    if not _is_install_run():
        # Database-global routine: tenant schemas carry no copy.
        return
    schema = _prefix(_target_schema())
    execute_unless_owned(
        "banks_with_attachments_in_window",
        f"""
        CREATE OR REPLACE FUNCTION {schema}banks_with_attachments_in_window(
            p_after timestamptz, p_before timestamptz
        )
        RETURNS TABLE(schema_name text, bank_id text)
        LANGUAGE plpgsql STABLE
        AS $fn$
        DECLARE
            sch text;
            prev_lock_timeout text;
        BEGIN
            prev_lock_timeout := current_setting('lock_timeout');
            PERFORM set_config('lock_timeout', '{_LOCK_TIMEOUT}', true);
            FOR sch IN
                SELECT n.nspname
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = 'attachments' AND c.relkind = 'r'
                  -- A table that never held a row has no pages: skip it unopened.
                  AND pg_relation_size(c.oid) > 0
            LOOP
                BEGIN
                    RETURN QUERY EXECUTE format($q$
                        SELECT DISTINCT %1$L::text, a.bank_id
                        FROM %1$I.attachments a
                        WHERE a.created_at >= $1 AND a.created_at < $2
                    $q$, sch) USING p_after, p_before;
                EXCEPTION
                    -- Schema or its table vanished between the pg_class
                    -- snapshot and this query (tenant dropped or migrating).
                    WHEN undefined_table OR invalid_schema_name OR undefined_column THEN
                        CONTINUE;
                    -- Schema is mid-DDL: skip it rather than wait (c8b4e2a71f95).
                    WHEN lock_not_available OR deadlock_detected THEN
                        CONTINUE;
                END;
            END LOOP;
            PERFORM set_config('lock_timeout', prev_lock_timeout, true);
        END;
        $fn$;
        """,
    )


def _pg_downgrade() -> None:
    # This migration is the sole creator of the routine.
    if not _is_install_run():
        return
    _drop_routine(_target_schema())


def upgrade() -> None:
    # Oracle slot intentionally absent: the maintenance loop, and every routine
    # it calls, is PostgreSQL-only.
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
