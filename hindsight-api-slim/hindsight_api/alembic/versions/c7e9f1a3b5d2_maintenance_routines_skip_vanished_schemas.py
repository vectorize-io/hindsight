"""Make maintenance routines resilient to schemas that vanish mid-scan.

``public.banks_needing_consolidation()`` and
``public.schemas_with_expired_rows(...)`` snapshot the set of schemas owning a
target table from ``pg_class`` and then run a dynamic query against each schema
in turn. That is a time-of-check/time-of-use race: a schema (or its tables) can
be dropped — a tenant being deleted, or a tenant migration that recreates
tables — between the snapshot and the per-schema query, which then aborts the
whole routine with::

    relation "<schema>.memory_units" does not exist
    relation "<schema>.audit_log" does not exist

In the test suite this surfaces as cross-worker contamination: the multi-tenant
maintenance test creates and drops ~100 ``mt<hash>_NNN`` schemas while
``test_maintenance_routines`` (on another xdist worker, same DB) calls the
routines. In production the background maintenance loop hits the same race when
a tenant is removed or mid-migration.

Wrap each per-schema query in its own ``BEGIN ... EXCEPTION`` block so a schema
that disappears (``undefined_table`` / ``invalid_schema_name`` /
``undefined_column``) is skipped instead of aborting the scan. The routines stay
``CREATE OR REPLACE`` and PostgreSQL-only, and are (re)installed only on the run
that targets the shared ``public`` schema — same gating as the original
install (``e5f6a7b8c9d0``) and its repair (``b2d4f6a8c1e3``).

Revision ID: c7e9f1a3b5d2
Revises: e1f2a3b4c5d6
Create Date: 2026-06-19
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "c7e9f1a3b5d2"
down_revision: str | Sequence[str] | None = "e1f2a3b4c5d6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _should_install_public_routines(target_schema: str | None) -> bool:
    """True for the run that must (re)create the shared ``public.*`` routines.

    The routines physically live in ``public``, so they are installed exactly
    once — on the base run (no ``target_schema``) or the run that explicitly
    targets ``public``. Mirrors ``b2d4f6a8c1e3``.
    """
    return not target_schema or target_schema == "public"


def _pg_upgrade() -> None:
    if not _should_install_public_routines(context.config.get_main_option("target_schema")):
        return

    # Same body as b2d4f6a8c1e3, but each per-schema query runs in its own
    # subtransaction so a schema dropped mid-scan is skipped, not fatal.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.banks_needing_consolidation()
        RETURNS TABLE(schema_name text, bank_id text)
        LANGUAGE plpgsql STABLE
        AS $fn$
        DECLARE
            sch text;
        BEGIN
            FOR sch IN
                SELECT n.nspname
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = 'memory_units' AND c.relkind = 'r'
            LOOP
                BEGIN
                    RETURN QUERY EXECUTE format($q$
                        SELECT %1$L::text, m.bank_id
                        FROM %1$I.memory_units m
                        JOIN %1$I.banks b ON b.bank_id = m.bank_id
                        WHERE m.consolidated_at IS NULL
                          AND m.consolidation_failed_at IS NULL
                          AND m.fact_type IN ('experience', 'world')
                          AND COALESCE(b.config -> 'enable_auto_consolidation', 'true'::jsonb) <> 'false'::jsonb
                          AND NOT EXISTS (
                              SELECT 1 FROM %1$I.async_operations o
                              WHERE o.bank_id = m.bank_id
                                AND o.operation_type = 'consolidation'
                                AND o.status IN ('pending', 'processing')
                          )
                        GROUP BY m.bank_id
                    $q$, sch);
                EXCEPTION
                    -- Schema or its tables vanished between the pg_class
                    -- snapshot and this query (tenant dropped or migrating).
                    WHEN undefined_table OR invalid_schema_name OR undefined_column THEN
                        CONTINUE;
                END;
            END LOOP;
        END;
        $fn$;
        """
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.schemas_with_expired_rows(
            p_table text, p_ts_col text, p_days int
        )
        RETURNS SETOF text
        LANGUAGE plpgsql STABLE
        AS $fn$
        DECLARE
            sch text;
            has_expired boolean;
        BEGIN
            IF p_days IS NULL OR p_days <= 0 THEN
                RETURN;
            END IF;
            FOR sch IN
                SELECT n.nspname
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = p_table AND c.relkind = 'r'
            LOOP
                BEGIN
                    EXECUTE format(
                        'SELECT EXISTS (SELECT 1 FROM %I.%I WHERE %I < NOW() - make_interval(days => $1))',
                        sch, p_table, p_ts_col
                    ) INTO has_expired USING p_days;
                EXCEPTION
                    -- Schema or its table vanished mid-scan; skip it.
                    WHEN undefined_table OR invalid_schema_name OR undefined_column THEN
                        CONTINUE;
                END;
                IF has_expired THEN
                    RETURN NEXT sch;
                END IF;
            END LOOP;
        END;
        $fn$;
        """
    )


def _pg_downgrade() -> None:
    # No-op: e5f6a7b8c9d0 owns these functions' lifecycle and drops them on its
    # own downgrade. This migration only re-installs them (the resilient body is
    # a strict superset of the previous behaviour), so there is nothing to undo
    # without racing that migration's DROP.
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
