"""Repair: install maintenance routines on the ``public`` / base-schema run.

The original maintenance-routines migration (``e5f6a7b8c9d0``) only created the
shared ``public.banks_needing_consolidation()`` and
``public.schemas_with_expired_rows(...)`` routines when the run had *no*
``target_schema`` at all. But the single-tenant runtime always migrates an
explicit schema — which defaults to ``public`` — so on every default
PostgreSQL deployment the migration was stamped as applied while the functions
were never created. Background maintenance then logs::

    Retention sweep failed for llm_requests: function public.schemas_with_expired_rows(...) does not exist
    Consolidation reconcile discovery failed: function public.banks_needing_consolidation() does not exist

See https://github.com/vectorize-io/hindsight/issues/2056.

Because ``e5f6a7b8c9d0`` is already stamped on affected ``0.8.0`` databases,
editing it would not re-run it there. This forward migration re-installs the
functions idempotently (``CREATE OR REPLACE``) on the run that targets the
shared ``public`` schema (base run with no ``target_schema``, or an explicit
``target_schema=public``), self-healing already-upgraded deployments and
covering fresh upgrades from earlier versions.

Per-tenant runs against a non-``public`` schema still skip it: re-issuing
``CREATE OR REPLACE FUNCTION public....`` from each concurrent tenant migration
aborts with ``tuple concurrently updated`` on the ``pg_proc`` catalog row, and
the base/public run has already created the functions for every tenant to use.
Runs that target ``public`` are serialized by the per-schema migration advisory
lock, so only one wins the create.

PostgreSQL only — the worker poller and these tables are not wired for Oracle,
so the Oracle slot is intentionally absent (mirrors ``e5f6a7b8c9d0``).

Revision ID: b2d4f6a8c1e3
Revises: e5f6a7b8c9d0
Create Date: 2026-06-08
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "b2d4f6a8c1e3"
down_revision: str | Sequence[str] | None = "e5f6a7b8c9d0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _should_install_public_routines(target_schema: str | None) -> bool:
    """True for the run that must (re)create the shared ``public.*`` routines.

    The routines physically live in ``public`` (hard-coded ``public.`` qualifier
    in the SQL below), so they must be installed exactly once — on the base run
    (no ``target_schema``) or on the run that explicitly targets ``public``. A
    run against any other tenant schema skips it to avoid concurrent
    ``CREATE OR REPLACE`` on the same ``pg_proc`` row.
    """
    return not target_schema or target_schema == "public"


def _pg_upgrade() -> None:
    if not _should_install_public_routines(context.config.get_main_option("target_schema")):
        return
    # Banks with eligible-but-unscheduled facts and no in-flight consolidation.
    # Auto-consolidation is filtered here only at the bank level (cheap prune);
    # the full hierarchical resolution (global -> tenant -> bank, plus
    # enable_observations) is done by the caller for the small returned set.
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
            END LOOP;
        END;
        $fn$;
        """
    )

    # Schemas holding at least one row of p_table older than p_days. p_ts_col is
    # the timestamp column to compare. Returns nothing when p_days <= 0
    # (retention disabled).
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
                EXECUTE format(
                    'SELECT EXISTS (SELECT 1 FROM %I.%I WHERE %I < NOW() - make_interval(days => $1))',
                    sch, p_table, p_ts_col
                ) INTO has_expired USING p_days;
                IF has_expired THEN
                    RETURN NEXT sch;
                END IF;
            END LOOP;
        END;
        $fn$;
        """
    )


def _pg_downgrade() -> None:
    # No-op: ``e5f6a7b8c9d0`` owns the lifecycle of these functions and drops
    # them on its own downgrade. This migration only ever (re)creates them, so
    # there is nothing to undo without racing that migration's DROP.
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
