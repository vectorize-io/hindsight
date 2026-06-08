"""Add server-side routines for background maintenance sweeps.

Installs two PL/pgSQL discovery routines in the ``public`` schema. Both loop
over every schema that actually holds the relevant table (via ``pg_class``), so
a single function call covers all tenants in one round-trip instead of the
per-tenant query storm that a client-side loop would create at thousands of
tenants.

- ``public.banks_needing_consolidation()`` -> (schema_name, bank_id) for banks
  that have eligible-but-unscheduled facts (``consolidated_at IS NULL AND
  consolidation_failed_at IS NULL`` for consolidatable fact types), have
  auto-consolidation not explicitly disabled at the bank level, and have no
  consolidation operation already pending/processing. Drives the periodic
  reconcile that re-schedules consolidation after a terminal failure left facts
  stranded (see HINDSIGHT_API_CONSOLIDATION_RECONCILE_INTERVAL_SECONDS).

- ``public.schemas_with_expired_rows(p_table, p_ts_col, p_days)`` -> schema
  names that hold at least one ``p_table`` row older than ``p_days``. Drives the
  cross-tenant retention sweeps for ``audit_log`` and ``llm_requests``; the loop
  then issues a DELETE only against the returned schemas.

These are read-only (STABLE) discovery routines — the caller performs the
enqueue/DELETE — so installing them never mutates data.

PostgreSQL only — the worker poller and these tables are not wired for Oracle,
so the Oracle slot is intentionally absent (mirrors the audit_log / llm_requests
table migrations). The routines live in ``public`` and are CREATE OR REPLACE, so
running this migration once per tenant schema is idempotent.

Revision ID: e5f6a7b8c9d0
Revises: a7b8c9d0e1f2
Create Date: 2026-06-05
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "e5f6a7b8c9d0"
down_revision: str | Sequence[str] | None = "a7b8c9d0e1f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _is_base_schema_run() -> bool:
    """True only for the base-schema migration (no per-tenant target_schema).

    These routines live in the shared ``public`` schema, so they must be created
    exactly once. Running ``CREATE OR REPLACE FUNCTION public....`` again from each
    concurrent per-tenant migration aborts with ``tuple concurrently updated`` on
    the ``pg_proc`` catalog row, so tenant runs skip it (the base run already
    created the function for every tenant to use).
    """
    return not context.config.get_main_option("target_schema")


def _pg_upgrade() -> None:
    if not _is_base_schema_run():
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
    if not _is_base_schema_run():
        return
    op.execute("DROP FUNCTION IF EXISTS public.banks_needing_consolidation()")
    op.execute("DROP FUNCTION IF EXISTS public.schemas_with_expired_rows(text, text, int)")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
