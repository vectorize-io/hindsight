"""Add server-side routine for cron-scheduled mental model refresh.

Installs ``public.mental_models_with_cron()`` — a discovery routine that returns
every mental model carrying a non-empty ``trigger->>'refresh_cron'`` across all
tenant schemas in one round-trip (the same per-schema scan as the other
maintenance routines from ``e5f6a7b8c9d0``). The maintenance loop evaluates each
candidate's cron expression in Python (``croniter``) against ``last_refreshed_at``
to decide whether a scheduled refresh is due — cron arithmetic isn't expressible
in plain SQL — and only the cron *candidate set* is discovered here.

Models that already have a ``refresh_mental_model`` operation pending/processing
are excluded so a slow refresh isn't double-queued (mirrors the in-flight guard
in ``banks_needing_consolidation``). Each per-schema query runs in its own
``BEGIN ... EXCEPTION`` subtransaction so a schema dropped mid-scan (tenant
deletion / migration) is skipped, not fatal — same resilience as
``c7e9f1a3b5d2``.

Read-only (STABLE) discovery routine — the caller performs the refresh enqueue —
so installing it never mutates data. PostgreSQL only: the worker poller and the
maintenance loop are PG-only (Oracle slot intentionally absent, mirroring
``e5f6a7b8c9d0``). The routine lives in ``public`` and is CREATE OR REPLACE, so
it is installed exactly once (base / ``public`` run) to avoid the
``tuple concurrently updated`` race on concurrent per-tenant runs.

Revision ID: f4d1c2b3a5e6
Revises: c7e9f1a3b5d2
Create Date: 2026-06-23
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "f4d1c2b3a5e6"
down_revision: str | Sequence[str] | None = "c7e9f1a3b5d2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _should_install_public_routines(target_schema: str | None) -> bool:
    """True for the run that must (re)create the shared ``public.*`` routine.

    The routine physically lives in ``public``, so it is installed exactly once —
    on the base run (no ``target_schema``) or the run that explicitly targets
    ``public``. Mirrors ``c7e9f1a3b5d2``.
    """
    return not target_schema or target_schema == "public"


def _pg_upgrade() -> None:
    if not _should_install_public_routines(context.config.get_main_option("target_schema")):
        return

    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.mental_models_with_cron()
        RETURNS TABLE(schema_name text, bank_id text, mental_model_id text,
                     refresh_cron text, last_refreshed_at timestamptz)
        LANGUAGE plpgsql STABLE
        AS $fn$
        DECLARE
            sch text;
        BEGIN
            FOR sch IN
                SELECT n.nspname
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = 'mental_models' AND c.relkind = 'r'
            LOOP
                BEGIN
                    RETURN QUERY EXECUTE format($q$
                        SELECT %1$L::text, mm.bank_id::text, mm.id::text,
                               mm.trigger->>'refresh_cron', mm.last_refreshed_at
                        FROM %1$I.mental_models mm
                        WHERE COALESCE(mm.trigger->>'refresh_cron', '') <> ''
                          AND NOT EXISTS (
                              SELECT 1 FROM %1$I.async_operations o
                              WHERE o.bank_id = mm.bank_id
                                AND o.operation_type = 'refresh_mental_model'
                                AND o.status IN ('pending', 'processing')
                                AND o.task_payload->>'mental_model_id' = mm.id::text
                          )
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


def _pg_downgrade() -> None:
    if not _should_install_public_routines(context.config.get_main_option("target_schema")):
        return
    op.execute("DROP FUNCTION IF EXISTS public.mental_models_with_cron()")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
