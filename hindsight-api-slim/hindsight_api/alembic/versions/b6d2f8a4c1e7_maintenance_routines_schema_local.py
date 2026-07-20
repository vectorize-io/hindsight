"""Install the maintenance discovery routines into the configured schema.

The three discovery routines driving the background maintenance loop —
``banks_needing_consolidation()``, ``schemas_with_expired_rows(...)`` and
``mental_models_with_cron()`` — were installed into ``public`` and gated on the
run being the base run (no ``target_schema``) or an explicit
``target_schema='public'`` run (``e5f6a7b8c9d0`` → ``b2d4f6a8c1e3`` →
``c7e9f1a3b5d2``, ``f4d1c2b3a5e6``).

That leaves a **single-tenant deployment migrated into a dedicated, non-**
``public`` **schema** (``HINDSIGHT_API_DATABASE_SCHEMA=<non-public>``) with no
routines at all: the runtime migrates only that one schema, so ``target_schema``
is never falsy or ``public``, the gate never opens, and the maintenance loop
logs, forever::

    function public.banks_needing_consolidation() does not exist
    function public.schemas_with_expired_rows(...) does not exist

The revision is stamped applied, so redeploying the same version does not help
(issue #2638; #2056 only fixed the ``public``/base-run case).

**The bug was the hardcoded literal, not the gating.** These routines are
database-global — each enumerates ``pg_class`` across every schema and dispatches
per schema — so exactly one copy should exist, and the maintenance loop calls the
one in ``get_config().database_schema`` (see ``fq_routine``). The old gate
installed into whichever schema was named ``public`` instead of whichever schema
the deployment is actually configured to use. Comparing ``target_schema`` against
the configured schema instead of the literal fixes #2638 at the source.

That also keeps the property the gate existed for: exactly one migration run
satisfies the predicate, so concurrent per-schema runs never issue competing
``CREATE OR REPLACE`` against the same ``pg_proc`` row and cannot hit
``tuple concurrently updated``. No cross-process coordination is required — in
particular no advisory lock, which is unusable here because Hindsight runs behind
connection poolers and managed PG services (see #2817).

Runs targeting any *other* schema drop the routines from that schema rather than
merely skipping. An earlier revision of this migration installed a copy into
every schema it touched, which left one dead duplicate per tenant on any database
that ran it; the drop makes the next migration pass clean those up instead of
leaving them behind forever.

PostgreSQL only: the maintenance loop and worker poller are PG-only, so the
Oracle slot is intentionally absent (mirrors ``e5f6a7b8c9d0``).

Revision ID: b6d2f8a4c1e7
Revises: a8c1e4f7b0d3
Create Date: 2026-07-20
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect
from hindsight_api.config import get_config

revision: str = "b6d2f8a4c1e7"
down_revision: str | Sequence[str] | None = "a8c1e4f7b0d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _configured_schema() -> str:
    """The one schema this deployment's routines live in and are called from."""
    return get_config().database_schema or "public"


def _target_schema() -> str | None:
    return context.config.get_main_option("target_schema")


def _is_install_run() -> bool:
    """True for the single run that owns the routines.

    The base run (no ``target_schema``) and the run targeting the configured
    schema are the same deployment-level run; every other target is a tenant
    schema that must not carry its own copy.
    """
    target = _target_schema()
    return not target or target == _configured_schema()


def _prefix(schema: str | None) -> str:
    """Qualifier for ``schema``, or ``""`` to fall back to ``search_path``."""
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    if not _is_install_run():
        _drop_stray_copies()
        return
    schema = _prefix(_target_schema())

    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {schema}banks_needing_consolidation()
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
        f"""
        CREATE OR REPLACE FUNCTION {schema}schemas_with_expired_rows(
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

    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {schema}mental_models_with_cron()
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


def _drop_routines(schema: str | None) -> None:
    prefix = _prefix(schema)
    op.execute(f"DROP FUNCTION IF EXISTS {prefix}mental_models_with_cron()")
    op.execute(f"DROP FUNCTION IF EXISTS {prefix}schemas_with_expired_rows(text, text, int)")
    op.execute(f"DROP FUNCTION IF EXISTS {prefix}banks_needing_consolidation()")


def _drop_stray_copies() -> None:
    """Remove per-tenant duplicates left by the first cut of this migration.

    That version installed a copy into every schema it touched, so a database
    that ran it carries one dead duplicate per tenant — only the copy in the
    configured schema is ever called. Dropping here means the next migration pass
    cleans them up; without it they would persist for the life of the database.

    Safe on a database that never had them: ``DROP FUNCTION IF EXISTS`` is a
    no-op, and this branch never runs for the configured schema.
    """
    _drop_routines(_target_schema())


def _pg_downgrade() -> None:
    # Only drop what this migration uniquely owns. When the configured schema is
    # ``public`` the copies there belong to e5f6a7b8c9d0 / f4d1c2b3a5e6, which are
    # still applied at this point and drop them on their own downgrade — removing
    # them here would strand those migrations without the functions they claim to
    # have installed.
    if not _is_install_run() or _configured_schema() == "public":
        return
    _drop_routines(_target_schema())


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
