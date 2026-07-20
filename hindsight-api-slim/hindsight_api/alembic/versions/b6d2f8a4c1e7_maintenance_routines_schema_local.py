"""Install the maintenance discovery routines into each run's own schema.

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

**Fix: stop putting them in a shared schema.** Install all three into *this
run's own schema*, unconditionally. What the routines return does not depend on
where they live — each enumerates ``pg_class`` across the whole database and
dispatches per schema — so a copy in any one schema is fully functional, and the
maintenance loop calls the copy in its configured schema
(``get_config().database_schema``), which is by definition a schema that got
migrated.

This also removes the concurrency hazard the old gate existed to dodge, rather
than locking around it. Each migration process only ever writes
``CREATE OR REPLACE FUNCTION "<its own target_schema>".fn()``, so two concurrent
per-schema runs never touch the same ``pg_proc`` row and the
``tuple concurrently updated`` race cannot occur. No cross-process coordination
is needed — notably no advisory lock, which is unusable here because Hindsight
runs behind connection poolers and managed PG services (see the revert of
#2690).

Existing installs self-heal: this revision runs on every schema and creates the
routine exactly where that deployment's loop looks for it. The base/``public``
run keeps creating the ``public.*`` copies, so nothing changes for a default
deployment. Function bodies are byte-identical to ``c7e9f1a3b5d2`` /
``f4d1c2b3a5e6``.

The cost is one duplicate routine per tenant schema in a multi-tenant install —
a few catalog rows each, and the honest price of needing no coordination.

PostgreSQL only: the maintenance loop and worker poller are PG-only, so the
Oracle slot is intentionally absent (mirrors ``e5f6a7b8c9d0``).

Revision ID: b6d2f8a4c1e7
Revises: a8c1e4f7b0d3
Create Date: 2026-07-20
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "b6d2f8a4c1e7"
down_revision: str | Sequence[str] | None = "a8c1e4f7b0d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _schema_prefix() -> str:
    """Qualifier for the schema this run targets, or ``""`` for the base run.

    An empty prefix leaves the object to ``search_path``, which env.py points at
    the target schema (base run: ``public``) — matching every other raw-SQL
    migration in this tree.
    """
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _schema_prefix()

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


def _pg_downgrade() -> None:
    # Drop only the copies this migration uniquely owns — the ones in a
    # non-``public`` schema. The ``public`` copies belong to e5f6a7b8c9d0 /
    # f4d1c2b3a5e6, which are still applied at this point and drop them on their
    # own downgrade; removing them here would strand those migrations without the
    # functions they claim to have installed.
    target_schema = context.config.get_main_option("target_schema")
    if not target_schema or target_schema == "public":
        return
    schema = f'"{target_schema}".'
    op.execute(f"DROP FUNCTION IF EXISTS {schema}mental_models_with_cron()")
    op.execute(f"DROP FUNCTION IF EXISTS {schema}schemas_with_expired_rows(text, text, int)")
    op.execute(f"DROP FUNCTION IF EXISTS {schema}banks_needing_consolidation()")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
