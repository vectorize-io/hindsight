"""Install the shared public.* maintenance routines on every PG run.

``public.banks_needing_consolidation()`` and
``public.schemas_with_expired_rows(...)`` physically live in ``public`` and are
consumed by the background maintenance loop through a hardcoded ``public.``
qualifier (``engine/maintenance.py``). Every prior install/repair
(``e5f6a7b8c9d0`` → ``b2d4f6a8c1e3`` → ``c7e9f1a3b5d2``) gated creation on
``_should_install_public_routines(target_schema)`` — i.e. the base run
(no ``target_schema``) or an explicit ``target_schema='public'`` run.

That leaves a gap for a **single-tenant deployment migrated into a dedicated,
non-``public`` schema** (``HINDSIGHT_API_DATABASE_SCHEMA=<non-public>``). The
runtime migrates only that one schema (``run_migrations_for_schemas`` fans out
per configured schema), so ``target_schema`` is never falsy or ``public`` and
the routines are never created. The maintenance loop then logs, forever::

    function public.banks_needing_consolidation() does not exist
    function public.schemas_with_expired_rows(...) does not exist

and the revision is stamped applied, so redeploying the same version does not
help (issue #2638; #2056 only fixed the ``public``/base-run case).

Fix: install the routines on **every** PG run regardless of ``target_schema``.
They are schema-agnostic (they enumerate ``pg_class`` and dispatch per schema),
so a non-``public`` run creating them in ``public`` is correct and idempotent
via ``CREATE OR REPLACE``. The reason the earlier migrations gated to a single
run was to avoid ``tuple concurrently updated`` when parallel per-schema
migration processes race on the same ``CREATE OR REPLACE``; we keep that safety
with a transaction-scoped advisory lock so exactly one concurrent run performs
the replace at a time. Function bodies are identical to ``c7e9f1a3b5d2`` (the
vanishing-schema-resilient versions).

Revision ID: f2a4b6c8d0e2
Revises: e7c3a9f1b2d5
Create Date: 2026-07-13
"""

from collections.abc import Sequence

from alembic import op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "f2a4b6c8d0e2"
down_revision: str | Sequence[str] | None = "e7c3a9f1b2d5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Stable 64-bit key for the transaction-scoped advisory lock that serializes
# the CREATE OR REPLACE against concurrent per-schema migration processes.
# Arbitrary constant; only needs to be identical across processes.
_ROUTINE_INSTALL_LOCK_KEY = 472638_00000001


def _pg_upgrade() -> None:
    # Serialize concurrent per-schema migration processes so only one performs
    # the CREATE OR REPLACE at a time (avoids `tuple concurrently updated`).
    # Transaction-scoped: released automatically at commit.
    op.execute(f"SELECT pg_advisory_xact_lock({_ROUTINE_INSTALL_LOCK_KEY})")

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
    # own downgrade. This migration only (re)installs them on more runs, so there
    # is nothing to undo without racing that migration's DROP.
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
