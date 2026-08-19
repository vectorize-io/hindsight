"""Make the cross-schema maintenance routines skip a schema under concurrent DDL.

``banks_needing_consolidation()``, ``mental_models_with_cron()``,
``schemas_with_expired_rows(...)`` and ``schemas_with_expired_operations(...)``
snapshot the schemas owning a target table from ``pg_class`` and then query each
schema in turn, inside one transaction. Every such query takes AccessShareLock on
two or three relations, and those locks are held until the caller commits.

``c7e9f1a3b5d2`` already handles the schema *vanishing* mid-scan. The same race
has a second outcome: the schema is not gone, it is being rewritten, and its DDL
holds — or is queued for — AccessExclusiveLock. A queued AccessExclusiveLock
blocks later AccessShareLock requests, so::

    routine  holds AccessShare(memory_units)  ->  wants AccessShare(banks)
    dropper  queued AccessExclusive(banks)    ->  wants AccessExclusive(memory_units)

is a cycle, and PostgreSQL breaks it by killing one side. When it picks the
routine the whole scan aborts, so one tenant being dropped takes out an entire
maintenance pass. Observed as a recurring ``DeadlockDetectedError`` in the test
suite, where xdist workers create and drop schemas continuously while
``test_maintenance_routines`` calls the routines against the same database; in
production the background maintenance loop races tenant deletion and migration
the same way.

Fix the routine's side of the cycle: give each per-schema query a short
``lock_timeout`` so it abandons the wait long before the deadlock detector runs,
and skip that schema. A schema mid-DDL has nothing useful to report anyway, and
the maintenance loop runs on a ticker, so it is picked up on the next pass. Locks
already held from earlier schemas stay until the caller commits — that is fine,
the point is only that this routine stops *waiting* on the other party.

``lock_timeout`` is set via ``set_config(..., is_local => true)`` rather than
``SET LOCAL``: PL/pgSQL rejects the ``SET`` command inside a non-volatile
function, and these are all ``STABLE``. The previous value is restored before
returning so the caller's transaction is left as it was found. Only conflicting
DDL can trigger it — AccessShareLock does not conflict with ordinary DML — so
this never fires on a merely busy table.

Downgrade is a no-op: the bodies here are the ones from ``b6d2f8a4c1e7`` /
``d7b2f8a1c934`` plus strictly-additive resilience, with identical signatures and
results, so leaving them in place is harmless. Downgrading past those migrations
restores or drops them as they define.

Revision ID: c8b4e2a71f95
Revises: e7c3a91f4b62
Create Date: 2026-08-17
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect
from hindsight_api.config import get_config

revision: str = "c8b4e2a71f95"
down_revision: str | Sequence[str] | None = "e7c3a91f4b62"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Short enough to abandon the wait before PostgreSQL's deadlock detector runs
# (deadlock_timeout defaults to 1s), long enough to ride out a brief DDL
# statement rather than skipping a healthy schema.
_LOCK_TIMEOUT = "250ms"

# Both outcomes of the same race, kept as separate arms so each reason is legible
# at the point it is handled.
_SKIP_ARMS = """
                EXCEPTION
                    -- Schema or its tables vanished between the pg_class
                    -- snapshot and this query (tenant dropped or migrating).
                    WHEN undefined_table OR invalid_schema_name OR undefined_column THEN
                        CONTINUE;
                    -- Schema is mid-DDL and holds (or has queued) an
                    -- AccessExclusiveLock. Skip it rather than wait: waiting is
                    -- what closes the deadlock cycle. deadlock_detected is
                    -- belt-and-braces for a cycle formed before lock_timeout.
                    WHEN lock_not_available OR deadlock_detected THEN
                        CONTINUE;
"""


def _configured_schema() -> str:
    """The one schema this deployment's routines live in and are called from."""
    return get_config().database_schema or "public"


def _target_schema() -> str | None:
    return context.config.get_main_option("target_schema")


def _is_install_run() -> bool:
    """True for the single run that owns the routines (mirrors b6d2f8a4c1e7)."""
    target = _target_schema()
    return not target or target == _configured_schema()


def _prefix(schema: str | None) -> str:
    """Qualifier for ``schema``, or ``""`` to fall back to ``search_path``."""
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    # Tenant schemas carry no copy of these routines; only the configured
    # schema's copy is ever called. Non-install runs have nothing to replace —
    # and unlike b6d2f8a4c1e7 there are no stray per-tenant copies to clean up,
    # that migration already did it.
    if not _is_install_run():
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
            prev_lock_timeout text;
        BEGIN
            prev_lock_timeout := current_setting('lock_timeout');
            PERFORM set_config('lock_timeout', '{_LOCK_TIMEOUT}', true);
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
{_SKIP_ARMS}                END;
            END LOOP;
            PERFORM set_config('lock_timeout', prev_lock_timeout, true);
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
            prev_lock_timeout text;
        BEGIN
            IF p_days IS NULL OR p_days <= 0 THEN
                RETURN;
            END IF;
            prev_lock_timeout := current_setting('lock_timeout');
            PERFORM set_config('lock_timeout', '{_LOCK_TIMEOUT}', true);
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
{_SKIP_ARMS}                END;
                IF has_expired THEN
                    RETURN NEXT sch;
                END IF;
            END LOOP;
            PERFORM set_config('lock_timeout', prev_lock_timeout, true);
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
            prev_lock_timeout text;
        BEGIN
            prev_lock_timeout := current_setting('lock_timeout');
            PERFORM set_config('lock_timeout', '{_LOCK_TIMEOUT}', true);
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
{_SKIP_ARMS}                END;
            END LOOP;
            PERFORM set_config('lock_timeout', prev_lock_timeout, true);
        END;
        $fn$;
        """
    )

    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {schema}schemas_with_expired_operations(p_days int)
        RETURNS SETOF text
        LANGUAGE plpgsql STABLE
        AS $fn$
        DECLARE
            sch text;
            has_expired boolean;
            prev_lock_timeout text;
        BEGIN
            -- Zero (or negative) retention means "keep forever": report nothing
            -- so the caller skips the sweep entirely.
            IF p_days IS NULL OR p_days <= 0 THEN
                RETURN;
            END IF;
            prev_lock_timeout := current_setting('lock_timeout');
            PERFORM set_config('lock_timeout', '{_LOCK_TIMEOUT}', true);
            FOR sch IN
                SELECT n.nspname
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = 'async_operations' AND c.relkind = 'r'
            LOOP
                BEGIN
                    -- Matches the worker's prune predicate: only terminal rows
                    -- are eligible, so a schema holding nothing but pending or
                    -- processing work is correctly reported as having nothing
                    -- to prune. Uses idx_async_operations_terminal_cleanup.
                    EXECUTE format(
                        'SELECT EXISTS ('
                        '  SELECT 1 FROM %I.async_operations'
                        '  WHERE status IN (''completed'', ''failed'', ''cancelled'')'
                        '    AND updated_at < NOW() - make_interval(days => $1)'
                        ')',
                        sch
                    ) INTO has_expired USING p_days;
{_SKIP_ARMS}                END;
                IF has_expired THEN
                    RETURN NEXT sch;
                END IF;
            END LOOP;
            PERFORM set_config('lock_timeout', prev_lock_timeout, true);
        END;
        $fn$;
        """
    )


def _pg_downgrade() -> None:
    # No-op by design — see the module docstring. The bodies installed here are
    # the previous ones plus a skip arm; dropping the routines would strand the
    # migrations that claim to own them, and re-installing the old bodies would
    # duplicate their definitions here.
    return


def upgrade() -> None:
    # Oracle slot intentionally absent: these routines are PostgreSQL-only, and
    # the Oracle worker keeps its per-schema sweep.
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
