"""Scheduled full refresh for mental models

Revision ID: d4f7a2c9e1b5
Revises: b2429c65cc20
Create Date: 2026-10-16

A delta-mode mental model can also carry ``trigger.full_refresh_cron``: a second,
independent schedule on which it is refreshed in full mode, to clear the drift that
many surgical delta edits accumulate (#4340). Its normal refresh mode is untouched.

Three parts:

- ``mental_models.last_full_refreshed_at`` — when a refresh last *ran* in full mode
  (requested or fallen back to). The cron scan needs its own clock for this
  schedule: ``last_refreshed_at`` advances on every delta refresh, so keyed off it a
  weekly full refresh would never come due on a model that deltas daily.
- ``idx_mental_models_full_refresh_cron`` — a partial index matching the new arm
  of the discovery predicate, for the same reason ``f2a7c9d4b168`` indexes
  ``refresh_cron``: a tenant with no scheduled models should cost an empty index
  scan. The planner ORs the two partial indexes.
- ``mental_models_with_cron()`` now also returns models that carry only a
  ``full_refresh_cron`` (e.g. delta + refresh after consolidation), plus the two
  new columns. Its result type changes, which ``CREATE OR REPLACE`` cannot do, so
  it is dropped and recreated; both statements are skipped when the deployment
  owns the routine (``_owned.py``). The maintenance loop reads the new columns
  optionally, so an owned routine that still returns the old shape keeps its
  ``refresh_cron`` schedules working.

The routine and the index are PostgreSQL-only, like everything the maintenance
loop runs (``f4d1c2b3a5e6``); Oracle gets the column so the shared engine code
can write it.
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect
from hindsight_api.alembic._owned import execute_unless_owned
from hindsight_api.config import get_config

revision: str = "d4f7a2c9e1b5"
down_revision: str | Sequence[str] | None = "b2429c65cc20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEX = "idx_mental_models_full_refresh_cron"
_ROUTINE = "mental_models_with_cron"

# Copied from c8b4e2a71f95, which explains both arms and the lock_timeout.
_LOCK_TIMEOUT = "250ms"
_SKIP_ARMS = """
                EXCEPTION
                    WHEN undefined_table OR invalid_schema_name OR undefined_column THEN
                        CONTINUE;
                    WHEN lock_not_available OR deadlock_detected THEN
                        CONTINUE;
"""


def _target_schema() -> str | None:
    return context.config.get_main_option("target_schema")


def _prefix(schema: str | None) -> str:
    return f'"{schema}".' if schema else ""


def _is_install_run() -> bool:
    """True for the single run that owns the routines (mirrors c8b4e2a71f95)."""
    target = _target_schema()
    return not target or target == (get_config().database_schema or "public")


def _routine_sql(schema: str, *, with_full_refresh: bool) -> str:
    if with_full_refresh:
        columns = "refresh_cron text, last_refreshed_at timestamptz, full_refresh_cron text, last_full_refreshed_at timestamptz"
        select = (
            "mm.trigger->>'refresh_cron', mm.last_refreshed_at,\n"
            "                               mm.trigger->>'full_refresh_cron', mm.last_full_refreshed_at"
        )
        where = (
            "(COALESCE(mm.trigger->>'refresh_cron', '') <> ''\n"
            "                               OR COALESCE(mm.trigger->>'full_refresh_cron', '') <> '')"
        )
    else:
        columns = "refresh_cron text, last_refreshed_at timestamptz"
        select = "mm.trigger->>'refresh_cron', mm.last_refreshed_at"
        where = "COALESCE(mm.trigger->>'refresh_cron', '') <> ''"
    return f"""
        CREATE OR REPLACE FUNCTION {schema}{_ROUTINE}()
        RETURNS TABLE(schema_name text, bank_id text, mental_model_id text,
                     {columns})
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
                               {select}
                        FROM %1$I.mental_models mm
                        WHERE {where}
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


def _replace_routine(*, with_full_refresh: bool) -> None:
    if not _is_install_run():
        return
    schema = _prefix(_target_schema())
    execute_unless_owned(_ROUTINE, f"DROP FUNCTION IF EXISTS {schema}{_ROUTINE}()")
    execute_unless_owned(_ROUTINE, _routine_sql(schema, with_full_refresh=with_full_refresh))


def _pg_upgrade() -> None:
    schema = _prefix(_target_schema())
    op.execute(
        f"ALTER TABLE {schema}mental_models ADD COLUMN IF NOT EXISTS last_full_refreshed_at TIMESTAMP WITH TIME ZONE"
    )
    # Predicate is character-for-character the routine's new WHERE arm, which is
    # what lets the planner match the partial index (see f2a7c9d4b168).
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {_INDEX} ON {schema}mental_models (bank_id) "
        "WHERE COALESCE(\"trigger\"->>'full_refresh_cron', '') <> ''"
    )
    _replace_routine(with_full_refresh=True)


def _pg_downgrade() -> None:
    schema = _prefix(_target_schema())
    # Routine first: the new body names last_full_refreshed_at, and once the column
    # is gone its undefined_column arm would silently skip every schema.
    _replace_routine(with_full_refresh=False)
    op.execute(f"DROP INDEX IF EXISTS {schema}{_INDEX}")
    op.execute(f"ALTER TABLE {schema}mental_models DROP COLUMN IF EXISTS last_full_refreshed_at")


def _oracle_upgrade() -> None:
    op.execute("ALTER TABLE mental_models ADD (last_full_refreshed_at TIMESTAMP WITH TIME ZONE)")


def _oracle_downgrade() -> None:
    op.execute("ALTER TABLE mental_models DROP COLUMN last_full_refreshed_at")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
