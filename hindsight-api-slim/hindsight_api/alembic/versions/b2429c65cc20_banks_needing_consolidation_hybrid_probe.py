"""Make ``banks_needing_consolidation()`` probe each bank instead of scanning facts.

The ``c8b4e2a71f95`` body joined ``memory_units`` to ``banks`` and grouped by
``bank_id``. When a large share of facts is unconsolidated (a backlog after
consolidation fell behind), ``GROUP BY`` must visit every matching row, so the
planner skips ``idx_memory_units_unconsolidated`` and seq-scans the whole table
on every reconcile pass (#5425).

The rewrite starts from ``banks`` and asks, per bank, whether at least one fact
is eligible (unconsolidated, not failed, experience/world). It only uses
existing indexes — no new index on ``memory_units``, which is too costly to
build on large installs:

1. ``idx_memory_units_unconsolidated`` (bank_id, created_at): the bank has any
   unconsolidated fact at all.
2. The newest 100 unconsolidated facts, read backwards from that index: one of
   them is not failed. Consolidation works oldest-first, so failed facts sit at
   the old end and a bank with real work answers here.
3. Otherwise, count: more unconsolidated facts than failed ones
   (``idx_memory_units_consolidation_failed``). Both counts come from partial
   indexes, so a bank clogged with failed facts does not fetch each one from
   the heap — the case a plain ``EXISTS`` probe made slower than the old scan.

Step 3 relies on a failed fact never also carrying ``consolidated_at``: the
consolidator stamps one or the other, retry clears both, and the importer
drops ``consolidation_failed_at`` when ``consolidated_at`` is set. If both were
ever set, step 3 can only under-report a bank whose newest 100 facts all failed.

Results are identical to the old body: ``banks.bank_id`` is the primary key
(one row per bank, like the ``GROUP BY``), facts without a ``banks`` row are
excluded by both forms, and the config / in-flight-op predicates are unchanged.
The wrapper (lock_timeout, skip arms, install-run gating,
``execute_unless_owned``) is copied unchanged from ``c8b4e2a71f95``.

Downgrade is a no-op: same signature and same results as the previous body, so
leaving it in place is harmless; downgrading past earlier migrations restores or
drops the routine as they define.

Oracle slot intentionally absent: this routine is PostgreSQL-only, and the Oracle
worker keeps its per-schema sweep.

Revision ID: b2429c65cc20
Revises: c8e2f4a6b1d3
Create Date: 2026-10-16
"""

from collections.abc import Sequence

from alembic import context

from hindsight_api.alembic._dialect import run_for_dialect
from hindsight_api.alembic._owned import execute_unless_owned
from hindsight_api.config import get_config

revision: str = "b2429c65cc20"
down_revision: str | Sequence[str] | None = "c8e2f4a6b1d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Same values as c8b4e2a71f95 — see that migration for the reasoning.
_LOCK_TIMEOUT = "250ms"

# How many of a bank's newest unconsolidated facts to check before counting.
_NEWEST_PROBE = 100

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
    """True for the single run that owns the routines (mirrors c8b4e2a71f95)."""
    target = _target_schema()
    return not target or target == _configured_schema()


def _prefix(schema: str | None) -> str:
    """Qualifier for ``schema``, or ``""`` to fall back to ``search_path``."""
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    if not _is_install_run():
        return
    schema = _prefix(_target_schema())

    execute_unless_owned(
        "banks_needing_consolidation",
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
                        SELECT %1$L::text, b.bank_id
                        FROM %1$I.banks b
                        WHERE COALESCE(b.config -> 'enable_auto_consolidation', 'true'::jsonb) <> 'false'::jsonb
                          AND NOT EXISTS (
                              SELECT 1 FROM %1$I.async_operations o
                              WHERE o.bank_id = b.bank_id
                                AND o.operation_type = 'consolidation'
                                AND o.status IN ('pending', 'processing')
                          )
                          AND EXISTS (
                              SELECT 1 FROM %1$I.memory_units m
                              WHERE m.bank_id = b.bank_id
                                AND m.consolidated_at IS NULL
                                AND m.fact_type IN ('experience', 'world')
                          )
                          AND (
                              EXISTS (
                                  SELECT 1 FROM (
                                      SELECT m.consolidation_failed_at
                                      FROM %1$I.memory_units m
                                      WHERE m.bank_id = b.bank_id
                                        AND m.consolidated_at IS NULL
                                        AND m.fact_type IN ('experience', 'world')
                                      ORDER BY m.created_at DESC
                                      LIMIT {_NEWEST_PROBE}
                                  ) newest
                                  WHERE newest.consolidation_failed_at IS NULL
                              )
                              OR (
                                  SELECT count(*) FROM %1$I.memory_units m
                                  WHERE m.bank_id = b.bank_id
                                    AND m.consolidated_at IS NULL
                                    AND m.fact_type IN ('experience', 'world')
                              ) > (
                                  SELECT count(*) FROM %1$I.memory_units f
                                  WHERE f.bank_id = b.bank_id
                                    AND f.consolidation_failed_at IS NOT NULL
                                    AND f.fact_type IN ('experience', 'world')
                              )
                          )
                    $q$, sch);
{_SKIP_ARMS}                END;
            END LOOP;
            PERFORM set_config('lock_timeout', prev_lock_timeout, true);
        END;
        $fn$;
        """,
    )


def _pg_downgrade() -> None:
    # No-op by design — see the module docstring.
    return


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
