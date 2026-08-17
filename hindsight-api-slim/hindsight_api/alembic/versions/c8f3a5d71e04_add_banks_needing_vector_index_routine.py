"""Add the ``banks_needing_vector_index`` cross-tenant discovery routine.

Per-(bank, fact_type) partial vector indexes stopped being created
unconditionally at bank creation: they live on the shared ``memory_units``
table, so PostgreSQL locks and plans against every one of them for queries
belonging to every *other* bank, and three per bank exhausts the lock table at a
few thousand banks (#3485). They are now earned by size — a partition gets an
index once it holds ``HINDSIGHT_API_VECTOR_INDEX_MIN_ROWS`` rows — and a
background sweep on the maintenance loop reconciles the index set.

That sweep needs to know, across every tenant schema, which (bank, fact_type)
pairs are big enough to deserve an index. Asking per schema is the query storm
``d7b2f8a1c934`` and ``e5f6a7b8c9d0`` already removed for the other sweeps, so
this follows them: one round-trip returns just the qualifying pairs, along with
the ``banks.internal_id`` the caller needs to derive each index name.

The count is deliberately unfiltered by ``embedding IS NOT NULL``. That
predicate is not in ``idx_memory_units_bank_fact_type``, so including it would
turn an index-only scan into a heap scan of the whole table on every sweep, and
it cannot change a threshold decision by enough to matter — a partition sitting
within a rounding error of the boundary is one the hysteresis gap already
covers.

Install policy mirrors ``d7b2f8a1c934``: the routine is database-global (it
enumerates ``pg_class`` across every schema and dispatches per schema), so
exactly one copy exists, in the schema this deployment is configured to use,
called from there via ``fq_routine``. Exactly one migration run satisfies that
predicate, so concurrent per-schema runs cannot issue competing
``CREATE OR REPLACE`` against the same ``pg_proc`` row. No advisory lock — they
are unusable behind connection poolers (#2817), which is also why #2803's
background reconcile was rejected.

Each per-schema probe runs in its own ``BEGIN ... EXCEPTION`` block so a tenant
dropped mid-scan is skipped rather than aborting the sweep (``c7e9f1a3b5d2``),
and carries the short ``lock_timeout`` from ``c8b4e2a71f95`` so a schema under
concurrent DDL is skipped rather than waited on — waiting is what closes the
deadlock cycle between this scan's AccessShareLocks and a tenant drop's
AccessExclusiveLocks. This routine is more exposed to that race than its
siblings, not less: the sweep it feeds goes on to issue index DDL against the
very tables it just read.

This migration installs a routine only. It deliberately issues no index DDL:
an instance that has already hit the #3485 wall cannot plan a statement against
``memory_units``, so a migration that counted rows to decide what to drop would
fail before it could help. Convergence is left entirely to the sweep, which
drops in bounded batches using ``DROP INDEX CONCURRENTLY`` — a utility statement
that locks its own index plus the table rather than all of the table's indexes,
and therefore still runs when everything else on that relation is failing.

Revision ID: c8f3a5d71e04
Revises: f2a7c9d4b168
Create Date: 2026-08-17
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect
from hindsight_api.config import get_config

revision: str = "c8f3a5d71e04"
down_revision: str | Sequence[str] | None = "f2a7c9d4b168"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Matches the sibling routines (c8b4e2a71f95): short enough to abandon the wait
# before PostgreSQL's deadlock detector runs (deadlock_timeout defaults to 1s),
# long enough to ride out a brief DDL statement rather than skipping a healthy
# schema.
_LOCK_TIMEOUT = "250ms"

# Both outcomes of the same tenant-drop race, kept as separate arms so each
# reason is legible at the point it is handled.
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
    """True for the single run that owns the routine (mirrors d7b2f8a1c934)."""
    target = _target_schema()
    return not target or target == _configured_schema()


def _prefix(schema: str | None) -> str:
    """Qualifier for ``schema``, or ``""`` to fall back to ``search_path``."""
    return f'"{schema}".' if schema else ""


def _drop_routine(schema: str | None) -> None:
    op.execute(f"DROP FUNCTION IF EXISTS {_prefix(schema)}banks_needing_vector_index(bigint)")


def _pg_upgrade() -> None:
    if not _is_install_run():
        # Tenant schemas must not carry their own copy: the routine is
        # database-global and only the configured schema's copy is ever called.
        _drop_routine(_target_schema())
        return
    schema = _prefix(_target_schema())
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {schema}banks_needing_vector_index(p_min_rows bigint)
        RETURNS TABLE(schema_name text, bank_id text, internal_id uuid, fact_type text, row_count bigint)
        LANGUAGE plpgsql STABLE
        AS $fn$
        DECLARE
            sch text;
            prev_lock_timeout text;
        BEGIN
            -- A non-positive floor means "no partition qualifies": either the
            -- deployment disabled per-bank indexes outright, or the caller
            -- asked for a threshold that would return every pair in the
            -- database. Both are answered with the empty set; the caller's
            -- catalog scan still finds indexes that need dropping.
            IF p_min_rows IS NULL OR p_min_rows <= 0 THEN
                RETURN;
            END IF;
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
                        SELECT %1$L::text, g.bank_id, b.internal_id, g.fact_type, g.row_count
                        FROM (
                            SELECT m.bank_id, m.fact_type, COUNT(*) AS row_count
                            FROM %1$I.memory_units m
                            WHERE m.fact_type IN ('world', 'experience', 'observation')
                            GROUP BY m.bank_id, m.fact_type
                            HAVING COUNT(*) >= $1
                        ) g
                        JOIN %1$I.banks b ON b.bank_id = g.bank_id
                    $q$, sch) USING p_min_rows;
                    -- Sentinel: this schema was scanned successfully and simply
                    -- has nothing at or above the floor. Without it the caller
                    -- cannot tell "no qualifying partitions" from "skipped by an
                    -- arm below", and would read a skipped schema as an empty
                    -- one — dropping every vector index it owns. Emitted after
                    -- the query so a schema that raised never gets one.
                    RETURN QUERY SELECT sch, NULL::text, NULL::uuid, NULL::text, NULL::bigint;
{_SKIP_ARMS}                END;
            END LOOP;
            PERFORM set_config('lock_timeout', prev_lock_timeout, true);
        END;
        $fn$;
        """
    )


def _pg_downgrade() -> None:
    # Unconditional: the install run drops the real copy, and a tenant run drops
    # nothing because upgrade never created one there.
    _drop_routine(_target_schema())


def upgrade() -> None:
    # PostgreSQL-only: the maintenance loop that calls this routine does not run
    # on Oracle, which uses a single global vector index and has no per-bank
    # indexes to reconcile.
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
