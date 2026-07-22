"""Add the ``schemas_with_expired_operations`` cross-tenant discovery routine.

The worker's terminal-operation cleanup (``a8c1e4f7b0d3``) opens a connection
and a prune transaction against *every* tenant schema on every cleanup cycle,
whether or not that tenant has anything to prune. At thousands of tenants that
is a per-cycle query storm whose cost is paid entirely by idle schemas.

This is the same problem ``public.schemas_with_expired_rows`` already solves for
the ``audit_log`` / ``llm_requests`` retention sweeps (``e5f6a7b8c9d0``): one
round-trip returns just the schemas that actually hold expired rows, and the
caller then does real work only there. ``async_operations`` needs its own
routine rather than reusing that one because eligibility is not "row older than
N days" — pending and processing rows are never prunable, so the status filter
has to be part of the predicate.

Install policy mirrors ``b6d2f8a4c1e7`` (#2638/#2824), the current behaviour for
the sibling routines: the routine is database-global — it enumerates ``pg_class``
across every schema and dispatches per schema — so exactly one copy should exist,
installed into the schema this deployment is *configured* to use and called from
there via ``fq_routine``. Gating on the literal ``"public"`` instead of the
configured schema is what left single-tenant deployments in a dedicated
non-``public`` schema without the routine (#2638).

Exactly one migration run satisfies that predicate, so concurrent per-schema runs
never issue competing ``CREATE OR REPLACE`` against the same ``pg_proc`` row and
cannot hit ``tuple concurrently updated``. No cross-process coordination is
required — in particular no advisory lock, which is unusable here because
Hindsight runs behind connection poolers and managed PG services (see #2817).

Each per-schema probe runs in its own ``BEGIN ... EXCEPTION`` block so a tenant
dropped mid-scan is skipped instead of aborting the sweep (see ``c7e9f1a3b5d2``).

Revision ID: d7b2f8a1c934
Revises: b6d2f8a4c1e7
Create Date: 2026-07-20
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect
from hindsight_api.config import get_config

revision: str = "d7b2f8a1c934"
down_revision: str | Sequence[str] | None = "b6d2f8a4c1e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _configured_schema() -> str:
    """The one schema this deployment's routines live in and are called from."""
    return get_config().database_schema or "public"


def _target_schema() -> str | None:
    return context.config.get_main_option("target_schema")


def _is_install_run() -> bool:
    """True for the single run that owns the routine (mirrors b6d2f8a4c1e7)."""
    target = _target_schema()
    return not target or target == _configured_schema()


def _prefix(schema: str | None) -> str:
    """Qualifier for ``schema``, or ``""`` to fall back to ``search_path``."""
    return f'"{schema}".' if schema else ""


def _drop_routine(schema: str | None) -> None:
    op.execute(f"DROP FUNCTION IF EXISTS {_prefix(schema)}schemas_with_expired_operations(int)")


def _pg_upgrade() -> None:
    if not _is_install_run():
        # Tenant schemas must not carry their own copy: the routine is
        # database-global and only the configured schema's copy is ever called.
        # Dropping (rather than skipping) also cleans up after any interim build
        # of this branch that installed per-schema copies.
        _drop_routine(_target_schema())
        return
    schema = _prefix(_target_schema())
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {schema}schemas_with_expired_operations(p_days int)
        RETURNS SETOF text
        LANGUAGE plpgsql STABLE
        AS $fn$
        DECLARE
            sch text;
            has_expired boolean;
        BEGIN
            -- Zero (or negative) retention means "keep forever": report nothing
            -- so the caller skips the sweep entirely.
            IF p_days IS NULL OR p_days <= 0 THEN
                RETURN;
            END IF;
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
                EXCEPTION
                    -- Schema or its table vanished between the pg_class
                    -- snapshot and this probe (tenant dropped or migrating).
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
    # This migration is the sole creator of this routine — no older migration
    # owns a copy the way e5f6a7b8c9d0 owns the public sibling routines — so the
    # install run's own copy is always ours to drop.
    if not _is_install_run():
        return
    _drop_routine(_target_schema())


def upgrade() -> None:
    # Oracle slot intentionally absent: this mirrors the PostgreSQL-only
    # maintenance routines, and the Oracle worker keeps its per-schema sweep.
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
