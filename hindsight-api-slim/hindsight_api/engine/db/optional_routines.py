"""Registry of optional PostgreSQL server-side routines.

Some hot paths (currently only the worker poller's per-cycle scan) can be
sped up by a server-side PL/pgSQL routine that the API never installs
itself — operators install it out-of-band (e.g. a Helm hook in
hindsight-cloud) when they want the optimisation. When the routine is not
installed, callers must fall back to a pure-Python implementation.

This module centralises that pattern so we don't sprinkle ad-hoc
``try / except`` blocks (which silently log a server-side error on every
call) around the codebase. Each registered entry carries:

* a ``schema`` and ``name`` (used to probe ``pg_proc``)
* the install SQL, kept next to the registration so anyone who greps for
  the routine name finds the canonical definition

Probe behaviour:

* On first ``is_installed()`` call per backend instance we issue a single
  ``SELECT EXISTS(...) FROM pg_proc`` and cache the boolean result in
  memory for the life of the process.
* No TTL: if an operator installs a routine on a running cluster, workers
  pick it up only after restart. This is intentional — these routines are
  expected to be installed once at deploy time, and a probe-per-poll would
  defeat the optimisation.
* Non-PostgreSQL backends short-circuit to ``False`` without touching the
  database, so callers can use the same code path for Oracle.

PostgreSQL terminology note: ``CREATE FUNCTION ... RETURNS SETOF`` defines
a *function* (invoked via ``SELECT``); ``CREATE PROCEDURE`` defines a
*procedure* (invoked via ``CALL``). The SQL-standard umbrella term
covering both is *routine*, and the system catalog (``pg_proc``) stores
both. The module name uses "routine" so a future ``CREATE PROCEDURE``
entry slots in without a rename.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import DatabaseBackend, DatabaseConnection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptionalRoutine:
    """One optional server-side routine.

    Attributes:
        name: Unqualified routine name as it appears in ``pg_proc.proname``.
        schema: Schema the routine lives in (matched against
            ``pg_namespace.nspname``). Defaults to ``"public"``.
        install_sql: Canonical ``CREATE OR REPLACE`` statement, kept here
            so operators / Helm hooks have a single source of truth.
    """

    name: str
    schema: str
    install_sql: str


# Registry of known optional routines.
#
# Add new entries here when a hot path grows a server-side optimisation.
# Keep the install SQL inline — duplicating it elsewhere is how it goes
# stale.
SCHEMAS_WITH_PENDING_WORK = OptionalRoutine(
    name="schemas_with_pending_work",
    schema="public",
    install_sql="""
        CREATE OR REPLACE FUNCTION public.schemas_with_pending_work()
        RETURNS SETOF text AS $$
        DECLARE
          r RECORD; has_work BOOLEAN;
        BEGIN
          FOR r IN SELECT nspname FROM pg_namespace
                   WHERE nspname LIKE 'tenant_%' LOOP
            BEGIN
              EXECUTE format(
                'SELECT EXISTS(SELECT 1 FROM %I.async_operations '
                'WHERE status = ''pending'' '
                'AND task_payload IS NOT NULL LIMIT 1)',
                r.nspname) INTO has_work;
              IF has_work THEN RETURN NEXT r.nspname; END IF;
            EXCEPTION WHEN OTHERS THEN NULL;
            END;
          END LOOP;
        END $$ LANGUAGE plpgsql STABLE;
    """,
)

_REGISTRY: dict[str, OptionalRoutine] = {
    SCHEMAS_WITH_PENDING_WORK.name: SCHEMAS_WITH_PENDING_WORK,
}


class OptionalRoutines:
    """Per-backend cache of which optional routines are installed.

    One instance per long-lived consumer (e.g. one per ``WorkerPoller``).
    Probes ``pg_proc`` lazily on first lookup and caches the result in
    memory until the process restarts.
    """

    def __init__(self, backend: DatabaseBackend) -> None:
        self._backend = backend
        self._cache: dict[str, bool] = {}

    async def is_installed(self, conn: DatabaseConnection, routine_name: str) -> bool:
        """Return True iff *routine_name* exists in ``pg_proc``.

        On non-PostgreSQL backends always returns False without issuing a
        query. Result is memoised for the life of this instance.
        """
        if self._backend.backend_type != "postgresql":
            return False

        cached = self._cache.get(routine_name)
        if cached is not None:
            return cached

        routine = _REGISTRY.get(routine_name)
        if routine is None:
            raise KeyError(f"Unknown optional routine: {routine_name!r}")

        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_proc p "
            "JOIN pg_namespace n ON p.pronamespace = n.oid "
            "WHERE n.nspname = $1 AND p.proname = $2)",
            routine.schema,
            routine.name,
        )
        installed = bool(exists)
        self._cache[routine_name] = installed
        if installed:
            logger.info(
                "Optional PG routine %s.%s detected — using server-side path",
                routine.schema,
                routine.name,
            )
        else:
            logger.debug(
                "Optional PG routine %s.%s not installed — using fallback path",
                routine.schema,
                routine.name,
            )
        return installed

    def invalidate(self, routine_name: str | None = None) -> None:
        """Drop cached probe results (test helper).

        Without an argument, clears the entire cache.
        """
        if routine_name is None:
            self._cache.clear()
        else:
            self._cache.pop(routine_name, None)
