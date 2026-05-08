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
* a ``contract`` describing the expected signature and return shape

Bodies are deliberately not stored here. Hindsight never installs these
routines, so a body checked into this repo would (a) drift from whatever
operators actually deploy and (b) imply ownership we don't have. The
contract is the entire API surface: any operator-supplied implementation
that satisfies it is interchangeable.

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
    """One optional server-side routine Hindsight may call when present.

    Hindsight never installs these — operators do, out-of-band. The fields
    here describe the contract the API expects; any implementation that
    matches is interchangeable.

    Attributes:
        name: Unqualified routine name as it appears in ``pg_proc.proname``.
        schema: Schema the routine lives in (matched against
            ``pg_namespace.nspname``).
        contract: Free-form description of the expected signature,
            arguments, return type, and any semantic constraints. Read
            this before installing a custom implementation.
    """

    name: str
    schema: str
    contract: str


# Registry of known optional routines.
#
# Add new entries here when a hot path grows a server-side optimisation.
# Document the *contract* — not the implementation — so operator-supplied
# variants stay interchangeable and we don't pretend to own SQL we never
# install.
SCHEMAS_WITH_PENDING_WORK = OptionalRoutine(
    name="schemas_with_pending_work",
    schema="public",
    contract="""
    Signature:  public.schemas_with_pending_work() RETURNS SETOF text

    Called by the worker poller on every cycle to find schemas with
    claimable async_operations rows. The poller then runs FOR UPDATE
    SKIP LOCKED only against the returned schemas, so an empty set means
    "nothing to do, skip the expensive claim query".

    A schema is "claimable" iff at least one row matches:
        status = 'pending' AND task_payload IS NOT NULL
    in that schema's ``async_operations`` table.

    Required semantics:
      * No arguments.
      * Returns a set of schema names (``text``); each must match a real
        ``pg_namespace.nspname``. The poller passes them straight into
        the claim query — anything that isn't a valid schema will fail.
      * Operators choose the search scope (e.g. ``tenant_%`` only, or
        include ``public``). A schema omitted from the scan will *never*
        be serviced by the poller, so the implementation must cover
        every schema that holds an ``async_operations`` table in that
        deployment.
      * Should be cheap and idempotent — called every poll cycle (~30s).

    Fallback when the routine is absent: per-schema ``EXISTS`` queries
    from Python (~4ms per schema). The server-side path is a single-
    round-trip optimisation worth ~200ms in deployments with thousands
    of tenant schemas; everything else works correctly without it.
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
