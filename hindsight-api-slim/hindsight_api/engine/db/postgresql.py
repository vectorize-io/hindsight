"""PostgreSQL backend implementation using asyncpg.

Wraps asyncpg's pool and connection objects behind the DatabaseBackend
and DatabaseConnection interfaces.  Returns raw asyncpg.Record objects
from fetch/fetchrow — they satisfy the ResultRow protocol natively in C,
avoiding Python-level wrapping overhead (~570K __getitem__ calls per
20-query benchmark → measurable regression when wrapped).
"""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import parse_qs, urlparse

import asyncpg  # noqa: F401

from .base import DatabaseBackend, DatabaseConnection
from .pool_instrumentation import PoolStats, instrument_acquire

logger = logging.getLogger(__name__)


async def apply_session_settings(conn: asyncpg.Connection, settings: list[tuple[str, str]]) -> None:
    """Apply session-scoped GUCs to ``conn`` in a single round trip.

    Unless ``HINDSIGHT_API_DB_SESSION_SETUP_ON_ACQUIRE=false``, the pool passes
    its init callback as ``setup=`` too, so this runs on *every* acquire, not
    just on connection creation. Issued as N separate ``SET`` statements that
    was N round trips — and behind a transaction-mode pooler N server-side
    transactions — per acquire, which the worker's per-schema acquires
    multiplied into a sustained commit-rate burn (#3499). One
    ``SELECT set_config(...)`` collapses them into one statement.

    Some of the settings are extension-provided (``hnsw.ef_search``,
    ``pg_trgm.similarity_threshold``) and may not exist on the cluster; a single
    statement fails as a whole, so on error fall back to applying them one by
    one, skipping only the ones the server rejects.
    """
    if not settings:
        return

    args: list[str] = [value for pair in settings for value in pair]
    projection = ", ".join(f"set_config(${2 * i + 1}, ${2 * i + 2}, false)" for i in range(len(settings)))
    try:
        await conn.execute(f"SELECT {projection}", *args)
        return
    except asyncpg.exceptions.PostgresError:
        # Narrow to PostgresError so genuine bugs in the pool/conn layer surface
        # instead of being silently retried statement-by-statement.
        logger.debug("Batched session setup failed — applying settings individually")

    for name, value in settings:
        try:
            await conn.execute("SELECT set_config($1, $2, false)", name, value)
        except asyncpg.exceptions.PostgresError:
            logger.debug("Could not set %s — the server may not know this setting", name)


class PostgresConnection(DatabaseConnection):
    """DatabaseConnection wrapper around an asyncpg.Connection."""

    __slots__ = ("_conn",)

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator["PostgresConnection"]:
        async with self._conn.transaction():
            yield self

    async def execute(self, query: str, *args: Any, timeout: float | None = None) -> str:
        return await self._conn.execute(query, *args, timeout=timeout)

    async def executemany(self, query: str, args: list[tuple[Any, ...]], *, timeout: float | None = None) -> None:
        await self._conn.executemany(query, args, timeout=timeout)

    async def fetch(self, query: str, *args: Any, timeout: float | None = None) -> list:
        # Return raw asyncpg.Record objects — they satisfy the ResultRow
        # protocol natively (key access, .keys(), .get(), etc.) with zero
        # Python wrapping overhead.
        return await self._conn.fetch(query, *args, timeout=timeout)

    async def fetchrow(self, query: str, *args: Any, timeout: float | None = None):
        # Return raw asyncpg.Record — no wrapping needed.
        return await self._conn.fetchrow(query, *args, timeout=timeout)

    async def fetchval(self, query: str, *args: Any, column: int = 0, timeout: float | None = None) -> Any:
        return await self._conn.fetchval(query, *args, column=column, timeout=timeout)

    async def copy_records_to_table(
        self,
        table_name: str,
        *,
        records: list[tuple[Any, ...]],
        columns: list[str],
        timeout: float | None = None,
    ) -> None:
        """Use asyncpg's native COPY for fast bulk loading."""
        await self._conn.copy_records_to_table(table_name, records=records, columns=columns, timeout=timeout)


def application_name_from_dsn(dsn: str) -> str | None:
    """Extract the ``application_name`` query parameter from a PostgreSQL DSN.

    asyncpg already forwards this to the server in the startup packet (it
    passes unrecognized DSN query parameters through as ``server_settings``),
    so a direct connection is labelled correctly in ``pg_stat_activity``.
    The value is extracted here so it can be re-applied per acquire — see
    ``_application_name_setup``.
    """
    try:
        values = parse_qs(urlparse(dsn).query).get("application_name")
    except ValueError:
        return None
    if not values:
        return None
    # libpq semantics: the last occurrence of a repeated parameter wins.
    return values[-1] or None


def _application_name_setup(app_name: str, init_callback: Any | None) -> Callable[[Any], Awaitable[None]]:
    """Wrap ``init_callback`` so every acquire re-asserts ``application_name``.

    asyncpg runs ``RESET ALL`` when a connection is released back to the pool.
    Connected straight to PostgreSQL that is harmless: ``RESET ALL`` restores
    the value from the startup packet, which carried the DSN's name.

    Behind a connection pooler (pgbouncer) it is not. The server connection's
    startup packet is the *pooler's*, with no application_name; pgbouncer
    applies the client's value with a ``SET`` when it links client to server.
    ``RESET ALL`` therefore resets it to empty, and pgbouncer — which already
    believes the value is applied — does not re-issue it. Only the first
    acquire on each server connection is attributed; every later one reports
    an empty application_name, which is exactly the sort of gap that shows up
    in production but never under psql.

    Re-asserting it on every acquire fixes both topologies. ``set_config``
    rather than ``SET`` because the name is operator-supplied and ``SET`` does
    not accept bind parameters.
    """

    async def _setup(conn: Any) -> None:
        await conn.execute("SELECT set_config('application_name', $1, false)", app_name)
        if init_callback is not None:
            await init_callback(conn)

    return _setup


class PostgreSQLBackend(DatabaseBackend):
    """DatabaseBackend implementation wrapping an asyncpg connection pool."""

    def run_migrations(self, dsn: str, *, schema: str | None = None) -> None:
        """Run Alembic migrations for PostgreSQL."""
        from ...config import get_config
        from ...migrations import run_migrations

        config = get_config()
        run_migrations(dsn, schema=schema, migration_database_url=config.migration_database_url)

    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None
        self._acquire_warn_threshold_s: float = 1.0
        self._acquire_timeout_s: float | None = None
        self._dsn: str | None = None

    @property
    def dsn(self) -> str | None:
        """The DSN this backend's pool was opened with, if it has been initialized."""
        return self._dsn

    async def initialize(
        self,
        dsn: str,
        *,
        min_size: int = 5,
        max_size: int = 20,
        command_timeout: float = 300,
        acquire_timeout: float = 30,
        statement_cache_size: int = 0,
        init_callback: Any | None = None,
    ) -> None:
        from ...config import get_config

        config = get_config()
        # Kept so code that needs its *own* connection — CREATE/DROP INDEX
        # CONCURRENTLY cannot run on a pooled one inside a transaction — can
        # reach the database this engine is actually attached to. Re-deriving it
        # from HINDSIGHT_API_DATABASE_URL is wrong whenever the engine was handed
        # a DSN directly (embedders, and the test suite, which resolves pg0 in a
        # fixture and never sets the env var).
        self._dsn = dsn
        self._acquire_warn_threshold_s = config.db_acquire_warn_threshold_ms / 1000.0
        # Kept for acquire() below: asyncpg's ``timeout`` create_pool kwarg is a
        # *connect* kwarg (how long establishing a new connection may take), and
        # ``Pool.acquire()`` defaults to waiting for a free connection forever.
        # Passing it here alone made HINDSIGHT_API_DB_ACQUIRE_TIMEOUT a no-op for
        # the wait it names: a pool-exhaustion stall never surfaced as an error,
        # it just hung (#3002). 0 restores the unbounded behaviour.
        self._acquire_timeout_s = acquire_timeout if acquire_timeout > 0 else None
        # The DSN's application_name survives RESET ALL only on a direct
        # connection; behind pgbouncer it has to be re-asserted per acquire
        # (see _application_name_setup).
        app_name = application_name_from_dsn(dsn)
        pool_init = _application_name_setup(app_name, init_callback) if app_name else init_callback

        # init runs once per new connection; setup runs on every acquire, after
        # asyncpg's release-time RESET ALL. Re-running the session GUCs
        # (hnsw.ef_search, statement_timeout, …) there is what keeps a *reused*
        # connection from silently falling back to server defaults, so it is the
        # default. Deployments that pin those GUCs server-side (ALTER ROLE /
        # ALTER DATABASE ... SET) get them back from RESET ALL anyway, making the
        # re-apply a wasted round trip on every acquire — and behind a
        # transaction-mode pooler, a wasted transaction too (#3499); they can
        # drop it with HINDSIGHT_API_DB_SESSION_SETUP_ON_ACQUIRE=false.
        # application_name is NOT part of that trade-off: pgbouncer never
        # re-issues it after RESET ALL, so it keeps its per-acquire hook either
        # way (#3491).
        setup_on_acquire = config.db_session_setup_on_acquire
        if setup_on_acquire:
            pool_setup = pool_init
        else:
            pool_setup = _application_name_setup(app_name, None) if app_name else None

        self._pool = await asyncpg.create_pool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            command_timeout=command_timeout,
            statement_cache_size=statement_cache_size,
            timeout=acquire_timeout,
            init=pool_init,
            setup=pool_setup,
        )
        logger.info(
            f"PostgreSQL pool created (min={min_size}, max={max_size}, "
            f"cmd_timeout={command_timeout}s, acquire_timeout={acquire_timeout}s, "
            f"session_setup_on_acquire={setup_on_acquire})"
        )

    async def shutdown(self) -> None:
        # Drop the reference *before* awaiting close(): closing is not
        # instantaneous, and anything acquiring during that window would
        # otherwise get an asyncpg "pool is closing" error rather than seeing
        # is_ready False.
        pool, self._pool = self._pool, None
        if pool is not None:
            await pool.close()
            logger.info("PostgreSQL pool closed")

    @property
    def is_ready(self) -> bool:
        return self._pool is not None

    def _pool_stats(self) -> PoolStats | None:
        """Snapshot for slow-acquire logs. in_use = live connections minus idle ones."""
        pool = self._pool
        if pool is None:
            return None
        idle = pool.get_idle_size()
        return PoolStats(in_use=pool.get_size() - idle, max=pool.get_max_size(), idle=idle)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[PostgresConnection]:
        pool = self._ensure_pool()
        async with instrument_acquire(
            pool.acquire(timeout=self._acquire_timeout_s),
            pool_stats=self._pool_stats,
            warn_threshold_s=self._acquire_warn_threshold_s,
        ) as conn:
            yield PostgresConnection(conn)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[PostgresConnection]:
        pool = self._ensure_pool()
        async with instrument_acquire(
            pool.acquire(timeout=self._acquire_timeout_s),
            pool_stats=self._pool_stats,
            warn_threshold_s=self._acquire_warn_threshold_s,
        ) as conn:
            async with conn.transaction():
                yield PostgresConnection(conn)

    def get_pool(self) -> asyncpg.Pool:
        return self._ensure_pool()

    def _ensure_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgreSQLBackend is not initialized. Call initialize() first.")
        return self._pool
