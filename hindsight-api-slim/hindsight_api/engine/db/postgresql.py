"""PostgreSQL backend implementation using asyncpg.

Wraps asyncpg's pool and connection objects behind the DatabaseBackend
and DatabaseConnection interfaces.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import asyncpg  # noqa: F401

from .base import DatabaseBackend, DatabaseConnection
from .result import ResultRow

logger = logging.getLogger(__name__)


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

    async def fetch(self, query: str, *args: Any, timeout: float | None = None) -> list[ResultRow]:
        rows = await self._conn.fetch(query, *args, timeout=timeout)
        return [ResultRow(row) for row in rows]

    async def fetchrow(self, query: str, *args: Any, timeout: float | None = None) -> ResultRow | None:
        row = await self._conn.fetchrow(query, *args, timeout=timeout)
        if row is None:
            return None
        return ResultRow(row)

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
        self._pool = await asyncpg.create_pool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            command_timeout=command_timeout,
            statement_cache_size=statement_cache_size,
            timeout=acquire_timeout,
            init=init_callback,
        )
        logger.info(
            f"PostgreSQL pool created (min={min_size}, max={max_size}, "
            f"cmd_timeout={command_timeout}s, acquire_timeout={acquire_timeout}s)"
        )

    async def shutdown(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL pool closed")

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[PostgresConnection]:
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            yield PostgresConnection(conn)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[PostgresConnection]:
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                yield PostgresConnection(conn)

    def get_pool(self) -> asyncpg.Pool:
        return self._ensure_pool()

    def _ensure_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgreSQLBackend is not initialized. Call initialize() first.")
        return self._pool
