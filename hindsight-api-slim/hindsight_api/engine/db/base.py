"""Abstract base classes for database backend abstraction.

Defines the interfaces that all database backends (PostgreSQL, Oracle, etc.)
must implement. Business logic depends only on these interfaces.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

# TYPE_CHECKING-only import to avoid circular import at runtime.
# DataAccessOps lives in ops.py which imports nothing from base.py,
# so the cycle is: base -> ops (type-only) and ops -> (nothing from base).
from typing import TYPE_CHECKING, Any

from .result import ResultRow

if TYPE_CHECKING:
    from .ops import DataAccessOps


class DatabaseConnection(ABC):
    """Wraps a single connection from the pool.

    Provides a uniform interface over asyncpg.Connection, oracledb cursor, etc.
    Methods mirror asyncpg's connection API for minimal migration friction.
    """

    @property
    def backend_type(self) -> str:
        """Return ``"postgresql"`` or ``"oracle"``."""
        return "postgresql"

    def parse_json(self, value: Any) -> Any:
        """Parse a JSON column value into a Python object.

        PG (asyncpg) returns JSON columns as strings that need json.loads().
        Oracle returns them as pre-parsed dicts/lists (via OracleConnection
        row conversion). This method normalizes both to Python objects.
        """
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        # Already a dict/list (Oracle pre-parses JSON columns)
        return value

    async def bulk_insert_from_arrays(
        self,
        table: str,
        columns: list[str],
        arrays: list[list],
        *,
        column_types: list[str] | None = None,
        returning: str | None = None,
    ) -> list[ResultRow] | str:
        """Insert multiple rows from parallel arrays.

        Default implementation uses ``INSERT ... SELECT * FROM unnest(...)``
        (PostgreSQL).  Oracle overrides this with ``executemany``.

        Args:
            table: Fully-qualified table name.
            columns: Column names matching the arrays.
            arrays: Parallel lists of values, one per column.
            column_types: PG type suffixes for unnest casting (e.g. ``["text[]", "uuid[]"]``).
                          Ignored by backends that don't use unnest.
            returning: Optional column expression for a RETURNING clause.

        Returns:
            If *returning* is set, a list of ResultRow; otherwise a status string.
        """
        # Default: PostgreSQL unnest path
        col_list = ", ".join(columns)
        n_cols = len(columns)
        types = column_types or ["text[]"] * n_cols
        unnest_args = ", ".join(f"${i + 1}::{types[i]}" for i in range(n_cols))
        query = f"INSERT INTO {table} ({col_list}) SELECT * FROM unnest({unnest_args})"
        if returning:
            query += f" RETURNING {returning}"
            return await self.fetch(query, *arrays)
        result = await self.execute(query, *arrays)
        return result

    @abstractmethod
    @asynccontextmanager
    async def transaction(self) -> AsyncIterator["DatabaseConnection"]:
        """Start a transaction (or savepoint if already in a transaction).

        Yields:
            Self — the same connection, now inside a transaction scope.
            On clean exit the transaction is committed; on exception it is rolled back.
        """
        ...  # pragma: no cover
        yield  # type: ignore[misc]

    @abstractmethod
    async def execute(self, query: str, *args: Any, timeout: float | None = None) -> str:
        """Execute a query and return a status string (e.g. 'INSERT 0 1').

        Args:
            query: SQL query with dialect-appropriate placeholders.
            *args: Positional bind parameters.
            timeout: Optional statement timeout in seconds.

        Returns:
            Command status string.
        """
        ...

    @abstractmethod
    async def executemany(self, query: str, args: list[tuple[Any, ...]], *, timeout: float | None = None) -> None:
        """Execute a query for each set of arguments.

        Args:
            query: SQL query with dialect-appropriate placeholders.
            args: List of argument tuples, one per execution.
            timeout: Optional statement timeout in seconds.
        """
        ...

    @abstractmethod
    async def fetch(self, query: str, *args: Any, timeout: float | None = None) -> list[ResultRow]:
        """Execute a query and return all rows.

        Args:
            query: SQL query with dialect-appropriate placeholders.
            *args: Positional bind parameters.
            timeout: Optional statement timeout in seconds.

        Returns:
            List of ResultRow objects.
        """
        ...

    @abstractmethod
    async def fetchrow(self, query: str, *args: Any, timeout: float | None = None) -> ResultRow | None:
        """Execute a query and return a single row (or None).

        Args:
            query: SQL query with dialect-appropriate placeholders.
            *args: Positional bind parameters.
            timeout: Optional statement timeout in seconds.

        Returns:
            A single ResultRow, or None if no rows match.
        """
        ...

    @abstractmethod
    async def fetchval(self, query: str, *args: Any, column: int = 0, timeout: float | None = None) -> Any:
        """Execute a query and return a single value from the first row.

        Args:
            query: SQL query with dialect-appropriate placeholders.
            *args: Positional bind parameters.
            column: Column index to return (default 0).
            timeout: Optional statement timeout in seconds.

        Returns:
            The value from the specified column of the first row, or None.
        """
        ...

    async def copy_records_to_table(
        self,
        table_name: str,
        *,
        records: list[tuple[Any, ...]],
        columns: list[str],
        timeout: float | None = None,
    ) -> None:
        """Bulk-load records into a table.

        Default implementation uses executemany INSERT. Backends with native
        bulk-load support (e.g. asyncpg COPY) should override for performance.
        """
        cols = ", ".join(columns)
        placeholders = ", ".join(f"${i + 1}" for i in range(len(columns)))
        query = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"
        await self.executemany(query, list(records))


class DatabaseBackend(ABC):
    """Database pool lifecycle and connection acquisition.

    Manages the connection pool and provides context managers for
    acquiring connections and running transactions.

    The ``ops`` property provides backend-specific data access operations
    (the Strategy pattern — like Django's ``connection.ops``). All business
    logic should use ``backend.ops`` instead of creating DataAccessOps
    instances directly.
    """

    _ops_instance: "DataAccessOps | None" = None

    # -- Backend capabilities --------------------------------------------
    # Subclasses override these to advertise what the platform supports.
    # Callers use these instead of checking ``config.database_backend``.

    @property
    def backend_type(self) -> str:
        """Return ``"postgresql"`` or ``"oracle"``."""
        return "postgresql"

    @property
    def ops(self) -> "DataAccessOps":
        """Backend-specific data access operations (cached).

        Follows the Django pattern: ``connection.ops`` provides the
        operations handler for the current backend. Created lazily on
        first access and cached for the lifetime of the backend.
        """
        if self._ops_instance is None:
            from . import create_data_access_ops

            self._ops_instance = create_data_access_ops(self.backend_type)
        return self._ops_instance

    @property
    def supports_partial_indexes(self) -> bool:
        """Can CREATE INDEX … WHERE <predicate>."""
        return True

    @property
    def supports_bm25(self) -> bool:
        """Has BM25 / tsvector full-text search."""
        return True

    @property
    def supports_unnest(self) -> bool:
        """Supports ``unnest()`` for expanding arrays into rows."""
        return True

    @property
    def supports_pg_trgm(self) -> bool:
        """Platform *might* have pg_trgm (must still be checked at runtime)."""
        return True

    @property
    def supports_worker_poller(self) -> bool:
        """Whether this backend supports the async WorkerPoller.

        WorkerPoller is backend-agnostic (uses DatabaseBackend.acquire()).
        All current backends (PostgreSQL, Oracle) support it.
        """
        return True

    def normalize_schema(self, schema: str | None) -> str | None:
        """Normalize a schema name for this backend.

        Returns the schema as-is by default. Oracle overrides this to
        convert ``"public"`` (a PG-specific default) to ``None`` (use the
        connecting user's default schema).
        """
        return schema

    def run_migrations(self, dsn: str, *, schema: str | None = None) -> None:
        """Run database migrations for this backend.

        PG uses Alembic migrations. Oracle uses its own idempotent DDL runner.
        Subclasses must override this method.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement run_migrations()")

    def create_task_backend(self, *, pool_getter: Any = None, schema_getter: Any = None) -> Any:
        """Create the task backend for this database.

        All backends use BrokerTaskBackend for async worker/poller execution.
        """
        from ..task_backend import BrokerTaskBackend

        return BrokerTaskBackend(pool_getter=pool_getter, schema_getter=schema_getter)

    @abstractmethod
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
        """Create the connection pool.

        Args:
            dsn: Database connection string.
            min_size: Minimum number of connections in the pool.
            max_size: Maximum number of connections in the pool.
            command_timeout: Default command timeout in seconds.
            acquire_timeout: Timeout for acquiring a connection from the pool.
            statement_cache_size: Size of the prepared-statement cache (0 to disable).
            init_callback: Optional async callback invoked on each new connection.
        """
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Close the connection pool and release all resources."""
        ...

    @abstractmethod
    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[DatabaseConnection]:
        """Acquire a connection from the pool.

        Yields:
            A DatabaseConnection wrapper.
        """
        ...  # pragma: no cover
        yield  # type: ignore[misc]

    @abstractmethod
    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[DatabaseConnection]:
        """Acquire a connection and start a transaction.

        The transaction is committed on clean exit, rolled back on exception.

        Yields:
            A DatabaseConnection wrapper inside a transaction.
        """
        ...  # pragma: no cover
        yield  # type: ignore[misc]

    @abstractmethod
    def get_pool(self) -> Any:
        """Return the underlying raw pool object.

        Escape hatch for gradual migration — callers that still need direct
        pool access (e.g. asyncpg-specific features) can use this during
        the transition period.
        """
        ...
