"""Database backend abstraction layer.

Provides a uniform interface over different database drivers (asyncpg, oracledb, etc.)
so that business logic is decoupled from any specific database platform.

Usage:
    from hindsight_api.engine.db import create_database_backend, DatabaseBackend

    backend = create_database_backend("postgresql")
    await backend.initialize(dsn="postgresql://...")
    async with backend.acquire() as conn:
        rows = await conn.fetch("SELECT ...")
"""

from .base import DatabaseBackend, DatabaseConnection
from .ops import DataAccessOps
from .result import ResultRow

__all__ = [
    "DataAccessOps",
    "DatabaseBackend",
    "DatabaseConnection",
    "ResultRow",
    "create_data_access_ops",
    "create_database_backend",
]


def _get_backend_class(backend_type: str) -> type[DatabaseBackend]:
    """Resolve backend class by name using lazy imports."""
    if backend_type == "postgresql":
        from .postgresql import PostgreSQLBackend

        return PostgreSQLBackend
    if backend_type == "oracle":
        from .oracle import OracleBackend

        return OracleBackend
    raise ValueError(f"Unknown database backend: {backend_type!r}. Supported: 'postgresql', 'oracle'.")


def _get_ops_class(backend_type: str) -> type[DataAccessOps]:
    """Resolve ops class by name using lazy imports."""
    if backend_type == "postgresql":
        from .ops_postgresql import PostgreSQLOps

        return PostgreSQLOps
    if backend_type == "oracle":
        from .ops_oracle import OracleOps

        return OracleOps
    raise ValueError(f"Unknown data access ops: {backend_type!r}. Supported: 'postgresql', 'oracle'.")


def create_database_backend(backend_type: str) -> DatabaseBackend:
    """Factory: create a DatabaseBackend by name.

    Args:
        backend_type: One of "postgresql" or "oracle".

    Returns:
        An uninitialized DatabaseBackend instance.

    Raises:
        ValueError: If backend_type is not recognized.
    """
    return _get_backend_class(backend_type)()


_OPS_CACHE: dict[str, DataAccessOps] = {}


def create_data_access_ops(backend_type: str) -> DataAccessOps:
    """Factory: the DataAccessOps for a backend name.

    Returns a per-dialect SINGLETON: ``DataAccessOps`` is stateless (it only builds and runs SQL),
    so one shared instance per dialect is correct — and it means the database backend and the
    memories store hold the *same* ops object, so a test that patches a method on it (e.g.
    ``enqueue_graph_maintenance``) observes every caller regardless of which layer issued it.

    Args:
        backend_type: One of "postgresql" or "oracle".

    Returns:
        The shared DataAccessOps instance for that backend.

    Raises:
        ValueError: If backend_type is not recognized.
    """
    ops = _OPS_CACHE.get(backend_type)
    if ops is None:
        ops = _get_ops_class(backend_type)()
        _OPS_CACHE[backend_type] = ops
    return ops
