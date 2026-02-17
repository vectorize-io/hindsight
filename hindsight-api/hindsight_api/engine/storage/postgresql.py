"""PostgreSQL BYTEA-based file storage (default, zero-config)."""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg

from .base import FileStorage

logger = logging.getLogger(__name__)


def fq_table(table: str, schema: str | None = None) -> str:
    """Get fully-qualified table name with optional schema prefix."""
    if schema:
        return f'"{schema}".{table}'
    return table


class PostgreSQLFileStorage(FileStorage):
    """
    PostgreSQL BYTEA-based file storage.

    Stores files directly in PostgreSQL using BYTEA columns.
    This is the default storage backend - zero configuration required!

    Pros:
    - Works out of the box (no external dependencies)
    - Transactional consistency with database
    - Simple backups (included in pg_dump)
    - Good performance for <10MB files

    Cons:
    - Database bloat for large/many files
    - Not ideal for distributed deployments
    - Higher cost than object storage at scale

    For production/scale, consider S3FileStorage instead.
    """

    def __init__(self, pool_getter: Callable[[], "asyncpg.Pool"], schema: str | None = None):
        """
        Initialize PostgreSQL file storage.

        Args:
            pool_getter: Function that returns asyncpg connection pool
            schema: Database schema (for multi-tenant support)
        """
        self._pool_getter = pool_getter
        self._schema = schema

    async def store(
        self,
        file_data: bytes,
        key: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Store file in PostgreSQL."""
        pool = self._pool_getter()

        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {fq_table("file_storage", self._schema)}
                (storage_key, data)
                VALUES ($1, $2)
                ON CONFLICT (storage_key) DO UPDATE SET
                    data = EXCLUDED.data
                """,
                key,
                file_data,
            )

        logger.debug(f"Stored file {key} ({len(file_data)} bytes) in PostgreSQL")
        return key

    async def retrieve(self, key: str) -> bytes:
        """Retrieve file from PostgreSQL."""
        pool = self._pool_getter()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT data FROM {fq_table("file_storage", self._schema)}
                WHERE storage_key = $1
                """,
                key,
            )

            if not row:
                raise FileNotFoundError(f"File not found: {key}")

            return bytes(row["data"])

    async def delete(self, key: str) -> None:
        """Delete file from PostgreSQL."""
        pool = self._pool_getter()

        async with pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM {fq_table("file_storage", self._schema)}
                WHERE storage_key = $1
                """,
                key,
            )

            # Check if anything was deleted
            if result == "DELETE 0":
                logger.warning(f"Attempted to delete non-existent file: {key}")

    async def exists(self, key: str) -> bool:
        """Check if file exists in PostgreSQL."""
        pool = self._pool_getter()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT 1 FROM {fq_table("file_storage", self._schema)}
                WHERE storage_key = $1
                """,
                key,
            )

            return row is not None

    async def get_download_url(self, key: str, expires_in: int = 3600) -> str:
        """
        Get download URL for PostgreSQL-stored file.

        Returns an API endpoint path (not a pre-signed URL since the file
        is stored in the database). The expires_in parameter is ignored
        for PostgreSQL storage.
        """
        # Return API path for download endpoint
        # (expires_in ignored for database storage - auth handled at API level)
        return f"/v1/default/files/download/{key}"
