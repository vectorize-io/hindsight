"""PostgreSQL BYTEA-based file storage (default, zero-config)."""

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from ..db_utils import acquire_with_retry
from ..schema import _is_oracle
from ..schema import fq_table_explicit as fq_table
from .base import FileObjectInfo, FileStorage

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        pool_getter: Callable[[], Any],
        schema: str | None = None,
        schema_getter: Callable[[], str] | None = None,
    ):
        """
        Initialize PostgreSQL file storage.

        Args:
            pool_getter: Function that returns asyncpg connection pool
            schema: Static database schema (fallback for single-tenant / tests)
            schema_getter: Callable returning current schema at query time (for multi-tenant)
        """
        self._pool_getter = pool_getter
        self._static_schema = schema
        self._schema_getter = schema_getter

    @property
    def _schema(self) -> str | None:
        """Resolve schema dynamically per-request when schema_getter is provided."""
        if self._schema_getter:
            return self._schema_getter()
        return self._static_schema

    async def store(
        self,
        file_data: bytes,
        key: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Store file in PostgreSQL."""
        pool = self._pool_getter()

        async with acquire_with_retry(pool) as conn:
            async with conn.transaction():
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
                await conn.execute(
                    f"DELETE FROM {fq_table('file_storage_chunks', self._schema)} WHERE storage_key = $1", key
                )

        logger.debug(f"Stored file {key} ({len(file_data)} bytes) in PostgreSQL")
        return key

    async def retrieve(self, key: str) -> bytes:
        """Retrieve file from PostgreSQL."""
        pool = self._pool_getter()

        async with acquire_with_retry(pool) as conn:
            chunks = await conn.fetch(
                f"SELECT data FROM {fq_table('file_storage_chunks', self._schema)} "
                "WHERE storage_key = $1 ORDER BY ordinal",
                key,
            )
            if chunks:
                return b"".join(bytes(chunk["data"]) for chunk in chunks)
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

    async def store_stream(
        self,
        chunks: AsyncIterator[bytes],
        key: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Persist chunks incrementally; the empty parent keeps legacy FK/exists semantics."""
        pool = self._pool_getter()
        async with acquire_with_retry(pool) as conn:
            async with conn.transaction():
                await conn.execute(
                    f"INSERT INTO {fq_table('file_storage', self._schema)} (storage_key, data) VALUES ($1, $2) "
                    "ON CONFLICT (storage_key) DO UPDATE SET data = EXCLUDED.data",
                    key,
                    b"",
                )
                await conn.execute(
                    f"DELETE FROM {fq_table('file_storage_chunks', self._schema)} WHERE storage_key = $1", key
                )
                ordinal = 0
                async for chunk in chunks:
                    if not chunk:
                        continue
                    await conn.execute(
                        f"INSERT INTO {fq_table('file_storage_chunks', self._schema)} "
                        "(storage_key, ordinal, data) VALUES ($1, $2, $3)",
                        key,
                        ordinal,
                        chunk,
                    )
                    ordinal += 1
        return key

    async def delete(self, key: str) -> None:
        """Delete file from PostgreSQL."""
        pool = self._pool_getter()

        async with acquire_with_retry(pool) as conn:
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

        async with acquire_with_retry(pool) as conn:
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

    async def stat(self, key: str) -> FileObjectInfo:
        """Read the BYTEA length without loading the object into Python."""
        pool = self._pool_getter()
        # Oracle stores native objects as BLOBs and does not implement
        # PostgreSQL's octet_length. DBMS_LOB.GETLENGTH preserves the bounded
        # metadata-only read used for streamed image and transfer archives.
        length_fn = "DBMS_LOB.GETLENGTH" if _is_oracle() else "octet_length"
        async with acquire_with_retry(pool) as conn:
            size = await conn.fetchval(
                f"SELECT COALESCE((SELECT SUM({length_fn}(data)) "
                f"FROM {fq_table('file_storage_chunks', self._schema)} c "
                f"WHERE c.storage_key = f.storage_key), {length_fn}(f.data)) "
                f"FROM {fq_table('file_storage', self._schema)} f WHERE f.storage_key = $1",
                key,
            )
        if size is None:
            raise FileNotFoundError(f"File not found: {key}")
        return FileObjectInfo(size_bytes=int(size))

    async def iter_bytes(self, key: str, chunk_size: int = 1024 * 1024) -> AsyncIterator[bytes]:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        pool = self._pool_getter()
        async with acquire_with_retry(pool) as conn:
            has_chunks = await conn.fetchval(
                f"SELECT 1 FROM {fq_table('file_storage_chunks', self._schema)} WHERE storage_key = $1 FETCH FIRST 1 ROW ONLY",
                key,
            )
            if has_chunks:
                last_ordinal = -1
                while True:
                    rows = await conn.fetch(
                        f"SELECT ordinal, data FROM {fq_table('file_storage_chunks', self._schema)} "
                        "WHERE storage_key = $1 AND ordinal > $2 ORDER BY ordinal FETCH FIRST 16 ROWS ONLY",
                        key,
                        last_ordinal,
                    )
                    if not rows:
                        return
                    for row in rows:
                        last_ordinal = int(row["ordinal"])
                        yield bytes(row["data"])
            row = await conn.fetchrow(
                f"SELECT data FROM {fq_table('file_storage', self._schema)} WHERE storage_key = $1", key
            )
            if row is None:
                raise FileNotFoundError(f"File not found: {key}")
            data = bytes(row["data"])
        for offset in range(0, len(data), chunk_size):
            yield data[offset : offset + chunk_size]
