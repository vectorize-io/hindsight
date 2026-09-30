"""PostgreSQL BYTEA-based file storage (default, zero-config)."""

import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel

from ..db_utils import acquire_with_retry
from ..schema import fq_table_explicit as fq_table
from .base import FileStorage

logger = logging.getLogger(__name__)

PG_STREAM_CHUNK_SIZE = 4 * 1024 * 1024  # 4MB chunks for streaming storage
# 36-byte magic signature prepended to the manifest row for chunked files.
# A file starting with these exact bytes is treated as a chunked manifest.
# Collision probability with valid user binary documents or text is virtually zero.
_CHUNKED_MANIFEST_SIGNATURE = b"HINDSIGHT_STORAGE_CHUNK_MANIFEST_V1\n"


class ChunkedStorageManifest(BaseModel):
    """Manifest describing a chunked file storage stream in PostgreSQL."""

    format: str = "chunked_v1"
    chunks: int
    total_bytes: int
    chunk_size: int = PG_STREAM_CHUNK_SIZE


@dataclass(frozen=True)
class _StoredFile:
    """What one probe of a stored key tells the readers: its row length and, if chunked, its manifest."""

    length: int
    manifest: ChunkedStorageManifest | None
    oracle: bool


def _chunk_key(key: str, index: int) -> str:
    return f"{key}#chunk={index:06d}"


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

    async def _upsert(self, conn: Any, key: str, data: bytes) -> None:
        await conn.execute(
            f"""
            INSERT INTO {fq_table("file_storage", self._schema)}
            (storage_key, data)
            VALUES ($1, $2)
            ON CONFLICT (storage_key) DO UPDATE SET
                data = EXCLUDED.data
            """,
            key,
            data,
        )

    async def _probe(self, conn: Any, key: str) -> _StoredFile | None:
        """Read a key's length and signature without pulling its bytes.

        Only a chunked file's manifest row (a few dozen bytes) is then read in full.
        """
        sig_len = len(_CHUNKED_MANIFEST_SIGNATURE)
        # The admin CLI hands this storage a raw asyncpg pool, whose connections carry no dialect.
        oracle = getattr(conn, "backend_type", "postgresql") == "oracle"
        if oracle:
            head = f"DBMS_LOB.GETLENGTH(data) AS len, DBMS_LOB.SUBSTR(data, {sig_len}, 1) AS sig"
        else:
            head = f"length(data) AS len, substring(data from 1 for {sig_len}) AS sig"
        row = await conn.fetchrow(
            f"SELECT {head} FROM {fq_table('file_storage', self._schema)} WHERE storage_key = $1",
            key,
        )
        if not row:
            return None
        manifest = None
        if row["sig"] is not None and bytes(row["sig"]) == _CHUNKED_MANIFEST_SIGNATURE:
            data = await conn.fetchval(
                f"SELECT data FROM {fq_table('file_storage', self._schema)} WHERE storage_key = $1",
                key,
            )
            try:
                manifest = ChunkedStorageManifest.model_validate_json(bytes(data)[sig_len:])
            except ValueError:
                # A row that only looks like a manifest: treat it as the plain bytes it
                # is, so it can still be read back and deleted rather than wedging both.
                logger.warning("Unparseable chunk manifest at %s; treating it as a plain file", key)
        return _StoredFile(length=row["len"], manifest=manifest, oracle=oracle)

    async def _delete_with_chunks(self, conn: Any, key: str, *, keep_key: bool = False) -> bool:
        """Delete ``key`` and, when it is a chunked file, every chunk row its manifest names.

        ``keep_key`` removes only the chunk rows: a write about to replace ``key``
        calls this first, since overwriting the manifest row alone would strand the
        chunks of the previous version under keys nothing references any more.
        Returns whether ``key`` existed.
        """
        stored = await self._probe(conn, key)
        if stored is None:
            return False
        keys = [] if keep_key else [key]
        if stored.manifest is not None:
            keys.extend(_chunk_key(key, i) for i in range(stored.manifest.chunks))
        if keys:
            await conn.execute(
                f"DELETE FROM {fq_table('file_storage', self._schema)} WHERE storage_key = ANY($1)",
                keys,
            )
        return True

    async def store(
        self,
        file_data: bytes,
        key: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Store file in PostgreSQL."""
        pool = self._pool_getter()

        async with acquire_with_retry(pool) as conn:
            await self._delete_with_chunks(conn, key, keep_key=True)
            await self._upsert(conn, key, file_data)

        logger.debug(f"Stored file {key} ({len(file_data)} bytes) in PostgreSQL")
        return key

    async def store_stream(
        self,
        key: str,
        stream: AsyncIterator[bytes],
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Store a byte stream as 4 MiB chunk rows plus a manifest row under ``key``.

        A stream that ends under one chunk is stored as a plain single row, exactly
        what :meth:`store` would write. Only one chunk is buffered at a time, and a
        connection is held only for each write, never across the producer's awaits.
        """
        pool = self._pool_getter()
        chunk_idx = 0
        total_bytes = 0
        buffer = bytearray()

        async with acquire_with_retry(pool) as conn:
            await self._delete_with_chunks(conn, key, keep_key=True)
        try:
            async for chunk in stream:
                buffer.extend(chunk)
                while len(buffer) >= PG_STREAM_CHUNK_SIZE:
                    async with acquire_with_retry(pool) as conn:
                        await self._upsert(conn, _chunk_key(key, chunk_idx), bytes(buffer[:PG_STREAM_CHUNK_SIZE]))
                    del buffer[:PG_STREAM_CHUNK_SIZE]
                    total_bytes += PG_STREAM_CHUNK_SIZE
                    chunk_idx += 1

            if chunk_idx == 0:
                async with acquire_with_retry(pool) as conn:
                    await self._upsert(conn, key, bytes(buffer))
                logger.debug(f"Stored stream {key} ({len(buffer)} bytes) in PostgreSQL as single row")
                return key

            if buffer:
                async with acquire_with_retry(pool) as conn:
                    await self._upsert(conn, _chunk_key(key, chunk_idx), bytes(buffer))
                total_bytes += len(buffer)
                chunk_idx += 1

            manifest = ChunkedStorageManifest(chunks=chunk_idx, total_bytes=total_bytes)
            async with acquire_with_retry(pool) as conn:
                await self._upsert(conn, key, _CHUNKED_MANIFEST_SIGNATURE + manifest.model_dump_json().encode("utf-8"))
            logger.debug(f"Stored chunked stream {key} ({total_bytes} bytes in {chunk_idx} chunks) in PostgreSQL")
            return key
        except BaseException:
            # No manifest points at the chunks written so far, so nothing else would
            # ever find them: remove them (and a single-row write, if one landed).
            try:
                async with acquire_with_retry(pool) as conn:
                    await conn.execute(
                        f"DELETE FROM {fq_table('file_storage', self._schema)} WHERE storage_key = ANY($1)",
                        [key] + [_chunk_key(key, i) for i in range(chunk_idx + 1)],
                    )
            except Exception:
                logger.warning("Failed to clean up partially stored stream %s", key, exc_info=True)
            raise

    async def retrieve(self, key: str) -> bytes:
        """Retrieve file from PostgreSQL."""
        pool = self._pool_getter()

        async with acquire_with_retry(pool) as conn:
            row = await conn.fetchrow(
                f"""
                SELECT data FROM {fq_table("file_storage", self._schema)}
                WHERE storage_key = $1
                """,
                key,
            )

        if not row:
            raise FileNotFoundError(f"File not found: {key}")

        data = bytes(row["data"])
        if data.startswith(_CHUNKED_MANIFEST_SIGNATURE):
            return b"".join([chunk async for chunk in self.retrieve_stream(key)])
        return data

    async def retrieve_stream(self, key: str) -> AsyncIterator[bytes]:
        """Stream a file at most 4 MiB at a time.

        A chunked file is read one chunk row per query. A plain single-row file (every
        attachment written by :meth:`store`) is read in ``substring`` ranges of the
        same size, so a large attachment never has to sit in memory whole. Each range
        re-reads the row from TOAST: an incompressible blob (images, PDFs) is stored
        uncompressed and sliced directly; a compressible one is decompressed up to
        the range's end on every read, which is CPU the 4 MiB range size keeps to a
        few dozen passes over even a large file. Oracle has no SQL form that returns
        more than 32 KB of a BLOB, so it still reads a plain row in one query.
        """
        pool = self._pool_getter()
        table = fq_table("file_storage", self._schema)
        async with acquire_with_retry(pool) as conn:
            stored = await self._probe(conn, key)
        if stored is None:
            raise FileNotFoundError(f"File not found: {key}")

        if stored.manifest is not None:
            for i in range(stored.manifest.chunks):
                async with acquire_with_retry(pool) as conn:
                    data = await conn.fetchval(f"SELECT data FROM {table} WHERE storage_key = $1", _chunk_key(key, i))
                if data is None:
                    raise FileNotFoundError(f"Missing chunk {i} for {key}")
                yield bytes(data)
        elif stored.oracle:
            async with acquire_with_retry(pool) as conn:
                data = await conn.fetchval(f"SELECT data FROM {table} WHERE storage_key = $1", key)
            if data is None:
                raise FileNotFoundError(f"File not found: {key}")
            yield bytes(data)
        else:
            for offset in range(0, stored.length, PG_STREAM_CHUNK_SIZE):
                async with acquire_with_retry(pool) as conn:
                    data = await conn.fetchval(
                        f"SELECT substring(data from $2 for $3) FROM {table} WHERE storage_key = $1",
                        key,
                        offset + 1,
                        PG_STREAM_CHUNK_SIZE,
                    )
                if data is None:
                    raise FileNotFoundError(f"File not found: {key}")
                yield bytes(data)

    async def delete(self, key: str) -> None:
        """Delete file and any chunk rows from PostgreSQL."""
        pool = self._pool_getter()
        async with acquire_with_retry(pool) as conn:
            if not await self._delete_with_chunks(conn, key):
                logger.warning(f"Attempted to delete non-existent file: {key}")

    async def delete_prefix(self, prefix: str) -> int:
        """Delete every file under ``prefix`` in PostgreSQL."""
        pool = self._pool_getter()
        # Escape LIKE's wildcards: keys carry percent-encoded segments.
        pattern = prefix.replace("!", "!!").replace("%", "!%").replace("_", "!_") + "%"
        async with acquire_with_retry(pool) as conn:
            result = await conn.execute(
                f"DELETE FROM {fq_table('file_storage', self._schema)} WHERE storage_key LIKE $1 ESCAPE '!'",
                pattern,
            )
        return int(result.split()[-1]) if isinstance(result, str) else 0

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
        # Quoted so a key's own percent-encoded segments survive the server's
        # path decoding and arrive back as the stored key.
        return f"/v1/default/files/download/{quote(key)}"

    async def get_size(self, key: str) -> int | None:
        """Return the file's size in bytes, or ``None`` if it does not exist."""
        pool = self._pool_getter()
        async with acquire_with_retry(pool) as conn:
            stored = await self._probe(conn, key)
        if stored is None:
            return None
        return stored.manifest.total_bytes if stored.manifest is not None else stored.length
