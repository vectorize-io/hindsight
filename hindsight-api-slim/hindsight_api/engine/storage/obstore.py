"""Shared streaming operations for obstore-backed FileStorage providers."""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator
from typing import Any

import obstore as obs

from .base import FileObjectInfo, is_not_found_error


class ObstoreStreamingMixin:
    """Common bounded-memory methods for S3, GCS, and Azure backends."""

    _store: Any

    async def store_stream(
        self,
        chunks: AsyncIterator[bytes],
        key: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        # obstore multipart upload currently accepts a local path, so spool the
        # incoming async stream to disk without materializing it in memory.
        path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False) as output:
                path = output.name
                async for chunk in chunks:
                    output.write(chunk)
            await obs.put_async(self._store, key, path, use_multipart=True)
            return key
        finally:
            if path:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass

    async def stat(self, key: str) -> FileObjectInfo:
        try:
            meta = await obs.head_async(self._store, key)
        except Exception as exc:
            if is_not_found_error(exc):
                raise FileNotFoundError(f"File not found: {key}") from exc
            raise
        return FileObjectInfo(size_bytes=int(meta["size"]), etag=meta.get("e_tag"))

    async def iter_bytes(self, key: str, chunk_size: int = 1024 * 1024) -> AsyncIterator[bytes]:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        try:
            response = await obs.get_async(self._store, key)
            async for chunk in response.stream(min_chunk_size=chunk_size):
                yield chunk
        except Exception as exc:
            if is_not_found_error(exc):
                raise FileNotFoundError(f"File not found: {key}") from exc
            raise
