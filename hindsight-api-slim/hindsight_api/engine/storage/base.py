"""Abstract base class for file storage backends."""

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import obstore as obs

logger = logging.getLogger(__name__)


class FileStorage(ABC):
    """Abstract base for file storage backends."""

    @abstractmethod
    async def store(
        self,
        file_data: bytes,
        key: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """
        Store file and return storage key.

        Args:
            file_data: Raw file bytes
            key: Storage key (e.g., "banks/{bank_id}/files/{file_id}.pdf")
            metadata: Optional metadata to store with file

        Returns:
            Storage key that can be used to retrieve the file
        """
        pass

    async def store_stream(
        self,
        key: str,
        stream: AsyncIterator[bytes],
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Store file from an async byte stream.

        Default implementation buffers stream into memory and calls :meth:`store`.
        Streaming-capable backends override this so a large file is never held whole.
        """
        chunks: list[bytes] = []
        async for chunk in stream:
            if chunk:
                chunks.append(chunk)
        return await self.store(b"".join(chunks), key, metadata=metadata)

    @abstractmethod
    async def retrieve(self, key: str) -> bytes:
        """
        Retrieve file by storage key.

        Args:
            key: Storage key

        Returns:
            File data as bytes

        Raises:
            FileNotFoundError: If file does not exist
        """
        pass

    async def retrieve_stream(self, key: str) -> AsyncIterator[bytes]:
        """Retrieve file as an async stream of bytes.

        Default implementation calls :meth:`retrieve` and yields the full bytes.
        Streaming-capable backends override this so a large file is never held whole.
        """
        data = await self.retrieve(key)
        yield data

    @abstractmethod
    async def delete(self, key: str) -> None:
        """
        Delete file by storage key.

        Args:
            key: Storage key
        """
        pass

    async def delete_prefix(self, prefix: str) -> int:
        """Delete every file whose key starts with ``prefix``; return how many.

        How a bank's files go when the bank does: its keys share one prefix
        (see :func:`~hindsight_api.engine.storage.bank_storage_prefix`), so the
        sweep needs no row to enumerate them. Not abstract, so a backend written
        before this existed still loads; callers treat it as best-effort.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support delete_prefix")

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if file exists.

        Args:
            key: Storage key

        Returns:
            True if file exists, False otherwise
        """
        pass

    @abstractmethod
    async def get_download_url(self, key: str, expires_in: int = 3600) -> str:
        """
        Get a URL for downloading the file.

        For PostgreSQL storage, this might be a relative API path.
        For S3, this would be a pre-signed URL.

        Args:
            key: Storage key
            expires_in: Expiration time in seconds (may be ignored for some backends)

        Returns:
            Download URL or path
        """
        pass

    async def get_size(self, key: str) -> int | None:
        """Return the size of the file in bytes if known/supported, or None."""
        return None


async def delete_object_store_prefix(store, prefix: str) -> int:
    """:meth:`FileStorage.delete_prefix` for the obstore-backed stores (S3, GCS, Azure)."""
    deleted = 0
    # One page per bulk delete: S3's DeleteObjects takes up to 1000 keys.
    async for page in obs.list(store, prefix=prefix, chunk_size=1000):
        await obs.delete_async(store, [meta["path"] for meta in page])
        deleted += len(page)
    return deleted


def _is_not_found(error: Exception) -> bool:
    """obstore raises one generic error type; S3, GCS and Azure each word a missing key differently."""
    message = str(error).lower()
    return "not found" in message or "nosuchkey" in message or "blobnotfound" in message


class ObstoreFileStorage(FileStorage):
    """Shared base class for obstore-backed object storage backends (S3, GCS, Azure)."""

    _store: Any

    async def store(
        self,
        file_data: bytes,
        key: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        await obs.put_async(self._store, key, file_data)
        return key

    async def store_stream(
        self,
        key: str,
        stream: AsyncIterator[bytes],
        metadata: dict[str, str] | None = None,
    ) -> str:
        writer = obs.open_writer_async(self._store, key)
        try:
            async for chunk in stream:
                if chunk:
                    await writer.write(chunk)
            await writer.close()
        except BaseException:
            # Do NOT call writer.close() on failure as it commits the truncated object.
            # Note: obstore's writer does not expose an explicit abort() method, so abandoning
            # the writer leaves uncommitted multipart upload parts in the target bucket.
            # Attempting delete_async(key) cleans up any completed/overwritten object that
            # may exist, but cloud storage buckets (AWS S3, GCS, Azure) MUST configure an
            # AbortIncompleteMultipartUpload lifecycle rule (e.g. 7 days) to automatically
            # purge orphaned multipart parts.
            try:
                await obs.delete_async(self._store, key)
            except Exception:
                logger.warning("Failed to delete partially uploaded object %s from storage", key, exc_info=True)
            raise
        return key

    async def retrieve(self, key: str) -> bytes:
        try:
            response = await obs.get_async(self._store, key)
            # obstore returns its own Bytes buffer, not a Python `bytes`. Callers rely on
            # the `-> bytes` return type, and handing a non-`bytes` to something strict about
            # the type (e.g. a Starlette Response, whose render() calls `.encode()` on a
            # non-`bytes`) fails at runtime. Copy into native bytes so the backend honours
            # its declared contract.
            return bytes(await response.bytes_async())
        except Exception as e:
            if _is_not_found(e):
                raise FileNotFoundError(f"File not found: {key}") from e
            raise

    async def retrieve_stream(self, key: str) -> AsyncIterator[bytes]:
        try:
            response = await obs.get_async(self._store, key)
            async for chunk in response.stream():
                yield bytes(chunk)
        except Exception as e:
            if _is_not_found(e):
                raise FileNotFoundError(f"File not found: {key}") from e
            raise

    async def delete(self, key: str) -> None:
        await obs.delete_async(self._store, key)

    async def delete_prefix(self, prefix: str) -> int:
        return await delete_object_store_prefix(self._store, prefix)

    async def exists(self, key: str) -> bool:
        try:
            await obs.head_async(self._store, key)
            return True
        except Exception:
            return False

    async def get_download_url(self, key: str, expires_in: int = 3600) -> str:
        return await obs.sign_async(self._store, "GET", key, timedelta(seconds=expires_in))

    async def get_size(self, key: str) -> int | None:
        try:
            head = await obs.head_async(self._store, key)
        except Exception as e:
            if _is_not_found(e):
                return None
            raise
        return head["size"]
