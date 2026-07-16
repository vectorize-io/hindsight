"""Abstract base class for file storage backends."""

import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True)
class FileObjectInfo:
    """Metadata needed to serve an object without exposing a storage URL."""

    size_bytes: int
    etag: str | None = None


def is_not_found_error(exc: Exception) -> bool:
    """Conservatively classify object-store not-found errors.

    Permission, timeout, and transport failures must not be presented as a
    missing object because callers use that distinction for reconciliation.
    """
    message = str(exc).lower()
    return any(marker in message for marker in ("not found", "nosuchkey", "blobnotfound", "404"))


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

    @abstractmethod
    async def delete(self, key: str) -> None:
        """
        Delete file by storage key.

        Args:
            key: Storage key
        """
        pass

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

    async def stat(self, key: str) -> FileObjectInfo:
        """Return object metadata.

        Backends may override this with a native HEAD query. The fallback keeps
        third-party FileStorage implementations source compatible while the
        image API has a strict per-object size limit.
        """
        data = await self.retrieve(key)
        return FileObjectInfo(size_bytes=len(data))

    async def store_stream(
        self,
        chunks: AsyncIterator[bytes],
        key: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Adapt an async byte stream to the legacy byte-oriented store method.

        The compatibility implementation bounds ingestion memory by spooling to
        disk, but the final legacy ``store`` call still materializes the complete
        object. Built-in backends override this method for end-to-end streaming.
        """
        path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False) as output:
                path = output.name
                async for chunk in chunks:
                    output.write(chunk)
            with open(path, "rb") as source:
                return await self.store(source.read(), key, metadata)
        finally:
            if path:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass

    async def iter_bytes(self, key: str, chunk_size: int = 1024 * 1024) -> AsyncIterator[bytes]:
        """Yield an object in bounded chunks without creating a download URL."""
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        data = await self.retrieve(key)
        for offset in range(0, len(data), chunk_size):
            yield data[offset : offset + chunk_size]
