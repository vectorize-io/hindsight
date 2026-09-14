"""Abstract base class for file storage backends."""

from abc import ABC, abstractmethod


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


async def delete_object_store_prefix(store, prefix: str) -> int:
    """:meth:`FileStorage.delete_prefix` for the obstore-backed stores (S3, GCS, Azure)."""
    import obstore as obs

    deleted = 0
    # One page per bulk delete: S3's DeleteObjects takes up to 1000 keys.
    async for page in obs.list(store, prefix=prefix, chunk_size=1000):
        await obs.delete_async(store, [meta["path"] for meta in page])
        deleted += len(page)
    return deleted
