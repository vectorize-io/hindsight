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
