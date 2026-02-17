"""Azure Blob Storage backend using obstore."""

import logging
from datetime import timedelta

import obstore as obs
from obstore.store import AzureStore

from .base import FileStorage

logger = logging.getLogger(__name__)


class AzureFileStorage(FileStorage):
    """
    Azure Blob Storage backend.

    Uses obstore (Rust-backed) for high-throughput async access to Azure Blob Storage.
    Supports account key, SAS token, and default Azure credentials.
    """

    def __init__(
        self,
        container_name: str,
        account_name: str | None = None,
        account_key: str | None = None,
    ):
        kwargs: dict = {}
        if account_name:
            kwargs["account_name"] = account_name
        if account_key:
            kwargs["account_key"] = account_key

        self._store = AzureStore(container_name, **kwargs)
        logger.info(f"Initialized Azure file storage: container={container_name}, account={account_name}")

    async def store(self, file_data: bytes, key: str, metadata: dict[str, str] | None = None) -> str:
        await obs.put_async(self._store, key, file_data)
        logger.debug(f"Stored file {key} ({len(file_data)} bytes) in Azure")
        return key

    async def retrieve(self, key: str) -> bytes:
        try:
            response = await obs.get_async(self._store, key)
            return await response.bytes_async()
        except Exception as e:
            if "not found" in str(e).lower() or "BlobNotFound" in str(e):
                raise FileNotFoundError(f"File not found: {key}") from e
            raise

    async def delete(self, key: str) -> None:
        await obs.delete_async(self._store, key)

    async def exists(self, key: str) -> bool:
        try:
            await obs.head_async(self._store, key)
            return True
        except Exception:
            return False

    async def get_download_url(self, key: str, expires_in: int = 3600) -> str:
        return await obs.sign_async(self._store, "GET", key, timedelta(seconds=expires_in))
