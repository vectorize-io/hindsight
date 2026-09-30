"""Azure Blob Storage backend using obstore."""

import logging

from obstore.store import AzureStore

from .base import ObstoreFileStorage

logger = logging.getLogger(__name__)


class AzureFileStorage(ObstoreFileStorage):
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
