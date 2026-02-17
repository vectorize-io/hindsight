"""S3 object storage backend using obstore."""

import logging
from datetime import timedelta

import obstore as obs
from obstore.store import S3Store

from .base import FileStorage

logger = logging.getLogger(__name__)


class S3FileStorage(FileStorage):
    """
    S3-compatible object storage backend.

    Uses obstore (Rust-backed) for high-throughput async access to
    Amazon S3, MinIO, Cloudflare R2, and other S3-compliant APIs.
    """

    def __init__(
        self,
        bucket: str,
        region: str | None = None,
        endpoint: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
    ):
        kwargs: dict = {}
        if region:
            kwargs["region"] = region
        if endpoint:
            kwargs["endpoint"] = endpoint
            # Allow plain HTTP for local S3-compatible services (MinIO, LocalStack, etc.)
            if endpoint.startswith("http://"):
                kwargs["allow_http"] = True
        if access_key_id:
            kwargs["access_key_id"] = access_key_id
        if secret_access_key:
            kwargs["secret_access_key"] = secret_access_key

        self._store = S3Store(bucket, **kwargs)
        logger.info(f"Initialized S3 file storage: bucket={bucket}, region={region}, endpoint={endpoint}")

    async def store(self, file_data: bytes, key: str, metadata: dict[str, str] | None = None) -> str:
        await obs.put_async(self._store, key, file_data)
        logger.debug(f"Stored file {key} ({len(file_data)} bytes) in S3")
        return key

    async def retrieve(self, key: str) -> bytes:
        try:
            response = await obs.get_async(self._store, key)
            return await response.bytes_async()
        except Exception as e:
            if "not found" in str(e).lower() or "NoSuchKey" in str(e):
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
