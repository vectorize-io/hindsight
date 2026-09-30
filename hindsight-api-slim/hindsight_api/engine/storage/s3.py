"""S3 object storage backend using obstore."""

import logging

from obstore.store import S3Store

from .base import ObstoreFileStorage

logger = logging.getLogger(__name__)


class S3FileStorage(ObstoreFileStorage):
    """
    S3-compatible object storage backend.

    Uses obstore (Rust-backed) for high-throughput async access to
    Amazon S3, MinIO, Cloudflare R2, Tigris, and other S3-compliant APIs.
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
