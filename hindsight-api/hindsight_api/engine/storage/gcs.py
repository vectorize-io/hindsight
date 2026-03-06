"""Google Cloud Storage backend using obstore."""

import logging
import os
from datetime import datetime, timedelta, timezone

import obstore as obs
from obstore.store import GCSStore

from .base import FileStorage

logger = logging.getLogger(__name__)


def _make_google_auth_credential_provider():
    """Create a credential provider using google.auth (supports all credential types).

    obstore's built-in credential parsing only supports service_account and
    authorized_user JSON types.  This provider uses the google-auth library
    which additionally handles external_account (Workload Identity Federation),
    impersonated credentials, and metadata-server credentials.
    """
    import google.auth
    import google.auth.transport.requests

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    request = google.auth.transport.requests.Request()

    def _provide():
        credentials.refresh(request)
        expiry = credentials.expiry
        if expiry and expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        return {"token": credentials.token, "expires_at": expiry}

    return _provide


class GCSFileStorage(FileStorage):
    """
    Google Cloud Storage backend.

    Uses obstore (Rust-backed) for high-throughput async access to GCS.
    Supports Application Default Credentials, service account keys, and explicit credentials.
    """

    def __init__(
        self,
        bucket: str,
        service_account_key: str | None = None,
    ):
        kwargs: dict = {}
        if service_account_key:
            kwargs["service_account_key"] = service_account_key
        else:
            # Use google.auth credential provider for broad credential type support
            # (service_account, authorized_user, external_account, metadata server, etc.)
            try:
                kwargs["credential_provider"] = _make_google_auth_credential_provider()
                logger.info("Using google.auth credential provider for GCS")
            except Exception as e:
                logger.warning(
                    f"Failed to create google.auth credential provider, falling back to obstore defaults: {e}"
                )

        # When using a custom credential_provider, temporarily hide
        # GOOGLE_APPLICATION_CREDENTIALS so GCSStore doesn't try to parse
        # credential types it doesn't understand (e.g. external_account).
        saved_creds = None
        if "credential_provider" in kwargs and "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            saved_creds = os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS")

        try:
            self._store = GCSStore(bucket, **kwargs)
        finally:
            if saved_creds is not None:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = saved_creds
        logger.info(f"Initialized GCS file storage: bucket={bucket}")

    async def store(self, file_data: bytes, key: str, metadata: dict[str, str] | None = None) -> str:
        await obs.put_async(self._store, key, file_data)
        logger.debug(f"Stored file {key} ({len(file_data)} bytes) in GCS")
        return key

    async def retrieve(self, key: str) -> bytes:
        try:
            response = await obs.get_async(self._store, key)
            return await response.bytes_async()
        except Exception as e:
            if "not found" in str(e).lower():
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
