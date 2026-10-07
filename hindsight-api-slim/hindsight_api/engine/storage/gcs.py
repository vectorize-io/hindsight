"""Google Cloud Storage backend using obstore."""

import logging
import os
from datetime import timezone

from obstore.store import GCSStore

from .base import ObstoreFileStorage

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


class GCSFileStorage(ObstoreFileStorage):
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

        # Workaround for https://github.com/developmentseed/obstore/issues/605
        # obstore's Rust layer doesn't support external_account credentials (Workload
        # Identity Federation) and eagerly parses GOOGLE_APPLICATION_CREDENTIALS even
        # when credential_provider is given. Per the obstore maintainer's guidance,
        # remove env vars so the Rust code doesn't try to authenticate itself.
        # google.auth (used by credential_provider above) has already loaded credentials.
        gac = os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        try:
            self._store = GCSStore(bucket, **kwargs)
        finally:
            if gac is not None:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gac
        logger.info(f"Initialized GCS file storage: bucket={bucket}")

    async def get_download_url(self, key: str, expires_in: int = 3600) -> str:
        # Storage keys percent-encode their segments (``key_segment``), so a bank id
        # with a space, a dot or non-ASCII is stored under an object name that
        # literally contains ``%`` -- "Express%20AI" is the name, not an escaped
        # "Express AI". obstore's GCS signer puts the key into the URL path as-is, so
        # GCS decodes that ``%20`` and looks up a different object: NoSuchKey on a
        # download whose export succeeded. Escaping ``%`` makes the path decode back to
        # the literal name, and the signature is computed over that same path.
        #
        # GCS only: obstore's S3 and Azure signers already escape ``%`` (pre-escaping
        # there would double-encode). tests/test_file_storage_signed_urls.py signs with
        # all three and asserts the path decodes to the stored key, so a sibling that
        # regresses -- or a fourth backend that arrives -- fails there.
        return await super().get_download_url(key.replace("%", "%25"), expires_in)
