"""A presigned download URL must address the object's literal name, on every backend.

Storage keys percent-encode their segments (``key_segment``), so a bank id with a
space, a dot or non-ASCII is stored under an object name that literally contains
``%``. An object store decodes the request path once, so a signed URL has to carry
``%25`` for every literal ``%`` -- otherwise "Express%20AI" is read as "Express AI",
a different object, and the download fails with NoSuchKey even though the export
succeeded. That is what GCS did (obstore's GCS signer passes the key through as-is,
unlike its S3 and Azure signers), and why ``GCSFileStorage`` pre-escapes.

These cover all three obstore backends, because the sibling nobody tested is the one
that breaks: they sign for real but offline, with throwaway credentials, so no
network, bucket or container is needed. A fourth backend is caught by
``test_every_obstore_backend_is_covered``, which enumerates the package.
"""

import importlib
import json
import pkgutil
import uuid
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from hindsight_api.engine import storage as storage_pkg
from hindsight_api.engine.storage import key_segment
from hindsight_api.engine.storage.azure import AzureFileStorage
from hindsight_api.engine.storage.base import ObstoreFileStorage
from hindsight_api.engine.storage.gcs import GCSFileStorage
from hindsight_api.engine.storage.s3 import S3FileStorage

CONTAINER = "test-bucket"
# The backends the signer fixture builds; test_every_obstore_backend_is_covered
# checks the package holds no others.
BACKENDS = {"gcs": GCSFileStorage, "s3": S3FileStorage, "azure": AzureFileStorage}


@dataclass(frozen=True)
class Signer:
    """A storage backend, and the URL path prefix its objects sit under."""

    storage: ObstoreFileStorage
    path_prefix: str


def _service_account_key() -> str:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    return json.dumps(
        {
            "type": "service_account",
            "project_id": "test-project",
            "private_key_id": "test-key-id",
            "private_key": pem,
            "client_email": "signer@test-project.iam.gserviceaccount.com",
            "client_id": "1",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    )


@pytest.fixture(scope="module", params=list(BACKENDS))
def signer(request) -> Signer:
    if request.param == "gcs":
        storage: ObstoreFileStorage = GCSFileStorage(bucket=CONTAINER, service_account_key=_service_account_key())
    elif request.param == "s3":
        storage = S3FileStorage(
            bucket=CONTAINER, region="us-east-1", access_key_id="test-key", secret_access_key="test-secret"
        )
    else:
        # Azure's shared-key signer needs a base64 account key; the value is never used.
        storage = AzureFileStorage(container_name=CONTAINER, account_name="testaccount", account_key="dGVzdA==")
    return Signer(storage=storage, path_prefix=f"/{CONTAINER}/")


def _object_name_the_store_will_read(signer: Signer, url: str) -> str:
    """The object name a store resolves from a signed URL: the path, decoded once."""
    path = urlsplit(url).path
    assert path.startswith(signer.path_prefix), url
    return unquote(path[len(signer.path_prefix) :])


@pytest.mark.asyncio
@pytest.mark.parametrize("bank_id", ["plain-bank", "Express AI — Trial", "team.notes", "100% done"])
async def test_signed_url_resolves_to_the_stored_object_name(signer, bank_id):
    key = f"tenants/tenant_test/banks/{key_segment(bank_id)}/exports/{uuid.uuid4()}/transfer.zip"

    url = await signer.storage.get_download_url(key, expires_in=300)

    assert _object_name_the_store_will_read(signer, url) == key


@pytest.mark.asyncio
async def test_key_without_percent_is_signed_unchanged(signer):
    key = f"tenants/tenant_test/banks/plain-bank/exports/{uuid.uuid4()}/transfer.zip"

    url = await signer.storage.get_download_url(key, expires_in=300)

    assert urlsplit(url).path == f"{signer.path_prefix}{key}"


def test_every_obstore_backend_is_covered():
    """A new obstore backend must be signed here too, not discovered broken in production.

    ``BACKENDS`` is hand-written, so enumerate the package instead of trusting it:
    whoever adds the next backend gets a failure here rather than a NoSuchKey from a
    presigned URL.
    """
    for module in pkgutil.iter_modules(storage_pkg.__path__):
        importlib.import_module(f"{storage_pkg.__name__}.{module.name}")

    assert set(ObstoreFileStorage.__subclasses__()) == set(BACKENDS.values())
