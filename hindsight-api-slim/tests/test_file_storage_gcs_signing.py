"""GCS presigned download URLs must address the object's literal name.

Storage keys percent-encode their segments (``key_segment``), so a bank id with a
space, a dot or non-ASCII is stored under an object name that literally contains
``%``. GCS decodes the request path once, so the signed URL has to carry ``%25``
for every literal ``%`` -- otherwise "Express%20AI" is read as "Express AI", a
different object, and the download fails with NoSuchKey even though the export
succeeded.

These sign for real, offline: obstore's GCS signer runs against a throwaway
service-account key, so no network or bucket is needed.
"""

import json
import uuid
from urllib.parse import unquote, urlsplit

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from hindsight_api.engine.storage import key_segment
from hindsight_api.engine.storage.gcs import GCSFileStorage

BUCKET = "test-bucket"


@pytest.fixture(scope="module")
def gcs_storage() -> GCSFileStorage:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    service_account_key = json.dumps(
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
    return GCSFileStorage(bucket=BUCKET, service_account_key=service_account_key)


def _object_name_gcs_will_read(url: str) -> str:
    """The object name GCS resolves from a signed URL: the path, decoded once."""
    path = urlsplit(url).path
    prefix = f"/{BUCKET}/"
    assert path.startswith(prefix), url
    return unquote(path[len(prefix) :])


@pytest.mark.asyncio
@pytest.mark.parametrize("bank_id", ["plain-bank", "Express AI — Trial", "team.notes", "100% done"])
async def test_signed_url_resolves_to_the_stored_object_name(gcs_storage, bank_id):
    key = f"tenants/tenant_test/banks/{key_segment(bank_id)}/exports/{uuid.uuid4()}/transfer.zip"

    url = await gcs_storage.get_download_url(key, expires_in=300)

    assert _object_name_gcs_will_read(url) == key


@pytest.mark.asyncio
async def test_key_without_percent_is_signed_unchanged(gcs_storage):
    key = f"tenants/tenant_test/banks/plain-bank/exports/{uuid.uuid4()}/transfer.zip"

    url = await gcs_storage.get_download_url(key, expires_in=300)

    assert urlsplit(url).path == f"/{BUCKET}/{key}"
