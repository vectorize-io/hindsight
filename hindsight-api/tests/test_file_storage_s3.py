"""
Integration tests for S3FileStorage against a SeaweedFS Docker container.

SeaweedFS (Apache 2.0) provides an S3-compatible API via `weed server -s3`.
Requires Docker to be running. Tests are skipped automatically if Docker is unavailable.
"""

import json
import logging
import os
import subprocess
import tempfile
import time
import uuid

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

logger = logging.getLogger(__name__)

try:
    from testcontainers.core.container import DockerContainer

    _has_testcontainers = True
except ImportError:
    _has_testcontainers = False

pytestmark = [
    pytest.mark.skipif(not _has_testcontainers, reason="testcontainers not installed"),
    pytest.mark.timeout(300),
]

SEAWEEDFS_S3_PORT = 8333
TEST_BUCKET = "hindsight-test"
ACCESS_KEY = "test_access_key"
SECRET_KEY = "test_secret_key"

# SeaweedFS S3 IAM config granting full access to our test credentials
_S3_CONFIG = {
    "identities": [
        {
            "name": "test-user",
            "credentials": [{"accessKey": ACCESS_KEY, "secretKey": SECRET_KEY}],
            "actions": ["Admin", "Read", "Write", "List"],
        }
    ]
}


def _docker_available() -> bool:
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _wait_for_seaweedfs(endpoint: str, timeout: int = 30) -> None:
    """Poll SeaweedFS S3 endpoint until ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(endpoint, timeout=2)
            # 200 = no auth, 403 = auth enabled but gateway is up — either means ready
            if resp.status_code in (200, 403):
                logger.info("SeaweedFS S3 is ready at %s", endpoint)
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"SeaweedFS did not become ready at {endpoint} within {timeout}s")


@pytest.fixture(scope="module")
def seaweedfs_container():
    """Start a SeaweedFS container for the test module, shared across all tests.

    Mounts an s3.json config file to set up S3 credentials for the test user.
    """
    if not _docker_available():
        pytest.skip("Docker is not available")

    # Write S3 IAM config to a temp file that persists for the module scope
    s3_config_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(_S3_CONFIG, s3_config_file)
    s3_config_file.flush()

    container = (
        DockerContainer(image="chrislusf/seaweedfs:latest")
        .with_exposed_ports(SEAWEEDFS_S3_PORT)
        .with_volume_mapping(s3_config_file.name, "/etc/seaweedfs/s3.json", "ro")
        .with_command(
            f"server -s3 -s3.port={SEAWEEDFS_S3_PORT} -s3.config=/etc/seaweedfs/s3.json -ip.bind=0.0.0.0"
        )
    )

    container.start()

    try:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(SEAWEEDFS_S3_PORT)
        endpoint = f"http://{host}:{port}"

        _wait_for_seaweedfs(endpoint, timeout=120)

        # Create test bucket using obstore (proper SigV4 signing)
        import obstore as obs
        from obstore.store import S3Store

        admin_store = S3Store(
            TEST_BUCKET,
            endpoint=endpoint,
            region="us-east-1",
            access_key_id=ACCESS_KEY,
            secret_access_key=SECRET_KEY,
            allow_http=True,
        )
        # SeaweedFS auto-creates buckets on first write
        obs.put(admin_store, ".bucket-init", b"")
        obs.delete(admin_store, ".bucket-init")
        logger.info("Test bucket '%s' is ready", TEST_BUCKET)

        yield {
            "endpoint": endpoint,
            "access_key": ACCESS_KEY,
            "secret_key": SECRET_KEY,
            "bucket": TEST_BUCKET,
        }
    finally:
        container.stop()
        import os

        os.unlink(s3_config_file.name)


@pytest.fixture
def s3_storage(seaweedfs_container):
    """Create an S3FileStorage instance pointing at the SeaweedFS container."""
    from hindsight_api.engine.storage.s3 import S3FileStorage

    return S3FileStorage(
        bucket=seaweedfs_container["bucket"],
        region="us-east-1",
        endpoint=seaweedfs_container["endpoint"],
        access_key_id=seaweedfs_container["access_key"],
        secret_access_key=seaweedfs_container["secret_key"],
    )


@pytest.mark.asyncio
async def test_s3_storage_store_and_retrieve(s3_storage):
    """Store a file, retrieve it, verify bytes match."""
    content = b"Hello, SeaweedFS! This is a test file."
    key = f"test/{uuid.uuid4()}.txt"

    stored_key = await s3_storage.store(
        file_data=content,
        key=key,
        metadata={"content_type": "text/plain"},
    )
    assert stored_key == key

    retrieved = await s3_storage.retrieve(key)
    assert retrieved == content


@pytest.mark.asyncio
async def test_s3_storage_exists_and_delete(s3_storage):
    """Store, check exists=True, delete, check exists=False."""
    content = b"File to be deleted."
    key = f"test/{uuid.uuid4()}.txt"

    await s3_storage.store(file_data=content, key=key)

    assert await s3_storage.exists(key) is True

    await s3_storage.delete(key)

    assert await s3_storage.exists(key) is False


@pytest.mark.asyncio
async def test_s3_storage_file_not_found(s3_storage):
    """Retrieve a non-existent key, expect FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        await s3_storage.retrieve(f"nonexistent/{uuid.uuid4()}.txt")


@pytest.mark.asyncio
async def test_s3_storage_get_download_url(s3_storage):
    """Store a file, get a presigned URL, verify it's a valid URL string."""
    content = b"Presigned URL test content."
    key = f"test/{uuid.uuid4()}.txt"

    await s3_storage.store(file_data=content, key=key)

    url = await s3_storage.get_download_url(key, expires_in=300)
    assert isinstance(url, str)
    assert url.startswith("http")
    assert key in url


@pytest.mark.asyncio
async def test_s3_file_retain_api_end_to_end(seaweedfs_container, memory_no_llm_verify):
    """Full HTTP API flow: upload file via /files/retain with S3 storage backend."""
    from hindsight_api.api.http import create_app
    from hindsight_api.engine.storage.s3 import S3FileStorage

    # Swap the engine's file storage to use the SeaweedFS-backed S3 storage
    original_storage = memory_no_llm_verify._file_storage
    s3_storage = S3FileStorage(
        bucket=seaweedfs_container["bucket"],
        region="us-east-1",
        endpoint=seaweedfs_container["endpoint"],
        access_key_id=seaweedfs_container["access_key"],
        secret_access_key=seaweedfs_container["secret_key"],
    )
    memory_no_llm_verify._file_storage = s3_storage

    try:
        app = create_app(memory_no_llm_verify, initialize_memory=False)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            bank_id = f"test-s3-bank-{uuid.uuid4().hex[:8]}"
            bank_response = await client.put(f"/v1/default/banks/{bank_id}", json={"name": "S3 Test Bank"})
            assert bank_response.status_code in (200, 201)

            txt_content = b"Alice works at Acme Corp. She joined in 2024."
            request_data = {
                "document_tags": ["s3-test"],
                "async": True,
            }

            files = {"files": ("notes.txt", txt_content, "text/plain")}
            data = {"request": json.dumps(request_data)}

            response = await client.post(
                f"/v1/default/banks/{bank_id}/files/retain",
                files=files,
                data=data,
            )

            assert response.status_code == 200
            result = response.json()
            assert "operation_ids" in result
            assert len(result["operation_ids"]) == 1
    finally:
        memory_no_llm_verify._file_storage = original_storage
