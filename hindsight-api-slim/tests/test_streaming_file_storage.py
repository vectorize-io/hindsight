"""Streaming exports: the ZIP writer and the file-storage backends it streams through."""

import io
import json
import uuid
import zipfile
import zlib
from collections.abc import AsyncIterator

import pytest
from obstore.store import MemoryStore
from starlette.responses import StreamingResponse

from hindsight_api.engine.storage.azure import AzureFileStorage
from hindsight_api.engine.storage.gcs import GCSFileStorage
from hindsight_api.engine.storage.postgresql import _CHUNKED_MANIFEST_SIGNATURE, PG_STREAM_CHUNK_SIZE
from hindsight_api.engine.storage.s3 import S3FileStorage
from hindsight_api.engine.transfer.stream_archive import (
    CHUNK_SIZE,
    SIG_ZIP64_EOCD,
    SIG_ZIP64_LOCATOR,
    ZipStreamer,
)

BACKENDS = [GCSFileStorage, S3FileStorage, AzureFileStorage]
PAYLOAD = b"Hello world! " * 1024 * 64  # ~832 KB payload


def _backend_over_memory_store(cls):
    fs = cls.__new__(cls)
    fs._store = MemoryStore()
    return fs


@pytest.mark.parametrize("cls", BACKENDS, ids=lambda c: c.__name__)
async def test_store_and_retrieve_stream_cloud(cls):
    """Test store_stream and retrieve_stream across S3, GCS, and Azure."""
    fs = _backend_over_memory_store(cls)

    async def chunk_gen() -> AsyncIterator[bytes]:
        for i in range(0, len(PAYLOAD), 65536):
            yield PAYLOAD[i : i + 65536]

    key = await fs.store_stream("test_stream.bin", chunk_gen())
    assert key == "test_stream.bin"
    assert await fs.exists(key)

    # Read back via retrieve_stream
    chunks = []
    async for chunk in fs.retrieve_stream(key):
        assert isinstance(chunk, bytes)
        chunks.append(chunk)
    assert b"".join(chunks) == PAYLOAD

    # Also test retrieve() returns exact bytes
    data = await fs.retrieve(key)
    assert isinstance(data, bytes)
    assert data == PAYLOAD

    # Test streaming through Starlette StreamingResponse
    resp = StreamingResponse(fs.retrieve_stream(key), media_type="application/octet-stream")
    body_chunks = []
    async for b in resp.body_iterator:
        body_chunks.append(b if isinstance(b, bytes) else b.encode())
    assert b"".join(body_chunks) == PAYLOAD

    # Clean up
    await fs.delete(key)
    assert not await fs.exists(key)


async def test_zip_streamer_roundtrip_and_deflate():
    """Test ZipStreamer creates valid PKZIP archives parseable by standard zipfile."""
    zs = ZipStreamer()
    chunks = []

    # Stream file 1 with Deflate (default level)
    file1_content = b"Hello, this is file 1 content repeated! " * 500
    async for c in zs.write_file_bytes("folder/file1.txt", file1_content, compress=True):
        chunks.append(c)

    # Stream file 2 with Deflate level 0 (pass-through Deflate blocks for attachments)
    file2_content = b"RAW BINARY DATA 1234567890" * 20
    async for c in zs.write_file_bytes("images/photo.bin", file2_content, compress=True, compression_level=0):
        chunks.append(c)

    # Stream file 3 without compression (stored)
    file3_content = b"STORED DATA"
    async for c in zs.write_file_bytes("data/raw.bin", file3_content, compress=False):
        chunks.append(c)

    # Finish archive
    chunks.append(zs.finish())

    zip_bytes = b"".join(chunks)
    assert len(zip_bytes) > 0

    # Parse and verify with Python's zipfile
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")
    assert zf.namelist() == ["folder/file1.txt", "images/photo.bin", "data/raw.bin"]
    assert zf.read("folder/file1.txt") == file1_content
    assert zf.read("images/photo.bin") == file2_content
    assert zf.read("data/raw.bin") == file3_content
    assert zf.testzip() is None
    assert zf.getinfo("images/photo.bin").compress_type == zipfile.ZIP_DEFLATED


@pytest.mark.parametrize("cls", BACKENDS, ids=lambda c: c.__name__)
async def test_store_and_retrieve_empty_file_cloud(cls):
    """Empty files (0 bytes) can be stored, sized, and retrieved across cloud stores."""
    fs = _backend_over_memory_store(cls)

    async def empty_gen() -> AsyncIterator[bytes]:
        if False:
            yield b""

    key = await fs.store_stream("empty.bin", empty_gen())
    assert await fs.exists(key)
    assert await fs.get_size(key) == 0
    assert await fs.retrieve(key) == b""
    assert b"".join([c async for c in fs.retrieve_stream(key)]) == b""
    await fs.delete(key)
    assert not await fs.exists(key)


async def test_zip_streamer_empty_entries_and_zero_bytes():
    """ZipStreamer properly handles 0-byte files with and without compression."""
    zs = ZipStreamer()
    chunks = []

    async for c in zs.write_file_bytes("empty_deflated.txt", b"", compress=True):
        chunks.append(c)

    async for c in zs.write_file_bytes("empty_stored.bin", b"", compress=False):
        chunks.append(c)

    chunks.append(zs.finish())

    zip_bytes = b"".join(chunks)
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")
    assert zf.namelist() == ["empty_deflated.txt", "empty_stored.bin"]
    assert zf.read("empty_deflated.txt") == b""
    assert zf.read("empty_stored.bin") == b""
    assert zf.testzip() is None


@pytest.mark.parametrize("cls", BACKENDS, ids=lambda c: c.__name__)
async def test_cloud_storage_cancellation_cleanup(cls):
    """When a stream aborts with an exception, the incomplete upload is deleted."""
    fs = _backend_over_memory_store(cls)

    async def failing_gen():
        yield b"chunk 1"
        yield b"chunk 2"
        raise RuntimeError("Stream aborted prematurely")

    with pytest.raises(RuntimeError, match="Stream aborted prematurely"):
        await fs.store_stream("aborted.bin", failing_gen())

    # Key should have been cleaned up and not exist
    assert not await fs.exists("aborted.bin")


async def test_stream_export_bank_empty_bank_manifest():
    """Verify stream_export_bank outputs a valid ZIP archive with an empty manifest for an empty bank."""
    from unittest.mock import AsyncMock

    from hindsight_api.engine.transfer import TransferScope, stream_export_bank

    fake_conn = AsyncMock(spec=["fetch"])
    fake_conn.fetch = AsyncMock(return_value=[])

    chunks = []
    async for chunk in stream_export_bank(
        fake_conn,
        "test-bank",
        scope=TransferScope(data=True, bank_config=False, history=False),
        memories=None,
    ):
        chunks.append(chunk)

    zip_bytes = b"".join(chunks)
    assert len(zip_bytes) > 0

    zf = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")
    assert "manifest.json" in zf.namelist()
    manifest_data = json.loads(zf.read("manifest.json").decode("utf-8"))
    assert manifest_data["source_bank_id"] == "test-bank"
    assert manifest_data["document_count"] == 0
    assert manifest_data["archive_type"] == "bank"
    assert zf.testzip() is None


async def test_zip_streamer_deflate_level_0_sequential_unpack():
    """Verify Deflate level 0 entries produce valid Method 8 streams for sequential readers."""
    import shutil
    import subprocess

    zs = ZipStreamer()
    chunks = []
    sample_payload = b"MULTIMODAL_BLOB_RAW_BYTES_12345" * 100

    async def sample_stream():
        for i in range(0, len(sample_payload), 500):
            yield sample_payload[i : i + 500]

    async for c in zs.write_file_chunks("blobs/000000.bin", sample_stream(), compress=True, compression_level=0):
        chunks.append(c)
    chunks.append(zs.finish())

    zip_bytes = b"".join(chunks)

    # 1. Local header inspection: method must be 8 (Deflate) with Bit 3 set
    import struct

    sig, ver, flags, method = struct.unpack("<4sHHH", zip_bytes[:10])
    assert method == 8, f"Expected method 8 (Deflate), got {method}"
    assert flags & 0x0008, "Bit 3 (Data Descriptor) must be set"

    # 2. Random-access verification with Python standard library
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")
    assert zf.testzip() is None
    assert zf.read("blobs/000000.bin") == sample_payload

    # 3. Sequential streaming decompressor verification (zlib wbits=-15)
    # Header is 30 bytes + len("blobs/000000.bin") = 46 bytes
    header_len = 30 + len("blobs/000000.bin")
    # Descriptor is 16 bytes at the end of entry data
    entry = zs.entries[0]
    raw_deflated = zip_bytes[header_len : header_len + entry.comp_size]
    decomp = zlib.decompressobj(wbits=-15)
    extracted = decomp.decompress(raw_deflated) + decomp.flush()
    assert extracted == sample_payload

    # 4. If funzip is available on the system, verify streaming from stdin
    if shutil.which("funzip"):
        proc = subprocess.run(["funzip"], input=zip_bytes, capture_output=True)
        assert proc.returncode == 0
        assert proc.stdout == sample_payload


@pytest.mark.asyncio
async def test_zipstreamer_zip64_many_entries():
    """Verify Zip64 EOCD and Locator emission when member count >= 65,535."""
    zs = ZipStreamer()
    buf = io.BytesIO()
    num_entries = 65536  # Exceeds 16-bit limit 65,535

    for i in range(num_entries):
        async for chunk in zs.write_file_bytes(f"{i}.txt", b"", compress=False):
            buf.write(chunk)
    buf.write(zs.finish())

    data = buf.getvalue()

    # 1. Assert Zip64 magic signatures are emitted
    assert SIG_ZIP64_EOCD in data, "Zip64 End of Central Directory record must be present"
    assert SIG_ZIP64_LOCATOR in data, "Zip64 Locator record must be present"

    # 2. Verify standard zipfile reader decodes all entries and parses Zip64 structure
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        assert len(zf.infolist()) == num_entries
        assert zf.testzip() is None
        assert zf.read("0.txt") == b""
        assert zf.read(f"{num_entries - 1}.txt") == b""


# --- PostgreSQL file storage, against the real database -------------------------


def _pg_key(name: str) -> str:
    return f"tests/streaming/{uuid.uuid4().hex}/{name}"


async def _pieces(data: bytes, size: int = 1024 * 1024) -> AsyncIterator[bytes]:
    for i in range(0, len(data), size):
        yield data[i : i + size]


async def test_postgres_short_stream_is_one_plain_row(memory):
    """A stream under one chunk is stored exactly as store() would store it."""
    fs = memory._file_storage
    key = _pg_key("small.txt")
    try:
        await fs.store_stream(key, _pieces(b"small data"))
        assert await fs.retrieve(key) == b"small data"
        assert not await fs.exists(f"{key}#chunk=000000")
    finally:
        await fs.delete(key)


async def test_postgres_empty_file(memory):
    fs = memory._file_storage
    key = _pg_key("empty.bin")
    try:
        await fs.store_stream(key, _pieces(b""))
        assert await fs.exists(key)
        assert await fs.get_size(key) == 0
        assert await fs.retrieve(key) == b""
        assert b"".join([c async for c in fs.retrieve_stream(key)]) == b""
    finally:
        await fs.delete(key)
    assert not await fs.exists(key)


async def test_postgres_plain_row_is_streamed_in_bounded_ranges(memory):
    """A large file written by store() — every attachment — never comes back in one piece."""
    fs = memory._file_storage
    key = _pg_key("attachment.bin")
    data = bytes(range(256)) * (10 * 1024 * 1024 // 256)  # 10 MiB, not a chunked file
    try:
        await fs.store(data, key)
        pieces = [c async for c in fs.retrieve_stream(key)]
        assert [len(p) for p in pieces] == [PG_STREAM_CHUNK_SIZE, PG_STREAM_CHUNK_SIZE, 2 * 1024 * 1024]
        assert b"".join(pieces) == data
        assert await fs.get_size(key) == len(data)
    finally:
        await fs.delete(key)


async def test_postgres_aborted_stream_leaves_nothing(memory):
    fs = memory._file_storage
    key = _pg_key("aborted.bin")

    async def failing() -> AsyncIterator[bytes]:
        yield b"A" * PG_STREAM_CHUNK_SIZE
        yield b"B" * 1024
        raise RuntimeError("disconnect")

    with pytest.raises(RuntimeError, match="disconnect"):
        await fs.store_stream(key, failing())
    assert not await fs.exists(key)
    assert not await fs.exists(f"{key}#chunk=000000")


@pytest.mark.parametrize("rewrite", ["store", "store_stream"])
async def test_postgres_overwriting_a_chunked_file_drops_its_old_chunks(memory, rewrite):
    """Replacing a chunked file must not strand the previous version's chunk rows."""
    fs = memory._file_storage
    key = _pg_key("rewritten.bin")
    try:
        await fs.store_stream(key, _pieces(b"X" * (9 * 1024 * 1024)))
        assert await fs.exists(f"{key}#chunk=000002")
        if rewrite == "store":
            await fs.store(b"short", key)
        else:
            await fs.store_stream(key, _pieces(b"Y" * (5 * 1024 * 1024)))
        assert not await fs.exists(f"{key}#chunk=000002")
        if rewrite == "store":
            assert not await fs.exists(f"{key}#chunk=000000")
            assert await fs.retrieve(key) == b"short"
        else:
            assert await fs.retrieve(key) == b"Y" * (5 * 1024 * 1024)
    finally:
        await fs.delete(key)


async def test_postgres_row_that_only_looks_like_a_manifest_is_a_plain_file(memory):
    fs = memory._file_storage
    key = _pg_key("lookalike.bin")
    data = _CHUNKED_MANIFEST_SIGNATURE + b"not-a-manifest"
    await fs.store(data, key)
    assert await fs.retrieve(key) == data
    assert await fs.get_size(key) == len(data)
    await fs.delete(key)
    assert not await fs.exists(key)


async def test_zip_streamer_never_hands_on_more_than_one_chunk_at_a_time():
    """A source that yields a blob in one piece still leaves the writer as bounded pieces."""
    blob = bytes(range(256)) * (3 * CHUNK_SIZE // 256 + 7)

    async def one_piece() -> AsyncIterator[bytes]:
        yield blob

    for level in (0, 6):
        zs = ZipStreamer()
        out = [c async for c in zs.write_file_chunks("blob.bin", one_piece(), compression_level=level)]
        assert max(len(c) for c in out) <= CHUNK_SIZE + 64  # deflate framing on a stored block
        archive = b"".join(out) + zs.finish()
        assert zipfile.ZipFile(io.BytesIO(archive)).read("blob.bin") == blob
