"""Object-store backends must return native ``bytes`` from ``retrieve()``.

The obstore-backed backends (S3, GCS, Azure) read an object with
``obstore``'s ``Response.bytes_async()``, which returns obstore's own ``Bytes``
buffer — a zero-copy view that implements the buffer protocol but is **not** an
instance of ``bytes``. Most callers consume it as a buffer and never notice, so
the ``-> bytes`` return annotation quietly drifted from the truth.

It stops being invisible the moment a caller hands the result to something that
is strict about the type. A Starlette ``Response`` is the canonical example: its
``render()`` does ``content if isinstance(content, bytes) else content.encode()``,
so a non-``bytes`` value raises ``'Bytes' object has no attribute 'encode'`` at
request time — a 500 that no mocked-storage unit test and no CI run without a live
object store would catch (the real SeaweedFS/S3 test is skipped in CI).

These tests exercise the real ``retrieve()`` code path against an in-memory
``obstore`` store — no cloud, no Docker, runs in CI — and pin the contract:
``retrieve()`` returns native ``bytes``, and its result serves cleanly through a
Starlette ``Response``. They fail if any backend returns obstore's ``Bytes`` raw.
"""

import pytest

obstore = pytest.importorskip("obstore")
import obstore as obs  # noqa: E402
from obstore.store import MemoryStore  # noqa: E402
from starlette.responses import Response  # noqa: E402

from hindsight_api.engine.storage.azure import AzureFileStorage  # noqa: E402
from hindsight_api.engine.storage.gcs import GCSFileStorage  # noqa: E402
from hindsight_api.engine.storage.s3 import S3FileStorage  # noqa: E402

# Non-trivial payload: PNG magic + every byte value, so a lossy/text coercion
# anywhere in the path would corrupt it.
PAYLOAD = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) + b"\x00\xff\x80"

BACKENDS = [GCSFileStorage, S3FileStorage, AzureFileStorage]


def _backend_over_memory_store(cls):
    """A real backend instance whose object store is an in-memory obstore store.

    ``__init__`` needs a bucket and cloud credentials, so bypass it and inject the
    store directly — ``store()``/``retrieve()`` only touch ``self._store`` and go
    through the same ``obstore`` calls as against real S3/GCS/Azure, so the return
    type under test is identical.
    """
    fs = cls.__new__(cls)
    fs._store = MemoryStore()
    return fs


@pytest.mark.parametrize("cls", BACKENDS, ids=lambda c: c.__name__)
async def test_retrieve_returns_native_bytes(cls):
    fs = _backend_over_memory_store(cls)
    await fs.store(PAYLOAD, "obj")

    data = await fs.retrieve("obj")

    assert type(data) is bytes, f"{cls.__name__}.retrieve() returned {type(data)!r}, not native bytes"
    assert data == PAYLOAD


@pytest.mark.parametrize("cls", BACKENDS, ids=lambda c: c.__name__)
async def test_retrieved_bytes_serve_through_starlette_response(cls):
    """The exact failure mode: a non-``bytes`` retrieve() result crashes Response.render()."""
    fs = _backend_over_memory_store(cls)
    await fs.store(PAYLOAD, "obj")

    # Would raise "'Bytes' object has no attribute 'encode'" if retrieve() returned
    # obstore's Bytes instead of native bytes.
    resp = Response(content=await fs.retrieve("obj"), media_type="image/png")

    assert resp.body == PAYLOAD


async def test_obstore_bytes_alone_is_not_native_bytes():
    """Guards the assumption behind the fix: obstore's own Bytes is not ``bytes``.

    If a future obstore version returns native ``bytes`` from ``bytes_async()``,
    this canary flips and the per-backend coercion can be revisited as a no-op.
    """
    store = MemoryStore()
    await obs.put_async(store, "obj", PAYLOAD)
    raw = await (await obs.get_async(store, "obj")).bytes_async()

    assert not isinstance(raw, bytes), "obstore now returns native bytes; revisit the retrieve() coercion"
    assert bytes(raw) == PAYLOAD  # buffer-protocol copy still round-trips
