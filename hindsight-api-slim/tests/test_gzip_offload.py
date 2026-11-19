"""Response gzip keeps large bodies off the event loop (#6667), and still compresses
exactly what Starlette's middleware would."""

import os
import threading

import httpx
import pytest
from starlette.applications import Starlette
from starlette.middleware.gzip import GZipResponder
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from hindsight_api.api.gzip_offload import _OFFLOAD_MIN_BYTES, OffloadingGZipMiddleware

# Compressible but not trivially so, and well past the offload floor.
LARGE = (os.urandom(512).hex() * (4 * _OFFLOAD_MIN_BYTES // 1024)).encode()
SMALL = b"x" * 1000


def _app() -> Starlette:
    async def large(request):
        return Response(LARGE, media_type="application/json")

    async def small(request):
        return Response(SMALL, media_type="application/json")

    async def streamed(request):
        async def chunks():
            yield LARGE
            yield SMALL
            yield LARGE

        return StreamingResponse(chunks(), media_type="application/json")

    async def events(request):
        return Response(LARGE, media_type="text/event-stream")

    app = Starlette(
        routes=[Route(f"/{fn.__name__}", fn) for fn in (large, small, streamed, events)],
    )
    app.add_middleware(OffloadingGZipMiddleware, minimum_size=500)
    return app


@pytest.fixture
def compress_threads(monkeypatch) -> list[int]:
    """The thread each compression ran on."""
    threads: list[int] = []
    real = GZipResponder.apply_compression

    def spy(self, body, *, more_body):
        threads.append(threading.get_ident())
        return real(self, body, more_body=more_body)

    monkeypatch.setattr(GZipResponder, "apply_compression", spy)
    return threads


async def _get(path: str) -> httpx.Response:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()), base_url="http://t") as client:
        return await client.get(path, headers={"Accept-Encoding": "gzip"})


@pytest.mark.asyncio
async def test_large_body_is_compressed_off_the_loop(compress_threads):
    resp = await _get("/large")
    assert resp.headers["content-encoding"] == "gzip"
    assert int(resp.headers["content-length"]) < len(LARGE)
    assert resp.content == LARGE  # httpx decodes gzip
    assert compress_threads and threading.get_ident() not in compress_threads


@pytest.mark.asyncio
async def test_small_body_is_compressed_inline(compress_threads):
    resp = await _get("/small")
    assert resp.headers["content-encoding"] == "gzip"
    assert resp.content == SMALL
    assert compress_threads == [threading.get_ident()]


@pytest.mark.asyncio
async def test_streamed_body_mixes_offloaded_and_inline_chunks_into_one_stream(compress_threads):
    resp = await _get("/streamed")
    assert resp.headers["content-encoding"] == "gzip"
    assert resp.content == LARGE + SMALL + LARGE
    loop = threading.get_ident()
    # Large chunks in a thread, the small one and the closing flush inline.
    assert [t == loop for t in compress_threads] == [False, True, False, True]


@pytest.mark.asyncio
async def test_excluded_content_type_is_left_alone(compress_threads):
    resp = await _get("/events")
    assert "content-encoding" not in resp.headers
    assert resp.content == LARGE
    assert compress_threads == []
