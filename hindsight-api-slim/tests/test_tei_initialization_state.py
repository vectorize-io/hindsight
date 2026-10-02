"""Failed and interrupted startup probes must leave TEI embeddings uninitialized."""

import asyncio

import pytest
from aiohttp import web

from hindsight_api.engine.embeddings import RemoteTEIEmbeddings
from tests.aiohttp_stub import stub_server


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_path", ["/info", "/embed"])
async def test_failed_probe_can_initialize_again(failure_path: str) -> None:
    failed = False

    async def handler(request: web.Request) -> web.StreamResponse:
        nonlocal failed
        if request.path == failure_path and not failed:
            failed = True
            return web.json_response({"error": "temporary failure"}, status=503)
        if request.path == "/info":
            return web.json_response({"model_id": "tei-test"})
        return web.json_response([[0.1, 0.2]])

    async with stub_server(handler) as base_url:
        backend = RemoteTEIEmbeddings(base_url=base_url, max_retries=0)
        try:
            with pytest.raises(RuntimeError, match="Failed to connect to TEI server"):
                await backend.initialize()
            with pytest.raises(RuntimeError, match="not initialized"):
                await backend.encode(["text"])
            await backend.initialize()
            assert backend.dimension == 2
            assert await backend.encode(["text"]) == [[0.1, 0.2]]
        finally:
            await backend._session.close()


@pytest.mark.asyncio
async def test_cancelled_probe_can_initialize_again() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def handler(request: web.Request) -> web.StreamResponse:
        nonlocal calls
        if request.path == "/info":
            calls += 1
            if calls == 1:
                started.set()
                await release.wait()
            return web.json_response({"model_id": "tei-test"})
        return web.json_response([[0.1, 0.2]])

    async with stub_server(handler) as base_url:
        backend = RemoteTEIEmbeddings(base_url=base_url)
        startup = asyncio.create_task(backend.initialize())
        try:
            await asyncio.wait_for(started.wait(), timeout=1.0)
            startup.cancel()
            with pytest.raises(asyncio.CancelledError):
                await startup
            release.set()
            await backend.initialize()
            assert backend.dimension == 2
            assert calls == 2
        finally:
            release.set()
            if not startup.done():
                startup.cancel()
                await asyncio.gather(startup, return_exceptions=True)
            await backend._session.close()
