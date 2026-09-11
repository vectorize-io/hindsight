"""The TEI client works from every event loop that uses it.

An ``aiohttp.ClientSession`` binds to the loop that created it, and a Hindsight process
runs more than one: provider ``initialize()`` is sometimes run under ``asyncio.run`` in an
executor thread, and tests and tooling spin up their own. A session shared across loops
fails with "attached to a different loop" or "Event loop is closed". The provider holds
one session per loop (``LoopLocalSession``), created on first use there and reused after.
"""

from __future__ import annotations

import asyncio

import aiohttp
from aiohttp import web

from hindsight_api.engine.embeddings import RemoteTEIEmbeddings
from tests.aiohttp_stub import stub_server


async def _embed_handler(request: web.Request) -> web.StreamResponse:
    inputs = (await request.json())["inputs"]
    return web.json_response([[0.5, 0.5] for _ in inputs])


async def test_each_loop_gets_its_own_session_and_both_work() -> None:
    async with stub_server(_embed_handler) as base_url:
        tei = RemoteTEIEmbeddings(base_url=base_url)
        tei._initialized = True
        tei._dimension = 2
        try:
            assert await tei.encode(["here"]) == [[0.5, 0.5]]
            main_session = tei._session.get()

            async def on_another_loop() -> tuple[list[list[float]], aiohttp.ClientSession]:
                session = tei._session.get()
                try:
                    return await tei.encode(["there"]), session
                finally:
                    await session.close()

            # A fresh loop in a worker thread, as `asyncio.run(provider.initialize())` in an
            # executor would be. The stub keeps serving on this loop meanwhile.
            vectors, other_session = await asyncio.to_thread(asyncio.run, on_another_loop())

            assert vectors == [[0.5, 0.5]]
            assert other_session is not main_session
            # The other loop's session is its own; this loop's is untouched and still usable.
            assert not main_session.closed
            assert await tei.encode(["again"]) == [[0.5, 0.5]]
        finally:
            await tei._session.close()


async def test_the_same_loop_reuses_its_session() -> None:
    """Per loop, not per call: a session per request would open a pool per request."""
    async with stub_server(_embed_handler) as base_url:
        tei = RemoteTEIEmbeddings(base_url=base_url)
        tei._initialized = True
        tei._dimension = 2
        try:
            first = tei._session.get()
            await tei.encode(["one"])
            await tei.encode(["two"])
            assert tei._session.get() is first
        finally:
            await tei._session.close()
