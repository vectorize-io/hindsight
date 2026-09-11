"""In-process HTTP upstream for tests of the aiohttp-based clients.

Production HTTP goes through aiohttp (see ``hindsight_api/engine/aiohttp_session.py``),
so tests stub the *upstream*, not the client: a real ``aiohttp.web`` server on a local
port, which exercises the actual transport (timeouts, status handling, body parsing).

    async def handler(request: web.Request) -> web.StreamResponse:
        return web.json_response([[0.1, 0.2]])

    async with stub_server(handler) as base_url:
        emb = RemoteTEIEmbeddings(base_url=base_url)
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from aiohttp import web
from aiohttp.test_utils import TestServer

Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


@asynccontextmanager
async def stub_server(handler: Handler) -> AsyncIterator[str]:
    """Serve every method and path with ``handler``; yield the base URL (no trailing slash)."""
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    server = TestServer(app)
    await server.start_server()
    try:
        yield str(server.make_url("")).rstrip("/")
    finally:
        await server.close()
