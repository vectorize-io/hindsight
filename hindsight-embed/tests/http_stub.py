"""A real loopback HTTP server for tests of the aiohttp probes.

The probes' sync entry point runs ``asyncio.run`` itself, so the stub has to
live on another loop: it is served from a background thread. Tests stub the
*upstream* (status, body, latency), not the client, so the actual transport —
timeouts, refused connections, body parsing — is exercised.

    with serve(reply(200, {"status": "healthy"})) as stub:
        assert DaemonEmbedManager._port_health_ok(stub.port)
"""

import asyncio
import socket
import threading
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


@dataclass
class StubServer:
    port: int = 0
    paths: list[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def reply(status: int = 200, json_body: Any = None, *, text: str | None = None, delay: float = 0.0) -> Handler:
    """Handler answering every request with ``status`` and a JSON (or raw text) body."""

    async def handler(_request: web.Request) -> web.StreamResponse:
        if delay:
            await asyncio.sleep(delay)
        if text is not None:
            return web.Response(status=status, text=text)
        return web.json_response(json_body if json_body is not None else {}, status=status)

    return handler


def replies(*handlers: Handler) -> Handler:
    """Answer the n-th request with the n-th handler; the last one repeats."""
    remaining = list(handlers)

    async def handler(request: web.Request) -> web.StreamResponse:
        current = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return await current(request)

    return handler


def closed_port() -> int:
    """A loopback port nothing listens on (connections are refused)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def serve(handler: Handler) -> Iterator[StubServer]:
    """Serve every path with ``handler`` on 127.0.0.1 from a background thread."""
    stub = StubServer()
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    async def dispatch(request: web.Request) -> web.StreamResponse:
        stub.paths.append(request.path)
        return await handler(request)

    async def start() -> web.AppRunner:
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", dispatch)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        stub.port = runner.addresses[0][1]
        return runner

    def run() -> None:
        asyncio.set_event_loop(loop)
        runner = loop.run_until_complete(start())
        ready.set()
        loop.run_forever()
        loop.run_until_complete(runner.cleanup())
        loop.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(10), "stub server did not start"
    try:
        yield stub
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(10)
