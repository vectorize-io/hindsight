"""Test stub for CodexLLM's streaming POST.

``CodexLLM`` streams its SSE body through an aiohttp session under a wall-clock
deadline (issue #3898), so tests stub the *upstream*, not the client: the canned
reply is served by a real local HTTP server and the provider's ``base_url`` is
pointed at it for the duration of the ``with`` block. The server runs on its own
event loop in a background thread, which keeps these context managers synchronous
around an ``await llm.call(...)``.

The mock they yield records one call per request the server received, shaped like
the old ``client.stream("POST", url, json=..., headers=...)`` call —
``mock.call_args.kwargs["json"]`` / ``["headers"]`` and ``mock.call_count`` still
work. ``headers`` are the headers that arrived on the wire (case-insensitive).

A reply is anything with an integer ``status_code`` (``httpx.Response``,
``MagicMock(status_code=200)``, :class:`CodexReply`); its body comes from
``content`` bytes or ``text`` when either is set, and is empty otherwise.
"""

import asyncio
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from aiohttp import web
from aiohttp.test_utils import TestServer
from multidict import CIMultiDict


@dataclass
class CodexReply:
    """A canned upstream reply."""

    status_code: int = 200
    text: str = ""
    content_type: str = "text/event-stream"


def _to_web_response(reply: Any) -> web.Response:
    status = reply.status_code if isinstance(getattr(reply, "status_code", None), int) else 200
    content = getattr(reply, "content", None)
    text = getattr(reply, "text", None)
    body = content if isinstance(content, bytes) else text.encode() if isinstance(text, str) else b""
    content_type = getattr(reply, "content_type", None)
    if not isinstance(content_type, str):
        headers = getattr(reply, "headers", None)
        content_type = headers.get("content-type") if hasattr(headers, "get") else None
        content_type = content_type if isinstance(content_type, str) else "text/event-stream"
    return web.Response(status=status, body=body, headers={"Content-Type": content_type})


class _ThreadedServer:
    """An aiohttp server on a private event loop in a daemon thread."""

    def __init__(self, recorder: MagicMock) -> None:
        self._recorder = recorder
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._server: TestServer | None = None

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        reply = self._recorder("POST", str(request.url), json=payload, headers=CIMultiDict(request.headers))
        return _to_web_response(reply)

    async def _start(self) -> TestServer:
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self._handle)
        server = TestServer(app)
        await server.start_server()
        return server

    def start(self) -> str:
        self._thread.start()
        self._server = asyncio.run_coroutine_threadsafe(self._start(), self._loop).result(timeout=10)
        return str(self._server.make_url("")).rstrip("/")

    def stop(self) -> None:
        if self._server is not None:
            asyncio.run_coroutine_threadsafe(self._server.close(), self._loop).result(timeout=10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)
        self._loop.close()


@contextmanager
def _serving(llm: Any, recorder: MagicMock) -> Iterator[MagicMock]:
    server = _ThreadedServer(recorder)
    original_base_url = llm.base_url
    llm.base_url = server.start()
    try:
        yield recorder
    finally:
        llm.base_url = original_base_url
        server.stop()


def stub_codex_stream(llm: Any, response: Any) -> Any:
    """Serve ``response`` to every Codex request ``llm`` makes; use as a context manager."""
    return _serving(llm, MagicMock(return_value=response))


def stub_codex_stream_with(llm: Any, handler: Callable[..., Any]) -> Any:
    """Like ``stub_codex_stream`` but ``handler(url, json=..., headers=...)`` builds each reply.

    For tests that vary the reply per attempt (auth-refresh retries).
    """
    return _serving(llm, MagicMock(side_effect=lambda _method, url, **kwargs: handler(url, **kwargs)))
