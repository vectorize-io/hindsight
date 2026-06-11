"""Client-disconnect detection that works behind ``BaseHTTPMiddleware``.

``Request.is_disconnected()`` is the obvious way to notice an abandoned HTTP
request, but it is silently broken once any ``@app.middleware("http")``
(Starlette ``BaseHTTPMiddleware``) is installed: that middleware runs the route
in a child task behind anyio memory streams, so the ``http.disconnect`` ASGI
event never reaches the route's ``Request``. This app has such middlewares, so
the recall/reflect cancellation in #2122/#2127 never actually fired in
production — the disconnect was never observed.

This pure-ASGI middleware sits *outside* the ``BaseHTTPMiddleware`` layer, where
it still owns the real ``receive`` channel. For the recall and reflect routes it
drains ``receive`` in a background task and trips a :class:`CancellationToken`
the moment ``http.disconnect`` arrives, stashing the token on the ASGI ``scope``.
The route copies that token onto its ``RequestContext`` and the engine checks it
at stage boundaries — so abandoned work stops instead of running to completion.

It only wraps recall/reflect (small JSON bodies); every other request — uploads,
MCP streams, etc. — passes straight through untouched, so there is no buffering
or latency cost elsewhere.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from ..cancellation import CancellationToken

# Key under which the per-request CancellationToken is stored on the ASGI scope.
# A dedicated top-level scope key (not scope["state"]) avoids any interaction
# with Starlette's per-request state copying.
SCOPE_CANCELLATION_TOKEN = "hindsight.cancellation_token"

_CLIENT_DISCONNECTED_REASON = "client disconnected"

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]


def _should_monitor(path: str) -> bool:
    """Only the two long-running, abandon-prone read endpoints need monitoring."""
    return path.endswith("/memories/recall") or path.endswith("/reflect")


class ClientDisconnectCancellationMiddleware:
    """Trip a scope-level CancellationToken when the client disconnects.

    Must be installed *outside* any ``BaseHTTPMiddleware`` so it owns the real
    ASGI ``receive`` channel.
    """

    def __init__(self, app: Callable) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _should_monitor(scope.get("path", "")):
            await self.app(scope, receive, send)
            return

        token = CancellationToken()
        scope[SCOPE_CANCELLATION_TOKEN] = token

        # The downstream app still needs to read the request body, so we cannot
        # simply consume `receive` ourselves. Instead a single pump task drains
        # the real channel, forwards every message to a queue the app reads from,
        # and trips the token the instant `http.disconnect` shows up — which the
        # app would otherwise never pull once it has finished reading the body.
        queue: asyncio.Queue = asyncio.Queue()

        async def pump() -> None:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    token.cancel(_CLIENT_DISCONNECTED_REASON)
                    await queue.put(message)
                    return
                await queue.put(message)

        async def proxied_receive() -> MutableMapping[str, Any]:
            return await queue.get()

        pump_task = asyncio.create_task(pump())
        try:
            await self.app(scope, proxied_receive, send)
        finally:
            pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump_task


def get_scope_cancellation_token(scope: Scope) -> CancellationToken | None:
    """Return the CancellationToken the middleware attached, if any."""
    return scope.get(SCOPE_CANCELLATION_TOKEN)
