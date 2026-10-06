"""Client-disconnect detection that works behind ``BaseHTTPMiddleware``.

``Request.is_disconnected()`` is the obvious way to notice an abandoned HTTP
request, but it is silently broken once any ``@app.middleware("http")``
(Starlette ``BaseHTTPMiddleware``) is installed: that middleware runs the route
in a child task behind anyio memory streams, so the ``http.disconnect`` ASGI
event never reaches the route's ``Request``. This app has such middlewares, so
the recall/reflect cancellation in #2122/#2127 never actually fired in
production — the disconnect was never observed.

This pure-ASGI middleware sits *outside* the ``BaseHTTPMiddleware`` layer, where
it still owns the real ``receive`` channel. For the routes it monitors (see
``_should_monitor``) it drains ``receive`` in a background task and trips a
:class:`CancellationToken` the moment ``http.disconnect`` arrives, stashing the
token on the ASGI ``scope``.
Recall and reflect copy that token onto their ``RequestContext`` and the engine
checks it at stage boundaries. The retain POST cancels its task outright instead,
because a sync retain spends its wall time inside one await (the LLM semaphore,
then the provider) where no checkpoint gets a turn — see
``run_task_cancellable_on_disconnect``. Either way, abandoned work stops instead
of running to completion.

It only wraps recall, reflect and the retain POST (small-to-moderate JSON
bodies); every other request — uploads, MCP streams, etc. — passes straight
through untouched, so there is no buffering or latency cost elsewhere. The pump
queue is bounded at one message so a monitored request's body is never held in
memory ahead of the app reading it.
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


def _should_monitor(scope: Scope) -> bool:
    """Which requests get a disconnect token: the long-running, abandon-prone ones.

    Recall and reflect are the reads. A synchronous retain (``async: false``) is the
    write: it runs the whole extraction inline, so an abandoned one keeps its place
    in the admission queue, then holds an LLM slot for its full run and commits —
    possibly into a bank deleted while it ran (issue #4526). DELETE shares the same
    path and is not a retain, hence the method check.
    """
    path = scope.get("path", "")
    if path.endswith("/memories/recall") or path.endswith("/reflect"):
        return True
    return path.endswith("/memories") and scope.get("method") == "POST"


class ClientDisconnectCancellationMiddleware:
    """Trip a scope-level CancellationToken when the client disconnects.

    Must be installed *outside* any ``BaseHTTPMiddleware`` so it owns the real
    ASGI ``receive`` channel.
    """

    def __init__(self, app: Callable) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _should_monitor(scope):
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
