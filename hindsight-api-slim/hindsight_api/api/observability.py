"""Pure-ASGI HTTP observability: request metrics, and the ignored-params header.

This replaces two ``@app.middleware("http")`` handlers. That decorator installs a
Starlette ``BaseHTTPMiddleware``, which per request spawns a child task and pipes
the response through a pair of anyio memory-object streams. Measured against the
real API on a 2-CPU container at 32 concurrent clients, removing the two of them
took ``/health/live`` from ~1500 to ~5100 rps (p99 159ms -> 33ms) while *lowering*
CPU from a saturated core to ~65% -- the cost is scheduling hops, not compute, so
it grows exactly when the event loop is already contended.

A pure-ASGI middleware has no such overhead: it is one ``await`` in the same task,
wrapping ``send`` to observe the response.

Two responsibilities live here because both need the same hook -- the moment the
response status is known:

* record the request metrics (what ``http_metrics_middleware`` did);
* attach ``X-Ignored-Params`` when the route stashed unknown parameters on the
  scope (see :mod:`hindsight_api.api.unknown_params`). Doing it here rather than
  in the route keeps the header on error responses too, which is what the old
  middleware did -- a route handler cannot add a header to a response produced by
  an exception handler above it.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from ..metrics import get_metrics_collector, normalize_http_endpoint

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]

#: Scope key the route class uses to hand unknown parameter names to this middleware.
SCOPE_IGNORED_PARAMS = "hindsight.ignored_params"


def _header_safe(value: str) -> bytes:
    """Encode an attacker-supplied string for use as a header value.

    The names in this header come straight from the client's query string and JSON
    body, percent-decoded — so they can carry anything, including CR/LF (response
    splitting) and non-latin-1 characters, which are simply not encodable in a
    header and would otherwise raise mid-``send`` and turn a harmless typo'd query
    param into a 500. Keep printable ASCII, drop the rest.
    """
    return "".join(ch if " " <= ch <= "~" else "?" for ch in value).encode("ascii")


class HttpObservabilityMiddleware:
    """Record per-request metrics and inject ``X-Ignored-Params``."""

    def __init__(self, app: Callable) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Template id segments (bank ids, UUIDs, numeric ids) so the endpoint
        # metric label stays bounded-cardinality.
        # ASGI entry time, stashed for handlers that want to know how much of a request was spent
        # BEFORE their body ran. `handler_start` is set inside the endpoint, so routing, body
        # parsing and dependency resolution (auth among them) are invisible to any timer the
        # endpoint sets — which is exactly where an unexplained request cost can hide.
        scope["hs_asgi_t0"] = time.time()
        path = normalize_http_endpoint(scope.get("path", ""))
        method = scope.get("method", "GET")

        # Default to 500: if the app raises before sending a response start, the
        # request did fail, and the metric should say so rather than go unrecorded.
        status_code = [500]
        metrics_collector = get_metrics_collector()

        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status_code[0] = message["status"]
                ignored = scope.get(SCOPE_IGNORED_PARAMS)
                if ignored:
                    # Headers are a raw list of byte pairs at this layer; the route
                    # has already joined and logged the names.
                    message.setdefault("headers", []).append((b"x-ignored-params", _header_safe(ignored)))
            await send(message)

        with metrics_collector.record_http_request(method, path, lambda: status_code[0]):
            await self.app(scope, receive, send_wrapper)
