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

* record the request metrics (what ``http_metrics_middleware`` did), labelled by the
  template of the route the request hits (see :func:`http_endpoint_label`);
* attach ``X-Ignored-Params`` when the route stashed unknown parameters on the
  scope (see :mod:`hindsight_api.api.unknown_params`). Doing it here rather than
  in the route keeps the header on error responses too, which is what the old
  middleware did -- a route handler cannot add a header to a response produced by
  an exception handler above it.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Iterator, MutableMapping, Sequence
from typing import Any

import fastapi.routing
from starlette.routing import Match

from ..metrics import get_metrics_collector

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]

#: Scope key the route class uses to hand unknown parameter names to this middleware.
SCOPE_IGNORED_PARAMS = "hindsight.ignored_params"

#: Set by the route class when the request named a bank by an alias, to the alias it
#: used. Nothing downstream routes on it -- the path param already carries the real
#: bank id by then -- it is here so logs and traces can tell which id a caller sent,
#: which is the whole question during a phased migration off an old one.
SCOPE_RESOLVED_ALIAS = "hindsight.resolved_alias"

#: ``endpoint`` label of a request no route matches: scanner 404s, trailing-slash
#: redirects. One constant rather than the path, which the client fully controls.
UNMATCHED_ENDPOINT = "unmatched"

#: ``endpoint`` label of a request a route matched that has no path template to show
#: (a ``Host`` route, a wrapper type). Bounded like ``unmatched``, but kept apart from it
#: so ``unmatched`` keeps meaning "nothing matched".
UNKNOWN_ROUTE_ENDPOINT = "unknown_route"

# FastAPI >= 0.137 no longer copies an included router's routes into the app's route
# list: include_router() appends one _IncludedRouter node (nested for nested includes)
# that matches as a whole but carries no path template. Its routes have to be
# flattened into their effective route contexts -- each with matches() and the full,
# prefixed path_format -- before they can be matched one by one. >= 0.137.2 exposes
# that as the public iter_route_contexts(); 0.137.0-0.137.1 only as the private
# effective_route_contexts(); older versions need nothing. Looked up with getattr so
# the code type-checks against whichever FastAPI is installed. The OpenTelemetry
# FastAPI instrumentation flattens the same way for ``http.route``.
_iter_route_contexts: Callable[[Sequence[Any]], Iterator[Any]] | None = getattr(
    fastapi.routing, "iter_route_contexts", None
)


def _flatten_routes(routes: Sequence[Any]) -> Iterator[Any]:
    for route in routes:
        if hasattr(route, "path_format"):
            # An ordinary route (or Mount) is matched as-is: wrapping every route in a
            # RouteContext, as iter_route_contexts(routes) would, makes the lookup ~2.5x slower.
            yield route
        elif _iter_route_contexts is not None:
            yield from _iter_route_contexts([route])
        elif (effective_route_contexts := getattr(route, "effective_route_contexts", None)) is not None:
            yield from effective_route_contexts()
        else:
            yield route


def _route_template(route: Any) -> str:
    # path_format is the template with converters stripped ("{document_id:path}"
    # renders as "{document_id}").
    return getattr(route, "path_format", None) or UNKNOWN_ROUTE_ENDPOINT


def http_endpoint_label(scope: Scope) -> str:
    """The ``endpoint`` metric label for a request: the template of the route it hits.

    Previously the raw path with bank ids, UUIDs and numeric segments regex-templated
    (#2191). Every other path parameter -- document ids (32-hex or free-form, and
    ``:path`` so they span segments), memory/entity/operation ids, file keys -- passed
    through raw, one never-evicted series per id. The route template is bounded by the
    number of routes, and is what traces use for ``http.route``.

    Resolved before the request runs, not read from ``scope["route"]`` afterwards,
    because the in-progress up/down counter has to be incremented under the label it
    is later decremented with. It walks the (flattened, see ``_flatten_routes``) routes
    the way Starlette's router does -- first full match, else the first method-only
    match, i.e. a 405 -- which is the lookup the OpenTelemetry FastAPI instrumentation
    runs for ``http.route``, at a few to a few tens of microseconds per request.
    """
    router = getattr(scope.get("app"), "router", None)
    partial: Any = None
    for route in _flatten_routes(getattr(router, "routes", ())):
        match, _ = route.matches(scope)
        if match is Match.FULL:
            return _route_template(route)
        if match is Match.PARTIAL and partial is None:
            partial = route
    return _route_template(partial) if partial is not None else UNMATCHED_ENDPOINT


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

        # ASGI entry time, stashed for handlers that want to know how much of a request was spent
        # BEFORE their body ran. `handler_start` is set inside the endpoint, so routing, body
        # parsing and dependency resolution (auth among them) are invisible to any timer the
        # endpoint sets — which is exactly where an unexplained request cost can hide.
        scope["hs_asgi_t0"] = time.time()
        endpoint = http_endpoint_label(scope)
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

        with metrics_collector.record_http_request(method, endpoint, lambda: status_code[0]):
            await self.app(scope, receive, send_wrapper)
