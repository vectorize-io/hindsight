"""Report unknown query params and body fields, without paying for them per request.

The check is worth keeping -- a client sending ``max_token`` instead of
``max_tokens`` otherwise gets a silently different result -- but the way it used
to be done was expensive on every request, whether or not anything was wrong:

* a full second ``json.loads`` of the request body, *before* the route ran, so
  every payload was parsed twice;
* ``inspect.signature(endpoint)`` per request, uncached;
* two walks of ``app.routes`` calling ``route.matches(scope)`` to re-discover the
  route the router was about to resolve anyway;
* all of it inside a ``@app.middleware("http")``, i.e. a Starlette
  ``BaseHTTPMiddleware``, which adds a child task and two memory-stream hops per
  request (see :mod:`hindsight_api.api.observability` for what that costs).

Doing it as an ``APIRoute`` subclass removes all four. The route is *already*
resolved, so no matching or signature inspection is needed at request time: the
known names are computed once, at startup, from FastAPI's own ``dependant`` --
which is a more faithful notion of "known" than reading the raw signature was.
And the body is read through the same ``Request`` object FastAPI will use, whose
``json()`` caches, so the payload is parsed exactly once for both consumers.

The names are handed to the observability middleware through the ASGI scope
rather than set as a header here, so that -- as before -- the header survives on
responses produced by an exception handler above this layer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Coroutine
from typing import Any, cast

from fastapi import APIRouter, Request, Response
from fastapi.routing import APIRoute, request_response
from pydantic import BaseModel

from .observability import SCOPE_IGNORED_PARAMS

logger = logging.getLogger(__name__)


def _known_body_names(body_field: Any) -> set[str] | None:
    """Field names (and JSON aliases) of the request body model, or None if not a model.

    Mirrors the previous behaviour, which looked for the first parameter annotated
    with a pydantic model and used its fields.
    """
    if body_field is None:
        return None
    # FastAPI's ModelField exposes the annotation differently across its pydantic
    # compat layers: `field_info.annotation` on the v2 shim, `type_` on the older
    # one. Try both rather than pinning to whichever this version ships.
    model = getattr(getattr(body_field, "field_info", None), "annotation", None)
    if model is None:
        model = getattr(body_field, "type_", None)
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        return None
    names = set(model.model_fields.keys())
    for field in model.model_fields.values():
        # Pydantic models can expose public JSON names via aliases (for example
        # RetainRequest.async_ is sent as "async"). Treat aliases as known fields so
        # valid client payloads are not reported as ignored parameters.
        if isinstance(field.alias, str):
            names.add(field.alias)
    return names


class UnknownParamsRoute(APIRoute):
    """Route class that flags query params and body fields the endpoint ignores."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._compute_known_params()

    def _compute_known_params(self) -> None:
        """Cache what this endpoint accepts. Called once per route, at startup.

        `dependant` is FastAPI's resolved view of the endpoint, so both the
        parameter name and any alias a client may legitimately send are covered.
        """
        self._known_query: set[str] = set()
        for param in self.dependant.query_params:
            self._known_query.add(param.name)
            if isinstance(getattr(param, "alias", None), str):
                self._known_query.add(param.alias)
        self._path_params: set[str] = {p.name for p in self.dependant.path_params}
        self._known_body: set[str] | None = _known_body_names(self.body_field)

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            ignored = await self._collect_ignored(request)
            if ignored:
                joined = ", ".join(ignored)
                logger.warning(
                    "Unknown parameters ignored: [%s] for %s %s",
                    joined,
                    request.method,
                    request.url.path,
                )
                # Stashed on the scope; the observability middleware turns it into
                # the X-Ignored-Params header once the status is known.
                request.scope[SCOPE_IGNORED_PARAMS] = joined
            return await original_route_handler(request)

        return custom_route_handler

    async def _collect_ignored(self, request: Request) -> list[str]:
        ignored: list[str] = []

        for name in request.query_params:
            if name not in self._known_query and name not in self._path_params:
                ignored.append(name)

        if self._known_body is not None and request.method in ("POST", "PUT", "PATCH"):
            if "application/json" in request.headers.get("content-type", ""):
                try:
                    # Starlette caches both body() and json() on the Request, and
                    # FastAPI parses the body from this same object -- so this does
                    # not add a parse, it just gets there first.
                    body = await request.json()
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # A malformed body is FastAPI's error to report, not ours.
                    return ignored
                if isinstance(body, dict):
                    ignored.extend(key for key in body if key not in self._known_body)

        return ignored


def use_unknown_params_routes(router: APIRouter) -> None:
    """Make an about-to-be-included router contribute ``UnknownParamsRoute`` routes.

    ``app.router.route_class`` only covers routes declared directly on the app.
    Routes an HTTP extension hands over via ``include_router`` would otherwise
    arrive as plain ``APIRoute`` and silently lose unknown-param reporting, which
    the old middleware — sitting above the router — did cover.

    FastAPI resolves the class of an included route in two different ways
    depending on version, so both are covered here:

    * <= 0.140 builds the sub-router's routes eagerly and re-registers each one
      with ``route_class_override=type(route)``, so the *existing objects* decide.
      Re-class them, and redo the two things ``APIRoute.__init__`` derives from the
      class: the cached parameter sets and the ASGI app wrapping the route handler
      (FastAPI's ``request_response``, not Starlette's — it also opens the
      per-request dependency ``AsyncExitStack``).
    * >= 0.141 keeps the included router lazily and materialises its routes later
      from ``router.route_class``, so the *attribute* decides.

    Call before ``include_router``; if neither path applies on a future version
    the routes merely lose the header again, which a test guards against.
    """
    router.route_class = UnknownParamsRoute
    for route in router.routes:
        if type(route) is not APIRoute:
            continue
        route.__class__ = UnknownParamsRoute
        adopted = cast(UnknownParamsRoute, route)
        adopted._compute_known_params()
        adopted.app = request_response(adopted.get_route_handler())
