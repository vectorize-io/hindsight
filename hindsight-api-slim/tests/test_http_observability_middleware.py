"""Tests for the pure-ASGI HTTP observability middleware.

It took over two jobs from the `@app.middleware("http")` handlers it replaced, and
the second one exists for a reason that is easy to regress: the ignored-params
header must survive on error responses. A route handler cannot add a header to a
response produced by an exception handler above it, which is why the route stashes
the names on the scope and this middleware attaches them.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.testclient import TestClient

from hindsight_api.api.http import create_app
from hindsight_api.api import observability
from hindsight_api.api.observability import (
    UNKNOWN_ROUTE_ENDPOINT,
    UNMATCHED_ENDPOINT,
    HttpObservabilityMiddleware,
)
from hindsight_api.api.unknown_params import UnknownParamsRoute, use_unknown_params_routes
from hindsight_api.extensions import HttpExtension


def _app() -> FastAPI:
    app = FastAPI()
    app.router.route_class = UnknownParamsRoute
    app.add_middleware(HttpObservabilityMiddleware)

    @app.get("/ok")
    async def ok(limit: int = 10):
        return {"limit": limit}

    @app.get("/boom")
    async def boom(limit: int = 10):
        raise HTTPException(status_code=418, detail="teapot")

    return app


def test_no_header_when_everything_is_known():
    client = TestClient(_app())
    response = client.get("/ok", params={"limit": 5})
    assert response.status_code == 200
    assert "X-Ignored-Params" not in response.headers


def test_header_present_on_success():
    client = TestClient(_app())
    response = client.get("/ok", params={"limit": 5, "nope": 1})
    assert response.status_code == 200
    assert response.headers["X-Ignored-Params"] == "nope"


def test_header_survives_an_error_response():
    """The old BaseHTTPMiddleware tagged error responses too; keep that."""
    client = TestClient(_app(), raise_server_exceptions=False)
    response = client.get("/boom", params={"nope": 1})
    assert response.status_code == 418
    assert response.headers["X-Ignored-Params"] == "nope"


def test_metrics_are_recorded_for_each_request():
    from hindsight_api.metrics import get_metrics_collector

    client = TestClient(_app())
    collector = get_metrics_collector()
    # The collector is a process singleton; assert it is exercised rather than
    # asserting an absolute count, which other tests in the session would perturb.
    calls: list[tuple[str, str]] = []
    original = collector.record_http_request

    def spy(method: str, endpoint: str, status_code_getter):
        calls.append((method, endpoint))
        return original(method, endpoint, status_code_getter)

    collector.record_http_request = spy  # type: ignore[method-assign]
    try:
        client.get("/ok")
    finally:
        collector.record_http_request = original  # type: ignore[method-assign]

    assert ("GET", "/ok") in calls


def test_hostile_param_names_do_not_break_the_response():
    """The names are client-controlled, so they cannot be trusted as a header value.

    A non-latin-1 name is unencodable in a header and a name carrying CR/LF would
    split the response; either one used to happen inside `send`, turning a typo'd
    query param into a 500.
    """
    client = TestClient(_app())
    response = client.get("/ok", params={"\u65e5\u672c": 1, "a\r\nX-Evil: 1": 2})
    assert response.status_code == 200
    value = response.headers["X-Ignored-Params"]
    assert "\r" not in value and "\n" not in value
    assert "X-Evil" not in response.headers


def test_routes_from_an_included_router_still_report_unknown_params():
    """`include_router` does not apply the app's route_class; the helper does."""
    app = FastAPI()
    app.router.route_class = UnknownParamsRoute

    sub = APIRouter()

    @sub.get("/thing")
    async def thing(limit: int = 10):
        return {"limit": limit}

    use_unknown_params_routes(sub)
    app.include_router(sub, prefix="/ext")
    app.add_middleware(HttpObservabilityMiddleware)

    response = TestClient(app).get("/ext/thing", params={"nope": 1})
    assert response.status_code == 200
    assert response.headers["X-Ignored-Params"] == "nope"


# --- endpoint label -----------------------------------------------------------


class _ExtensionRoutes(HttpExtension):
    def get_router(self, memory) -> APIRouter:
        router = APIRouter()

        @router.get("/things/{thing_id}")
        async def thing(thing_id: str):
            return {"thing_id": thing_id}

        # A nested include: FastAPI >= 0.137 keeps it as a router node inside a router node.
        nested = APIRouter()

        @nested.get("/items/{item_id:path}")
        async def item(item_id: str):
            return {"item_id": item_id}

        router.include_router(nested, prefix="/nested/{group}")
        return router


def _recorded_endpoints(app: FastAPI, method: str, paths: list[str]) -> list[str]:
    """Send each request through ``app`` and return the endpoint label it was recorded under."""
    from hindsight_api.metrics import get_metrics_collector

    collector = get_metrics_collector()
    endpoints: list[str] = []
    original = collector.record_http_request

    def spy(method: str, endpoint: str, status_code_getter):
        endpoints.append(endpoint)
        return original(method, endpoint, status_code_getter)

    # The handlers run against a mock engine and may well fail; only the label matters.
    client = TestClient(app, raise_server_exceptions=False)
    collector.record_http_request = spy  # type: ignore[method-assign]
    try:
        for path in paths:
            client.request(method, path)
    finally:
        collector.record_http_request = original  # type: ignore[method-assign]
    return endpoints


@pytest.fixture(scope="module")
def real_app() -> FastAPI:
    """The real route table, with an HTTP extension, so the labels come from the routes users hit."""
    return create_app(MagicMock(), initialize_memory=False, http_extension=_ExtensionRoutes({}))


@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        # Bank ids, numeric and non-numeric (what #2191 templated).
        ("GET", "/v1/default/banks/user-1680/config", "/v1/default/banks/{bank_id}/config"),
        ("GET", "/v1/default/banks/42/config", "/v1/default/banks/{bank_id}/config"),
        ("GET", "/v1/default/banks", "/v1/default/banks"),
        # Document ids: dashless hex, free-form, and -- `{document_id:path}` -- containing slashes.
        (
            "GET",
            "/v1/default/banks/b1/documents/0123456789abcdef0123456789abcdef",
            "/v1/default/banks/{bank_id}/documents/{document_id}",
        ),
        (
            "GET",
            "/v1/default/banks/b1/documents/abcde_fghijk_123456",
            "/v1/default/banks/{bank_id}/documents/{document_id}",
        ),
        (
            "GET",
            "/v1/default/banks/b1/documents/a/b/c/chunks",
            "/v1/default/banks/{bank_id}/documents/{document_id}/chunks",
        ),
        ("GET", "/v1/default/files/download/some/nested/key.pdf", "/v1/default/files/download/{key}"),
        ("GET", "/v1/default/banks/b1/memories/mem_7f3a", "/v1/default/banks/{bank_id}/memories/{memory_id}"),
        # A method the route does not serve (405) still names the route.
        ("GET", "/v1/default/banks/b1/reflect", "/v1/default/banks/{bank_id}/reflect"),
        ("GET", "/health", "/health"),
        # Extension routes are mounted under /ext and resolve to their template like any other.
        ("GET", "/ext/things/t-123", "/ext/things/{thing_id}"),
        ("GET", "/ext/nested/g1/items/a/b", "/ext/nested/{group}/items/{item_id}"),
        # Nothing matches: one constant, never the client-chosen path.
        ("GET", "/wp-admin/setup-config.php", UNMATCHED_ENDPOINT),
        ("GET", "/v1/default/banks/b1/no-such-thing/xyz", UNMATCHED_ENDPOINT),
    ],
)
def test_endpoint_label_is_the_route_template(real_app: FastAPI, method: str, path: str, expected: str):
    assert _recorded_endpoints(real_app, method, [path]) == [expected]


def test_in_progress_returns_to_zero_under_the_route_template(real_app: FastAPI, monkeypatch):
    """In-progress is +1 at the start and -1 at the end; both must land on the same series."""
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    from hindsight_api import metrics

    reader = InMemoryMetricReader()
    monkeypatch.setattr(metrics, "get_meter", lambda: MeterProvider(metric_readers=[reader]).get_meter("test"))
    monkeypatch.setattr(metrics, "_metrics_collector", metrics.MetricsCollector())

    client = TestClient(real_app, raise_server_exceptions=False)
    for doc_id in ("0123456789abcdef0123456789abcdef", "abcde_fghijk_123456", "a/b/c"):
        client.get(f"/v1/default/banks/b1/documents/{doc_id}")
    client.get("/wp-admin/setup-config.php")

    points = {
        (metric.name, dict(point.attributes)["endpoint"]): point
        for rm in reader.get_metrics_data().resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
        if metric.name.startswith("hindsight.http.")
        for point in metric.data.data_points
    }
    doc = "/v1/default/banks/{bank_id}/documents/{document_id}"
    assert {endpoint for _, endpoint in points} == {doc, UNMATCHED_ENDPOINT}
    assert points[("hindsight.http.requests.in_progress", doc)].value == 0
    assert points[("hindsight.http.requests.in_progress", UNMATCHED_ENDPOINT)].value == 0


def test_a_matched_route_without_a_template_is_not_unmatched():
    """A Host route matches but has no path; it must not read as a 404."""
    from starlette.routing import Host

    app = FastAPI()
    app.add_middleware(HttpObservabilityMiddleware)
    app.router.routes.append(Host("testserver", app=FastAPI()))

    assert _recorded_endpoints(app, "GET", ["/anything"]) == [UNKNOWN_ROUTE_ENDPOINT]


def test_included_router_nodes_are_flattened_without_the_public_helper(monkeypatch):
    """FastAPI 0.137.0-0.137.1 has the router node but not iter_route_contexts().

    Whatever FastAPI the suite runs on, check that fallback descends into a node
    exposing effective_route_contexts() instead of matching (or skipping) the node.
    """

    class _Node:
        def __init__(self, *children):
            self.children = children

        def effective_route_contexts(self):
            for child in self.children:
                if isinstance(child, _Node):
                    yield from child.effective_route_contexts()
                else:
                    yield child

    monkeypatch.setattr(observability, "_iter_route_contexts", None)
    plain, inner_a, inner_b = object(), object(), object()
    assert list(observability._flatten_routes([plain, _Node(inner_a, _Node(inner_b))])) == [plain, inner_a, inner_b]
