"""Tests for the pure-ASGI HTTP observability middleware.

It took over two jobs from the `@app.middleware("http")` handlers it replaced, and
the second one exists for a reason that is easy to regress: the ignored-params
header must survive on error responses. A route handler cannot add a header to a
response produced by an exception handler above it, which is why the route stashes
the names on the scope and this middleware attaches them.
"""

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from hindsight_api.api.observability import SCOPE_IGNORED_PARAMS, HttpObservabilityMiddleware
from hindsight_api.api.unknown_params import UnknownParamsRoute


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


def test_scope_key_is_not_leaked_as_a_header_when_empty():
    client = TestClient(_app())
    response = client.get("/ok")
    assert SCOPE_IGNORED_PARAMS not in response.headers
    assert "X-Ignored-Params" not in response.headers


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
