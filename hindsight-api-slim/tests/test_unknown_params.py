"""Tests for unknown parameter detection (X-Ignored-Params header).

These drive the REAL implementation -- `UnknownParamsRoute` plus the pure-ASGI
`HttpObservabilityMiddleware` that turns the scope entry into a header. The file
previously defined its own inline copy of the old `@app.middleware("http")`
version, so it passed no matter what the shipped code did; the behaviour it
describes is the contract, so the cases are unchanged and only the app under test
is now the real one.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from hindsight_api.api.observability import HttpObservabilityMiddleware
from hindsight_api.api.unknown_params import UnknownParamsRoute


class ItemRequest(BaseModel):
    name: str
    value: int = 0


class AliasedRequest(BaseModel):
    name: str
    async_: bool = Field(default=False, alias="async")


def _make_test_app() -> FastAPI:
    """A minimal app wired exactly as `create_app` wires the real one."""
    app = FastAPI()
    # Must be set before any route is registered -- the route class is applied at
    # registration time, not at request time.
    app.router.route_class = UnknownParamsRoute
    app.add_middleware(HttpObservabilityMiddleware)

    @app.get("/items")
    async def list_items(limit: int = 10, offset: int = 0):
        return {"items": [], "limit": limit, "offset": offset}

    @app.get("/items/{item_id}")
    async def get_item(item_id: str, details: bool = False):
        return {"id": item_id, "details": details}

    @app.post("/items")
    async def create_item(request: ItemRequest):
        return {"name": request.name, "value": request.value}

    @app.post("/aliased")
    async def create_aliased(request: AliasedRequest):
        return {"name": request.name, "async": request.async_}

    return app


@pytest.fixture
def client():
    return TestClient(_make_test_app())


class TestUnknownQueryParams:
    def test_known_params_no_header(self, client):
        resp = client.get("/items", params={"limit": 5, "offset": 0})
        assert resp.status_code == 200
        assert "X-Ignored-Params" not in resp.headers

    def test_unknown_query_param_sets_header(self, client):
        resp = client.get("/items", params={"limit": 5, "tag": "foo"})
        assert resp.status_code == 200
        assert "X-Ignored-Params" in resp.headers
        assert "tag" in resp.headers["X-Ignored-Params"]

    def test_multiple_unknown_query_params(self, client):
        resp = client.get("/items", params={"limit": 5, "tag": "foo", "created_after": "2024-01-01"})
        assert resp.status_code == 200
        ignored = resp.headers["X-Ignored-Params"]
        assert "tag" in ignored
        assert "created_after" in ignored

    def test_path_params_not_flagged(self, client):
        resp = client.get("/items/abc123", params={"details": "true"})
        assert resp.status_code == 200
        assert "X-Ignored-Params" not in resp.headers

    def test_unknown_with_path_param(self, client):
        resp = client.get("/items/abc123", params={"details": "true", "unknown": "x"})
        assert resp.status_code == 200
        assert "X-Ignored-Params" in resp.headers
        assert "unknown" in resp.headers["X-Ignored-Params"]

    def test_no_query_params_no_header(self, client):
        resp = client.get("/items")
        assert resp.status_code == 200
        assert "X-Ignored-Params" not in resp.headers


class TestUnknownBodyFields:
    def test_known_body_fields_no_header(self, client):
        resp = client.post("/items", json={"name": "test", "value": 42})
        assert resp.status_code == 200
        assert "X-Ignored-Params" not in resp.headers

    def test_body_field_alias_no_header(self, client):
        resp = client.post("/aliased", json={"name": "test", "async": True})
        assert resp.status_code == 200
        assert resp.json()["async"] is True
        assert "X-Ignored-Params" not in resp.headers

    def test_unknown_body_field_sets_header(self, client):
        resp = client.post("/items", json={"name": "test", "value": 42, "extra_field": "surprise"})
        assert resp.status_code == 200
        assert "X-Ignored-Params" in resp.headers
        assert "extra_field" in resp.headers["X-Ignored-Params"]

    def test_multiple_unknown_body_fields(self, client):
        resp = client.post("/items", json={"name": "test", "foo": 1, "bar": 2})
        assert resp.status_code == 200
        ignored = resp.headers["X-Ignored-Params"]
        assert "foo" in ignored
        assert "bar" in ignored
