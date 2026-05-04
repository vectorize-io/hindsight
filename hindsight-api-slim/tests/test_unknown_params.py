"""Tests for unknown parameter detection middleware (X-Ignored-Params header)."""

import pytest
from fastapi import FastAPI, Query
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field


def _make_test_app() -> FastAPI:
    """Create a minimal FastAPI app with the unknown params middleware."""
    import json
    import logging

    app = FastAPI()
    logger = logging.getLogger(__name__)

    @app.middleware("http")
    async def unknown_params_middleware(request, call_next):
        from starlette.routing import Match

        ignored_params: list[str] = []

        if request.query_params:
            for route in app.routes:
                match, _ = route.matches(request.scope)
                if match == Match.FULL:
                    endpoint = getattr(route, "endpoint", None)
                    if endpoint:
                        import inspect

                        sig = inspect.signature(endpoint)
                        declared = set(sig.parameters.keys())
                        path_params = set(getattr(route, "param_convertors", {}).keys()) | set(
                            request.path_params.keys()
                        )
                        known_query = declared - path_params
                        for name in request.query_params:
                            if name not in known_query and name not in path_params:
                                ignored_params.append(name)
                    break

        body_ignored: list[str] = []
        content_type = request.headers.get("content-type", "")
        if request.method in ("POST", "PUT", "PATCH") and "application/json" in content_type:
            try:
                body_bytes = await request.body()
                if body_bytes:
                    body_json = json.loads(body_bytes)
                    if isinstance(body_json, dict):
                        for route in app.routes:
                            match, _ = route.matches(request.scope)
                            if match == Match.FULL:
                                endpoint = getattr(route, "endpoint", None)
                                if endpoint:
                                    import inspect

                                    sig = inspect.signature(endpoint)
                                    for param in sig.parameters.values():
                                        ann = param.annotation
                                        if isinstance(ann, type) and issubclass(ann, BaseModel):
                                            known_fields = set(ann.model_fields.keys())
                                            for field in ann.model_fields.values():
                                                if isinstance(field.alias, str):
                                                    known_fields.add(field.alias)
                                            for key in body_json:
                                                if key not in known_fields:
                                                    body_ignored.append(key)
                                            break
                                break
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        all_ignored = ignored_params + body_ignored
        response = await call_next(request)

        if all_ignored:
            ignored_str = ", ".join(all_ignored)
            logger.warning(
                "Unknown parameters ignored: [%s] for %s %s",
                ignored_str,
                request.method,
                request.url.path,
            )
            response.headers["X-Ignored-Params"] = ignored_str

        return response

    class ItemRequest(BaseModel):
        name: str
        value: int = 0

    class AliasedRequest(BaseModel):
        name: str
        async_: bool = Field(default=False, alias="async")

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
