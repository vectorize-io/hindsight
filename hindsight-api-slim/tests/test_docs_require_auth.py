"""HINDSIGHT_API_DOCS_REQUIRE_AUTH gates /docs, /redoc and /openapi.json.

By default the three are public (an open, browsable API is friendly to
self-hosters). When the flag is set they are re-served behind the same
authentication the data routes use, so the endpoint/schema surface cannot be
enumerated anonymously. With the default no-auth tenant extension that
authentication is a pass-through, so these tests drive both sides: the keyless
default (served) and a configured extension that rejects (401).
"""

import os

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.config import clear_config_cache

DOCS_PATHS = ("/openapi.json", "/docs", "/redoc")


def _app(memory):
    clear_config_cache()  # re-read HINDSIGHT_API_DOCS_REQUIRE_AUTH from the env
    return create_app(memory, initialize_memory=False)


@pytest_asyncio.fixture
async def docs_auth_env():
    os.environ["HINDSIGHT_API_DOCS_REQUIRE_AUTH"] = "true"
    yield
    os.environ.pop("HINDSIGHT_API_DOCS_REQUIRE_AUTH", None)
    clear_config_cache()


async def _get(app, path):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


@pytest.mark.asyncio
async def test_docs_are_public_by_default(memory):
    """Unset flag: the built-in docs stay open, as they always have."""
    os.environ.pop("HINDSIGHT_API_DOCS_REQUIRE_AUTH", None)
    app = _app(memory)
    for path in DOCS_PATHS:
        assert (await _get(app, path)).status_code == 200


@pytest.mark.asyncio
async def test_docs_served_to_an_authenticated_caller(memory, docs_auth_env):
    """Flag on, auth passes (the default no-auth extension): still served."""
    app = _app(memory)
    for path in DOCS_PATHS:
        r = await _get(app, path)
        assert r.status_code == 200, path
    # And it is our schema, not a stub.
    assert (await _get(app, "/openapi.json")).json()["info"]["title"] == "Hindsight HTTP API"


@pytest.mark.asyncio
async def test_openapi_and_docs_carry_the_base_path(memory):
    """base_path + flag on: the served spec still advertises the base path in
    ``servers`` and the UI fetches the spec under it, as FastAPI's built-in
    routes would — the custom routes must not drop the root_path."""
    os.environ["HINDSIGHT_API_DOCS_REQUIRE_AUTH"] = "true"
    os.environ["HINDSIGHT_API_BASE_PATH"] = "/hindsight"
    try:
        app = _app(memory)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test/hindsight") as client:
            spec = await client.get("/openapi.json")
            assert spec.status_code == 200
            servers = spec.json().get("servers", [])
            assert any(s.get("url") == "/hindsight" for s in servers), servers
            docs = await client.get("/docs")
            assert docs.status_code == 200
            assert "/hindsight/openapi.json" in docs.text
    finally:
        os.environ.pop("HINDSIGHT_API_DOCS_REQUIRE_AUTH", None)
        os.environ.pop("HINDSIGHT_API_BASE_PATH", None)
        clear_config_cache()


@pytest.mark.asyncio
async def test_docs_rejected_when_authentication_fails(memory, docs_auth_env, monkeypatch):
    """Flag on, auth rejects (as a configured extension would for an anonymous
    caller): all three 401 instead of disclosing the surface."""
    from hindsight_api.extensions import AuthenticationError

    app = _app(memory)

    async def _reject(_request_context):
        raise AuthenticationError("API key required")

    monkeypatch.setattr(app.state.memory, "_authenticate_tenant", _reject)
    for path in DOCS_PATHS:
        assert (await _get(app, path)).status_code == 401, path
