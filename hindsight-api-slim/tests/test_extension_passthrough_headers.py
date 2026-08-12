"""Tests for HINDSIGHT_API_EXTENSION_PASSTHROUGH_HEADERS.

Covers config parsing plus the two transports that build a RequestContext from a
live request: the HTTP dependency and the MCP middleware.
"""

import pytest
from fastapi.testclient import TestClient

from hindsight_api.config import HindsightConfig, clear_config_cache
from hindsight_api.extensions import (
    AuthenticationError,
    RequestContext,
    Tenant,
    TenantContext,
    TenantExtension,
)

ASSERTION_HEADER = "x-user-assertion"
ENV_VAR = "HINDSIGHT_API_EXTENSION_PASSTHROUGH_HEADERS"


class RecordingTenantExtension(TenantExtension):
    """Captures every RequestContext it authenticates, then rejects the request.

    Rejecting keeps these tests on the auth path only — the request never reaches
    an engine method, so no database work is needed to observe what the transport
    put in the context.
    """

    def __init__(self):
        super().__init__({})
        self.contexts: list[RequestContext] = []

    async def authenticate(self, context: RequestContext) -> TenantContext:
        self.contexts.append(context)
        raise AuthenticationError("recorded")

    async def list_tenants(self) -> list[Tenant]:
        return [Tenant(schema="public")]


@pytest.fixture
def set_passthrough(monkeypatch):
    """Set the allowlist env var and invalidate the cached config around the test."""

    def _set(value: str | None) -> None:
        if value is None:
            monkeypatch.delenv(ENV_VAR, raising=False)
        else:
            monkeypatch.setenv(ENV_VAR, value)
        clear_config_cache()

    yield _set
    clear_config_cache()


@pytest.fixture
def recording_client(memory, set_passthrough):
    """Build a TestClient whose tenant extension records the request context."""
    from hindsight_api.api.http import create_app

    def _build(allowlist: str | None) -> tuple[TestClient, RecordingTenantExtension]:
        set_passthrough(allowlist)
        ext = RecordingTenantExtension()
        memory._tenant_extension = ext
        return TestClient(create_app(memory, initialize_memory=False)), ext

    return _build


def _last_context(ext: RecordingTenantExtension) -> RequestContext:
    assert ext.contexts, "tenant extension was never called"
    return ext.contexts[-1]


class TestConfigParsing:
    """HindsightConfig parsing of the allowlist."""

    def test_empty_by_default(self, set_passthrough):
        set_passthrough(None)
        assert HindsightConfig.from_env().extension_passthrough_headers == []

    def test_parses_and_lowercases_entries(self, set_passthrough):
        set_passthrough("X-User-Assertion, X-Request-Origin")
        assert HindsightConfig.from_env().extension_passthrough_headers == [
            "x-user-assertion",
            "x-request-origin",
        ]

    def test_ignores_blank_entries(self, set_passthrough):
        set_passthrough(" , x-user-assertion , ")
        assert HindsightConfig.from_env().extension_passthrough_headers == ["x-user-assertion"]


class TestHttpTransport:
    """RequestContext.extra_headers as built by the HTTP dependency."""

    def test_forwards_allowlisted_header(self, recording_client):
        client, ext = recording_client(ASSERTION_HEADER)

        client.get("/v1/default/banks", headers={ASSERTION_HEADER: "token-abc"})

        assert _last_context(ext).extra_headers == {ASSERTION_HEADER: "token-abc"}

    def test_empty_when_unset(self, recording_client):
        client, ext = recording_client(None)

        client.get("/v1/default/banks", headers={ASSERTION_HEADER: "token-abc"})

        assert _last_context(ext).extra_headers == {}

    def test_empty_when_header_absent(self, recording_client):
        client, ext = recording_client(ASSERTION_HEADER)

        client.get("/v1/default/banks")

        assert _last_context(ext).extra_headers == {}

    def test_matches_header_name_case_insensitively(self, recording_client):
        client, ext = recording_client("X-User-Assertion")

        client.get("/v1/default/banks", headers={"X-USER-ASSERTION": "token-abc"})

        assert _last_context(ext).extra_headers == {ASSERTION_HEADER: "token-abc"}

    def test_does_not_forward_unlisted_headers(self, recording_client):
        client, ext = recording_client(ASSERTION_HEADER)

        client.get(
            "/v1/default/banks",
            headers={ASSERTION_HEADER: "token-abc", "x-secret": "nope"},
        )

        assert _last_context(ext).extra_headers == {ASSERTION_HEADER: "token-abc"}

    def test_authorization_still_parsed_into_api_key(self, recording_client):
        client, ext = recording_client(ASSERTION_HEADER)

        client.get(
            "/v1/default/banks",
            headers={"Authorization": "Bearer shared-key", ASSERTION_HEADER: "token-abc"},
        )

        context = _last_context(ext)
        assert context.api_key == "shared-key"
        assert context.extra_headers == {ASSERTION_HEADER: "token-abc"}


class TestMcpTransport:
    """Header collection in the MCP ASGI middleware."""

    @staticmethod
    def _middleware(memory):
        from hindsight_api.api.mcp import MCPMiddleware

        # Pre-created app slots skip MCP server construction — this only exercises
        # header collection, which needs no server.
        return MCPMiddleware(
            app=None,
            memory=memory,
            multi_bank_app=object(),
            single_bank_app=object(),
        )

    @staticmethod
    def _scope(*headers: tuple[str, str]) -> dict:
        return {"headers": [(name.encode(), value.encode()) for name, value in headers]}

    def test_forwards_allowlisted_header(self, memory, set_passthrough):
        set_passthrough(ASSERTION_HEADER)
        middleware = self._middleware(memory)

        extra = middleware._get_extra_headers(self._scope((ASSERTION_HEADER, "token-abc")))

        assert extra == {ASSERTION_HEADER: "token-abc"}

    def test_empty_when_unset(self, memory, set_passthrough):
        set_passthrough(None)
        middleware = self._middleware(memory)

        extra = middleware._get_extra_headers(self._scope((ASSERTION_HEADER, "token-abc")))

        assert extra == {}

    def test_empty_when_header_absent(self, memory, set_passthrough):
        set_passthrough(ASSERTION_HEADER)
        middleware = self._middleware(memory)

        extra = middleware._get_extra_headers(self._scope(("authorization", "Bearer shared-key")))

        assert extra == {}

    def test_matches_header_name_case_insensitively(self, memory, set_passthrough):
        set_passthrough("X-User-Assertion")
        middleware = self._middleware(memory)

        extra = middleware._get_extra_headers(self._scope(("X-USER-ASSERTION", "token-abc")))

        assert extra == {ASSERTION_HEADER: "token-abc"}

    def test_does_not_forward_unlisted_headers(self, memory, set_passthrough):
        set_passthrough(ASSERTION_HEADER)
        middleware = self._middleware(memory)

        extra = middleware._get_extra_headers(self._scope((ASSERTION_HEADER, "token-abc"), ("x-secret", "nope")))

        assert extra == {ASSERTION_HEADER: "token-abc"}


class TestMcpToolsConfig:
    """RequestContext built for MCP tool calls carries the resolved headers."""

    def test_resolver_populates_extra_headers(self):
        from hindsight_api.mcp_tools import MCPToolsConfig, _get_request_context

        config = MCPToolsConfig(
            bank_id_resolver=lambda: "test-bank",
            extra_headers_resolver=lambda: {ASSERTION_HEADER: "token-abc"},
        )

        assert _get_request_context(config).extra_headers == {ASSERTION_HEADER: "token-abc"}

    def test_empty_without_resolver(self):
        from hindsight_api.mcp_tools import MCPToolsConfig, _get_request_context

        config = MCPToolsConfig(bank_id_resolver=lambda: "test-bank")

        assert _get_request_context(config).extra_headers == {}
