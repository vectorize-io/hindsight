"""Tests for HINDSIGHT_API_EXTENSION_PASSTHROUGH_HEADERS.

Covers config parsing plus the two transports that build a RequestContext from a
live request: the HTTP dependency and the MCP middleware.
"""

import httpx
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

    def test_duplicated_header_is_not_forwarded(self, recording_client):
        """A second copy must not be able to override the one the proxy injected."""
        client, ext = recording_client(ASSERTION_HEADER)

        client.get(
            "/v1/default/banks",
            headers=[(ASSERTION_HEADER, "spoofed"), (ASSERTION_HEADER, "trusted")],
        )

        assert _last_context(ext).extra_headers == {}

    def test_unlisted_header_with_non_utf8_bytes_does_not_break_the_request(self, recording_client):
        """Header bytes are latin-1 on the wire; a stray one must not fail the request."""
        client, ext = recording_client(ASSERTION_HEADER)

        client.get(
            "/v1/default/banks",
            headers=[(b"user-agent", b"caf\xe9"), (ASSERTION_HEADER.encode(), b"token-abc")],
        )

        assert _last_context(ext).extra_headers == {ASSERTION_HEADER: "token-abc"}


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

    def test_duplicated_header_is_not_forwarded(self, memory, set_passthrough):
        """Same rule as HTTP: neither copy wins, so the two transports cannot disagree."""
        set_passthrough(ASSERTION_HEADER)
        middleware = self._middleware(memory)

        extra = middleware._get_extra_headers(self._scope((ASSERTION_HEADER, "spoofed"), (ASSERTION_HEADER, "trusted")))

        assert extra == {}

    def test_unlisted_header_with_non_utf8_bytes_does_not_raise(self, memory, set_passthrough):
        """ASGI header bytes are not necessarily UTF-8; decoding one must not 500 the request."""
        set_passthrough(ASSERTION_HEADER)
        middleware = self._middleware(memory)

        scope = {"headers": [(b"user-agent", b"caf\xe9"), (ASSERTION_HEADER.encode(), b"token-abc")]}

        assert middleware._get_extra_headers(scope) == {ASSERTION_HEADER: "token-abc"}


class TestReachesOperationValidator:
    """The headers survive the whole HTTP path, not just the auth hop."""

    @pytest.mark.asyncio
    async def test_validator_sees_the_headers(self, memory, set_passthrough):
        from hindsight_api.api.http import create_app
        from hindsight_api.extensions import (
            BankListContext,
            BankListResult,
            OperationValidatorExtension,
            ValidationResult,
        )

        captured: list[RequestContext] = []

        class CapturingValidator(OperationValidatorExtension):
            async def validate_retain(self, ctx) -> ValidationResult:
                return ValidationResult.accept()

            async def validate_recall(self, ctx) -> ValidationResult:
                return ValidationResult.accept()

            async def validate_reflect(self, ctx) -> ValidationResult:
                return ValidationResult.accept()

            async def filter_bank_list(self, ctx: BankListContext) -> BankListResult:
                captured.append(ctx.request_context)
                return BankListResult(banks=ctx.banks)

        set_passthrough(ASSERTION_HEADER)
        memory._operation_validator = CapturingValidator({})
        # An in-loop ASGI client: TestClient drives its own event loop, which the
        # engine's connection pool (created in this test's loop) cannot serve.
        transport = httpx.ASGITransport(app=create_app(memory, initialize_memory=False))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/v1/default/banks", headers={ASSERTION_HEADER: "token-abc"})

        assert response.status_code == 200
        assert captured, "operation validator was never called"
        assert captured[-1].extra_headers == {ASSERTION_HEADER: "token-abc"}


class TestSharedCollection:
    """Both transports go through collect_passthrough_headers, so it owns the rules."""

    @staticmethod
    def _collect(allowlist, *headers: tuple[bytes, bytes]) -> dict[str, str]:
        from hindsight_api.api.passthrough_headers import collect_passthrough_headers

        return collect_passthrough_headers(list(headers), allowlist)

    def test_empty_allowlist_forwards_nothing(self):
        assert self._collect([], (b"x-user-assertion", b"token-abc")) == {}

    def test_lower_cases_the_key(self):
        assert self._collect(["x-user-assertion"], (b"X-User-Assertion", b"token-abc")) == {
            ASSERTION_HEADER: "token-abc"
        }

    def test_drops_duplicates_and_keeps_the_rest(self):
        collected = self._collect(
            ["x-user-assertion", "x-request-origin"],
            (b"x-user-assertion", b"spoofed"),
            (b"x-user-assertion", b"trusted"),
            (b"x-request-origin", b"gateway"),
        )

        assert collected == {"x-request-origin": "gateway"}

    def test_decodes_values_as_latin1(self):
        assert self._collect(["x-user-assertion"], (b"x-user-assertion", b"caf\xe9")) == {ASSERTION_HEADER: "café"}


class TestMcpToolsConfig:
    """RequestContext built for MCP tool calls carries the resolved headers."""

    def test_resolver_populates_extra_headers(self):
        from hindsight_api.mcp_tools import MCPToolsConfig, _get_request_context

        config = MCPToolsConfig(
            bank_id_resolver=lambda: "test-bank",
            extra_headers_resolver=lambda: {ASSERTION_HEADER: "token-abc"},
        )

        assert _get_request_context(config).extra_headers == {ASSERTION_HEADER: "token-abc"}

    def test_each_context_owns_its_dict(self):
        """One tool call mutating extra_headers must not change what the next one sees."""
        from hindsight_api.api.mcp import _current_extra_headers, get_current_extra_headers

        token = _current_extra_headers.set({ASSERTION_HEADER: "token-abc"})
        try:
            first = get_current_extra_headers()
            first["injected"] = "nope"

            assert get_current_extra_headers() == {ASSERTION_HEADER: "token-abc"}
        finally:
            _current_extra_headers.reset(token)

    def test_empty_without_resolver(self):
        from hindsight_api.mcp_tools import MCPToolsConfig, _get_request_context

        config = MCPToolsConfig(bank_id_resolver=lambda: "test-bank")

        assert _get_request_context(config).extra_headers == {}
