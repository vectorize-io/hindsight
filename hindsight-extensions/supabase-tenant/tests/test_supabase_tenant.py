"""Tests for the Supabase Tenant Extension.

Supabase is stubbed as a real in-process ``aiohttp.web`` server, so the
extension's actual aiohttp transport (status handling, JSON parsing, timeouts,
connection errors) is exercised rather than a mocked client.
"""

import asyncio
import socket
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import jwt as pyjwt
import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from jwt import PyJWK

from hindsight_api.extensions.context import ExtensionContext
from hindsight_api.extensions.loader import load_extension
from hindsight_api.extensions.tenant import AuthenticationError, Tenant, TenantContext, TenantExtension
from hindsight_api.models import RequestContext
from hindsight_ext_supabase_tenant.extension import (
    JWKS_CACHE_TTL_SECONDS,
    JWKS_MIN_REFRESH_INTERVAL_SECONDS,
    MIN_TOKEN_LENGTH,
    REQUEST_TIMEOUT_SECONDS,
    SupabaseTenantExtension,
)

# A valid UUID for test user IDs
VALID_UUID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

JWKS_PATH = "/auth/v1/.well-known/jwks.json"
USER_PATH = "/auth/v1/user"
HEALTH_PATH = "/auth/v1/health"

# Minimal JWKS response with one RSA key
MOCK_JWKS_RESPONSE = {
    "keys": [
        {
            "kid": "test-key-1",
            "kty": "RSA",
            "alg": "RS256",
            "use": "sig",
            "n": "0vx7agoebGcQSuuPiLJXZptN9nndrQmbXEps2aiAFbWhM78LhWx4cbbfAAtVT86zwu1RK7aPFFxuhDR1L6tSoc_BJECPebWKRXjBZCiFV4n3oknjhMstn64tZ_2W-5JsGY4Hc5n9yBXArwl93lqt7_RN5w6Cf0h4QyQ5v-65YGjQR0_FDW2QvzqY368QQMicAtaSqzs8KJZgnYb9c7d0zgdAZHzu6qMQvRL5hajrn1n91CbOpbISD08qNLyrdkt-bFTWhAI4vMQFh6WeZu0fM4lFd2NcRwr3XPksINHaQ-G_xBniIqbw0Ls1jF44-csFCur-kEgU8awapJzKnqDKgw",
            "e": "AQAB",
        }
    ]
}


# ----------------------------------------------------------------------
# Supabase stub
# ----------------------------------------------------------------------

Responder = Callable[[web.Request], Awaitable[web.StreamResponse]]


@dataclass
class RecordedRequest:
    path: str
    headers: dict[str, str]


@dataclass
class SupabaseStub:
    """What the stub server has been asked, and how it answers (per path)."""

    base_url: str = ""
    requests: list[RecordedRequest] = field(default_factory=list)
    routes: dict[str, Responder] = field(default_factory=dict)

    def reply(self, path: str, status: int = 200, json_data: dict | None = None) -> None:
        async def responder(_request: web.Request) -> web.StreamResponse:
            return web.json_response(json_data or {}, status=status)

        self.routes[path] = responder

    def paths(self) -> list[str]:
        return [r.path for r in self.requests]


@pytest.fixture
async def supabase():
    stub = SupabaseStub()

    async def dispatch(request: web.Request) -> web.StreamResponse:
        stub.requests.append(RecordedRequest(path=request.path, headers=dict(request.headers)))
        responder = stub.routes.get(request.path)
        if responder is None:
            return web.json_response({"error": "not stubbed"}, status=404)
        return await responder(request)

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", dispatch)
    server = TestServer(app)
    await server.start_server()
    stub.base_url = str(server.make_url("")).rstrip("/")
    try:
        yield stub
    finally:
        await server.close()


def _closed_port_url() -> str:
    """A loopback URL on which nothing listens (connection refused)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"


@pytest.fixture
async def make_extension():
    """Build extensions; close any HTTP sessions they opened at teardown."""
    created: list[SupabaseTenantExtension] = []

    def factory(
        supabase_url: str = "https://test.supabase.co",
        service_key: str | None = "test-service-key",
        schema_prefix: str | None = None,
    ) -> SupabaseTenantExtension:
        config = {"supabase_url": supabase_url}
        if service_key is not None:
            config["supabase_service_key"] = service_key
        if schema_prefix is not None:
            config["schema_prefix"] = schema_prefix
        ext = SupabaseTenantExtension(config)
        created.append(ext)
        return ext

    yield factory
    for ext in created:
        await ext.on_shutdown()


def _started(ext: SupabaseTenantExtension, timeout: float = REQUEST_TIMEOUT_SECONDS) -> None:
    """Put the extension in the post-startup state without running on_startup."""
    ext._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, connect=timeout, sock_read=timeout))


def _make_valid_token() -> str:
    """Return a token that passes the MIN_TOKEN_LENGTH check."""
    return "a" * (MIN_TOKEN_LENGTH + 10)


def _setup_jwks_ext(make_extension, supabase_url: str = "https://test.supabase.co") -> SupabaseTenantExtension:
    """An extension in JWKS mode with a fresh key cache."""
    ext = make_extension(supabase_url=supabase_url)
    _started(ext)
    ext._use_jwks = True
    ext._jwks_keys = {"test-key-1": MagicMock(spec=PyJWK)}
    ext._jwks_keys["test-key-1"].key = "mock-public-key"
    ext._jwks_last_fetched = time.monotonic()
    return ext


def _setup_legacy_ext(make_extension, supabase_url: str, timeout: float = REQUEST_TIMEOUT_SECONDS):
    """An extension in legacy (/auth/v1/user) mode."""
    ext = make_extension(supabase_url=supabase_url)
    _started(ext, timeout)
    ext._use_jwks = False
    return ext


# ======================================================================
# Initialization
# ======================================================================


class TestSupabaseTenantExtensionInit:
    """Tests for extension initialization."""

    def test_init_with_valid_config(self, make_extension):
        ext = make_extension()
        assert ext.supabase_url == "https://test.supabase.co"
        assert ext.supabase_service_key == "test-service-key"
        assert ext.schema_prefix == "user"
        assert ext._initialized_schemas == set()
        assert ext._http is None
        assert ext._use_jwks is False
        assert ext._jwks_keys == {}

    def test_init_missing_supabase_url(self):
        with pytest.raises(ValueError, match="HINDSIGHT_API_TENANT_SUPABASE_URL is required"):
            SupabaseTenantExtension({})

    def test_init_without_service_key(self, make_extension):
        """Service key is optional — JWKS mode doesn't require it."""
        ext = make_extension(service_key=None)
        assert ext.supabase_service_key is None

    def test_init_default_schema_prefix(self, make_extension):
        ext = make_extension()
        assert ext.schema_prefix == "user"

    def test_init_custom_schema_prefix(self, make_extension):
        ext = make_extension(schema_prefix="tenant")
        assert ext.schema_prefix == "tenant"

    def test_init_strips_trailing_slash(self, make_extension):
        ext = make_extension(supabase_url="https://test.supabase.co/")
        assert ext.supabase_url == "https://test.supabase.co"

    def test_init_rejects_invalid_schema_prefix(self, make_extension):
        """Schema prefix with special characters should be rejected."""
        with pytest.raises(ValueError, match="Invalid schema_prefix"):
            make_extension(schema_prefix='"; DROP TABLE')

    def test_init_rejects_empty_schema_prefix(self, make_extension):
        with pytest.raises(ValueError, match="Invalid schema_prefix"):
            make_extension(schema_prefix="")

    def test_init_rejects_schema_prefix_starting_with_digit(self, make_extension):
        with pytest.raises(ValueError, match="Invalid schema_prefix"):
            make_extension(schema_prefix="123abc")

    def test_init_allows_underscore_prefix(self, make_extension):
        ext = make_extension(schema_prefix="_internal")
        assert ext.schema_prefix == "_internal"

    def test_is_tenant_extension_subclass(self, make_extension):
        ext = make_extension()
        assert isinstance(ext, TenantExtension)


# ======================================================================
# Startup — JWKS initialization
# ======================================================================


class TestSupabaseTenantExtensionStartup:
    """Tests for on_startup behavior."""

    async def test_on_startup_creates_http_client(self, supabase, make_extension):
        ext = make_extension(supabase_url=supabase.base_url)
        supabase.reply(JWKS_PATH, 200, MOCK_JWKS_RESPONSE)
        supabase.reply(HEALTH_PATH, 200)

        with patch("hindsight_ext_supabase_tenant.extension.PyJWK"):
            await ext.on_startup()

        assert isinstance(ext._http, aiohttp.ClientSession)

    async def test_on_startup_fetches_jwks(self, supabase, make_extension):
        ext = make_extension(supabase_url=supabase.base_url)
        supabase.reply(JWKS_PATH, 200, MOCK_JWKS_RESPONSE)
        supabase.reply(HEALTH_PATH, 200)

        with patch("hindsight_ext_supabase_tenant.extension.PyJWK") as mock_pyjwk:
            mock_pyjwk.return_value = MagicMock(spec=PyJWK)
            await ext.on_startup()

        assert ext._use_jwks is True
        # First call: JWKS fetch, second call: health check
        assert supabase.paths() == [JWKS_PATH, HEALTH_PATH]

    async def test_on_startup_falls_back_to_legacy_when_jwks_empty(self, supabase, make_extension):
        ext = make_extension(supabase_url=supabase.base_url)
        # JWKS returns empty keys, health check succeeds
        supabase.reply(JWKS_PATH, 200, {"keys": []})
        supabase.reply(HEALTH_PATH, 200)

        await ext.on_startup()

        assert ext._use_jwks is False

    async def test_on_startup_falls_back_to_legacy_when_jwks_fetch_fails(self, make_extension):
        # Nothing listens: the JWKS fetch (and the health check) cannot connect.
        ext = make_extension(supabase_url=_closed_port_url())

        await ext.on_startup()

        assert ext._use_jwks is False

    async def test_on_startup_falls_back_to_legacy_when_jwks_returns_error(self, supabase, make_extension):
        ext = make_extension(supabase_url=supabase.base_url)
        supabase.reply(JWKS_PATH, 503)
        supabase.reply(HEALTH_PATH, 200)

        await ext.on_startup()

        assert ext._use_jwks is False

    async def test_on_startup_raises_if_no_jwks_and_no_service_key(self, supabase, make_extension):
        ext = make_extension(supabase_url=supabase.base_url, service_key=None)
        supabase.reply(JWKS_PATH, 200, {"keys": []})

        with pytest.raises(ValueError, match="HINDSIGHT_API_TENANT_SUPABASE_SERVICE_KEY is required"):
            await ext.on_startup()

    async def test_on_startup_health_check_with_service_key(self, supabase, make_extension):
        ext = make_extension(supabase_url=supabase.base_url)
        supabase.reply(JWKS_PATH, 200, MOCK_JWKS_RESPONSE)
        supabase.reply(HEALTH_PATH, 200)

        with patch("hindsight_ext_supabase_tenant.extension.PyJWK"):
            await ext.on_startup()

        # Second call should be health check
        health_call = supabase.requests[1]
        assert health_call.path == HEALTH_PATH
        assert health_call.headers["apikey"] == "test-service-key"

    async def test_on_startup_skips_health_check_without_service_key(self, supabase, make_extension):
        ext = make_extension(supabase_url=supabase.base_url, service_key=None)
        supabase.reply(JWKS_PATH, 200, MOCK_JWKS_RESPONSE)

        with patch("hindsight_ext_supabase_tenant.extension.PyJWK"):
            await ext.on_startup()

        # Only one call: JWKS fetch, no health check
        assert supabase.paths() == [JWKS_PATH]


# ======================================================================
# JWKS cache management
# ======================================================================


class TestJWKSCacheManagement:
    """Tests for JWKS key fetching, caching, and rotation handling."""

    async def test_get_signing_key_from_cache(self, make_extension):
        ext = _setup_jwks_ext(make_extension)

        with patch("hindsight_ext_supabase_tenant.extension.pyjwt.get_unverified_header") as mock_header:
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}
            key = await ext._get_signing_key("fake-token")

        assert key is ext._jwks_keys["test-key-1"]

    async def test_get_signing_key_refreshes_stale_cache(self, supabase, make_extension):
        ext = _setup_jwks_ext(make_extension, supabase.base_url)
        # Make cache expired
        ext._jwks_last_fetched = time.monotonic() - JWKS_CACHE_TTL_SECONDS - 1

        new_key = MagicMock(spec=PyJWK)
        supabase.reply(JWKS_PATH, 200, MOCK_JWKS_RESPONSE)

        with (
            patch("hindsight_ext_supabase_tenant.extension.pyjwt.get_unverified_header") as mock_header,
            patch("hindsight_ext_supabase_tenant.extension.PyJWK", return_value=new_key),
        ):
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}
            key = await ext._get_signing_key("fake-token")

        assert key is new_key
        assert supabase.paths() == [JWKS_PATH]

    async def test_get_signing_key_handles_key_rotation(self, supabase, make_extension):
        """When kid not in cache and cache is old enough, refresh once for key rotation."""
        ext = _setup_jwks_ext(make_extension, supabase.base_url)
        # Make cache just old enough to allow a refresh
        ext._jwks_last_fetched = time.monotonic() - JWKS_MIN_REFRESH_INTERVAL_SECONDS - 1

        rotated_key = MagicMock(spec=PyJWK)
        supabase.reply(JWKS_PATH, 200, MOCK_JWKS_RESPONSE)

        with (
            patch("hindsight_ext_supabase_tenant.extension.pyjwt.get_unverified_header") as mock_header,
            patch("hindsight_ext_supabase_tenant.extension.PyJWK", return_value=rotated_key),
        ):
            mock_header.return_value = {"kid": "rotated-key-99", "alg": "RS256"}
            # The refreshed JWKS won't have "rotated-key-99" either, so this should raise
            with pytest.raises(AuthenticationError, match="Unable to find signing key"):
                await ext._get_signing_key("fake-token")

        # Should have attempted one refresh
        assert supabase.paths() == [JWKS_PATH]

    async def test_get_signing_key_missing_kid_header(self, make_extension):
        ext = _setup_jwks_ext(make_extension)

        with patch("hindsight_ext_supabase_tenant.extension.pyjwt.get_unverified_header") as mock_header:
            mock_header.return_value = {"alg": "RS256"}  # no kid
            with pytest.raises(AuthenticationError, match="Token missing key ID"):
                await ext._get_signing_key("fake-token")

    async def test_get_signing_key_refresh_network_error(self, make_extension):
        """If JWKS refresh fails during key rotation, error should propagate."""
        ext = _setup_jwks_ext(make_extension, _closed_port_url())
        ext._jwks_last_fetched = time.monotonic() - JWKS_MIN_REFRESH_INTERVAL_SECONDS - 1

        with patch("hindsight_ext_supabase_tenant.extension.pyjwt.get_unverified_header") as mock_header:
            mock_header.return_value = {"kid": "unknown-key", "alg": "RS256"}
            with pytest.raises(Exception):
                await ext._get_signing_key("fake-token")


# ======================================================================
# Authentication — JWKS mode
# ======================================================================


def _mock_context() -> AsyncMock:
    context = AsyncMock(spec=ExtensionContext)
    context.run_migration = AsyncMock()
    return context


class TestAuthenticateJWKS:
    """Tests for JWKS-based JWT verification."""

    async def test_authenticate_valid_token(self, make_extension):
        ext = _setup_jwks_ext(make_extension)
        ext._context = _mock_context()

        with (
            patch("hindsight_ext_supabase_tenant.extension.pyjwt.get_unverified_header") as mock_header,
            patch("hindsight_ext_supabase_tenant.extension.pyjwt.decode") as mock_decode,
        ):
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}
            mock_decode.return_value = {"sub": VALID_UUID, "aud": "authenticated"}

            result = await ext.authenticate(RequestContext(api_key=_make_valid_token()))

        assert isinstance(result, TenantContext)
        expected_schema = "user_" + VALID_UUID.replace("-", "_")
        assert result.schema_name == expected_schema

    async def test_authenticate_custom_prefix(self, make_extension):
        ext = _setup_jwks_ext(make_extension)
        ext.schema_prefix = "org"
        ext._context = _mock_context()

        with (
            patch("hindsight_ext_supabase_tenant.extension.pyjwt.get_unverified_header") as mock_header,
            patch("hindsight_ext_supabase_tenant.extension.pyjwt.decode") as mock_decode,
        ):
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}
            mock_decode.return_value = {"sub": VALID_UUID}

            result = await ext.authenticate(RequestContext(api_key=_make_valid_token()))

        assert result.schema_name.startswith("org_")

    @pytest.mark.parametrize(
        ("error", "message"),
        [
            (pyjwt.ExpiredSignatureError(), "Token has expired"),
            (pyjwt.InvalidAudienceError(), "Invalid token audience"),
            (pyjwt.InvalidIssuerError(), "Invalid token issuer"),
            (pyjwt.DecodeError(), "Invalid token"),
            (RuntimeError("unexpected internal error"), "Token verification failed"),
        ],
    )
    async def test_authenticate_decode_failures(self, make_extension, error, message):
        ext = _setup_jwks_ext(make_extension)

        with (
            patch("hindsight_ext_supabase_tenant.extension.pyjwt.get_unverified_header") as mock_header,
            patch("hindsight_ext_supabase_tenant.extension.pyjwt.decode", side_effect=error),
        ):
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}

            with pytest.raises(AuthenticationError, match=message):
                await ext.authenticate(RequestContext(api_key=_make_valid_token()))

    @pytest.mark.parametrize("claims", [{"email": "test@example.com"}, {"sub": ""}])
    async def test_authenticate_missing_or_empty_sub_claim(self, make_extension, claims):
        ext = _setup_jwks_ext(make_extension)

        with (
            patch("hindsight_ext_supabase_tenant.extension.pyjwt.get_unverified_header") as mock_header,
            patch("hindsight_ext_supabase_tenant.extension.pyjwt.decode") as mock_decode,
        ):
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}
            mock_decode.return_value = claims

            with pytest.raises(AuthenticationError, match="missing subject"):
                await ext.authenticate(RequestContext(api_key=_make_valid_token()))


# ======================================================================
# Authentication — Legacy mode
# ======================================================================


class TestAuthenticateLegacy:
    """Tests for legacy /auth/v1/user endpoint verification."""

    async def test_authenticate_valid_token(self, supabase, make_extension):
        ext = _setup_legacy_ext(make_extension, supabase.base_url)
        supabase.reply(USER_PATH, 200, {"id": VALID_UUID})
        ext._context = _mock_context()

        result = await ext.authenticate(RequestContext(api_key=_make_valid_token()))

        assert isinstance(result, TenantContext)
        expected_schema = "user_" + VALID_UUID.replace("-", "_")
        assert result.schema_name == expected_schema

    async def test_authenticate_calls_user_endpoint(self, supabase, make_extension):
        ext = _setup_legacy_ext(make_extension, supabase.base_url)
        supabase.reply(USER_PATH, 200, {"id": VALID_UUID})
        ext._context = _mock_context()

        token = _make_valid_token()
        await ext.authenticate(RequestContext(api_key=token))

        assert supabase.paths() == [USER_PATH]
        headers = supabase.requests[0].headers
        assert headers["Authorization"] == f"Bearer {token}"
        assert headers["apikey"] == "test-service-key"

    async def test_authenticate_expired_token_401(self, supabase, make_extension):
        ext = _setup_legacy_ext(make_extension, supabase.base_url)
        supabase.reply(USER_PATH, 401)

        with pytest.raises(AuthenticationError, match="Invalid or expired token"):
            await ext.authenticate(RequestContext(api_key=_make_valid_token()))

    async def test_authenticate_supabase_error_500(self, supabase, make_extension):
        ext = _setup_legacy_ext(make_extension, supabase.base_url)
        supabase.reply(USER_PATH, 500)

        with pytest.raises(AuthenticationError, match="Authentication failed: 500"):
            await ext.authenticate(RequestContext(api_key=_make_valid_token()))

    async def test_authenticate_no_user_id(self, supabase, make_extension):
        ext = _setup_legacy_ext(make_extension, supabase.base_url)
        supabase.reply(USER_PATH, 200, {"email": "test@example.com"})

        with pytest.raises(AuthenticationError, match="no user ID found"):
            await ext.authenticate(RequestContext(api_key=_make_valid_token()))

    async def test_authenticate_timeout(self, supabase, make_extension):
        ext = _setup_legacy_ext(make_extension, supabase.base_url, timeout=0.2)

        async def slow(_request: web.Request) -> web.StreamResponse:
            await asyncio.sleep(2.0)
            return web.json_response({"id": VALID_UUID})

        supabase.routes[USER_PATH] = slow

        with pytest.raises(AuthenticationError, match="Authentication timeout"):
            await ext.authenticate(RequestContext(api_key=_make_valid_token()))

    async def test_authenticate_connection_error(self, make_extension):
        ext = _setup_legacy_ext(make_extension, _closed_port_url())

        with pytest.raises(AuthenticationError, match="Connection error"):
            await ext.authenticate(RequestContext(api_key=_make_valid_token()))


# ======================================================================
# Authentication — common (both modes)
# ======================================================================


class TestAuthenticateCommon:
    """Tests that apply regardless of verification mode."""

    async def test_authenticate_missing_token(self, make_extension):
        ext = _setup_jwks_ext(make_extension)

        with pytest.raises(AuthenticationError, match="Missing Authorization header"):
            await ext.authenticate(RequestContext(api_key=None))

    async def test_authenticate_empty_token(self, make_extension):
        ext = _setup_jwks_ext(make_extension)

        with pytest.raises(AuthenticationError, match="Missing Authorization header"):
            await ext.authenticate(RequestContext(api_key=""))

    async def test_authenticate_short_token(self, make_extension):
        ext = _setup_jwks_ext(make_extension)

        with pytest.raises(AuthenticationError, match="Invalid token format"):
            await ext.authenticate(RequestContext(api_key="short"))

    async def test_authenticate_not_initialized(self, make_extension):
        ext = make_extension()
        # on_startup has not run, so there is no HTTP session

        with pytest.raises(AuthenticationError, match="Extension not initialized"):
            await ext.authenticate(RequestContext(api_key=_make_valid_token()))

    @pytest.mark.parametrize("sub", ["not-a-uuid", "'; DROP TABLE users;--"])
    async def test_authenticate_rejects_non_uuid_user_id(self, make_extension, sub):
        """User IDs that aren't valid UUIDs (incl. injection attempts) are rejected for schema safety."""
        ext = _setup_jwks_ext(make_extension)

        with (
            patch("hindsight_ext_supabase_tenant.extension.pyjwt.get_unverified_header") as mock_header,
            patch("hindsight_ext_supabase_tenant.extension.pyjwt.decode") as mock_decode,
        ):
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}
            mock_decode.return_value = {"sub": sub}

            with pytest.raises(AuthenticationError, match="Invalid user ID format"):
                await ext.authenticate(RequestContext(api_key=_make_valid_token()))


# ======================================================================
# Schema management
# ======================================================================


class TestSupabaseTenantExtensionSchemaManagement:
    """Tests for schema initialization and caching."""

    async def _authenticate_once(self, ext: SupabaseTenantExtension) -> None:
        with (
            patch("hindsight_ext_supabase_tenant.extension.pyjwt.get_unverified_header") as mock_header,
            patch("hindsight_ext_supabase_tenant.extension.pyjwt.decode") as mock_decode,
        ):
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}
            mock_decode.return_value = {"sub": VALID_UUID}
            await ext.authenticate(RequestContext(api_key=_make_valid_token()))

    async def test_schema_initialized_on_first_access(self, make_extension):
        ext = _setup_jwks_ext(make_extension)
        ext._context = _mock_context()

        await self._authenticate_once(ext)

        expected_schema = "user_" + VALID_UUID.replace("-", "_")
        ext._context.run_migration.assert_called_once_with(expected_schema)
        assert expected_schema in ext._initialized_schemas

    async def test_schema_cached_on_second_access(self, make_extension):
        ext = _setup_jwks_ext(make_extension)
        ext._context = _mock_context()

        await self._authenticate_once(ext)
        await self._authenticate_once(ext)

        # run_migration should only be called once
        expected_schema = "user_" + VALID_UUID.replace("-", "_")
        ext._context.run_migration.assert_called_once_with(expected_schema)

    async def test_schema_init_failure(self, make_extension):
        ext = _setup_jwks_ext(make_extension)
        context = AsyncMock(spec=ExtensionContext)
        context.run_migration = AsyncMock(side_effect=RuntimeError("Migration failed"))
        ext._context = context

        with pytest.raises(AuthenticationError, match="Failed to initialize tenant"):
            await self._authenticate_once(ext)

        # Schema should NOT be cached on failure
        expected_schema = "user_" + VALID_UUID.replace("-", "_")
        assert expected_schema not in ext._initialized_schemas


# ======================================================================
# List tenants
# ======================================================================


class TestSupabaseTenantExtensionListTenants:
    """Tests for list_tenants behavior."""

    async def test_list_tenants_empty(self, make_extension):
        ext = make_extension()
        tenants = await ext.list_tenants()
        assert tenants == []

    async def test_list_tenants_after_auth(self, make_extension):
        ext = _setup_jwks_ext(make_extension)
        ext._context = _mock_context()

        with (
            patch("hindsight_ext_supabase_tenant.extension.pyjwt.get_unverified_header") as mock_header,
            patch("hindsight_ext_supabase_tenant.extension.pyjwt.decode") as mock_decode,
        ):
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}
            mock_decode.return_value = {"sub": VALID_UUID}
            await ext.authenticate(RequestContext(api_key=_make_valid_token()))

        tenants = await ext.list_tenants()
        assert len(tenants) == 1
        assert isinstance(tenants[0], Tenant)
        expected_schema = "user_" + VALID_UUID.replace("-", "_")
        assert tenants[0].schema == expected_schema


# ======================================================================
# Shutdown
# ======================================================================


class TestSupabaseTenantExtensionShutdown:
    """Tests for on_shutdown behavior."""

    async def test_on_shutdown_closes_client(self, supabase, make_extension):
        ext = _setup_legacy_ext(make_extension, supabase.base_url)
        supabase.reply(USER_PATH, 200, {"id": VALID_UUID})
        ext._context = _mock_context()
        await ext.authenticate(RequestContext(api_key=_make_valid_token()))
        session = ext._http
        assert not session.closed

        await ext.on_shutdown()

        assert session.closed
        assert ext._http is None

    async def test_on_shutdown_no_client(self, make_extension):
        ext = make_extension()
        # No HTTP session before on_startup — should not raise
        await ext.on_shutdown()


# ======================================================================
# Extension loader integration
# ======================================================================


class TestSupabaseTenantExtensionLoader:
    """Tests for loading via the extension loader."""

    def test_load_via_extension_loader(self, monkeypatch):
        monkeypatch.setenv(
            "HINDSIGHT_API_TENANT_EXTENSION",
            "hindsight_ext_supabase_tenant.extension:SupabaseTenantExtension",
        )
        monkeypatch.setenv("HINDSIGHT_API_TENANT_SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("HINDSIGHT_API_TENANT_SUPABASE_SERVICE_KEY", "test-key")
        monkeypatch.setenv("HINDSIGHT_API_TENANT_SCHEMA_PREFIX", "custom")

        ext = load_extension("TENANT", TenantExtension)

        assert ext is not None
        assert isinstance(ext, SupabaseTenantExtension)
        assert ext.supabase_url == "https://test.supabase.co"
        assert ext.supabase_service_key == "test-key"
        assert ext.schema_prefix == "custom"

    def test_load_without_service_key(self, monkeypatch):
        """Extension should load without service key — JWKS mode doesn't need it."""
        monkeypatch.setenv(
            "HINDSIGHT_API_TENANT_EXTENSION",
            "hindsight_ext_supabase_tenant.extension:SupabaseTenantExtension",
        )
        monkeypatch.setenv("HINDSIGHT_API_TENANT_SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.delenv("HINDSIGHT_API_TENANT_SUPABASE_SERVICE_KEY", raising=False)

        ext = load_extension("TENANT", TenantExtension)

        assert ext is not None
        assert isinstance(ext, SupabaseTenantExtension)
        assert ext.supabase_service_key is None
