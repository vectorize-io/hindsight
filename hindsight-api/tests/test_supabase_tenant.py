"""Tests for the Supabase Tenant Extension."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import jwt as pyjwt
import pytest
from jwt import PyJWK

from hindsight_api.extensions.builtin.supabase_tenant import (
    JWKS_CACHE_TTL_SECONDS,
    JWKS_MIN_REFRESH_INTERVAL_SECONDS,
    MIN_TOKEN_LENGTH,
    SupabaseTenantExtension,
)
from hindsight_api.extensions.context import ExtensionContext
from hindsight_api.extensions.loader import load_extension
from hindsight_api.extensions.tenant import AuthenticationError, Tenant, TenantContext, TenantExtension
from hindsight_api.models import RequestContext

# A valid UUID for test user IDs
VALID_UUID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

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


def _make_extension(
    supabase_url: str = "https://test.supabase.co",
    service_key: str | None = "test-service-key",
    schema_prefix: str | None = None,
) -> SupabaseTenantExtension:
    """Helper to create a SupabaseTenantExtension with test config."""
    config = {
        "supabase_url": supabase_url,
    }
    if service_key is not None:
        config["supabase_service_key"] = service_key
    if schema_prefix is not None:
        config["schema_prefix"] = schema_prefix
    return SupabaseTenantExtension(config)


def _make_mock_response(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    """Helper to create a mock httpx.Response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = json_data or {}
    response.raise_for_status = MagicMock()
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError("error", request=MagicMock(), response=response)
    return response


def _make_valid_token() -> str:
    """Return a token that passes the MIN_TOKEN_LENGTH check."""
    return "a" * (MIN_TOKEN_LENGTH + 10)


def _setup_jwks_ext() -> tuple[SupabaseTenantExtension, AsyncMock]:
    """Create an extension in JWKS mode with mocked internals."""
    ext = _make_extension()
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    ext._http_client = mock_client
    ext._use_jwks = True
    ext._jwks_keys = {"test-key-1": MagicMock(spec=PyJWK)}
    ext._jwks_keys["test-key-1"].key = "mock-public-key"
    ext._jwks_last_fetched = time.monotonic()
    return ext, mock_client


def _setup_legacy_ext() -> tuple[SupabaseTenantExtension, AsyncMock]:
    """Create an extension in legacy mode with mocked internals."""
    ext = _make_extension()
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    ext._http_client = mock_client
    ext._use_jwks = False
    return ext, mock_client


# ======================================================================
# Initialization
# ======================================================================


class TestSupabaseTenantExtensionInit:
    """Tests for extension initialization."""

    def test_init_with_valid_config(self):
        ext = _make_extension()
        assert ext.supabase_url == "https://test.supabase.co"
        assert ext.supabase_service_key == "test-service-key"
        assert ext.schema_prefix == "user"
        assert ext._initialized_schemas == set()
        assert ext._http_client is None
        assert ext._use_jwks is False
        assert ext._jwks_keys == {}

    def test_init_missing_supabase_url(self):
        with pytest.raises(ValueError, match="HINDSIGHT_API_TENANT_SUPABASE_URL is required"):
            SupabaseTenantExtension({})

    def test_init_without_service_key(self):
        """Service key is optional — JWKS mode doesn't require it."""
        ext = _make_extension(service_key=None)
        assert ext.supabase_service_key is None

    def test_init_default_schema_prefix(self):
        ext = _make_extension()
        assert ext.schema_prefix == "user"

    def test_init_custom_schema_prefix(self):
        ext = _make_extension(schema_prefix="tenant")
        assert ext.schema_prefix == "tenant"

    def test_init_strips_trailing_slash(self):
        ext = _make_extension(supabase_url="https://test.supabase.co/")
        assert ext.supabase_url == "https://test.supabase.co"

    def test_init_rejects_invalid_schema_prefix(self):
        """Schema prefix with special characters should be rejected."""
        with pytest.raises(ValueError, match="Invalid schema_prefix"):
            _make_extension(schema_prefix='"; DROP TABLE')

    def test_init_rejects_empty_schema_prefix(self):
        with pytest.raises(ValueError, match="Invalid schema_prefix"):
            _make_extension(schema_prefix="")

    def test_init_rejects_schema_prefix_starting_with_digit(self):
        with pytest.raises(ValueError, match="Invalid schema_prefix"):
            _make_extension(schema_prefix="123abc")

    def test_init_allows_underscore_prefix(self):
        ext = _make_extension(schema_prefix="_internal")
        assert ext.schema_prefix == "_internal"

    def test_is_tenant_extension_subclass(self):
        ext = _make_extension()
        assert isinstance(ext, TenantExtension)


# ======================================================================
# Startup — JWKS initialization
# ======================================================================


class TestSupabaseTenantExtensionStartup:
    """Tests for on_startup behavior."""

    @pytest.mark.asyncio
    async def test_on_startup_creates_http_client(self):
        ext = _make_extension()
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        # JWKS fetch returns keys
        mock_client.get.return_value = _make_mock_response(200, MOCK_JWKS_RESPONSE)

        with patch("hindsight_api.extensions.builtin.supabase_tenant.httpx.AsyncClient", return_value=mock_client):
            with patch("hindsight_api.extensions.builtin.supabase_tenant.PyJWK"):
                await ext.on_startup()

        assert ext._http_client is mock_client

    @pytest.mark.asyncio
    async def test_on_startup_fetches_jwks(self):
        ext = _make_extension()
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _make_mock_response(200, MOCK_JWKS_RESPONSE)

        with patch("hindsight_api.extensions.builtin.supabase_tenant.httpx.AsyncClient", return_value=mock_client):
            with patch("hindsight_api.extensions.builtin.supabase_tenant.PyJWK") as mock_pyjwk:
                mock_pyjwk.return_value = MagicMock(spec=PyJWK)
                await ext.on_startup()

        assert ext._use_jwks is True
        # First call: JWKS fetch, second call: health check
        assert mock_client.get.call_count == 2
        jwks_call = mock_client.get.call_args_list[0]
        assert jwks_call.args[0] == "https://test.supabase.co/auth/v1/.well-known/jwks.json"

    @pytest.mark.asyncio
    async def test_on_startup_falls_back_to_legacy_when_jwks_empty(self):
        ext = _make_extension()
        mock_client = AsyncMock(spec=httpx.AsyncClient)

        # JWKS returns empty keys, health check succeeds
        def mock_get(url, **kwargs):
            if "jwks" in url:
                return _make_mock_response(200, {"keys": []})
            return _make_mock_response(200)

        mock_client.get.side_effect = mock_get

        with patch("hindsight_api.extensions.builtin.supabase_tenant.httpx.AsyncClient", return_value=mock_client):
            await ext.on_startup()

        assert ext._use_jwks is False

    @pytest.mark.asyncio
    async def test_on_startup_falls_back_to_legacy_when_jwks_fetch_fails(self):
        ext = _make_extension()
        mock_client = AsyncMock(spec=httpx.AsyncClient)

        call_count = 0

        def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # JWKS fetch fails
                raise httpx.ConnectError("Connection refused")
            # health check
            return _make_mock_response(200)

        mock_client.get.side_effect = mock_get

        with patch("hindsight_api.extensions.builtin.supabase_tenant.httpx.AsyncClient", return_value=mock_client):
            await ext.on_startup()

        assert ext._use_jwks is False

    @pytest.mark.asyncio
    async def test_on_startup_raises_if_no_jwks_and_no_service_key(self):
        ext = _make_extension(service_key=None)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _make_mock_response(200, {"keys": []})

        with patch("hindsight_api.extensions.builtin.supabase_tenant.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ValueError, match="HINDSIGHT_API_TENANT_SUPABASE_SERVICE_KEY is required"):
                await ext.on_startup()

    @pytest.mark.asyncio
    async def test_on_startup_health_check_with_service_key(self):
        ext = _make_extension()
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _make_mock_response(200, MOCK_JWKS_RESPONSE)

        with patch("hindsight_api.extensions.builtin.supabase_tenant.httpx.AsyncClient", return_value=mock_client):
            with patch("hindsight_api.extensions.builtin.supabase_tenant.PyJWK"):
                await ext.on_startup()

        # Second call should be health check
        health_call = mock_client.get.call_args_list[1]
        assert health_call.args[0] == "https://test.supabase.co/auth/v1/health"
        assert health_call.kwargs["headers"] == {"apikey": "test-service-key"}

    @pytest.mark.asyncio
    async def test_on_startup_skips_health_check_without_service_key(self):
        ext = _make_extension(service_key=None)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _make_mock_response(200, MOCK_JWKS_RESPONSE)

        with patch("hindsight_api.extensions.builtin.supabase_tenant.httpx.AsyncClient", return_value=mock_client):
            with patch("hindsight_api.extensions.builtin.supabase_tenant.PyJWK"):
                await ext.on_startup()

        # Only one call: JWKS fetch, no health check
        assert mock_client.get.call_count == 1


# ======================================================================
# JWKS cache management
# ======================================================================


class TestJWKSCacheManagement:
    """Tests for JWKS key fetching, caching, and rotation handling."""

    @pytest.mark.asyncio
    async def test_get_signing_key_from_cache(self):
        ext, _ = _setup_jwks_ext()

        with patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.get_unverified_header") as mock_header:
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}
            key = await ext._get_signing_key("fake-token")

        assert key is ext._jwks_keys["test-key-1"]

    @pytest.mark.asyncio
    async def test_get_signing_key_refreshes_stale_cache(self):
        ext, mock_client = _setup_jwks_ext()
        # Make cache expired
        ext._jwks_last_fetched = time.monotonic() - JWKS_CACHE_TTL_SECONDS - 1

        new_key = MagicMock(spec=PyJWK)
        mock_client.get.return_value = _make_mock_response(200, MOCK_JWKS_RESPONSE)

        with (
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.get_unverified_header") as mock_header,
            patch("hindsight_api.extensions.builtin.supabase_tenant.PyJWK", return_value=new_key),
        ):
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}
            key = await ext._get_signing_key("fake-token")

        assert key is new_key
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_signing_key_handles_key_rotation(self):
        """When kid not in cache and cache is old enough, refresh once for key rotation."""
        ext, mock_client = _setup_jwks_ext()
        # Make cache just old enough to allow a refresh
        ext._jwks_last_fetched = time.monotonic() - JWKS_MIN_REFRESH_INTERVAL_SECONDS - 1

        rotated_key = MagicMock(spec=PyJWK)
        mock_client.get.return_value = _make_mock_response(200, MOCK_JWKS_RESPONSE)

        with (
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.get_unverified_header") as mock_header,
            patch("hindsight_api.extensions.builtin.supabase_tenant.PyJWK", return_value=rotated_key),
        ):
            mock_header.return_value = {"kid": "rotated-key-99", "alg": "RS256"}
            # The refreshed JWKS won't have "rotated-key-99" either, so this should raise
            with pytest.raises(AuthenticationError, match="Unable to find signing key"):
                await ext._get_signing_key("fake-token")

        # Should have attempted one refresh
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_signing_key_missing_kid_header(self):
        ext, _ = _setup_jwks_ext()

        with patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.get_unverified_header") as mock_header:
            mock_header.return_value = {"alg": "RS256"}  # no kid
            with pytest.raises(AuthenticationError, match="Token missing key ID"):
                await ext._get_signing_key("fake-token")

    @pytest.mark.asyncio
    async def test_get_signing_key_refresh_network_error(self):
        """If JWKS refresh fails during key rotation, error should propagate."""
        ext, mock_client = _setup_jwks_ext()
        ext._jwks_last_fetched = time.monotonic() - JWKS_MIN_REFRESH_INTERVAL_SECONDS - 1

        mock_client.get.side_effect = httpx.ConnectError("Connection refused")

        with patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.get_unverified_header") as mock_header:
            mock_header.return_value = {"kid": "unknown-key", "alg": "RS256"}
            with pytest.raises(Exception):
                await ext._get_signing_key("fake-token")


# ======================================================================
# Authentication — JWKS mode
# ======================================================================


class TestAuthenticateJWKS:
    """Tests for JWKS-based JWT verification."""

    @pytest.mark.asyncio
    async def test_authenticate_valid_token(self):
        ext, _ = _setup_jwks_ext()
        mock_context = AsyncMock(spec=ExtensionContext)
        mock_context.run_migration = AsyncMock()
        ext._context = mock_context

        with (
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.get_unverified_header") as mock_header,
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.decode") as mock_decode,
        ):
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}
            mock_decode.return_value = {"sub": VALID_UUID, "aud": "authenticated"}

            result = await ext.authenticate(RequestContext(api_key=_make_valid_token()))

        assert isinstance(result, TenantContext)
        expected_schema = "user_" + VALID_UUID.replace("-", "_")
        assert result.schema_name == expected_schema

    @pytest.mark.asyncio
    async def test_authenticate_custom_prefix(self):
        ext, _ = _setup_jwks_ext()
        ext.schema_prefix = "org"
        mock_context = AsyncMock(spec=ExtensionContext)
        mock_context.run_migration = AsyncMock()
        ext._context = mock_context

        with (
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.get_unverified_header") as mock_header,
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.decode") as mock_decode,
        ):
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}
            mock_decode.return_value = {"sub": VALID_UUID}

            result = await ext.authenticate(RequestContext(api_key=_make_valid_token()))

        assert result.schema_name.startswith("org_")

    @pytest.mark.asyncio
    async def test_authenticate_expired_token(self):
        ext, _ = _setup_jwks_ext()

        with (
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.get_unverified_header") as mock_header,
            patch(
                "hindsight_api.extensions.builtin.supabase_tenant.pyjwt.decode",
                side_effect=pyjwt.ExpiredSignatureError(),
            ),
        ):
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}

            with pytest.raises(AuthenticationError, match="Token has expired"):
                await ext.authenticate(RequestContext(api_key=_make_valid_token()))

    @pytest.mark.asyncio
    async def test_authenticate_invalid_audience(self):
        ext, _ = _setup_jwks_ext()

        with (
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.get_unverified_header") as mock_header,
            patch(
                "hindsight_api.extensions.builtin.supabase_tenant.pyjwt.decode",
                side_effect=pyjwt.InvalidAudienceError(),
            ),
        ):
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}

            with pytest.raises(AuthenticationError, match="Invalid token audience"):
                await ext.authenticate(RequestContext(api_key=_make_valid_token()))

    @pytest.mark.asyncio
    async def test_authenticate_invalid_issuer(self):
        ext, _ = _setup_jwks_ext()

        with (
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.get_unverified_header") as mock_header,
            patch(
                "hindsight_api.extensions.builtin.supabase_tenant.pyjwt.decode",
                side_effect=pyjwt.InvalidIssuerError(),
            ),
        ):
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}

            with pytest.raises(AuthenticationError, match="Invalid token issuer"):
                await ext.authenticate(RequestContext(api_key=_make_valid_token()))

    @pytest.mark.asyncio
    async def test_authenticate_decode_error(self):
        ext, _ = _setup_jwks_ext()

        with (
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.get_unverified_header") as mock_header,
            patch(
                "hindsight_api.extensions.builtin.supabase_tenant.pyjwt.decode",
                side_effect=pyjwt.DecodeError(),
            ),
        ):
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}

            with pytest.raises(AuthenticationError, match="Invalid token"):
                await ext.authenticate(RequestContext(api_key=_make_valid_token()))

    @pytest.mark.asyncio
    async def test_authenticate_missing_sub_claim(self):
        ext, _ = _setup_jwks_ext()

        with (
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.get_unverified_header") as mock_header,
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.decode") as mock_decode,
        ):
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}
            mock_decode.return_value = {"email": "test@example.com"}  # no sub

            with pytest.raises(AuthenticationError, match="missing subject"):
                await ext.authenticate(RequestContext(api_key=_make_valid_token()))

    @pytest.mark.asyncio
    async def test_authenticate_empty_sub_claim(self):
        """Empty string sub claim should be treated as missing."""
        ext, _ = _setup_jwks_ext()

        with (
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.get_unverified_header") as mock_header,
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.decode") as mock_decode,
        ):
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}
            mock_decode.return_value = {"sub": ""}

            with pytest.raises(AuthenticationError, match="missing subject"):
                await ext.authenticate(RequestContext(api_key=_make_valid_token()))

    @pytest.mark.asyncio
    async def test_authenticate_generic_exception(self):
        """Unexpected exceptions during decode should be caught and wrapped."""
        ext, _ = _setup_jwks_ext()

        with (
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.get_unverified_header") as mock_header,
            patch(
                "hindsight_api.extensions.builtin.supabase_tenant.pyjwt.decode",
                side_effect=RuntimeError("unexpected internal error"),
            ),
        ):
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}

            with pytest.raises(AuthenticationError, match="Token verification failed"):
                await ext.authenticate(RequestContext(api_key=_make_valid_token()))


# ======================================================================
# Authentication — Legacy mode
# ======================================================================


class TestAuthenticateLegacy:
    """Tests for legacy /auth/v1/user endpoint verification."""

    @pytest.mark.asyncio
    async def test_authenticate_valid_token(self):
        ext, mock_client = _setup_legacy_ext()
        mock_client.get.return_value = _make_mock_response(200, {"id": VALID_UUID})

        mock_context = AsyncMock(spec=ExtensionContext)
        mock_context.run_migration = AsyncMock()
        ext._context = mock_context

        result = await ext.authenticate(RequestContext(api_key=_make_valid_token()))

        assert isinstance(result, TenantContext)
        expected_schema = "user_" + VALID_UUID.replace("-", "_")
        assert result.schema_name == expected_schema

    @pytest.mark.asyncio
    async def test_authenticate_calls_user_endpoint(self):
        ext, mock_client = _setup_legacy_ext()
        mock_client.get.return_value = _make_mock_response(200, {"id": VALID_UUID})

        mock_context = AsyncMock(spec=ExtensionContext)
        mock_context.run_migration = AsyncMock()
        ext._context = mock_context

        token = _make_valid_token()
        await ext.authenticate(RequestContext(api_key=token))

        mock_client.get.assert_called_once_with(
            "https://test.supabase.co/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": "test-service-key",
            },
        )

    @pytest.mark.asyncio
    async def test_authenticate_expired_token_401(self):
        ext, mock_client = _setup_legacy_ext()
        mock_client.get.return_value = _make_mock_response(401)

        with pytest.raises(AuthenticationError, match="Invalid or expired token"):
            await ext.authenticate(RequestContext(api_key=_make_valid_token()))

    @pytest.mark.asyncio
    async def test_authenticate_supabase_error_500(self):
        ext, mock_client = _setup_legacy_ext()
        mock_client.get.return_value = _make_mock_response(500)

        with pytest.raises(AuthenticationError, match="Authentication failed: 500"):
            await ext.authenticate(RequestContext(api_key=_make_valid_token()))

    @pytest.mark.asyncio
    async def test_authenticate_no_user_id(self):
        ext, mock_client = _setup_legacy_ext()
        mock_client.get.return_value = _make_mock_response(200, {"email": "test@example.com"})

        with pytest.raises(AuthenticationError, match="no user ID found"):
            await ext.authenticate(RequestContext(api_key=_make_valid_token()))

    @pytest.mark.asyncio
    async def test_authenticate_timeout(self):
        ext, mock_client = _setup_legacy_ext()
        mock_client.get.side_effect = httpx.TimeoutException("Request timed out")

        with pytest.raises(AuthenticationError, match="Authentication timeout"):
            await ext.authenticate(RequestContext(api_key=_make_valid_token()))

    @pytest.mark.asyncio
    async def test_authenticate_connection_error(self):
        ext, mock_client = _setup_legacy_ext()
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(AuthenticationError, match="Connection error"):
            await ext.authenticate(RequestContext(api_key=_make_valid_token()))


# ======================================================================
# Authentication — common (both modes)
# ======================================================================


class TestAuthenticateCommon:
    """Tests that apply regardless of verification mode."""

    @pytest.mark.asyncio
    async def test_authenticate_missing_token(self):
        ext, _ = _setup_jwks_ext()

        with pytest.raises(AuthenticationError, match="Missing Authorization header"):
            await ext.authenticate(RequestContext(api_key=None))

    @pytest.mark.asyncio
    async def test_authenticate_empty_token(self):
        ext, _ = _setup_jwks_ext()

        with pytest.raises(AuthenticationError, match="Missing Authorization header"):
            await ext.authenticate(RequestContext(api_key=""))

    @pytest.mark.asyncio
    async def test_authenticate_short_token(self):
        ext, _ = _setup_jwks_ext()

        with pytest.raises(AuthenticationError, match="Invalid token format"):
            await ext.authenticate(RequestContext(api_key="short"))

    @pytest.mark.asyncio
    async def test_authenticate_not_initialized(self):
        ext = _make_extension()
        # _http_client is None by default

        with pytest.raises(AuthenticationError, match="Extension not initialized"):
            await ext.authenticate(RequestContext(api_key=_make_valid_token()))

    @pytest.mark.asyncio
    async def test_authenticate_rejects_non_uuid_user_id(self):
        """User IDs that aren't valid UUIDs should be rejected for schema safety."""
        ext, _ = _setup_jwks_ext()

        with (
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.get_unverified_header") as mock_header,
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.decode") as mock_decode,
        ):
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}
            mock_decode.return_value = {"sub": "not-a-uuid"}

            with pytest.raises(AuthenticationError, match="Invalid user ID format"):
                await ext.authenticate(RequestContext(api_key=_make_valid_token()))

    @pytest.mark.asyncio
    async def test_authenticate_rejects_malicious_user_id(self):
        """User IDs with SQL injection attempts should be rejected."""
        ext, _ = _setup_jwks_ext()

        with (
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.get_unverified_header") as mock_header,
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.decode") as mock_decode,
        ):
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}
            mock_decode.return_value = {"sub": "'; DROP TABLE users;--"}

            with pytest.raises(AuthenticationError, match="Invalid user ID format"):
                await ext.authenticate(RequestContext(api_key=_make_valid_token()))


# ======================================================================
# Schema management
# ======================================================================


class TestSupabaseTenantExtensionSchemaManagement:
    """Tests for schema initialization and caching."""

    @pytest.mark.asyncio
    async def test_schema_initialized_on_first_access(self):
        ext, _ = _setup_jwks_ext()
        mock_context = AsyncMock(spec=ExtensionContext)
        mock_context.run_migration = AsyncMock()
        ext._context = mock_context

        with (
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.get_unverified_header") as mock_header,
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.decode") as mock_decode,
        ):
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}
            mock_decode.return_value = {"sub": VALID_UUID}
            await ext.authenticate(RequestContext(api_key=_make_valid_token()))

        expected_schema = "user_" + VALID_UUID.replace("-", "_")
        mock_context.run_migration.assert_called_once_with(expected_schema)
        assert expected_schema in ext._initialized_schemas

    @pytest.mark.asyncio
    async def test_schema_cached_on_second_access(self):
        ext, _ = _setup_jwks_ext()
        mock_context = AsyncMock(spec=ExtensionContext)
        mock_context.run_migration = AsyncMock()
        ext._context = mock_context

        with (
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.get_unverified_header") as mock_header,
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.decode") as mock_decode,
        ):
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}
            mock_decode.return_value = {"sub": VALID_UUID}

            await ext.authenticate(RequestContext(api_key=_make_valid_token()))
            await ext.authenticate(RequestContext(api_key=_make_valid_token()))

        # run_migration should only be called once
        expected_schema = "user_" + VALID_UUID.replace("-", "_")
        mock_context.run_migration.assert_called_once_with(expected_schema)

    @pytest.mark.asyncio
    async def test_schema_init_failure(self):
        ext, _ = _setup_jwks_ext()
        mock_context = AsyncMock(spec=ExtensionContext)
        mock_context.run_migration = AsyncMock(side_effect=RuntimeError("Migration failed"))
        ext._context = mock_context

        with (
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.get_unverified_header") as mock_header,
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.decode") as mock_decode,
        ):
            mock_header.return_value = {"kid": "test-key-1", "alg": "RS256"}
            mock_decode.return_value = {"sub": VALID_UUID}

            with pytest.raises(AuthenticationError, match="Failed to initialize tenant"):
                await ext.authenticate(RequestContext(api_key=_make_valid_token()))

        # Schema should NOT be cached on failure
        expected_schema = "user_" + VALID_UUID.replace("-", "_")
        assert expected_schema not in ext._initialized_schemas


# ======================================================================
# List tenants
# ======================================================================


class TestSupabaseTenantExtensionListTenants:
    """Tests for list_tenants behavior."""

    @pytest.mark.asyncio
    async def test_list_tenants_empty(self):
        ext = _make_extension()
        tenants = await ext.list_tenants()
        assert tenants == []

    @pytest.mark.asyncio
    async def test_list_tenants_after_auth(self):
        ext, _ = _setup_jwks_ext()
        mock_context = AsyncMock(spec=ExtensionContext)
        mock_context.run_migration = AsyncMock()
        ext._context = mock_context

        with (
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.get_unverified_header") as mock_header,
            patch("hindsight_api.extensions.builtin.supabase_tenant.pyjwt.decode") as mock_decode,
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

    @pytest.mark.asyncio
    async def test_on_shutdown_closes_client(self):
        ext = _make_extension()
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        ext._http_client = mock_client

        await ext.on_shutdown()

        mock_client.aclose.assert_called_once()
        assert ext._http_client is None

    @pytest.mark.asyncio
    async def test_on_shutdown_no_client(self):
        ext = _make_extension()
        # _http_client is None by default — should not raise
        await ext.on_shutdown()


# ======================================================================
# Extension loader integration
# ======================================================================


class TestSupabaseTenantExtensionLoader:
    """Tests for loading via the extension loader."""

    def test_load_via_extension_loader(self, monkeypatch):
        monkeypatch.setenv(
            "HINDSIGHT_API_TENANT_EXTENSION",
            "hindsight_api.extensions.builtin.supabase_tenant:SupabaseTenantExtension",
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
            "hindsight_api.extensions.builtin.supabase_tenant:SupabaseTenantExtension",
        )
        monkeypatch.setenv("HINDSIGHT_API_TENANT_SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.delenv("HINDSIGHT_API_TENANT_SUPABASE_SERVICE_KEY", raising=False)

        ext = load_extension("TENANT", TenantExtension)

        assert ext is not None
        assert isinstance(ext, SupabaseTenantExtension)
        assert ext.supabase_service_key is None
