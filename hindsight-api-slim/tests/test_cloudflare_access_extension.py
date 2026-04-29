"""Tests for the built-in Cloudflare Access OAuth extension.

Covers:
  - JWT mint/verify helpers and PKCE S256
  - CloudflareAccessTenantExtension: API key for HTTP, OAuth JWT for MCP
  - CloudflareAccessHttpExtension HTTP routes: well-known, /register, /authorize, /token
  - Full OAuth dance end-to-end (register → authorize → token → authenticate_mcp)
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from unittest.mock import AsyncMock, MagicMock, patch

import jwt as pyjwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hindsight_api.extensions.builtin.cloudflare_access import (
    CloudflareAccessHttpExtension,
    CloudflareAccessTenantExtension,
    mint_access_token,
    mint_client_id,
    mint_code,
    verify_access_token,
    verify_client_id,
    verify_code,
    verify_pkce_s256,
)
from hindsight_api.extensions.tenant import AuthenticationError
from hindsight_api.models import RequestContext

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_SECRET = "test-signing-secret-32bytes-long!!"
_ISSUER = "https://hindsight.example.com"
_RESOURCE = "https://hindsight.example.com/mcp"
_EMAIL = "test@example.com"
_REDIRECT_URI = "https://claude.ai/api/mcp/auth_callback"
_LOOPBACK_URI = "http://127.0.0.1:3000/callback"


def _make_pkce_pair() -> tuple[str, str]:
    """Return (code_verifier, code_challenge) pair for S256."""
    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def _make_tenant_ext() -> CloudflareAccessTenantExtension:
    return CloudflareAccessTenantExtension(
        {
            "api_key": "http-api-key",
            "oauth_signing_secret": _SECRET,
            "oauth_issuer": _ISSUER,
        }
    )


def _make_http_ext(
    allowed_emails: str = "",
    access_team: str = "testteam",
    access_aud: str = "test-aud",
) -> CloudflareAccessHttpExtension:
    return CloudflareAccessHttpExtension(
        {
            "oauth_signing_secret": _SECRET,
            "oauth_issuer": _ISSUER,
            "oauth_resource": _RESOURCE,
            "oauth_access_team": access_team,
            "oauth_access_aud": access_aud,
            "oauth_allowed_emails": allowed_emails,
            "oauth_access_token_ttl": "3600",
            "oauth_auth_code_ttl": "600",
        }
    )


def _make_test_client(ext: CloudflareAccessHttpExtension) -> TestClient:
    """Mount the extension's root router on a minimal FastAPI app."""
    app = FastAPI()
    router = ext.get_root_router(memory=None)
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# tokens.py unit tests
# ---------------------------------------------------------------------------


class TestTokens:
    def test_client_id_roundtrip(self):
        uris = [_REDIRECT_URI, _LOOPBACK_URI]
        token = mint_client_id(_SECRET, uris, "test-client")
        payload = verify_client_id(_SECRET, token)
        assert payload["redirect_uris"] == uris
        assert payload["client_name"] == "test-client"

    def test_client_id_wrong_secret(self):
        token = mint_client_id(_SECRET, [_REDIRECT_URI], "c")
        with pytest.raises(pyjwt.InvalidTokenError):
            verify_client_id("wrong-secret", token)

    def test_code_roundtrip(self):
        verifier, challenge = _make_pkce_pair()
        token = mint_code(
            _SECRET,
            client_id="cid",
            redirect_uri=_REDIRECT_URI,
            code_challenge=challenge,
            code_challenge_method="S256",
            email=_EMAIL,
            scope="mcp:full",
        )
        payload = verify_code(_SECRET, token)
        assert payload["email"] == _EMAIL
        assert payload["code_challenge"] == challenge
        assert payload["redirect_uri"] == _REDIRECT_URI

    def test_code_expired(self):
        _, challenge = _make_pkce_pair()
        token = mint_code(
            _SECRET,
            client_id="cid",
            redirect_uri=_REDIRECT_URI,
            code_challenge=challenge,
            code_challenge_method="S256",
            email=_EMAIL,
            scope="mcp:full",
            ttl_seconds=-1,  # already expired
        )
        with pytest.raises(pyjwt.ExpiredSignatureError):
            verify_code(_SECRET, token)

    def test_access_token_roundtrip(self):
        token = mint_access_token(_SECRET, email=_EMAIL, scope="mcp:full", issuer=_ISSUER)
        payload = verify_access_token(_SECRET, token, _ISSUER)
        assert payload["sub"] == _EMAIL
        assert payload["scope"] == "mcp:full"

    def test_access_token_expired(self):
        token = mint_access_token(_SECRET, email=_EMAIL, scope="mcp:full", issuer=_ISSUER, ttl_seconds=-1)
        with pytest.raises(pyjwt.ExpiredSignatureError):
            verify_access_token(_SECRET, token, _ISSUER)

    def test_access_token_wrong_issuer(self):
        token = mint_access_token(_SECRET, email=_EMAIL, scope="mcp:full", issuer=_ISSUER)
        with pytest.raises(pyjwt.InvalidTokenError):
            verify_access_token(_SECRET, token, "https://other.example.com")

    def test_pkce_s256_happy(self):
        verifier, challenge = _make_pkce_pair()
        assert verify_pkce_s256(challenge, verifier) is True

    def test_pkce_s256_wrong_verifier(self):
        _, challenge = _make_pkce_pair()
        assert verify_pkce_s256(challenge, "wrong-verifier") is False

    def test_pkce_s256_empty_verifier(self):
        _, challenge = _make_pkce_pair()
        assert verify_pkce_s256(challenge, "") is False


# ---------------------------------------------------------------------------
# CloudflareAccessTenantExtension tests
# ---------------------------------------------------------------------------


class TestCloudflareAccessTenantExtension:
    def test_init_missing_signing_secret(self):
        with pytest.raises(ValueError, match="OAUTH_SIGNING_SECRET"):
            CloudflareAccessTenantExtension({"api_key": "k", "oauth_issuer": _ISSUER})

    def test_init_missing_issuer(self):
        with pytest.raises(ValueError, match="OAUTH_ISSUER"):
            CloudflareAccessTenantExtension({"api_key": "k", "oauth_signing_secret": _SECRET})

    def test_init_missing_api_key(self):
        with pytest.raises(ValueError, match="API_KEY"):
            CloudflareAccessTenantExtension({"oauth_signing_secret": _SECRET, "oauth_issuer": _ISSUER})

    @pytest.mark.asyncio
    async def test_authenticate_http_valid_api_key(self):
        ext = _make_tenant_ext()
        ctx = await ext.authenticate(RequestContext(api_key="http-api-key"))
        assert ctx.schema_name  # non-empty

    @pytest.mark.asyncio
    async def test_authenticate_http_wrong_api_key(self):
        ext = _make_tenant_ext()
        with pytest.raises(AuthenticationError):
            await ext.authenticate(RequestContext(api_key="wrong"))

    @pytest.mark.asyncio
    async def test_authenticate_mcp_valid_token(self):
        ext = _make_tenant_ext()
        token = mint_access_token(_SECRET, email=_EMAIL, scope="mcp:full", issuer=_ISSUER)
        ctx = await ext.authenticate_mcp(RequestContext(api_key=token))
        assert ctx.schema_name

    @pytest.mark.asyncio
    async def test_authenticate_mcp_missing_token(self):
        ext = _make_tenant_ext()
        with pytest.raises(AuthenticationError) as exc_info:
            await ext.authenticate_mcp(RequestContext(api_key=None))
        assert "WWW-Authenticate" in exc_info.value.headers
        assert "resource_metadata" in exc_info.value.headers["WWW-Authenticate"]

    @pytest.mark.asyncio
    async def test_authenticate_mcp_expired_token(self):
        ext = _make_tenant_ext()
        token = mint_access_token(_SECRET, email=_EMAIL, scope="mcp:full", issuer=_ISSUER, ttl_seconds=-1)
        with pytest.raises(AuthenticationError) as exc_info:
            await ext.authenticate_mcp(RequestContext(api_key=token))
        assert "expired" in exc_info.value.reason.lower()
        assert "WWW-Authenticate" in exc_info.value.headers

    @pytest.mark.asyncio
    async def test_authenticate_mcp_wrong_signature(self):
        ext = _make_tenant_ext()
        token = mint_access_token("different-secret", email=_EMAIL, scope="mcp:full", issuer=_ISSUER)
        with pytest.raises(AuthenticationError):
            await ext.authenticate_mcp(RequestContext(api_key=token))

    @pytest.mark.asyncio
    async def test_authenticate_mcp_api_key_rejected(self):
        """HTTP API key must not work for MCP — different audience."""
        ext = _make_tenant_ext()
        with pytest.raises(AuthenticationError):
            await ext.authenticate_mcp(RequestContext(api_key="http-api-key"))


# ---------------------------------------------------------------------------
# CloudflareAccessHttpExtension: well-known metadata
# ---------------------------------------------------------------------------


class TestWellKnownEndpoints:
    def test_authorization_server_metadata(self):
        ext = _make_http_ext()
        client = _make_test_client(ext)
        resp = client.get("/.well-known/oauth-authorization-server")
        assert resp.status_code == 200
        data = resp.json()
        assert data["issuer"] == _ISSUER
        assert data["authorization_endpoint"] == f"{_ISSUER}/authorize"
        assert data["token_endpoint"] == f"{_ISSUER}/token"
        assert data["registration_endpoint"] == f"{_ISSUER}/register"
        assert "S256" in data["code_challenge_methods_supported"]
        assert "authorization_code" in data["grant_types_supported"]
        assert "none" in data["token_endpoint_auth_methods_supported"]

    def test_protected_resource_metadata(self):
        ext = _make_http_ext()
        client = _make_test_client(ext)
        resp = client.get("/.well-known/oauth-protected-resource")
        assert resp.status_code == 200
        data = resp.json()
        assert data["resource"] == _RESOURCE
        assert _ISSUER in data["authorization_servers"]
        assert "header" in data["bearer_methods_supported"]


# ---------------------------------------------------------------------------
# CloudflareAccessHttpExtension: /register (DCR)
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_https_redirect_uri(self):
        ext = _make_http_ext()
        client = _make_test_client(ext)
        resp = client.post("/register", json={"redirect_uris": [_REDIRECT_URI], "client_name": "Claude"})
        assert resp.status_code == 201
        data = resp.json()
        assert "client_id" in data
        assert data["redirect_uris"] == [_REDIRECT_URI]
        assert data["token_endpoint_auth_method"] == "none"
        # client_id is a verifiable JWT
        payload = verify_client_id(_SECRET, data["client_id"])
        assert payload["redirect_uris"] == [_REDIRECT_URI]

    def test_register_loopback_redirect_uri(self):
        ext = _make_http_ext()
        client = _make_test_client(ext)
        resp = client.post("/register", json={"redirect_uris": [_LOOPBACK_URI]})
        assert resp.status_code == 201

    def test_register_rejects_plain_http(self):
        ext = _make_http_ext()
        client = _make_test_client(ext)
        resp = client.post("/register", json={"redirect_uris": ["http://evil.com/callback"]})
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_client_metadata"

    def test_register_missing_redirect_uris(self):
        ext = _make_http_ext()
        client = _make_test_client(ext)
        resp = client.post("/register", json={"client_name": "x"})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# CloudflareAccessHttpExtension: /authorize
# ---------------------------------------------------------------------------


def _make_cf_jwt(email: str = _EMAIL) -> str:
    """Mint a mock CF Access JWT (signed with a fake key for injection via mock)."""
    return f"mock-cf-jwt-{email}"


class TestAuthorize:
    def _ext_with_mock_verifier(
        self,
        email: str = _EMAIL,
        verify_raises: Exception | None = None,
    ) -> CloudflareAccessHttpExtension:
        """Create extension with _cf_verify mocked."""
        ext = _make_http_ext()
        if verify_raises:
            ext._cf_verify = AsyncMock(side_effect=verify_raises)
        else:
            ext._cf_verify = AsyncMock(return_value={"email": email, "sub": email})
        return ext

    def _register_client(self, client: TestClient, redirect_uri: str = _REDIRECT_URI) -> str:
        resp = client.post("/register", json={"redirect_uris": [redirect_uri]})
        assert resp.status_code == 201
        return resp.json()["client_id"]

    def test_happy_path_returns_302_with_code(self):
        verifier, challenge = _make_pkce_pair()
        ext = self._ext_with_mock_verifier()
        tc = _make_test_client(ext)
        client_id = self._register_client(tc)

        resp = tc.get(
            "/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": _REDIRECT_URI,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": "abc123",
            },
            headers={"Cf-Access-Jwt-Assertion": "mock-cf-jwt"},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        location = resp.headers["location"]
        assert location.startswith(_REDIRECT_URI)
        assert "code=" in location
        assert "state=abc123" in location

    def test_missing_cf_jwt_returns_401_html(self):
        ext = _make_http_ext()
        tc = _make_test_client(ext)
        client_id = self._register_client(tc)
        _, challenge = _make_pkce_pair()

        resp = tc.get(
            "/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": _REDIRECT_URI,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 401
        assert "text/html" in resp.headers["content-type"]

    def test_invalid_cf_jwt_returns_403(self):
        ext = self._ext_with_mock_verifier(verify_raises=ValueError("expired"))
        tc = _make_test_client(ext)
        client_id = self._register_client(tc)
        _, challenge = _make_pkce_pair()

        resp = tc.get(
            "/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": _REDIRECT_URI,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            },
            headers={"Cf-Access-Jwt-Assertion": "bad-jwt"},
            follow_redirects=False,
        )
        assert resp.status_code == 403

    def test_email_not_in_allowlist_returns_403(self):
        ext = self._ext_with_mock_verifier(email="unauthorized@example.com")
        ext._allowed_emails = frozenset(["allowed@example.com"])
        tc = _make_test_client(ext)
        client_id = self._register_client(tc)
        _, challenge = _make_pkce_pair()

        resp = tc.get(
            "/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": _REDIRECT_URI,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            },
            headers={"Cf-Access-Jwt-Assertion": "jwt"},
            follow_redirects=False,
        )
        assert resp.status_code == 403

    def test_unregistered_redirect_uri_rejected(self):
        ext = self._ext_with_mock_verifier()
        tc = _make_test_client(ext)
        client_id = self._register_client(tc, redirect_uri=_REDIRECT_URI)
        _, challenge = _make_pkce_pair()

        resp = tc.get(
            "/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": "https://attacker.example.com/callback",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            },
            headers={"Cf-Access-Jwt-Assertion": "jwt"},
            follow_redirects=False,
        )
        assert resp.status_code == 400

    def test_plain_s256_required(self):
        ext = self._ext_with_mock_verifier()
        tc = _make_test_client(ext)
        client_id = self._register_client(tc)

        resp = tc.get(
            "/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": _REDIRECT_URI,
                "code_challenge": "abc",
                "code_challenge_method": "plain",  # not supported
            },
            headers={"Cf-Access-Jwt-Assertion": "jwt"},
            follow_redirects=False,
        )
        assert resp.status_code == 400

    def test_missing_code_challenge_rejected(self):
        ext = self._ext_with_mock_verifier()
        tc = _make_test_client(ext)
        client_id = self._register_client(tc)

        resp = tc.get(
            "/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": _REDIRECT_URI,
            },
            headers={"Cf-Access-Jwt-Assertion": "jwt"},
            follow_redirects=False,
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# CloudflareAccessHttpExtension: /token
# ---------------------------------------------------------------------------


class TestToken:
    def _get_code(self, tc: TestClient, ext: CloudflareAccessHttpExtension) -> tuple[str, str]:
        """Register a client, run /authorize, extract code from redirect. Returns (code, verifier)."""
        verifier, challenge = _make_pkce_pair()
        client_id = tc.post("/register", json={"redirect_uris": [_REDIRECT_URI]}).json()["client_id"]

        resp = tc.get(
            "/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": _REDIRECT_URI,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            },
            headers={"Cf-Access-Jwt-Assertion": "mock"},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        location = resp.headers["location"]
        code = dict(p.split("=", 1) for p in location.split("?", 1)[1].split("&"))["code"]
        return code, verifier

    def _make_ext_with_mock(self) -> CloudflareAccessHttpExtension:
        ext = _make_http_ext()
        ext._cf_verify = AsyncMock(return_value={"email": _EMAIL, "sub": _EMAIL})
        return ext

    def test_happy_path_json_body(self):
        ext = self._make_ext_with_mock()
        tc = _make_test_client(ext)
        code, verifier = self._get_code(tc, ext)
        client_id = verify_code(_SECRET, code)["client_id"]

        resp = tc.post(
            "/token",
            json={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": _REDIRECT_URI,
                "code_verifier": verifier,
                "client_id": client_id,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["token_type"] == "Bearer"
        assert "access_token" in data
        assert data["expires_in"] == 3600
        # Verify the token round-trips through authenticate_mcp
        payload = verify_access_token(_SECRET, data["access_token"], _ISSUER)
        assert payload["sub"] == _EMAIL

    def test_happy_path_form_body(self):
        ext = self._make_ext_with_mock()
        tc = _make_test_client(ext)
        code, verifier = self._get_code(tc, ext)
        client_id = verify_code(_SECRET, code)["client_id"]

        resp = tc.post(
            "/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": _REDIRECT_URI,
                "code_verifier": verifier,
                "client_id": client_id,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_wrong_pkce_verifier(self):
        ext = self._make_ext_with_mock()
        tc = _make_test_client(ext)
        code, _ = self._get_code(tc, ext)
        client_id = verify_code(_SECRET, code)["client_id"]

        resp = tc.post(
            "/token",
            json={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": _REDIRECT_URI,
                "code_verifier": "wrong-verifier",
                "client_id": client_id,
            },
        )
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_grant"

    def test_redirect_uri_mismatch(self):
        ext = self._make_ext_with_mock()
        tc = _make_test_client(ext)
        code, verifier = self._get_code(tc, ext)

        resp = tc.post(
            "/token",
            json={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": "https://other.example.com/callback",
                "code_verifier": verifier,
            },
        )
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_grant"

    def test_expired_code(self):
        ext = self._make_ext_with_mock()
        ext._auth_code_ttl = -1  # immediately expired
        tc = _make_test_client(ext)
        code, verifier = self._get_code(tc, ext)
        client_id = verify_code.__module__  # just need any string; code already expired

        resp = tc.post(
            "/token",
            json={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": _REDIRECT_URI,
                "code_verifier": verifier,
            },
        )
        # The code was minted with ttl=-1 but auth_code_ttl was changed AFTER mint_code
        # so we need to test with a pre-expired code instead
        expired_code = mint_code(
            _SECRET,
            client_id="cid",
            redirect_uri=_REDIRECT_URI,
            code_challenge="abc",
            code_challenge_method="S256",
            email=_EMAIL,
            scope="mcp:full",
            ttl_seconds=-1,
        )
        resp2 = tc.post(
            "/token",
            json={
                "grant_type": "authorization_code",
                "code": expired_code,
                "redirect_uri": _REDIRECT_URI,
                "code_verifier": verifier,
            },
        )
        assert resp2.status_code == 400
        assert resp2.json()["error"] == "invalid_grant"

    def test_missing_code_verifier(self):
        ext = self._make_ext_with_mock()
        tc = _make_test_client(ext)
        code, _ = self._get_code(tc, ext)

        resp = tc.post(
            "/token",
            json={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": _REDIRECT_URI,
            },
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Full end-to-end OAuth dance
# ---------------------------------------------------------------------------


class TestFullOAuthDance:
    def test_register_authorize_token_authenticate_mcp(self):
        """Full flow: DCR → /authorize → /token → authenticate_mcp succeeds."""
        http_ext = _make_http_ext()
        http_ext._cf_verify = AsyncMock(return_value={"email": _EMAIL, "sub": _EMAIL})

        tenant_ext = _make_tenant_ext()
        tc = _make_test_client(http_ext)

        # Step 1: Register
        reg_resp = tc.post("/register", json={"redirect_uris": [_REDIRECT_URI], "client_name": "TestClient"})
        assert reg_resp.status_code == 201
        client_id = reg_resp.json()["client_id"]

        # Step 2: Authorize
        verifier, challenge = _make_pkce_pair()
        auth_resp = tc.get(
            "/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": _REDIRECT_URI,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": "xyz",
            },
            headers={"Cf-Access-Jwt-Assertion": "mock-cf-jwt"},
            follow_redirects=False,
        )
        assert auth_resp.status_code == 302
        location = auth_resp.headers["location"]
        params_raw = location.split("?", 1)[1]
        code = dict(p.split("=", 1) for p in params_raw.split("&"))["code"]

        # Step 3: Token exchange
        token_resp = tc.post(
            "/token",
            json={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": _REDIRECT_URI,
                "code_verifier": verifier,
                "client_id": client_id,
            },
        )
        assert token_resp.status_code == 200
        access_token = token_resp.json()["access_token"]

        # Step 4: authenticate_mcp succeeds with the issued token
        import asyncio

        ctx = asyncio.get_event_loop().run_until_complete(
            tenant_ext.authenticate_mcp(RequestContext(api_key=access_token))
        )
        assert ctx.schema_name
