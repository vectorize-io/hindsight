"""Built-in OAuth 2.1 extensions for MCP clients using Cloudflare Access SSO.

Implements an OAuth 2.1 authorization server directly inside Hindsight, delegating
human authentication to Cloudflare Access. No separate proxy container or KV store
required.

Two extensions must be loaded together:

  CloudflareAccessTenantExtension — API key auth for HTTP routes (/v1/*), OAuth JWT
    bearer token auth for MCP routes (/mcp/*).

  CloudflareAccessHttpExtension — mounts the OAuth 2.1 authorization server endpoints
    at the app root: /.well-known/*, /register, /authorize, /token.

Cloudflare Access must protect GET /authorize with SSO. All other OAuth endpoints
(/token, /register, /.well-known/*, /mcp/*) must bypass Access so MCP clients can
reach them directly with their own bearer credentials.

Flow:
  1. MCP client hits GET /.well-known/oauth-protected-resource (no auth)
  2. Client registers: POST /register → receives client_id (signed JWT)
  3. Client redirects browser to GET /authorize → CF Access SSO → 302 back with code
  4. Client exchanges: POST /token + PKCE verifier → receives access_token (signed JWT)
  5. Client calls /mcp/* with Authorization: Bearer <access_token>

All tokens are HMAC-HS256 signed JWTs — no database writes, no KV storage.

Configuration:
    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.cloudflare_access:CloudflareAccessTenantExtension
    HINDSIGHT_API_TENANT_API_KEY=<existing-http-api-key>
    HINDSIGHT_API_TENANT_OAUTH_SIGNING_SECRET=<random-256-bit, must-match-http-extension>
    HINDSIGHT_API_TENANT_OAUTH_ISSUER=https://hindsight.yourdomain.com

    HINDSIGHT_API_HTTP_EXTENSION=hindsight_api.extensions.builtin.cloudflare_access:CloudflareAccessHttpExtension
    HINDSIGHT_API_HTTP_OAUTH_SIGNING_SECRET=<same-as-tenant>
    HINDSIGHT_API_HTTP_OAUTH_ISSUER=https://hindsight.yourdomain.com
    HINDSIGHT_API_HTTP_OAUTH_RESOURCE=https://hindsight.yourdomain.com/mcp   (optional)
    HINDSIGHT_API_HTTP_OAUTH_ACCESS_TEAM=<cloudflare-team-name>
    HINDSIGHT_API_HTTP_OAUTH_ACCESS_AUD=<access-application-aud-tag>
    HINDSIGHT_API_HTTP_OAUTH_ALLOWED_EMAILS=alice@example.com   (comma-separated; empty = any Access user)
    HINDSIGHT_API_HTTP_OAUTH_ACCESS_TOKEN_TTL=3600   (seconds, default 1h)
    HINDSIGHT_API_HTTP_OAUTH_AUTH_CODE_TTL=600       (seconds, default 10m)
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
import secrets
import time
from datetime import datetime, timezone
from urllib.parse import urlencode, urlparse

import httpx
import jwt as pyjwt
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from jwt import PyJWK

from hindsight_api.config import get_config
from hindsight_api.extensions.builtin.tenant import ApiKeyTenantExtension
from hindsight_api.extensions.http import HttpExtension
from hindsight_api.extensions.tenant import AuthenticationError, TenantContext
from hindsight_api.models import RequestContext

logger = logging.getLogger(__name__)

# ── JWT constants ──────────────────────────────────────────────────────────────

_ALGORITHM = "HS256"
_AUD_CLIENT = "hindsight:client"
_AUD_CODE = "hindsight:code"
_AUD_ACCESS = "hindsight:mcp"

# ── JWKS cache tuning ──────────────────────────────────────────────────────────

_JWKS_CACHE_TTL = 600
_JWKS_MIN_REFRESH_INTERVAL = 30
_SUPPORTED_CF_ALGORITHMS = ["RS256"]

# ── Redirect URI validation ────────────────────────────────────────────────────

# RFC 8252 §8.3 — loopback redirect URIs (http://127.0.0.1 with optional port/path)
_LOOPBACK_RE = re.compile(r"^http://127\.0\.0\.1(:\d+)?(/.*)?$")


# ── Low-level helpers ──────────────────────────────────────────────────────────


def _now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _jti() -> str:
    return secrets.token_hex(16)


def _is_valid_redirect_uri(uri: str) -> bool:
    parsed = urlparse(uri)
    if parsed.scheme == "https":
        return True
    if _LOOPBACK_RE.match(uri):
        return True
    # Custom URI schemes for native apps (e.g. "com.example.app://callback")
    if parsed.scheme and "." in parsed.scheme:
        return True
    return False


def _oauth_error(error: str, description: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": error, "error_description": description}, status_code=status)


# ── Token mint / verify helpers (exported — used by tests) ────────────────────


def mint_client_id(signing_key: str, redirect_uris: list[str], client_name: str) -> str:
    """Issue a signed client_id JWT encoding the registered redirect_uris."""
    return pyjwt.encode(
        {
            "aud": _AUD_CLIENT,
            "iat": _now(),
            "jti": _jti(),
            "redirect_uris": redirect_uris,
            "client_name": client_name,
        },
        signing_key,
        algorithm=_ALGORITHM,
    )


def verify_client_id(signing_key: str, token: str) -> dict:
    """Decode and verify a client_id JWT. Raises pyjwt.InvalidTokenError on failure."""
    return pyjwt.decode(
        token,
        signing_key,
        algorithms=[_ALGORITHM],
        audience=_AUD_CLIENT,
        options={"verify_exp": False},
    )


def mint_code(
    signing_key: str,
    *,
    client_id: str,
    redirect_uri: str,
    code_challenge: str,
    code_challenge_method: str,
    email: str,
    scope: str,
    ttl_seconds: int = 600,
) -> str:
    """Issue a short-lived authorization code JWT with embedded PKCE challenge."""
    now = _now()
    return pyjwt.encode(
        {
            "aud": _AUD_CODE,
            "iat": now,
            "exp": now + ttl_seconds,
            "jti": _jti(),
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
            "email": email,
            "scope": scope,
        },
        signing_key,
        algorithm=_ALGORITHM,
    )


def verify_code(signing_key: str, token: str) -> dict:
    """Decode and verify an authorization code JWT. Raises on expiry or bad signature."""
    return pyjwt.decode(
        token,
        signing_key,
        algorithms=[_ALGORITHM],
        audience=_AUD_CODE,
    )


def mint_access_token(
    signing_key: str,
    *,
    email: str,
    scope: str,
    issuer: str,
    ttl_seconds: int = 3600,
) -> str:
    """Issue an access token JWT for use as MCP bearer."""
    now = _now()
    return pyjwt.encode(
        {
            "iss": issuer,
            "sub": email,
            "aud": _AUD_ACCESS,
            "iat": now,
            "exp": now + ttl_seconds,
            "jti": _jti(),
            "scope": scope,
        },
        signing_key,
        algorithm=_ALGORITHM,
    )


def verify_access_token(signing_key: str, token: str, issuer: str) -> dict:
    """Decode and verify an access token JWT. Raises pyjwt.InvalidTokenError on failure."""
    return pyjwt.decode(
        token,
        signing_key,
        algorithms=[_ALGORITHM],
        audience=_AUD_ACCESS,
        issuer=issuer,
    )


def verify_pkce_s256(code_challenge: str, code_verifier: str) -> bool:
    """Return True iff base64url(sha256(code_verifier)) == code_challenge (RFC 7636 §4.6)."""
    digest = hashlib.sha256(code_verifier.encode()).digest()
    return _b64url(digest) == code_challenge


# ── Well-known metadata builders ───────────────────────────────────────────────


def _authorization_server_metadata(issuer: str, resource: str) -> dict:
    """RFC 8414 OAuth 2.0 Authorization Server Metadata."""
    return {
        "issuer": issuer,
        "authorization_endpoint": f"{issuer}/authorize",
        "token_endpoint": f"{issuer}/token",
        "registration_endpoint": f"{issuer}/register",
        "scopes_supported": ["mcp:full"],
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "token_endpoint_auth_methods_supported": ["none"],
        "code_challenge_methods_supported": ["S256"],
    }


def _protected_resource_metadata(issuer: str, resource: str) -> dict:
    """RFC 9728 OAuth 2.0 Protected Resource Metadata."""
    return {
        "resource": resource,
        "authorization_servers": [issuer],
        "bearer_methods_supported": ["header"],
        "scopes_supported": ["mcp:full"],
    }


# ── CloudflareAccessTenantExtension ────────────────────────────────────────────────────


class CloudflareAccessTenantExtension(ApiKeyTenantExtension):
    """TenantExtension: API key auth for HTTP routes, OAuth JWT auth for MCP routes.

    Inherits API key validation from ApiKeyTenantExtension. MCP requests must carry
    a bearer token issued by CloudflareAccessHttpExtension's /token endpoint.

    On MCP auth failure the WWW-Authenticate header points to the OAuth protected-resource
    metadata endpoint so MCP clients can auto-discover the authorization server.
    """

    def __init__(self, config: dict[str, str]) -> None:
        super().__init__(config)
        self._signing_secret = config.get("oauth_signing_secret", "")
        self._issuer = config.get("oauth_issuer", "").rstrip("/")
        if not self._signing_secret:
            raise ValueError("HINDSIGHT_API_TENANT_OAUTH_SIGNING_SECRET is required for CloudflareAccessTenantExtension")
        if not self._issuer:
            raise ValueError("HINDSIGHT_API_TENANT_OAUTH_ISSUER is required for CloudflareAccessTenantExtension")

    def _resource_metadata_url(self) -> str:
        return f"{self._issuer}/.well-known/oauth-protected-resource"

    async def authenticate_mcp(self, context: RequestContext) -> TenantContext:
        token = context.api_key
        if not token:
            raise AuthenticationError(
                "Authorization header with Bearer token is required for MCP",
                headers={
                    "WWW-Authenticate": (
                        f'Bearer realm="hindsight-mcp", resource_metadata="{self._resource_metadata_url()}"'
                    )
                },
            )

        try:
            payload = verify_access_token(self._signing_secret, token, self._issuer)
        except pyjwt.ExpiredSignatureError:
            raise AuthenticationError(
                "Access token has expired — reconnect to refresh",
                headers={
                    "WWW-Authenticate": (
                        f'Bearer realm="hindsight-mcp" error="invalid_token"'
                        f' error_description="Token expired",'
                        f' resource_metadata="{self._resource_metadata_url()}"'
                    )
                },
            )
        except pyjwt.InvalidTokenError as exc:
            raise AuthenticationError(
                f"Invalid access token: {exc}",
                headers={
                    "WWW-Authenticate": (
                        f'Bearer realm="hindsight-mcp" error="invalid_token",'
                        f' resource_metadata="{self._resource_metadata_url()}"'
                    )
                },
            )

        schema = get_config().database_schema
        logger.debug("MCP authenticated: sub=%s schema=%s", payload.get("sub"), schema)
        return TenantContext(schema_name=schema)


# ── CloudflareAccessHttpExtension ──────────────────────────────────────────────────────


class CloudflareAccessHttpExtension(HttpExtension):
    """HttpExtension: mounts OAuth 2.1 authorization server endpoints at the app root.

    Endpoints: /.well-known/oauth-authorization-server, /.well-known/oauth-protected-resource,
    POST /register, GET /authorize, POST /token.

    CF Access JWKS verification is handled inline — no separate verifier class.
    JWKS keys are cached with a 10-minute TTL; on kid miss the cache refreshes once per 30s.
    """

    def __init__(self, config: dict[str, str]) -> None:
        super().__init__(config)
        self._signing_secret = config.get("oauth_signing_secret", "")
        self._issuer = config.get("oauth_issuer", "").rstrip("/")
        resource = config.get("oauth_resource", "").rstrip("/")
        self._resource = resource or (f"{self._issuer}/mcp" if self._issuer else "")
        access_team = config.get("oauth_access_team", "")
        access_aud = config.get("oauth_access_aud", "")
        allowed_raw = config.get("oauth_allowed_emails", "")
        self._allowed_emails: frozenset[str] = (
            frozenset(e.strip() for e in allowed_raw.split(",") if e.strip()) if allowed_raw.strip() else frozenset()
        )
        self._access_token_ttl = int(config.get("oauth_access_token_ttl", "3600"))
        self._auth_code_ttl = int(config.get("oauth_auth_code_ttl", "600"))

        if not self._signing_secret:
            raise ValueError("HINDSIGHT_API_HTTP_OAUTH_SIGNING_SECRET is required")
        if not self._issuer:
            raise ValueError("HINDSIGHT_API_HTTP_OAUTH_ISSUER is required")

        # CF Access JWKS state
        self._cf_team = access_team
        self._cf_aud = access_aud
        self._cf_jwks_url = f"https://{access_team}.cloudflareaccess.com/cdn-cgi/access/certs" if access_team else ""
        self._cf_issuer = f"https://{access_team}.cloudflareaccess.com" if access_team else ""
        self._cf_keys: dict[str, PyJWK] = {}
        self._cf_last_fetched: float = 0
        self._cf_http: httpx.AsyncClient | None = None
        self._cf_configured = bool(access_team and access_aud)

        if not self._cf_configured:
            logger.warning(
                "HINDSIGHT_API_HTTP_OAUTH_ACCESS_TEAM / OAUTH_ACCESS_AUD not configured; "
                "GET /authorize will reject all requests"
            )

    # ── CF Access JWKS methods ─────────────────────────────────────────────────

    async def on_startup(self) -> None:
        if self._cf_configured:
            self._cf_http = httpx.AsyncClient(timeout=10.0)
            try:
                await self._cf_fetch_jwks()
                logger.info("CF Access JWKS loaded (%d keys) from %s", len(self._cf_keys), self._cf_jwks_url)
            except Exception as exc:
                logger.warning("Could not pre-fetch CF Access JWKS (will retry on first request): %s", exc)

    async def on_shutdown(self) -> None:
        if self._cf_http:
            await self._cf_http.aclose()
            self._cf_http = None

    async def _cf_fetch_jwks(self) -> None:
        if self._cf_http is None:
            raise RuntimeError("CF Access JWKS: no HTTP client — on_startup() not called")
        resp = await self._cf_http.get(self._cf_jwks_url)
        resp.raise_for_status()
        self._cf_keys = {kd["kid"]: PyJWK(kd) for kd in resp.json().get("keys", []) if "kid" in kd}
        self._cf_last_fetched = time.monotonic()

    async def _cf_resolve_key(self, token: str) -> PyJWK:
        header = pyjwt.get_unverified_header(token)
        kid = header.get("kid")
        if not kid:
            raise ValueError("CF Access JWT missing kid header")

        now = time.monotonic()
        if now - self._cf_last_fetched > _JWKS_CACHE_TTL:
            await self._cf_fetch_jwks()

        if kid in self._cf_keys:
            return self._cf_keys[kid]

        if now - self._cf_last_fetched > _JWKS_MIN_REFRESH_INTERVAL:
            logger.info("Signing key %r not cached — refreshing JWKS (possible key rotation)", kid)
            await self._cf_fetch_jwks()
            if kid in self._cf_keys:
                return self._cf_keys[kid]

        raise ValueError(f"CF Access signing key {kid!r} not found in JWKS at {self._cf_jwks_url}")

    async def _cf_verify(self, token: str) -> dict:
        """Verify a CF Access JWT. Returns payload dict. Raises ValueError on failure."""
        key = await self._cf_resolve_key(token)
        try:
            payload = pyjwt.decode(
                token,
                key.key,
                algorithms=_SUPPORTED_CF_ALGORITHMS,
                audience=self._cf_aud,
                issuer=self._cf_issuer,
            )
        except pyjwt.ExpiredSignatureError:
            raise ValueError("CF Access JWT has expired")
        except pyjwt.InvalidAudienceError:
            raise ValueError(f"CF Access JWT audience does not match configured aud={self._cf_aud!r}")
        except pyjwt.InvalidIssuerError:
            raise ValueError(f"CF Access JWT issuer does not match {self._cf_issuer!r}")
        except pyjwt.DecodeError as exc:
            raise ValueError(f"CF Access JWT decode failed: {exc}")
        return payload

    # ── FastAPI routers ────────────────────────────────────────────────────────

    def get_router(self, memory) -> APIRouter:
        return APIRouter()

    def get_root_router(self, memory) -> APIRouter:
        router = APIRouter(tags=["OAuth MCP"])

        signing_secret = self._signing_secret
        issuer = self._issuer
        resource = self._resource
        allowed_emails = self._allowed_emails
        access_token_ttl = self._access_token_ttl
        auth_code_ttl = self._auth_code_ttl
        cf_configured = self._cf_configured

        @router.get("/.well-known/oauth-authorization-server")
        async def oauth_authorization_server_metadata():
            return _authorization_server_metadata(issuer, resource)

        @router.get("/.well-known/oauth-protected-resource")
        async def oauth_protected_resource_metadata():
            return _protected_resource_metadata(issuer, resource)

        @router.post("/register")
        async def register(request: Request):
            try:
                body = await request.json()
            except Exception:
                return _oauth_error("invalid_request", "Request body must be JSON")

            redirect_uris: list[str] = body.get("redirect_uris") or []
            client_name: str = str(body.get("client_name") or "unknown")

            if not redirect_uris:
                return _oauth_error("invalid_client_metadata", "redirect_uris is required")

            invalid = [u for u in redirect_uris if not _is_valid_redirect_uri(u)]
            if invalid:
                return _oauth_error(
                    "invalid_client_metadata",
                    f"Redirect URI(s) must be HTTPS or loopback http://127.0.0.1: {invalid}",
                )

            client_id = mint_client_id(signing_secret, redirect_uris, client_name)
            return JSONResponse(
                {
                    "client_id": client_id,
                    "client_id_issued_at": int(time.time()),
                    "redirect_uris": redirect_uris,
                    "client_name": client_name,
                    "token_endpoint_auth_method": "none",
                    "grant_types": ["authorization_code"],
                    "response_types": ["code"],
                },
                status_code=201,
            )

        @router.get("/authorize")
        async def authorize(request: Request):
            params = dict(request.query_params)
            response_type = params.get("response_type")
            client_id = params.get("client_id")
            redirect_uri = params.get("redirect_uri")
            code_challenge = params.get("code_challenge")
            code_challenge_method = params.get("code_challenge_method", "S256")
            state = params.get("state", "")
            scope = params.get("scope", "mcp:full")

            if response_type != "code":
                return _oauth_error("unsupported_response_type", "Only response_type=code is supported")
            if not client_id:
                return _oauth_error("invalid_request", "client_id is required")
            if not redirect_uri:
                return _oauth_error("invalid_request", "redirect_uri is required")
            if not code_challenge:
                return _oauth_error("invalid_request", "code_challenge is required (PKCE)")
            if code_challenge_method != "S256":
                return _oauth_error("invalid_request", "Only code_challenge_method=S256 is supported")

            try:
                client_payload = verify_client_id(signing_secret, client_id)
            except pyjwt.InvalidTokenError:
                return _oauth_error("invalid_client", "Invalid or unrecognized client_id")

            registered_uris: list[str] = client_payload.get("redirect_uris", [])
            if redirect_uri not in registered_uris:
                return _oauth_error("invalid_request", "redirect_uri not registered for this client")

            if not cf_configured:
                return HTMLResponse(
                    "<h1>Server Misconfiguration</h1>"
                    "<p>Cloudflare Access is not configured. "
                    "Set HINDSIGHT_API_HTTP_OAUTH_ACCESS_TEAM and OAUTH_ACCESS_AUD.</p>",
                    status_code=500,
                )

            cf_jwt = request.headers.get("Cf-Access-Jwt-Assertion") or request.cookies.get("CF_Authorization")
            if not cf_jwt:
                return HTMLResponse(
                    "<h1>Authentication Required</h1>"
                    "<p>This endpoint requires Cloudflare Access authentication. "
                    "Ensure /authorize is protected by a Cloudflare Access application policy.</p>",
                    status_code=401,
                )

            try:
                cf_payload = await self._cf_verify(cf_jwt)
            except ValueError as exc:
                logger.warning("CF Access JWT verification failed: %s", exc)
                return HTMLResponse(
                    f"<h1>Access Denied</h1><p>{exc}</p>",
                    status_code=403,
                )

            email: str = cf_payload.get("email") or cf_payload.get("sub", "")
            if not email:
                return _oauth_error("access_denied", "CF Access JWT missing email claim", status=403)

            if allowed_emails and email not in allowed_emails:
                logger.warning("CF Access email %r not in allowed list", email)
                return HTMLResponse(
                    "<h1>Access Denied</h1><p>Your account is not authorized to use this server.</p>",
                    status_code=403,
                )

            code = mint_code(
                signing_secret,
                client_id=client_id,
                redirect_uri=redirect_uri,
                code_challenge=code_challenge,
                code_challenge_method=code_challenge_method,
                email=email,
                scope=scope,
                ttl_seconds=auth_code_ttl,
            )
            qs = urlencode({"code": code, "state": state} if state else {"code": code})
            return RedirectResponse(f"{redirect_uri}?{qs}", status_code=302)

        @router.post("/token")
        async def token(request: Request):
            content_type = request.headers.get("content-type", "")
            if "application/x-www-form-urlencoded" in content_type:
                form = await request.form()
                params: dict = dict(form)
            else:
                try:
                    params = await request.json()
                except Exception:
                    return _oauth_error("invalid_request", "Request body must be form-encoded or JSON")

            grant_type = params.get("grant_type")
            code_val = params.get("code")
            redirect_uri = params.get("redirect_uri")
            code_verifier = params.get("code_verifier")
            client_id = params.get("client_id")

            if grant_type != "authorization_code":
                return _oauth_error("unsupported_grant_type", "Only authorization_code is supported")
            if not code_val:
                return _oauth_error("invalid_request", "code is required")
            if not redirect_uri:
                return _oauth_error("invalid_request", "redirect_uri is required")
            if not code_verifier:
                return _oauth_error("invalid_request", "code_verifier is required (PKCE)")

            try:
                code_payload = verify_code(signing_secret, code_val)
            except pyjwt.ExpiredSignatureError:
                return _oauth_error("invalid_grant", "Authorization code has expired", status=400)
            except pyjwt.InvalidTokenError:
                return _oauth_error("invalid_grant", "Invalid authorization code", status=400)

            if not verify_pkce_s256(code_payload["code_challenge"], code_verifier):
                return _oauth_error("invalid_grant", "PKCE code_verifier mismatch", status=400)

            if code_payload.get("redirect_uri") != redirect_uri:
                return _oauth_error("invalid_grant", "redirect_uri does not match authorization request", status=400)

            if client_id and code_payload.get("client_id") != client_id:
                return _oauth_error("invalid_client", "client_id does not match authorization request", status=400)

            access_token = mint_access_token(
                signing_secret,
                email=code_payload["email"],
                scope=code_payload.get("scope", "mcp:full"),
                issuer=issuer,
                ttl_seconds=access_token_ttl,
            )
            return JSONResponse(
                {
                    "access_token": access_token,
                    "token_type": "Bearer",
                    "expires_in": access_token_ttl,
                    "scope": code_payload.get("scope", "mcp:full"),
                }
            )

        return router
