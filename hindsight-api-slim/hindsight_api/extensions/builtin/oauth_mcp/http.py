"""OAuthMcpHttpExtension — OAuth 2.1 authorization server endpoints for MCP clients.

Mounts at the app root (not under /ext/) via get_root_router() so well-known paths
and /authorize, /token, /register are at the standard locations.

Cloudflare Access must protect GET /authorize with SSO. All other OAuth endpoints
(/token, /register, /.well-known/*) must bypass Access authentication so MCP clients
can reach them directly.

Flow:
  1. MCP client hits GET /.well-known/oauth-protected-resource (no auth)
  2. Client registers: POST /register → receives client_id (signed JWT)
  3. Client redirects browser to GET /authorize → CF Access SSO → 302 back with code
  4. Client exchanges: POST /token + PKCE verifier → receives access_token (signed JWT)
  5. Client calls /mcp/* with Authorization: Bearer <access_token>
     → validated by OAuthMcpTenantExtension.authenticate_mcp()

Configuration:
    HINDSIGHT_API_HTTP_EXTENSION=hindsight_api.extensions.builtin.oauth_mcp:OAuthMcpHttpExtension
    HINDSIGHT_API_HTTP_OAUTH_SIGNING_SECRET=<must-match-tenant-extension>
    HINDSIGHT_API_HTTP_OAUTH_ISSUER=https://hindsight.yourdomain.com
    HINDSIGHT_API_HTTP_OAUTH_RESOURCE=https://hindsight.yourdomain.com/mcp   (optional, defaults to issuer/mcp)
    HINDSIGHT_API_HTTP_OAUTH_ACCESS_TEAM=<cloudflare-team-name>
    HINDSIGHT_API_HTTP_OAUTH_ACCESS_AUD=<access-application-aud-tag>
    HINDSIGHT_API_HTTP_OAUTH_ALLOWED_EMAILS=alice@example.com,bob@example.com  (empty = any Access user)
    HINDSIGHT_API_HTTP_OAUTH_ACCESS_TOKEN_TTL=3600   (seconds, default 1h)
    HINDSIGHT_API_HTTP_OAUTH_AUTH_CODE_TTL=600       (seconds, default 10m)
"""

from __future__ import annotations

import logging
import re
import time
from urllib.parse import urlencode, urlparse

import jwt as pyjwt
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from hindsight_api.extensions.builtin.oauth_mcp.cf_access import CfAccessVerifier
from hindsight_api.extensions.builtin.oauth_mcp.metadata import (
    authorization_server_metadata,
    protected_resource_metadata,
)
from hindsight_api.extensions.builtin.oauth_mcp.tokens import (
    mint_access_token,
    mint_client_id,
    mint_code,
    verify_client_id,
    verify_code,
    verify_pkce_s256,
)
from hindsight_api.extensions.http import HttpExtension

logger = logging.getLogger(__name__)

# RFC 8252 §8.3 — loopback redirect URIs (http://127.0.0.1 with optional port and path)
_LOOPBACK_RE = re.compile(r"^http://127\.0\.0\.1(:\d+)?(/.*)?$")


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


class OAuthMcpHttpExtension(HttpExtension):
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

        if access_team and access_aud:
            self._cf_verifier: CfAccessVerifier | None = CfAccessVerifier(access_team, access_aud)
        else:
            self._cf_verifier = None
            logger.warning(
                "HINDSIGHT_API_HTTP_OAUTH_ACCESS_TEAM / OAUTH_ACCESS_AUD not configured; "
                "GET /authorize will reject all requests"
            )

    async def on_startup(self) -> None:
        if self._cf_verifier:
            await self._cf_verifier.startup()

    async def on_shutdown(self) -> None:
        if self._cf_verifier:
            await self._cf_verifier.shutdown()

    def get_router(self, memory) -> APIRouter:
        return APIRouter()

    def get_root_router(self, memory) -> APIRouter:
        router = APIRouter(tags=["OAuth MCP"])

        signing_secret = self._signing_secret
        issuer = self._issuer
        resource = self._resource
        cf_verifier = self._cf_verifier
        allowed_emails = self._allowed_emails
        access_token_ttl = self._access_token_ttl
        auth_code_ttl = self._auth_code_ttl

        @router.get("/.well-known/oauth-authorization-server")
        async def oauth_authorization_server_metadata():
            return authorization_server_metadata(issuer, resource)

        @router.get("/.well-known/oauth-protected-resource")
        async def oauth_protected_resource_metadata():
            return protected_resource_metadata(issuer, resource)

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

            if cf_verifier is None:
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
                cf_payload = await cf_verifier.verify(cf_jwt)
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
