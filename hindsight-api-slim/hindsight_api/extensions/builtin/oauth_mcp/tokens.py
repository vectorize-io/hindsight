"""Stateless JWT minting and verification for OAuth 2.1 flows.

All tokens are HMAC-HS256 signed with a shared signing secret. No database writes.

Token types:
  client_id   — encodes registered redirect_uris; no expiry (clients are long-lived)
  code        — short-lived auth code with embedded PKCE challenge
  access_token — bearer token validated by OAuthMcpTenantExtension on every /mcp request
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from datetime import datetime, timezone

import jwt as pyjwt

_ALGORITHM = "HS256"
_AUD_CLIENT = "hindsight:client"
_AUD_CODE = "hindsight:code"
_AUD_ACCESS = "hindsight:mcp"


def _now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _jti() -> str:
    return secrets.token_hex(16)


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
