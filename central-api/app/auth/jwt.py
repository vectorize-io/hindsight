"""Authentik JWT validation — module boundary only.

The real implementation will fetch JWKS from `settings.authentik_jwks_url`, verify
RS256 signatures, and check the issuer. The scaffold intentionally makes no network
calls and returns None (no claims), so callers fall back to the dev context.
"""

from __future__ import annotations

from app.config import settings


async def validate_jwt(token: str) -> dict | None:
    """Validate a bearer JWT and return its claims, or None if unverifiable.

    Scaffold: always returns None (no JWKS call). The boundary, config fields
    (`authentik_issuer`, `authentik_jwks_url`, `jwt_algorithms`), and signature
    are in place for the real implementation.
    """
    _ = (token, settings.authentik_jwks_url, settings.jwt_algorithms)
    return None
