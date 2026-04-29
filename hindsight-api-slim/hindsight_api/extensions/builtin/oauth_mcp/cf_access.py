"""Cloudflare Access JWT verifier using cached JWKS.

Mirrors the JWKS pattern from SupabaseTenantExtension:
  - Cache public keys locally (10 min TTL)
  - Refresh on kid miss (once per 30s minimum)
  - Validate iss, aud, exp

Usage:
    verifier = CfAccessVerifier(team="myteam", aud="<application-aud>")
    await verifier.startup()      # pre-fetches JWKS; call from on_startup()
    payload = await verifier.verify(jwt_string)   # {email, sub, ...}
    await verifier.shutdown()     # closes httpx client
"""

from __future__ import annotations

import logging
import time

import httpx
import jwt as pyjwt
from jwt import PyJWK

logger = logging.getLogger(__name__)

_JWKS_CACHE_TTL = 600
_JWKS_MIN_REFRESH_INTERVAL = 30
_SUPPORTED_ALGORITHMS = ["RS256"]


class CfAccessVerifier:
    def __init__(self, team: str, aud: str) -> None:
        self._team = team
        self._aud = aud
        self._jwks_url = f"https://{team}.cloudflareaccess.com/cdn-cgi/access/certs"
        self._issuer = f"https://{team}.cloudflareaccess.com"
        self._keys: dict[str, PyJWK] = {}
        self._last_fetched: float = 0
        self._http: httpx.AsyncClient | None = None

    async def startup(self) -> None:
        self._http = httpx.AsyncClient(timeout=10.0)
        try:
            await self._fetch_jwks()
            logger.info("CF Access JWKS loaded (%d keys) from %s", len(self._keys), self._jwks_url)
        except Exception as exc:
            logger.warning("Could not pre-fetch CF Access JWKS (will retry on first request): %s", exc)

    async def shutdown(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None

    async def _fetch_jwks(self) -> None:
        if self._http is None:
            raise RuntimeError("CfAccessVerifier not started — call startup() first")
        resp = await self._http.get(self._jwks_url)
        resp.raise_for_status()
        self._keys = {kd["kid"]: PyJWK(kd) for kd in resp.json().get("keys", []) if "kid" in kd}
        self._last_fetched = time.monotonic()

    async def _resolve_key(self, token: str) -> PyJWK:
        header = pyjwt.get_unverified_header(token)
        kid = header.get("kid")
        if not kid:
            raise ValueError("CF Access JWT missing kid header")

        now = time.monotonic()
        if now - self._last_fetched > _JWKS_CACHE_TTL:
            await self._fetch_jwks()

        if kid in self._keys:
            return self._keys[kid]

        if now - self._last_fetched > _JWKS_MIN_REFRESH_INTERVAL:
            logger.info("Signing key %r not cached — refreshing JWKS (possible key rotation)", kid)
            await self._fetch_jwks()
            if kid in self._keys:
                return self._keys[kid]

        raise ValueError(f"CF Access signing key {kid!r} not found in JWKS at {self._jwks_url}")

    async def verify(self, token: str) -> dict:
        """Verify CF Access JWT. Returns payload dict (contains at least email and sub).

        Raises ValueError with a human-readable reason on failure.
        """
        key = await self._resolve_key(token)
        try:
            payload = pyjwt.decode(
                token,
                key.key,
                algorithms=_SUPPORTED_ALGORITHMS,
                audience=self._aud,
                issuer=self._issuer,
            )
        except pyjwt.ExpiredSignatureError:
            raise ValueError("CF Access JWT has expired")
        except pyjwt.InvalidAudienceError:
            raise ValueError(f"CF Access JWT audience does not match configured aud={self._aud!r}")
        except pyjwt.InvalidIssuerError:
            raise ValueError(f"CF Access JWT issuer does not match {self._issuer!r}")
        except pyjwt.DecodeError as exc:
            raise ValueError(f"CF Access JWT decode failed: {exc}")
        return payload
