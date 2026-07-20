"""Shared local JWT verification for Supabase-backed extensions."""

from __future__ import annotations

import time
from typing import ClassVar

import httpx
import jwt as pyjwt
from jwt import PyJWK

from hindsight_api.extensions.tenant import AuthenticationError

JWKS_CACHE_TTL_SECONDS = 600
JWKS_MIN_REFRESH_INTERVAL_SECONDS = 30
SUPPORTED_ALGORITHMS = ["RS256", "ES256"]


class SupabaseJwksVerifierMixin:
    """Provide cached JWKS verification to extensions with a Supabase client."""

    supabase_url: str
    _http_client: httpx.AsyncClient | None
    _jwks_keys: dict[str, PyJWK]
    _jwks_last_fetched: float
    _use_jwks: bool
    supported_algorithms: ClassVar[list[str]] = SUPPORTED_ALGORITHMS

    async def _fetch_jwks(self) -> None:
        if self._http_client is None:
            raise RuntimeError("HTTP client not initialized")
        response = await self._http_client.get(f"{self.supabase_url}/auth/v1/.well-known/jwks.json")
        response.raise_for_status()
        self._jwks_keys = {
            key_data["kid"]: self._create_jwk(key_data)
            for key_data in response.json().get("keys", [])
            if key_data.get("kid")
        }
        self._jwks_last_fetched = time.monotonic()

    def _create_jwk(self, key_data: dict[str, object]) -> PyJWK:
        return PyJWK(key_data)

    async def _get_signing_key(self, token: str) -> PyJWK:
        kid = pyjwt.get_unverified_header(token).get("kid")
        if not kid:
            raise AuthenticationError("Token missing key ID (kid) header")
        now = time.monotonic()
        if now - self._jwks_last_fetched > JWKS_CACHE_TTL_SECONDS:
            await self._fetch_jwks()
        if kid in self._jwks_keys:
            return self._jwks_keys[kid]
        if now - self._jwks_last_fetched > JWKS_MIN_REFRESH_INTERVAL_SECONDS:
            await self._fetch_jwks()
            if kid in self._jwks_keys:
                return self._jwks_keys[kid]
        raise AuthenticationError("Unable to find signing key for token")

    async def _verify_token_jwks(self, token: str) -> str:
        try:
            signing_key = await self._get_signing_key(token)
            payload = pyjwt.decode(
                token,
                signing_key.key,
                algorithms=self.supported_algorithms,
                audience="authenticated",
                issuer=f"{self.supabase_url}/auth/v1",
            )
        except pyjwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except pyjwt.InvalidAudienceError:
            raise AuthenticationError("Invalid token audience")
        except pyjwt.InvalidIssuerError:
            raise AuthenticationError("Invalid token issuer")
        except pyjwt.DecodeError:
            raise AuthenticationError("Invalid token")
        except AuthenticationError:
            raise
        except Exception as exc:
            raise AuthenticationError(f"Token verification failed: {exc!s}")

        user_id = payload.get("sub")
        if not user_id:
            raise AuthenticationError("Token valid but missing subject (sub) claim")
        return str(user_id)
