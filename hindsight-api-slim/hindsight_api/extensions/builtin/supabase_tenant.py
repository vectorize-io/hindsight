"""
Supabase Tenant Extension for Hindsight

Validates Supabase JWTs and maps authenticated users to isolated memory banks.
Each user gets their own PostgreSQL schema based on their Supabase user ID.

This extension enables multi-tenant memory isolation for applications using
Supabase Auth - each authenticated user's memories are stored in a separate
schema, ensuring complete data isolation.

Features:
    - Local JWT Verification: Validates tokens locally using JWKS public keys
      (no network call per request)
    - Automatic Schema Isolation: Each user gets {prefix}_{user_id} schema
    - Zero User Management: Leverages your existing Supabase Auth setup
    - Production Ready: Includes health checks, timeouts, key rotation handling,
      and error handling
    - Built-in: Ships with Hindsight, no extra installation needed
    - Legacy Support: Falls back to /auth/v1/user endpoint for HS256 projects

JWT Verification Strategy:
    By default, JWTs are verified locally using public keys from the Supabase
    JWKS endpoint (/auth/v1/.well-known/jwks.json). This is the Supabase-recommended
    approach: no network call per request, fast, and secure.

    If JWKS keys are unavailable (e.g., legacy HS256 projects), the extension
    falls back to calling /auth/v1/user per request for validation. This requires
    the service_role key to be configured.

Configuration via environment variables:
    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.supabase_tenant:SupabaseTenantExtension
    HINDSIGHT_API_TENANT_SUPABASE_URL=https://your-project.supabase.co

    # Optional - only required for legacy HS256 projects or health checks
    HINDSIGHT_API_TENANT_SUPABASE_SERVICE_KEY=your-service-role-key

    # Optional
    HINDSIGHT_API_TENANT_SCHEMA_PREFIX=user  # Default: "user" (creates user_<uuid> schemas)

Usage:
    Clients pass their Supabase JWT in the Authorization header:

    curl -H "Authorization: Bearer <supabase_jwt>" \\
        https://your-hindsight-server/v1/default/banks/my-bank/memories/recall

Author: BrighterBalance (https://brighterbalance.app)
License: MIT
"""

from __future__ import annotations

import logging
import re

import httpx
import jwt as pyjwt
from jwt import PyJWK

from hindsight_api.extensions.tenant import AuthenticationError, Tenant, TenantContext, TenantExtension
from hindsight_api.models import RequestContext

from .supabase_jwks import (
    JWKS_CACHE_TTL_SECONDS as JWKS_CACHE_TTL_SECONDS,
)
from .supabase_jwks import (
    JWKS_MIN_REFRESH_INTERVAL_SECONDS as JWKS_MIN_REFRESH_INTERVAL_SECONDS,
)
from .supabase_jwks import (
    SupabaseJwksVerifierMixin,
)

logger = logging.getLogger(__name__)

__all__ = ["JWKS_CACHE_TTL_SECONDS", "JWKS_MIN_REFRESH_INTERVAL_SECONDS", "SupabaseTenantExtension"]

# Minimum expected JWT length (JWTs are typically 100+ characters)
MIN_TOKEN_LENGTH = 20

# Timeout for Supabase API calls
REQUEST_TIMEOUT_SECONDS = 10.0

# JWKS cache TTL — Supabase Edge caches JWKS for 10 minutes, so we match that

# Supabase user IDs are UUIDs — validate before using in schema names
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)

# Schema prefix must be a valid Postgres identifier component (letters, digits, underscores)
_SCHEMA_PREFIX_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class SupabaseTenantExtension(SupabaseJwksVerifierMixin, TenantExtension):
    """
    TenantExtension that validates Supabase JWTs for multi-tenant isolation.

    Each authenticated user gets their own PostgreSQL schema, ensuring complete
    memory isolation between users. The schema name is derived from the user's
    Supabase user ID (the ``sub`` claim in the JWT).

    JWT verification uses JWKS (local, no network call per request) when
    asymmetric keys are configured in Supabase, and falls back to the
    ``/auth/v1/user`` endpoint for legacy HS256 projects.

    Example:
        User with ID "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        gets schema "user_a1b2c3d4_e5f6_7890_abcd_ef1234567890"
    """

    jwt_module = pyjwt

    def _create_jwk(self, key_data: dict[str, object]) -> PyJWK:
        # Keep construction local so extension-level instrumentation can observe it.
        return PyJWK(key_data)

    def __init__(self, config: dict[str, str]) -> None:
        """
        Initialize with configuration from environment variables.

        Config keys are derived from HINDSIGHT_API_TENANT_* env vars:
        - HINDSIGHT_API_TENANT_SUPABASE_URL -> config["supabase_url"] (required)
        - HINDSIGHT_API_TENANT_SUPABASE_SERVICE_KEY -> config["supabase_service_key"] (optional)
        - HINDSIGHT_API_TENANT_SCHEMA_PREFIX -> config["schema_prefix"] (optional)

        Args:
            config: Dictionary of configuration values from environment

        Raises:
            ValueError: If required configuration is missing
        """
        super().__init__(config)

        self.supabase_url = (config.get("supabase_url") or "").rstrip("/")
        self.supabase_service_key = config.get("supabase_service_key")
        self.schema_prefix = config.get("schema_prefix", "user")

        # Track initialized schemas to avoid redundant migrations
        self._initialized_schemas: set[str] = set()

        # Reusable HTTP client (created on startup)
        self._http_client: httpx.AsyncClient | None = None

        # JWKS state
        self._jwks_keys: dict[str, PyJWK] = {}
        self._jwks_last_fetched: float = 0
        self._use_jwks: bool = False

        if not self.supabase_url:
            raise ValueError(
                "HINDSIGHT_API_TENANT_SUPABASE_URL is required. "
                "Set it to your Supabase project URL (e.g., https://xxx.supabase.co)"
            )

        if not _SCHEMA_PREFIX_RE.match(self.schema_prefix):
            raise ValueError(
                f"Invalid schema_prefix '{self.schema_prefix}'. "
                "Must be a valid Postgres identifier (letters, digits, underscores, starting with a letter or underscore)."
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_startup(self) -> None:
        """
        Called when Hindsight starts.

        Creates a reusable HTTP client, fetches JWKS for local JWT verification,
        and optionally verifies connectivity to Supabase.
        """
        logger.info("Initializing Supabase tenant extension")
        logger.info("Supabase URL: %s", self.supabase_url)
        logger.info("Schema prefix: %s_", self.schema_prefix)

        self._http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)

        # Attempt to fetch JWKS for fast local JWT verification
        await self._try_init_jwks()

        # Optional health check using service key
        if self.supabase_service_key:
            await self._health_check()

    async def on_shutdown(self) -> None:
        """Called when Hindsight shuts down. Closes the HTTP client."""
        logger.info("Shutting down Supabase tenant extension")
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    # ------------------------------------------------------------------
    # JWKS management
    # ------------------------------------------------------------------

    async def _try_init_jwks(self) -> None:
        """Fetch JWKS and decide verification mode (local JWKS vs legacy endpoint)."""
        try:
            await self._fetch_jwks()
            if self._jwks_keys:
                self._use_jwks = True
                logger.info(
                    "JWKS loaded — using local JWT verification with %d key(s)",
                    len(self._jwks_keys),
                )
                return

            # JWKS endpoint returned no keys — project likely uses legacy HS256
            logger.warning(
                "JWKS endpoint returned no signing keys. "
                "Falling back to /auth/v1/user endpoint for JWT verification. "
                "For better performance, enable asymmetric JWT signing in your "
                "Supabase dashboard (Project Settings → Auth → JWT Algorithm)."
            )
        except Exception as e:
            logger.warning(
                "Could not fetch JWKS (%s). Falling back to /auth/v1/user endpoint for JWT verification.",
                e,
            )

        # Legacy mode requires service key
        if not self.supabase_service_key:
            raise ValueError(
                "HINDSIGHT_API_TENANT_SUPABASE_SERVICE_KEY is required when JWKS "
                "is not available. Either enable asymmetric JWT signing in your "
                "Supabase project or provide the service_role key."
            )
        self._use_jwks = False

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    async def authenticate(self, context: RequestContext) -> TenantContext:
        """
        Validate a Supabase JWT and return tenant context.

        Uses local JWKS verification when available (no network call per
        request), falling back to the ``/auth/v1/user`` endpoint for legacy
        HS256 projects.

        Args:
            context: Request context containing the API key (JWT)

        Returns:
            TenantContext with schema_name set to ``{prefix}_{user_uuid}``

        Raises:
            AuthenticationError: If token is missing, invalid, or expired
        """
        token = context.api_key

        if not token:
            raise AuthenticationError("Missing Authorization header. Expected: Bearer <supabase_jwt>")

        if len(token) < MIN_TOKEN_LENGTH:
            raise AuthenticationError("Invalid token format")

        if self._http_client is None:
            raise AuthenticationError("Extension not initialized")

        # Verify the JWT and extract user ID
        if self._use_jwks:
            user_id = await self._verify_token_jwks(token)
        else:
            user_id = await self._verify_token_legacy(token)

        # Validate user ID format before using in schema name
        if not _UUID_RE.match(user_id):
            raise AuthenticationError("Invalid user ID format in token")

        # Build isolated schema name — hyphens to underscores for Postgres compatibility
        safe_user_id = user_id.replace("-", "_")
        schema_name = f"{self.schema_prefix}_{safe_user_id}"

        # Initialize schema on first access
        if schema_name not in self._initialized_schemas:
            await self._initialize_schema(schema_name)

        return TenantContext(schema_name=schema_name)

    async def _verify_token_legacy(self, token: str) -> str:
        """
        Verify a JWT by calling the Supabase ``/auth/v1/user`` endpoint.

        This is the fallback for projects using legacy HS256 JWT signing.
        Adds a network round-trip per request.

        Raises:
            AuthenticationError: If the token is invalid or the request fails.
        """
        try:
            response = await self._http_client.get(
                f"{self.supabase_url}/auth/v1/user",
                headers={
                    "Authorization": f"Bearer {token}",
                    "apikey": self.supabase_service_key,
                },
            )

            if response.status_code == 401:
                raise AuthenticationError("Invalid or expired token")

            if response.status_code != 200:
                raise AuthenticationError(f"Authentication failed: {response.status_code}")

            user_data = response.json()
            user_id = user_data.get("id")

            if not user_id:
                raise AuthenticationError("Token valid but no user ID found")

            return user_id

        except AuthenticationError:
            raise
        except httpx.TimeoutException:
            raise AuthenticationError("Authentication timeout - please retry")
        except httpx.RequestError as e:
            raise AuthenticationError(f"Connection error: {e!s}")

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    async def _initialize_schema(self, schema_name: str) -> None:
        """Run migrations for a new tenant schema and cache the result."""
        logger.info("Initializing schema: %s", schema_name)
        try:
            await self.context.run_migration(schema_name)
            self._initialized_schemas.add(schema_name)
            logger.info("Schema ready: %s", schema_name)
        except Exception as e:
            logger.error("Schema initialization failed for %s: %s", schema_name, e)
            raise AuthenticationError(f"Failed to initialize tenant: {e!s}")

    async def list_tenants(self) -> list[Tenant]:
        """Return all tenant schemas that have been initialized."""
        return [Tenant(schema=schema) for schema in self._initialized_schemas]

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def _health_check(self) -> None:
        """Verify connectivity to Supabase using the auth health endpoint."""
        try:
            response = await self._http_client.get(
                f"{self.supabase_url}/auth/v1/health",
                headers={"apikey": self.supabase_service_key},
            )
            if response.status_code == 200:
                logger.info("Supabase connection verified")
            else:
                logger.warning("Supabase health check returned %d", response.status_code)
        except Exception as e:
            logger.warning("Could not verify Supabase connection: %s", e)
