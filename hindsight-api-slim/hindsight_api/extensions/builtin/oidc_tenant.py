"""
Generic OIDC / OAuth Tenant Extension for Hindsight.

Validates OIDC ID tokens / JWT access tokens from any standard OpenID Connect
provider (Auth0, Okta, Keycloak, Microsoft Entra ID, Google, ...) and maps each
authenticated user to an isolated PostgreSQL schema.

Each authenticated user gets their own schema (``{prefix}_{subject}``), ensuring
complete data isolation between users sharing one Hindsight instance.

Verification strategy:
    Tokens are verified locally using public keys discovered from the provider's
    OIDC discovery document (``{issuer}/.well-known/openid-configuration`` ->
    ``jwks_uri``). This requires no network call per request once JWKS is cached
    and matches the approach used by :class:`SupabaseTenantExtension`.

Extensibility:
    This class is intentionally written so provider-specific variants can be
    built as thin subclasses. The verification + identity extraction step is
    isolated in :meth:`_resolve_identity`, schema naming in
    :meth:`_schema_for_subject`, and role loading in :meth:`_load_roles`. The
    GitHub variant (:class:`hindsight_api.extensions.builtin.github_tenant.GitHubTenantExtension`)
    overrides these without reusing JWKS, because GitHub OAuth user tokens are
    opaque (not JWTs).

Configuration via environment variables:
    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.oidc_tenant:OidcTenantExtension
    HINDSIGHT_API_TENANT_OIDC_ISSUER=https://your-tenant.auth0.com/

    # Optional
    HINDSIGHT_API_TENANT_OIDC_AUDIENCE=your-api-audience      # expected `aud`
    HINDSIGHT_API_TENANT_OIDC_JWKS_URI=https://.../jwks.json  # skip discovery
    HINDSIGHT_API_TENANT_SUBJECT_CLAIM=sub                    # claim -> schema
    HINDSIGHT_API_TENANT_SCHEMA_PREFIX=user                   # schema prefix
    HINDSIGHT_API_TENANT_ALGORITHMS=RS256,ES256               # allowed algs

Usage:
    curl -H "Authorization: Bearer <id_or_access_token>" \\
        https://your-hindsight-server/v1/default/banks/my-bank/memories/recall

License: MIT
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field

import httpx
import jwt as pyjwt
from jwt import PyJWK

from hindsight_api.extensions.tenant import AuthenticationError, Tenant, TenantContext, TenantExtension
from hindsight_api.models import RequestContext

logger = logging.getLogger(__name__)

__all__ = ["Identity", "OidcTenantExtension"]

# Minimum expected token length (JWTs are typically 100+ characters).
MIN_TOKEN_LENGTH = 20

# Timeout for provider HTTP calls (discovery, JWKS, userinfo).
REQUEST_TIMEOUT_SECONDS = 10.0

# JWKS cache TTL.
JWKS_CACHE_TTL_SECONDS = 600

# Minimum interval between JWKS refreshes to avoid hammering the endpoint.
JWKS_MIN_REFRESH_INTERVAL_SECONDS = 30

# Default asymmetric algorithms supported for JWT signing.
DEFAULT_ALGORITHMS = ["RS256", "ES256"]

# Schema prefix must be a valid Postgres identifier component.
_SCHEMA_PREFIX_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Postgres identifier maximum length (bytes). Schema names must fit.
_MAX_IDENTIFIER_LENGTH = 63

# Characters allowed verbatim in a sanitized subject component.
_SAFE_SUBJECT_RE = re.compile(r"[^a-zA-Z0-9_]")


@dataclass
class Identity:
    """Resolved identity for an authenticated request.

    Attributes:
        subject: Stable unique identifier for the user (used to derive schema).
        claims: Raw claims/attributes resolved from the token or provider.
        roles: Optional resolved role names (populated by role-aware subclasses).
    """

    subject: str
    claims: dict = field(default_factory=dict)
    roles: list[str] = field(default_factory=list)


class OidcTenantExtension(TenantExtension):
    """TenantExtension that validates OIDC/JWT tokens for multi-tenant isolation.

    Each authenticated user gets their own PostgreSQL schema derived from a
    configurable subject claim, ensuring complete memory isolation between users
    on a shared Hindsight instance.
    """

    def __init__(self, config: dict[str, str]) -> None:
        super().__init__(config)

        self.issuer = (config.get("oidc_issuer") or "").rstrip("/")
        self.audience = config.get("oidc_audience") or None
        self.jwks_uri = config.get("oidc_jwks_uri") or None
        self.userinfo_endpoint = config.get("oidc_userinfo_endpoint") or None
        self.subject_claim = config.get("subject_claim", "sub")
        self.schema_prefix = config.get("schema_prefix", "user")

        algorithms_raw = config.get("algorithms", "")
        self.algorithms = (
            [a.strip() for a in algorithms_raw.split(",") if a.strip()] if algorithms_raw else list(DEFAULT_ALGORITHMS)
        )

        # Track initialized schemas to avoid redundant migrations.
        self._initialized_schemas: set[str] = set()

        # Reusable HTTP client (created on startup).
        self._http_client: httpx.AsyncClient | None = None

        # JWKS state.
        self._jwks_keys: dict[str, PyJWK] = {}
        self._jwks_last_fetched: float = 0.0

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration. Subclasses may relax issuer/JWKS requirements."""
        if not self.issuer and not self.jwks_uri:
            raise ValueError(
                "HINDSIGHT_API_TENANT_OIDC_ISSUER is required. "
                "Set it to your OIDC provider issuer URL (e.g. https://tenant.auth0.com/)."
            )
        if not _SCHEMA_PREFIX_RE.match(self.schema_prefix):
            raise ValueError(
                f"Invalid schema_prefix '{self.schema_prefix}'. "
                "Must be a valid Postgres identifier (letters, digits, underscores, "
                "starting with a letter or underscore)."
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_startup(self) -> None:
        logger.info("Initializing OIDC tenant extension")
        if self.issuer:
            logger.info("OIDC issuer: %s", self.issuer)
        logger.info("Schema prefix: %s_", self.schema_prefix)

        self._http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)

        # Discover JWKS URI if not explicitly provided.
        if not self.jwks_uri and self.issuer:
            await self._discover()

        if self.jwks_uri:
            try:
                await self._fetch_jwks()
                logger.info("JWKS loaded — using local JWT verification with %d key(s)", len(self._jwks_keys))
            except Exception as e:  # noqa: BLE001 - best-effort warm-up
                logger.warning("Could not pre-fetch JWKS (%s); will retry on first request.", e)

    async def on_shutdown(self) -> None:
        logger.info("Shutting down OIDC tenant extension")
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    # ------------------------------------------------------------------
    # OIDC discovery + JWKS management
    # ------------------------------------------------------------------

    async def _discover(self) -> None:
        """Fetch the OIDC discovery document and cache endpoints."""
        if self._http_client is None:
            raise RuntimeError("HTTP client not initialized")
        url = f"{self.issuer}/.well-known/openid-configuration"
        try:
            response = await self._http_client.get(url)
            response.raise_for_status()
            doc = response.json()
        except Exception as e:  # noqa: BLE001
            logger.warning("OIDC discovery failed at %s (%s)", url, e)
            return
        self.jwks_uri = self.jwks_uri or doc.get("jwks_uri")
        self.userinfo_endpoint = self.userinfo_endpoint or doc.get("userinfo_endpoint")
        # Prefer the issuer the provider reports, if present.
        self.issuer = (doc.get("issuer") or self.issuer).rstrip("/")

    async def _fetch_jwks(self) -> None:
        """Fetch public signing keys from the provider's JWKS endpoint."""
        if self._http_client is None:
            raise RuntimeError("HTTP client not initialized")
        if not self.jwks_uri:
            raise AuthenticationError("No JWKS URI configured or discovered")

        response = await self._http_client.get(self.jwks_uri)
        response.raise_for_status()

        jwks_data = response.json()
        keys: dict[str, PyJWK] = {}
        for key_data in jwks_data.get("keys", []):
            kid = key_data.get("kid")
            if kid:
                keys[kid] = PyJWK(key_data)

        self._jwks_keys = keys
        self._jwks_last_fetched = time.monotonic()

    async def _get_signing_key(self, token: str) -> PyJWK:
        """Resolve the signing key for a token, refreshing JWKS on key rotation."""
        try:
            header = pyjwt.get_unverified_header(token)
        except pyjwt.DecodeError:
            raise AuthenticationError("Invalid token header")
        kid = header.get("kid")
        if not kid:
            raise AuthenticationError("Token missing key ID (kid) header")

        now = time.monotonic()
        if not self._jwks_keys or now - self._jwks_last_fetched > JWKS_CACHE_TTL_SECONDS:
            await self._fetch_jwks()

        if kid in self._jwks_keys:
            return self._jwks_keys[kid]

        if time.monotonic() - self._jwks_last_fetched > JWKS_MIN_REFRESH_INTERVAL_SECONDS:
            logger.info("Signing key %s not cached, refreshing JWKS for possible rotation", kid)
            await self._fetch_jwks()
            if kid in self._jwks_keys:
                return self._jwks_keys[kid]

        raise AuthenticationError("Unable to find signing key for token")

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    async def authenticate(self, context: RequestContext) -> TenantContext:
        """Validate a token and return tenant context with an isolated schema."""
        token = context.api_key
        if not token:
            raise AuthenticationError(
                "Missing Authorization header. Expected: Bearer <token>",
                headers={"WWW-Authenticate": 'Bearer realm="hindsight"'},
            )
        if len(token) < MIN_TOKEN_LENGTH:
            raise AuthenticationError("Invalid token format")
        if self._http_client is None:
            raise AuthenticationError("Extension not initialized")

        identity = await self._resolve_identity(token)
        if not identity.subject:
            raise AuthenticationError("Token valid but missing subject")

        # Load roles (no-op in the base class) and let subclasses stash state.
        await self._load_roles(token, identity, context)

        schema_name = self._schema_for_subject(identity.subject)

        if schema_name not in self._initialized_schemas:
            await self._initialize_schema(schema_name)

        return TenantContext(schema_name=schema_name)

    async def _resolve_identity(self, token: str) -> Identity:
        """Verify the token and resolve the user identity.

        Default implementation performs local JWKS-based JWT verification.
        Provider-specific subclasses (e.g. GitHub) override this.
        """
        try:
            signing_key = await self._get_signing_key(token)
            options = {}
            decode_kwargs: dict = {"algorithms": self.algorithms}
            if self.audience:
                decode_kwargs["audience"] = self.audience
            else:
                options["verify_aud"] = False
            if self.issuer:
                decode_kwargs["issuer"] = self.issuer
            decode_kwargs["options"] = options
            payload = pyjwt.decode(token, signing_key.key, **decode_kwargs)
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
        except Exception as e:  # noqa: BLE001
            raise AuthenticationError(f"Token verification failed: {e!s}")

        subject = payload.get(self.subject_claim)
        if not subject:
            raise AuthenticationError(f"Token missing subject claim '{self.subject_claim}'")
        return Identity(subject=str(subject), claims=payload)

    async def _load_roles(self, token: str, identity: Identity, context: RequestContext) -> None:
        """Hook for role-aware subclasses. No-op in the generic extension."""
        return None

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def _schema_for_subject(self, subject: str) -> str:
        """Derive a safe, isolated Postgres schema name for a subject."""
        safe = _SAFE_SUBJECT_RE.sub("_", subject)
        schema = f"{self.schema_prefix}_{safe}"
        if len(schema) > _MAX_IDENTIFIER_LENGTH:
            # Hash long/opaque subjects to stay within the identifier limit while
            # remaining stable and collision-resistant.
            digest = hashlib.sha256(subject.encode("utf-8")).hexdigest()[:32]
            schema = f"{self.schema_prefix}_{digest}"
        return schema

    async def _initialize_schema(self, schema_name: str) -> None:
        """Run migrations for a new tenant schema and cache the result."""
        logger.info("Initializing schema: %s", schema_name)
        try:
            await self.context.run_migration(schema_name)
            self._initialized_schemas.add(schema_name)
            logger.info("Schema ready: %s", schema_name)
        except Exception as e:  # noqa: BLE001
            logger.error("Schema initialization failed for %s: %s", schema_name, e)
            raise AuthenticationError(f"Failed to initialize tenant: {e!s}")

    async def list_tenants(self) -> list[Tenant]:
        """Return all tenant schemas that have been initialized."""
        return [Tenant(schema=schema) for schema in self._initialized_schemas]
