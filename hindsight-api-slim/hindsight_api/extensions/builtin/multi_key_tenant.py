"""Multi-key tenant extension: env-configured static API keys with per-user schema isolation.

Bridges ApiKeyTenantExtension (one shared key) and SupabaseTenantExtension (external
IdP): a fully self-hosted, single-node multi-user mode where users and their API keys
are declared in environment variables. Each user maps to their own PostgreSQL schema
({prefix}_{user_id}), provisioned lazily on first access — memory isolation at the
database level, with the worker processing every tenant via list_tenants().

Features:
    - In-memory user store from env vars (no users table, no external IdP)
    - Multiple API keys may map to the same user (same isolated schema)
    - Per-user schema isolation with lazy provisioning + caching
    - Constant-time key comparison (hmac.compare_digest)
    - Works for HTTP and MCP auth
    - Usage-metering: sets RequestContext.tenant_id / api_key_id after auth
    - Fail-fast on misconfiguration

Configuration via environment variables:
    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.multi_key_tenant:StaticKeysTenantExtension
    HINDSIGHT_API_TENANT_USERS=user1:key1,user1:key2,user2:key3   # required, comma-separated user:key pairs
    HINDSIGHT_API_TENANT_SCHEMA_PREFIX=user            # optional, default: "user" (creates user_<user_id> schemas)
    HINDSIGHT_API_TENANT_MCP_AUTH_DISABLED=true        # optional, disable auth for MCP endpoints

Usage:
    Clients pass their API key in the Authorization header:

    curl -H "Authorization: Bearer <api_key>" \\
        http://localhost:8888/v1/default/banks

Authors: Rafael Kallis
License: MIT
"""

from __future__ import annotations

import hmac
import logging
import re
from dataclasses import dataclass

from hindsight_api.config import get_config
from hindsight_api.extensions.tenant import AuthenticationError, Tenant, TenantContext, TenantExtension
from hindsight_api.models import RequestContext

logger = logging.getLogger(__name__)

__all__ = ["StaticKeysTenantExtension"]

# Schema prefix must be a valid Postgres identifier component (letters, digits, underscores)
_SCHEMA_PREFIX_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# User IDs appear in schema names, so they must be safe as a Postgres identifier
# component: start with a letter/underscore, then letters/digits/underscore/dash.
# Dashes are normalized to underscores and the whole id is lowercased before
# building the schema name (Postgres folds unquoted identifiers to lowercase).
_USER_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")

# PostgreSQL truncates identifiers to NAMEDATALEN (63) bytes. An operator can
# configure two distinct user ids that differ only past that boundary; both
# would land on the same truncated schema and silently share memories. Reject
# at init rather than discovering the collision at runtime.
_MAX_SCHEMA_LENGTH = 63


@dataclass(frozen=True)
class _KeyEntry:
    """A configured API key's mapping: the owning user and its isolated schema."""

    user_id: str
    schema_name: str


class StaticKeysTenantExtension(TenantExtension):
    """
    TenantExtension mapping env-configured static API keys to per-user schemas.

    Each entry in ``HINDSIGHT_API_TENANT_USERS`` is a ``user_id:api_key`` pair.
    Multiple keys may map to the same user. Authenticated requests are mapped to
    schema ``{prefix}_{user_id}`` (user ids are lowercased, dashes normalized to
    underscores), provisioned lazily on first access.

    Example:
        HINDSIGHT_API_TENANT_USERS=rafael:key-a,sophie:key-b
        HINDSIGHT_API_TENANT_SCHEMA_PREFIX=user

        User "rafael" with key "key-a" gets schema "user_rafael";
        user "sophie" with key "key-b" gets schema "user_sophie".
    """

    def __init__(self, config: dict[str, str]) -> None:
        """
        Initialize with configuration from environment variables.

        Config keys are derived from HINDSIGHT_API_TENANT_* env vars:
        - HINDSIGHT_API_TENANT_USERS -> config["users"] (required)
        - HINDSIGHT_API_TENANT_SCHEMA_PREFIX -> config["schema_prefix"] (optional, default "user")
        - HINDSIGHT_API_TENANT_MCP_AUTH_DISABLED -> config["mcp_auth_disabled"] (optional)

        Args:
            config: Dictionary of configuration values from environment.

        Raises:
            ValueError: If required configuration is missing or invalid.
        """
        super().__init__(config)

        users_raw = config.get("users", "")
        self.schema_prefix = config.get("schema_prefix", "user")

        if not users_raw.strip():
            raise ValueError(
                "HINDSIGHT_API_TENANT_USERS is required when using StaticKeysTenantExtension. "
                'Format: "user1:key1,user2:key2"'
            )

        if not _SCHEMA_PREFIX_RE.match(self.schema_prefix):
            raise ValueError(
                f"Invalid schema_prefix '{self.schema_prefix}'. "
                "Must be a valid Postgres identifier (letters, digits, underscores, starting with a letter or underscore)."
            )

        # Parse users into an API-key -> _KeyEntry map. Multiple keys may map
        # to the same user. Also keep a per-user map so list_tenants() can
        # return every configured tenant, even before any authentication has
        # happened.
        self._key_to_user: dict[str, _KeyEntry] = {}
        self._users: dict[str, str] = {}  # user_id -> schema_name

        for entry in users_raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if ":" not in entry:
                raise ValueError(
                    f"Invalid HINDSIGHT_API_TENANT_USERS entry '{entry}'. Expected format \"user_id:api_key\"."
                )
            user_id, api_key = entry.split(":", 1)
            user_id = user_id.strip()
            api_key = api_key.strip()
            if not user_id or not api_key:
                raise ValueError(
                    f"Invalid HINDSIGHT_API_TENANT_USERS entry '{entry}'. user_id and api_key must be non-empty."
                )
            if not _USER_ID_RE.match(user_id):
                raise ValueError(
                    f"Invalid user_id '{user_id}' in HINDSIGHT_API_TENANT_USERS. "
                    "Must start with a letter or underscore, then letters, digits, underscores or dashes."
                )

            # Normalize the user id to lowercase. PostgreSQL folds unquoted
            # identifiers to lowercase at query time (the runtime's fq_table()
            # does not quote the schema, while the migration path creates it
            # quoted), so a mixed-case id like "Rafael" and its lowercase twin
            # "rafael" would collapse to the same schema at runtime — either
            # breaking every query or, worse, silently sharing one schema
            # between two users. Lowercasing makes case variants the same
            # canonical user (consistent with Postgres folding), so they merge
            # onto one schema rather than drifting apart.
            user_id = user_id.lower()

            # Each schema must be a unique isolation boundary. Two distinct
            # users whose ids differ in non-case ways yet normalize to the same
            # schema (e.g. "jane-doe" and "jane_doe"), or that collide past the
            # 63-byte identifier limit, would silently read and write each
            # other's memories — reject loudly instead of breaking the
            # extension's isolation guarantee.
            safe_user_id = user_id.replace("-", "_")
            schema_name = f"{self.schema_prefix}_{safe_user_id}"
            if len(schema_name) > _MAX_SCHEMA_LENGTH:
                raise ValueError(
                    f"Schema name '{schema_name}' for user_id '{user_id}' exceeds the PostgreSQL "
                    f"identifier limit of {_MAX_SCHEMA_LENGTH} characters."
                )
            existing = self._users.get(user_id)
            if existing is not None and existing != schema_name:
                raise ValueError(
                    f"Schema name collision for user_id '{user_id}': "
                    f"'{existing}' vs '{schema_name}'. User ids must map to distinct schemas."
                )
            claimed_by = next((uid for uid, s in self._users.items() if uid != user_id and s == schema_name), None)
            if claimed_by is not None:
                raise ValueError(
                    f"Schema name '{schema_name}' for user_id '{user_id}' is already claimed by "
                    f"user_id '{claimed_by}'. Two distinct users cannot share one schema."
                )
            if api_key in self._key_to_user:
                raise ValueError(
                    f"Duplicate API key '{api_key}' in HINDSIGHT_API_TENANT_USERS. Each key must be unique."
                )

            self._key_to_user[api_key] = _KeyEntry(user_id=user_id, schema_name=schema_name)
            self._users[user_id] = schema_name

        # Track initialized schemas to avoid redundant migrations
        self._initialized_schemas: set[str] = set()

        self.mcp_auth_disabled = config.get("mcp_auth_disabled", "").lower() in ("true", "1", "yes")

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    async def authenticate(self, context: RequestContext) -> TenantContext:
        """
        Validate the API key and return tenant context.

        Args:
            context: Request context containing the API key (the Authorization header).

        Returns:
            TenantContext with schema_name set to ``{prefix}_{user_id}``.

        Raises:
            AuthenticationError: If the key is missing or unknown.
        """
        key = context.api_key
        if not key:
            raise AuthenticationError("Missing Authorization header. Expected: Bearer <api_key>")

        # Compare against every configured key in constant time. The dict fast
        # path is deliberately absent: an exact-equality lookup would leak the
        # key's position/size via timing AND mean unknown keys never ran the
        # constant-time loop — compare_digest must always run over all entries.
        # Keys (and header values) arrive latin-1-decoded, so encode with
        # "surrogateescape" so any byte sequence round-trips losslessly
        # instead of raising TypeError (which would surface as a 500, not a 401)
        # for non-ASCII bearer tokens.
        key_bytes = key.encode("utf-8", "surrogateescape")
        match: _KeyEntry | None = None
        for configured_key, entry in self._key_to_user.items():
            if hmac.compare_digest(key_bytes, configured_key.encode("utf-8", "surrogateescape")):
                match = entry
                break

        if match is None:
            raise AuthenticationError("Invalid API key")

        user_id, schema_name = match.user_id, match.schema_name

        # Initialize schema on first access
        if schema_name not in self._initialized_schemas:
            await self._initialize_schema(schema_name)

        # Usage metering: the HTTP/MCP layers read these fields back after auth
        # to attribute operations to a tenant / API key.
        context.tenant_id = user_id
        context.api_key_id = user_id

        return TenantContext(schema_name=schema_name)

    async def authenticate_mcp(self, context: RequestContext) -> TenantContext:
        """
        Authenticate MCP requests.

        If mcp_auth_disabled is set, skip authentication and land in the base
        schema (parity with ApiKeyTenantExtension). Otherwise, delegate to
        authenticate().
        """
        if self.mcp_auth_disabled:
            return TenantContext(schema_name=get_config().database_schema)
        return await self.authenticate(context)

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

    # ------------------------------------------------------------------
    # Worker discovery
    # ------------------------------------------------------------------

    async def list_tenants(self) -> list[Tenant]:
        """Return all configured tenants for worker processing."""
        return [Tenant(schema=schema, tenant_id=user_id) for user_id, schema in self._users.items()]
