"""
API Key Tenant Extension with Schema Isolation for Hindsight

Maps API keys to isolated PostgreSQL schemas, providing database-level
memory isolation between tenants. Each key gets its own schema containing
independent banks, memories, and entities — no application-layer access
checks required.

Why schema isolation instead of application-layer bank filtering?
    The primary threat model is **prompt injection against AI agents**.
    Agents execute tool calls (including Hindsight recall/retain) based on
    conversation content. A prompt injection delivered via chat message,
    email, or web search result can trick an agent into querying any bank
    on the same Hindsight instance.

    Application-layer access control (checking bank_id in a validator
    extension) is defense-in-depth but not a security boundary — it depends
    on every code path calling the validator, and a single missed path or
    engine bug grants cross-tenant access.

    Schema isolation is a security boundary. The API key determines the
    PostgreSQL schema at authentication time, before any bank lookup or
    memory query. Even if an agent is fully compromised by injection, its
    queries are physically scoped to its schema. Banks from other schemas
    don't exist in its view of the database.

Configuration:
    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.bank_scoped_tenant:ApiKeySchemaTenantExtension

    # Semicolon-separated entries: api_key:schema_name
    HINDSIGHT_API_TENANT_KEY_MAP=key1:tenant_alpha;key2:tenant_beta

    # Optional: prefix for schema names (default: none, uses schema name as-is)
    HINDSIGHT_API_TENANT_SCHEMA_PREFIX=hs

    # Optional: disable auth for MCP endpoints
    HINDSIGHT_API_TENANT_MCP_AUTH_DISABLED=true

Example:
    Two AI agent deployments sharing one Hindsight instance:

    HINDSIGHT_API_TENANT_KEY_MAP=abc123:team_alpha;xyz789:team_beta

    - Agent with key "abc123" → schema "team_alpha" (its own banks, memories)
    - Agent with key "xyz789" → schema "team_beta" (its own banks, memories)
    - A prompt-injected agent sending recall requests with the wrong bank name
      gets "bank not found" — the bank doesn't exist in its schema
    - Schemas are auto-created with full table migrations on first access

License: MIT
"""

from __future__ import annotations

import logging
import re

from hindsight_api.config import get_config
from hindsight_api.extensions.tenant import AuthenticationError, Tenant, TenantContext, TenantExtension
from hindsight_api.models import RequestContext

logger = logging.getLogger(__name__)

__all__ = ["ApiKeySchemaTenantExtension"]

# Schema names must be valid Postgres identifiers
_SCHEMA_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _parse_key_map(raw: str) -> dict[str, str]:
    """
    Parse a key map string into a dict of API key → schema name.

    Format: key1:schema1;key2:schema2

    Returns:
        Dict mapping API key strings to schema name strings.

    Raises:
        ValueError: If the format is invalid or a schema name is not a valid
            Postgres identifier.
    """
    result: dict[str, str] = {}
    if not raw or not raw.strip():
        return result

    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(
                f"Invalid key_map entry '{entry}'. "
                f"Expected format: 'apikey:schema_name'. "
                f"Full format: 'key1:schema1;key2:schema2'"
            )
        key, schema = entry.split(":", 1)
        key = key.strip()
        schema = schema.strip()
        if not key:
            raise ValueError("Empty API key in key_map")
        if not schema:
            raise ValueError("Empty schema name for key in key_map")
        if not _SCHEMA_RE.match(schema):
            raise ValueError(
                f"Invalid schema name '{schema}'. "
                f"Must be a valid Postgres identifier "
                f"(letters, digits, underscores, starting with a letter or underscore)."
            )
        result[key] = schema

    return result


class ApiKeySchemaTenantExtension(TenantExtension):
    """
    Tenant extension that maps API keys to isolated PostgreSQL schemas.

    Each API key resolves to a dedicated schema. All database operations
    (bank creation, memory storage, recall, reflect) are scoped to that
    schema. Schemas are auto-created with full table migrations on first
    access.

    This provides database-level isolation — tenants cannot access each
    other's data regardless of bank names, query parameters, or
    application-layer bugs.

    Configuration:
        HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.bank_scoped_tenant:ApiKeySchemaTenantExtension
        HINDSIGHT_API_TENANT_KEY_MAP=key1:schema1;key2:schema2
        HINDSIGHT_API_TENANT_SCHEMA_PREFIX=hs (optional)
        HINDSIGHT_API_TENANT_MCP_AUTH_DISABLED=true (optional)
    """

    def __init__(self, config: dict[str, str]) -> None:
        super().__init__(config)

        raw_key_map = config.get("key_map", "")
        self.schema_prefix = config.get("schema_prefix", "")
        self.key_map = _parse_key_map(raw_key_map)

        if not self.key_map:
            raise ValueError("HINDSIGHT_API_TENANT_KEY_MAP is required. Format: key1:schema1;key2:schema2")

        if self.schema_prefix and not _SCHEMA_RE.match(self.schema_prefix):
            raise ValueError(f"Invalid schema_prefix '{self.schema_prefix}'. Must be a valid Postgres identifier.")

        self.mcp_auth_disabled = config.get("mcp_auth_disabled", "").lower() in (
            "true",
            "1",
            "yes",
        )

        # Track initialized schemas to avoid redundant migrations
        self._initialized_schemas: set[str] = set()

        # Build full schema names (with optional prefix)
        self._key_to_schema: dict[str, str] = {}
        for key, schema in self.key_map.items():
            full_schema = f"{self.schema_prefix}_{schema}" if self.schema_prefix else schema
            self._key_to_schema[key] = full_schema

        # Log configuration (without revealing full keys)
        for key, schema in self._key_to_schema.items():
            masked = key[:4] + "..." + key[-4:] if len(key) > 12 else key[:4] + "..."
            logger.info("Tenant key %s -> schema '%s'", masked, schema)

    async def authenticate(self, context: RequestContext) -> TenantContext:
        """
        Authenticate API key and return tenant context with isolated schema.

        On first access for a schema, runs database migrations to create
        all required tables.

        Args:
            context: Request context containing the API key.

        Returns:
            TenantContext with schema_name for database isolation.

        Raises:
            AuthenticationError: If the API key is missing or not recognized.
        """
        if not context.api_key:
            raise AuthenticationError("Missing API key. Pass via Authorization: Bearer <key>")

        schema_name = self._key_to_schema.get(context.api_key)
        if schema_name is None:
            raise AuthenticationError("Invalid API key")

        # Initialize schema on first access (creates tables via migration)
        if schema_name not in self._initialized_schemas:
            await self._initialize_schema(schema_name)

        return TenantContext(schema_name=schema_name)

    async def list_tenants(self) -> list[Tenant]:
        """Return all initialized tenant schemas for worker discovery."""
        return [Tenant(schema=schema) for schema in self._initialized_schemas]

    async def authenticate_mcp(self, context: RequestContext) -> TenantContext:
        """
        Authenticate MCP requests.

        If mcp_auth_disabled is set, falls back to the default schema
        from HINDSIGHT_API_DATABASE_SCHEMA. Otherwise delegates to
        authenticate().

        Note: Disabling MCP auth when using schema isolation means MCP
        requests hit the default schema, not a tenant schema. This is
        appropriate for admin MCP clients but not for tenant-facing ones.
        """
        if self.mcp_auth_disabled:
            return TenantContext(schema_name=get_config().database_schema)
        return await self.authenticate(context)

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
