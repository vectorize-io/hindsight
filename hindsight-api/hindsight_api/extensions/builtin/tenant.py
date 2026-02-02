"""Built-in tenant extension implementations."""

from hindsight_api.config import get_config
from hindsight_api.extensions.tenant import AuthenticationError, Tenant, TenantContext, TenantExtension
from hindsight_api.models import RequestContext


class DefaultTenantExtension(TenantExtension):
    """
    Default single-tenant extension with no authentication.

    This is the default extension used when no tenant extension is configured.
    It provides single-tenant behavior using the configured schema from
    HINDSIGHT_API_DATABASE_SCHEMA (defaults to 'public').

    Features:
    - No authentication required (passes all requests)
    - Uses configured schema from environment
    - Perfect for single-tenant deployments without auth

    Configuration:
        HINDSIGHT_API_DATABASE_SCHEMA=your-schema (optional, defaults to 'public')

    This is automatically enabled by default. To use custom authentication,
    configure a different tenant extension:
        HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.tenant:ApiKeyTenantExtension
    """

    def __init__(self, config: dict[str, str]):
        super().__init__(config)
        # Cache the schema at initialization for consistency
        # Support explicit schema override via config, otherwise use environment
        self._schema = config.get("schema", get_config().database_schema)

    async def authenticate(self, context: RequestContext) -> TenantContext:
        """Return configured schema without any authentication."""
        return TenantContext(schema_name=self._schema)

    async def list_tenants(self) -> list[Tenant]:
        """Return configured schema for single-tenant setup."""
        return [Tenant(schema=self._schema)]


class ApiKeyTenantExtension(TenantExtension):
    """
    Built-in tenant extension that validates API key against an environment variable.

    This is a simple implementation that:
    1. Validates the API key matches HINDSIGHT_API_TENANT_API_KEY
    2. Returns the configured schema (HINDSIGHT_API_DATABASE_SCHEMA, default 'public')
       for all authenticated requests

    Configuration:
        HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.tenant:ApiKeyTenantExtension
        HINDSIGHT_API_TENANT_API_KEY=your-secret-key
        HINDSIGHT_API_DATABASE_SCHEMA=your-schema (optional, defaults to 'public')

    For multi-tenant setups with separate schemas per tenant, implement a custom
    TenantExtension that looks up the schema based on the API key or token claims.
    """

    def __init__(self, config: dict[str, str]):
        super().__init__(config)
        self.expected_api_key = config.get("api_key")
        if not self.expected_api_key:
            raise ValueError("HINDSIGHT_API_TENANT_API_KEY is required when using ApiKeyTenantExtension")

    async def authenticate(self, context: RequestContext) -> TenantContext:
        """Validate API key and return configured schema context."""
        if context.api_key != self.expected_api_key:
            raise AuthenticationError("Invalid API key")
        return TenantContext(schema_name=get_config().database_schema)

    async def list_tenants(self) -> list[Tenant]:
        """Return configured schema for single-tenant setup."""
        return [Tenant(schema=get_config().database_schema)]
