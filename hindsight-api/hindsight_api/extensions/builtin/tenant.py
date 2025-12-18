"""Built-in tenant extension implementations."""

from hindsight_api.extensions.tenant import AuthenticationError, TenantContext, TenantExtension
from hindsight_api.models import RequestContext


class ApiKeyTenantExtension(TenantExtension):
    """
    Built-in tenant extension that validates API key against an environment variable.

    This is a simple implementation that:
    1. Validates the API key matches HINDSIGHT_API_TENANT_API_KEY
    2. Returns 'public' as the schema for all authenticated requests

    Configuration:
        HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.tenant:ApiKeyTenantExtension
        HINDSIGHT_API_TENANT_API_KEY=your-secret-key

    For multi-tenant setups with separate schemas per tenant, implement a custom
    TenantExtension that looks up the schema based on the API key or token claims.
    """

    def __init__(self, config: dict[str, str]):
        super().__init__(config)
        self.expected_api_key = config.get("api_key")
        if not self.expected_api_key:
            raise ValueError("HINDSIGHT_API_TENANT_API_KEY is required when using ApiKeyTenantExtension")

    async def authenticate(self, context: RequestContext) -> TenantContext:
        """Validate API key and return public schema context."""
        if context.api_key != self.expected_api_key:
            raise AuthenticationError("Invalid API key")
        return TenantContext(schema_name="public")
