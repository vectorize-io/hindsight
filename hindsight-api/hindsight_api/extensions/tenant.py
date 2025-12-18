"""Tenant Extension for multi-tenancy and API key authentication."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from hindsight_api.extensions.base import Extension
from hindsight_api.models import RequestContext


class AuthenticationError(Exception):
    """Raised when authentication fails."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Authentication failed: {reason}")


@dataclass
class TenantContext:
    """
    Tenant context returned by authentication.

    Contains the PostgreSQL schema name for tenant isolation.
    All database queries will use fully-qualified table names
    with this schema (e.g., schema_name.memory_units).
    """

    schema_name: str


class TenantExtension(Extension, ABC):
    """
    Extension for multi-tenancy and API key authentication.

    This extension validates incoming requests and returns the tenant context
    including the PostgreSQL schema to use for database operations.

    Built-in implementation:
        hindsight_api.extensions.builtin.tenant.ApiKeyTenantExtension

    Enable via environment variable:
        HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.tenant:ApiKeyTenantExtension
        HINDSIGHT_API_TENANT_API_KEY=your-secret-key

    The returned schema_name is used for fully-qualified table names in queries,
    enabling tenant isolation at the database level.
    """

    @abstractmethod
    async def authenticate(self, context: RequestContext) -> TenantContext:
        """
        Authenticate the action context and return tenant context.

        Args:
            context: The action context containing API key and other auth data.

        Returns:
            TenantContext with the schema_name for database operations.

        Raises:
            AuthenticationError: If authentication fails.
        """
        ...
