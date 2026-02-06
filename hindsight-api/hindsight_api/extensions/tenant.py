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


@dataclass
class Tenant:
    """
    Represents a tenant for worker discovery.

    Used by list_tenants() to return tenant information including
    the PostgreSQL schema name for database operations.
    """

    schema: str


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

    @abstractmethod
    async def list_tenants(self) -> list[Tenant]:
        """
        List all tenants that should be processed by workers.

        This method is used by the worker to discover all tenants that need
        task polling. Workers will poll for pending tasks in each tenant's schema.

        Returns:
            List of Tenant objects containing schema information.
            For single-tenant setups, return [Tenant(schema="public")].
        """
        ...

    async def authenticate_mcp(self, context: RequestContext) -> TenantContext:
        """
        Authenticate MCP requests.

        By default, this calls authenticate(). Override this method to provide
        different authentication behavior for MCP endpoints (e.g., to disable
        auth for backwards compatibility with existing MCP servers).

        Args:
            context: The action context containing API key and other auth data.

        Returns:
            TenantContext with the schema_name for database operations.

        Raises:
            AuthenticationError: If authentication fails.
        """
        return await self.authenticate(context)
