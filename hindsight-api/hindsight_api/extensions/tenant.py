"""Tenant Extension for multi-tenancy and API key authentication."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

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

    async def get_tenant_config(self, context: RequestContext) -> dict[str, Any]:
        """
        Get tenant-specific configuration overrides.

        This method is called during hierarchical configuration resolution to get
        tenant-level config overrides. The returned dict should contain Python field
        names (lowercase snake_case) as keys, not environment variable names.

        Example:
            {"llm_model": "gpt-4", "retain_extraction_mode": "verbose"}

        The default implementation returns an empty dict (no tenant-specific config).
        Override this method in custom extensions to provide tenant-specific configuration.

        Args:
            context: The request context containing tenant information.

        Returns:
            Dict of config field names to values (only configurable fields).
            Empty dict if no tenant-specific config.
        """
        return {}

    async def get_allowed_config_fields(self, context: RequestContext, bank_id: str) -> set[str] | None:
        """
        Get set of config fields that this tenant/bank is allowed to modify.

        This method controls which configurable fields can be modified via the bank config API.
        It enables fine-grained permission control per tenant or per bank.

        Examples:
            - Return None: Allow all configurable fields (default)
            - Return {"retain_chunk_size", "retain_custom_instructions"}: Allow only these fields
            - Return set(): Allow no modifications (read-only)

        The default implementation returns None (all configurable fields allowed).
        Override this method in custom extensions to implement custom permission logic.

        Args:
            context: The request context containing tenant information.
            bank_id: The bank identifier for per-bank permissions.

        Returns:
            Set of allowed field names, or None to allow all configurable fields.
            Returned fields must be a subset of HindsightConfig.get_configurable_fields().
        """
        return None

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
