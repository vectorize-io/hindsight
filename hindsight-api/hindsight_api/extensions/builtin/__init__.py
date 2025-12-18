"""
Built-in extension implementations.

These are ready-to-use implementations of the extension interfaces.
They can be used directly or serve as examples for custom implementations.

Available built-in extensions:
    - ApiKeyTenantExtension: Simple API key validation with public schema

Example usage:
    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.tenant:ApiKeyTenantExtension
"""

from hindsight_api.extensions.builtin.tenant import ApiKeyTenantExtension

__all__ = [
    "ApiKeyTenantExtension",
]
