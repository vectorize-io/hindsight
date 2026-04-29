"""
Built-in extension implementations.

These are ready-to-use implementations of the extension interfaces.
They can be used directly or serve as examples for custom implementations.

Available built-in extensions:
    - ApiKeyTenantExtension: Simple API key validation with public schema
    - SupabaseTenantExtension: Supabase JWT validation with per-user schema isolation
    - CloudflareAccessTenantExtension: OAuth JWT for MCP + API key for HTTP (use with CloudflareAccessHttpExtension)
    - CloudflareAccessHttpExtension: OAuth 2.1 server endpoints for MCP clients via Cloudflare Access

Example usage:
    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.tenant:ApiKeyTenantExtension
    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.supabase_tenant:SupabaseTenantExtension
    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.cloudflare_access:CloudflareAccessTenantExtension
    HINDSIGHT_API_HTTP_EXTENSION=hindsight_api.extensions.builtin.cloudflare_access:CloudflareAccessHttpExtension
"""

from hindsight_api.extensions.builtin.cloudflare_access import CloudflareAccessHttpExtension, CloudflareAccessTenantExtension
from hindsight_api.extensions.builtin.supabase_tenant import SupabaseTenantExtension
from hindsight_api.extensions.builtin.tenant import ApiKeyTenantExtension

__all__ = [
    "ApiKeyTenantExtension",
    "CloudflareAccessHttpExtension",
    "CloudflareAccessTenantExtension",
    "SupabaseTenantExtension",
]
