"""
Built-in extension implementations.

These are ready-to-use implementations of the extension interfaces.
They can be used directly or serve as examples for custom implementations.

Available built-in extensions:
    - ApiKeyTenantExtension: Simple API key validation with public schema
    - SupabaseTenantExtension: Supabase JWT validation with per-user schema isolation
    - OAuthMcpTenantExtension: OAuth JWT for MCP + API key for HTTP (use with OAuthMcpHttpExtension)
    - OAuthMcpHttpExtension: OAuth 2.1 server endpoints for MCP clients via Cloudflare Access

Example usage:
    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.tenant:ApiKeyTenantExtension
    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.supabase_tenant:SupabaseTenantExtension
    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.oauth_mcp:OAuthMcpTenantExtension
    HINDSIGHT_API_HTTP_EXTENSION=hindsight_api.extensions.builtin.oauth_mcp:OAuthMcpHttpExtension
"""

from hindsight_api.extensions.builtin.oauth_mcp import OAuthMcpHttpExtension, OAuthMcpTenantExtension
from hindsight_api.extensions.builtin.supabase_tenant import SupabaseTenantExtension
from hindsight_api.extensions.builtin.tenant import ApiKeyTenantExtension

__all__ = [
    "ApiKeyTenantExtension",
    "OAuthMcpHttpExtension",
    "OAuthMcpTenantExtension",
    "SupabaseTenantExtension",
]
