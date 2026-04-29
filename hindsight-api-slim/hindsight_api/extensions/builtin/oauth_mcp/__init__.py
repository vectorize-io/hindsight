"""Built-in OAuth 2.1 MCP extension.

Turns Hindsight itself into an OAuth 2.1 authorization server for MCP clients,
using Cloudflare Access for human SSO and stateless signed JWTs — no extra container
or database required.

Two extensions must be loaded together:

    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.oauth_mcp:OAuthMcpTenantExtension
    HINDSIGHT_API_HTTP_EXTENSION=hindsight_api.extensions.builtin.oauth_mcp:OAuthMcpHttpExtension

See each class's module docstring for the full configuration reference.
"""

from hindsight_api.extensions.builtin.oauth_mcp.http import OAuthMcpHttpExtension
from hindsight_api.extensions.builtin.oauth_mcp.tenant import OAuthMcpTenantExtension

__all__ = [
    "OAuthMcpHttpExtension",
    "OAuthMcpTenantExtension",
]
