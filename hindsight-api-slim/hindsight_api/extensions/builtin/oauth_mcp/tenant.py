"""OAuthMcpTenantExtension — API key for HTTP routes, OAuth JWT for MCP.

HTTP routes (/v1/*) keep the existing API key check from ApiKeyTenantExtension.
MCP routes (/mcp/*) require an access token issued by OAuthMcpHttpExtension.

Configuration:
    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.oauth_mcp:OAuthMcpTenantExtension
    HINDSIGHT_API_TENANT_API_KEY=<your-http-api-key>
    HINDSIGHT_API_TENANT_OAUTH_SIGNING_SECRET=<must-match-http-extension>
    HINDSIGHT_API_TENANT_OAUTH_ISSUER=https://hindsight.yourdomain.com
"""

from __future__ import annotations

import logging

import jwt as pyjwt

from hindsight_api.config import get_config
from hindsight_api.extensions.builtin.oauth_mcp.tokens import verify_access_token
from hindsight_api.extensions.builtin.tenant import ApiKeyTenantExtension
from hindsight_api.extensions.tenant import AuthenticationError, TenantContext
from hindsight_api.models import RequestContext

logger = logging.getLogger(__name__)


class OAuthMcpTenantExtension(ApiKeyTenantExtension):
    """TenantExtension that uses API key auth for HTTP and OAuth JWT auth for MCP.

    Inherits API key validation from ApiKeyTenantExtension. MCP requests must carry
    a bearer token issued by OAuthMcpHttpExtension's /token endpoint.

    On MCP auth failure the WWW-Authenticate header points to the OAuth protected-resource
    metadata endpoint so MCP clients can auto-discover the authorization server.
    """

    def __init__(self, config: dict[str, str]) -> None:
        super().__init__(config)
        self._signing_secret = config.get("oauth_signing_secret", "")
        self._issuer = config.get("oauth_issuer", "").rstrip("/")
        if not self._signing_secret:
            raise ValueError("HINDSIGHT_API_TENANT_OAUTH_SIGNING_SECRET is required for OAuthMcpTenantExtension")
        if not self._issuer:
            raise ValueError("HINDSIGHT_API_TENANT_OAUTH_ISSUER is required for OAuthMcpTenantExtension")

    def _resource_metadata_url(self) -> str:
        return f"{self._issuer}/.well-known/oauth-protected-resource"

    async def authenticate_mcp(self, context: RequestContext) -> TenantContext:
        token = context.api_key
        if not token:
            raise AuthenticationError(
                "Authorization header with Bearer token is required for MCP",
                headers={
                    "WWW-Authenticate": (
                        f'Bearer realm="hindsight-mcp", resource_metadata="{self._resource_metadata_url()}"'
                    )
                },
            )

        try:
            payload = verify_access_token(self._signing_secret, token, self._issuer)
        except pyjwt.ExpiredSignatureError:
            raise AuthenticationError(
                "Access token has expired — reconnect to refresh",
                headers={
                    "WWW-Authenticate": (
                        f'Bearer realm="hindsight-mcp" error="invalid_token"'
                        f' error_description="Token expired",'
                        f' resource_metadata="{self._resource_metadata_url()}"'
                    )
                },
            )
        except pyjwt.InvalidTokenError as exc:
            raise AuthenticationError(
                f"Invalid access token: {exc}",
                headers={
                    "WWW-Authenticate": (
                        f'Bearer realm="hindsight-mcp" error="invalid_token",'
                        f' resource_metadata="{self._resource_metadata_url()}"'
                    )
                },
            )

        schema = get_config().database_schema
        logger.debug("MCP authenticated: sub=%s schema=%s", payload.get("sub"), schema)
        return TenantContext(schema_name=schema)
