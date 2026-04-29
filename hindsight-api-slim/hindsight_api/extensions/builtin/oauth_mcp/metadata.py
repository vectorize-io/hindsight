"""RFC 8414 and RFC 9728 well-known metadata builders."""


def authorization_server_metadata(issuer: str, resource: str) -> dict:
    """RFC 8414 OAuth 2.0 Authorization Server Metadata."""
    return {
        "issuer": issuer,
        "authorization_endpoint": f"{issuer}/authorize",
        "token_endpoint": f"{issuer}/token",
        "registration_endpoint": f"{issuer}/register",
        "scopes_supported": ["mcp:full"],
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "token_endpoint_auth_methods_supported": ["none"],
        "code_challenge_methods_supported": ["S256"],
    }


def protected_resource_metadata(issuer: str, resource: str) -> dict:
    """RFC 9728 OAuth 2.0 Protected Resource Metadata."""
    return {
        "resource": resource,
        "authorization_servers": [issuer],
        "bearer_methods_supported": ["header"],
        "scopes_supported": ["mcp:full"],
    }
