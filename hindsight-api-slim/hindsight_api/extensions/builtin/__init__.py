"""
Built-in extension implementations.

These are ready-to-use implementations of the extension interfaces.
They can be used directly or serve as examples for custom implementations.

Available built-in extensions:
    - ApiKeyTenantExtension: Simple API key validation with public schema
    - SupabaseTenantExtension: Supabase JWT validation with per-user schema isolation
    - OidcTenantExtension: Generic OIDC/OAuth (JWKS) auth with per-user schema isolation
    - GitHubTenantExtension: GitHub OAuth with team-based roles and per-user schema isolation
    - GitHubRoleOperationValidator: Gates retain/recall/reflect by GitHub team-derived role

Example usage:
    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.tenant:ApiKeyTenantExtension
    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.supabase_tenant:SupabaseTenantExtension
    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.oidc_tenant:OidcTenantExtension
    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.github_tenant:GitHubTenantExtension
"""

from hindsight_api.extensions.builtin.github_role_validator import GitHubRoleOperationValidator
from hindsight_api.extensions.builtin.github_tenant import GitHubTenantExtension
from hindsight_api.extensions.builtin.memory_defense_regex import MemoryDefenseRegexExtension
from hindsight_api.extensions.builtin.oidc_tenant import OidcTenantExtension
from hindsight_api.extensions.builtin.supabase_tenant import SupabaseTenantExtension
from hindsight_api.extensions.builtin.tenant import ApiKeyTenantExtension

__all__ = [
    "ApiKeyTenantExtension",
    "GitHubRoleOperationValidator",
    "GitHubTenantExtension",
    "MemoryDefenseRegexExtension",
    "OidcTenantExtension",
    "SupabaseTenantExtension",
]
