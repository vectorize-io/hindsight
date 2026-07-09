"""
Built-in extension implementations.

These are ready-to-use implementations of the extension interfaces.
They can be used directly or serve as examples for custom implementations.

Available built-in extensions:
    - ApiKeyTenantExtension: Simple API key validation with public schema
    - SupabaseTenantExtension: Supabase JWT validation with per-user schema isolation
    - SupabaseOrgTenantExtension: Supabase Auth with organization schema isolation
    - SupabaseAuthorizationExtension: Organization role/API-key bank-scope authorization

Example usage:
    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.tenant:ApiKeyTenantExtension
    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.supabase_tenant:SupabaseTenantExtension
"""

from hindsight_api.extensions.builtin.memory_defense_regex import MemoryDefenseRegexExtension
from hindsight_api.extensions.builtin.supabase_org import SupabaseAuthorizationExtension, SupabaseOrgTenantExtension
from hindsight_api.extensions.builtin.supabase_tenant import SupabaseTenantExtension
from hindsight_api.extensions.builtin.tenant import ApiKeyTenantExtension

__all__ = [
    "ApiKeyTenantExtension",
    "MemoryDefenseRegexExtension",
    "SupabaseAuthorizationExtension",
    "SupabaseOrgTenantExtension",
    "SupabaseTenantExtension",
]
