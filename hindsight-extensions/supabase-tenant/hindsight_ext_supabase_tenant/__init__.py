"""Supabase Auth tenant extension for the Hindsight API server.

Configure the server to load it with::

    HINDSIGHT_API_TENANT_EXTENSION=hindsight_ext_supabase_tenant:SupabaseTenantExtension
"""

from hindsight_ext_supabase_tenant.extension import SupabaseTenantExtension

__all__ = ["SupabaseTenantExtension"]
