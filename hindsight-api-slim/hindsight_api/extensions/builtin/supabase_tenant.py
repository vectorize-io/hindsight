"""Removed: the Supabase tenant extension is no longer bundled with the server.

It moved to its own distribution, ``hindsight-ext-supabase-tenant``, so that a
third-party identity provider's client code (and its JWT/JWKS dependencies) is
not carried by every Hindsight install. See ``hindsight-extensions/`` in the
repository, and the extensions registry documentation.

This module is kept for one minor release so that deployments configured with
the old path fail at startup with actionable instructions rather than an opaque
``ImportError``.
"""

_MIGRATION_MESSAGE = (
    "SupabaseTenantExtension is no longer built into Hindsight. It now ships as the "
    "separate package 'hindsight-ext-supabase-tenant'. To restore it:\n"
    "  1. pip install hindsight-ext-supabase-tenant\n"
    "  2. set HINDSIGHT_API_TENANT_EXTENSION="
    "hindsight_ext_supabase_tenant:SupabaseTenantExtension\n"
    "All HINDSIGHT_API_TENANT_* settings keep their names and meaning."
)


def __getattr__(name: str) -> object:
    """Turn any access to the old symbol into the migration instructions."""
    if name == "SupabaseTenantExtension":
        raise ImportError(_MIGRATION_MESSAGE)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
