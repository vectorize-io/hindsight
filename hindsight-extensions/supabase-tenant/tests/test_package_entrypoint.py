"""The import path this extension is documented under must actually resolve.

``HINDSIGHT_API_TENANT_EXTENSION=hindsight_ext_supabase_tenant:SupabaseTenantExtension``
is the value in every README, Dockerfile and migration note. It is resolved by
``load_extension`` at server startup, so a missing re-export would only surface
as a boot failure in someone's deployment.
"""

import os
from unittest.mock import patch

from hindsight_api.extensions.loader import load_extension
from hindsight_api.extensions.tenant import TenantExtension

from hindsight_ext_supabase_tenant import SupabaseTenantExtension


def test_class_is_exported_from_the_package_root():
    from hindsight_ext_supabase_tenant.extension import (
        SupabaseTenantExtension as from_module,
    )

    assert SupabaseTenantExtension is from_module


def test_documented_env_value_loads_the_extension():
    env = {
        "HINDSIGHT_API_TENANT_EXTENSION": "hindsight_ext_supabase_tenant:SupabaseTenantExtension",
        "HINDSIGHT_API_TENANT_SUPABASE_URL": "https://xxx.supabase.co",
        "HINDSIGHT_API_TENANT_SCHEMA_PREFIX": "tenant",
    }
    with patch.dict(os.environ, env, clear=False):
        extension = load_extension("TENANT", TenantExtension)

    assert isinstance(extension, SupabaseTenantExtension)
    assert extension.supabase_url == "https://xxx.supabase.co"
    assert extension.schema_prefix == "tenant"
