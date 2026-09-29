"""The import path this extension is documented under must actually resolve.

``HINDSIGHT_API_TENANT_EXTENSION=hindsight_ext_static_keys_tenant:StaticKeysTenantExtension``
is the value in every README, Dockerfile and migration note. It is resolved by
``load_extension`` at server startup, so a missing re-export would only surface
as a boot failure in someone's deployment.
"""

import os
from unittest.mock import patch

from hindsight_api.extensions.loader import load_extension
from hindsight_api.extensions.tenant import TenantExtension

from hindsight_ext_static_keys_tenant import StaticKeysTenantExtension


def test_class_is_exported_from_the_package_root():
    from hindsight_ext_static_keys_tenant.extension import (
        StaticKeysTenantExtension as from_module,
    )

    assert StaticKeysTenantExtension is from_module


def test_documented_env_value_loads_the_extension():
    env = {
        "HINDSIGHT_API_TENANT_EXTENSION": "hindsight_ext_static_keys_tenant:StaticKeysTenantExtension",
        "HINDSIGHT_API_TENANT_USERS": "rafael:key-a,sophie:key-b",
        "HINDSIGHT_API_TENANT_SCHEMA_PREFIX": "tenant",
    }
    with patch.dict(os.environ, env, clear=False):
        extension = load_extension("TENANT", TenantExtension)

    assert isinstance(extension, StaticKeysTenantExtension)
    assert extension.schema_prefix == "tenant"
    assert extension._users == {"rafael": "tenant_rafael", "sophie": "tenant_sophie"}
