"""The server must not carry extension implementations into its import graph.

Extensions are resolved by import path at load time. Re-exporting a concrete
implementation from ``hindsight_api.extensions`` would make that implementation's
dependencies part of core's import graph, which is what kept the Supabase
extension (and its JWT stack) inside every install before it moved to
``hindsight-extensions/supabase-tenant``.
"""

from pathlib import Path

import pytest

import hindsight_api
import hindsight_api.extensions as extensions

_PACKAGE_ROOT = Path(hindsight_api.__file__).parent


@pytest.mark.parametrize(
    "name",
    ["ApiKeyTenantExtension", "MemoryDefenseRegexExtension", "SupabaseTenantExtension"],
)
def test_concrete_extensions_are_not_re_exported(name: str):
    assert name not in extensions.__all__
    assert not hasattr(extensions, name)


def test_builtin_package_exports_only_the_bundled_extensions():
    from hindsight_api.extensions import builtin

    assert sorted(builtin.__all__) == ["ApiKeyTenantExtension", "MemoryDefenseRegexExtension"]


def test_no_core_module_imports_jwt():
    """`jwt` was a direct dependency solely for the Supabase extension."""
    importers = [
        path.relative_to(_PACKAGE_ROOT).as_posix()
        for path in _PACKAGE_ROOT.rglob("*.py")
        if any(line.startswith(("import jwt", "from jwt ")) for line in path.read_text(encoding="utf-8").splitlines())
    ]
    assert importers == []


def test_old_supabase_path_raises_actionable_migration_error():
    """Deployments pinned to the old path fail loudly, not with a bare AttributeError."""
    from hindsight_api.extensions.builtin import supabase_tenant

    with pytest.raises(ImportError) as exc_info:
        supabase_tenant.SupabaseTenantExtension  # noqa: B018

    message = str(exc_info.value)
    assert "pip install hindsight-ext-supabase-tenant" in message
    assert "hindsight_ext_supabase_tenant:SupabaseTenantExtension" in message


def test_old_supabase_path_fails_extension_loading_with_that_message(monkeypatch):
    """The message reaches the operator through the loader, at server startup."""
    from hindsight_api.extensions.loader import load_extension
    from hindsight_api.extensions.tenant import TenantExtension

    monkeypatch.setenv(
        "HINDSIGHT_API_TENANT_EXTENSION",
        "hindsight_api.extensions.builtin.supabase_tenant:SupabaseTenantExtension",
    )

    with pytest.raises(ImportError) as exc_info:
        load_extension("TENANT", TenantExtension)

    assert "hindsight-ext-supabase-tenant" in str(exc_info.value)


def test_unknown_attribute_on_the_shim_is_still_an_attribute_error():
    from hindsight_api.extensions.builtin import supabase_tenant

    with pytest.raises(AttributeError):
        supabase_tenant.something_else  # noqa: B018
