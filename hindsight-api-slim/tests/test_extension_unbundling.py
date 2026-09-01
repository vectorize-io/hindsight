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
