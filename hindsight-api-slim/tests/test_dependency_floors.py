"""Guard against re-adding version floors for transitive-only packages.

``cryptography`` and ``pillow`` are not imported anywhere in ``hindsight_api``.
They only arrive transitively (cryptography via authlib / pyjwt[crypto] /
oracledb / pdfminer-six, pillow via markitdown's pdfplumber + python-pptx), so a
floor on them in *this package's* metadata cannot protect anything a lockfile
bump wouldn't: nothing in the tree caps either package, so a plain install
already resolves to the newest release.

What such a floor *does* do is make Hindsight impossible to co-install with any
application that pins those packages exactly. See #3251: hermes-agent 0.19.0
pins ``cryptography==46.0.7`` and ``Pillow==12.2.0``, and against
``cryptography>=48.0.1`` / ``pillow>=12.3.0`` the two are simply unsatisfiable —
``pip check`` fails permanently and every Hindsight upgrade forces a manual
re-pin.

The hardened versions belong in ``uv.lock``, which is what the published images
install via ``uv sync --locked``. This test exists because the floors were
originally added by Dependabot-alert triage, which is exactly the path that
would silently re-add them.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# Packages that must stay unconstrained here — pin them in uv.lock instead.
TRANSITIVE_ONLY = ("cryptography", "pillow")


def _requirement_names(spec: str) -> str:
    """Extract the distribution name from a PEP 508 requirement string."""
    name = spec.split(";")[0].strip()
    for sep in ("[", "(", "=", "<", ">", "!", "~", "@", " "):
        name = name.split(sep)[0]
    return name.strip().lower().replace("_", "-")


@pytest.mark.parametrize("package", TRANSITIVE_ONLY)
def test_no_floor_on_transitive_only_package(package: str) -> None:
    project = tomllib.loads(PYPROJECT.read_text())["project"]

    declared: list[str] = list(project.get("dependencies", []))
    for extra_specs in project.get("optional-dependencies", {}).values():
        declared.extend(extra_specs)

    offenders = [spec for spec in declared if _requirement_names(spec) == package]
    assert not offenders, (
        f"hindsight-api-slim declares {offenders!r}, but {package!r} is not imported by "
        f"hindsight_api — it is a transitive dependency only. A floor here cannot harden "
        f"anything a uv.lock bump wouldn't, and it makes Hindsight uninstallable alongside "
        f"applications that pin {package!r} exactly (see #3251). Bump uv.lock instead."
    )
