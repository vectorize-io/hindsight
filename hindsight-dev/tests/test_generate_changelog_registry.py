"""Regression tests for integration changelog registry coverage."""

import subprocess
import tomllib
from pathlib import Path

from hindsight_dev.generate_changelog import INTEGRATIONS

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_SCRIPT = REPO_ROOT / "scripts" / "release-integration.sh"
NEW_PYTHON_INTEGRATIONS = {"agno", "continue"}


def _release_script_integrations():
    result = subprocess.run(
        ["bash", str(RELEASE_SCRIPT), "--list-integrations"],
        check=True,
        capture_output=True,
        cwd=REPO_ROOT,
        text=True,
    )
    return result.stdout.splitlines()


def _validate_release_path(slug):
    subprocess.run(
        ["bash", str(RELEASE_SCRIPT), "--validate-only", slug],
        check=True,
        capture_output=True,
        cwd=REPO_ROOT,
        text=True,
    )


def test_release_script_integrations_match_changelog_metadata():
    release_integrations = _release_script_integrations()

    assert len(release_integrations) == len(set(release_integrations))
    assert set(release_integrations) == set(INTEGRATIONS)
    assert NEW_PYTHON_INTEGRATIONS <= set(release_integrations)


def test_python_integration_metadata_matches_package_manifests():
    pyproject_slugs = {path.parent.name for path in (REPO_ROOT / "hindsight-integrations").glob("*/pyproject.toml")}
    assert NEW_PYTHON_INTEGRATIONS <= pyproject_slugs

    for slug, meta in INTEGRATIONS.items():
        pyproject = REPO_ROOT / "hindsight-integrations" / slug / "pyproject.toml"
        if not pyproject.exists():
            continue
        manifest = tomllib.loads(pyproject.read_text())

        assert meta.package_name == manifest["project"]["name"]


def test_new_integration_display_names_match_user_facing_brands():
    assert INTEGRATIONS["agno"].display_name == "Agno"
    assert INTEGRATIONS["continue"].display_name == "Continue"


def test_new_integration_release_paths_validate_before_mutation():
    for slug in NEW_PYTHON_INTEGRATIONS:
        _validate_release_path(slug)
