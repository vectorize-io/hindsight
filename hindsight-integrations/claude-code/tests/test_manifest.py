"""Validate that JSON manifests are strict-valid JSON (no trailing commas, etc.)."""

import json
import tomllib
from pathlib import Path

INTEGRATION_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = INTEGRATION_ROOT.parent.parent


def test_hooks_json_is_valid():
    path = INTEGRATION_ROOT / "hooks" / "hooks.json"
    raw = path.read_text()
    parsed = json.loads(raw)
    assert "hooks" in parsed
    assert isinstance(parsed["hooks"], dict)


def test_plugin_manifest_version_is_not_behind_core_release():
    plugin_path = INTEGRATION_ROOT / ".claude-plugin" / "plugin.json"
    core_pyproject_path = REPO_ROOT / "hindsight-dev" / "pyproject.toml"

    plugin_version = json.loads(plugin_path.read_text())["version"]
    core_version = tomllib.loads(core_pyproject_path.read_text())["project"]["version"]

    assert tuple(map(int, plugin_version.split("."))) >= tuple(map(int, core_version.split(".")))
