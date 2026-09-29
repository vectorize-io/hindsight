"""Validate the Claude Code / ZCode plugin manifest.

The package ships two install paths that share the same hook scripts:
  1. pip installer (`hindsight-zcode install`) — writes ~/.zcode/cli/config.json
  2. Claude Code plugin — `.claude-plugin/plugin.json` + `hooks/hooks.json`,
     installed via a marketplace and auto-registered by the host.

This test guards the plugin path: the manifest is well-formed, its version
matches the package, and every hook command points at a script that exists.
"""

import json
import re
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_JSON = PKG_ROOT / ".claude-plugin" / "plugin.json"
HOOKS_JSON = PKG_ROOT / "hooks" / "hooks.json"
SCRIPTS_DIR = PKG_ROOT / "hindsight_zcode" / "hooks" / "scripts"


def _package_version() -> str:
    text = (PKG_ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, "version not found in pyproject.toml"
    return match.group(1)


def test_plugin_manifest_is_valid():
    manifest = json.loads(PLUGIN_JSON.read_text())
    assert manifest["name"] == "hindsight-zcode"
    assert manifest["description"]
    assert manifest["license"] == "MIT"


def test_plugin_version_matches_package():
    manifest = json.loads(PLUGIN_JSON.read_text())
    assert manifest["version"] == _package_version()


def test_hooks_reference_only_supported_zcode_events():
    hooks = json.loads(HOOKS_JSON.read_text())["hooks"]
    # ZCode supports exactly these events; notably there is no SessionEnd.
    assert set(hooks) == {"SessionStart", "UserPromptSubmit", "Stop"}


def test_hook_commands_point_at_existing_scripts():
    hooks = json.loads(HOOKS_JSON.read_text())["hooks"]
    referenced = re.findall(r"scripts/(\w+\.py)", json.dumps(hooks))
    assert referenced, "no script references found in hooks.json"
    for script in referenced:
        assert (SCRIPTS_DIR / script).is_file(), f"hooks.json references missing script: {script}"
