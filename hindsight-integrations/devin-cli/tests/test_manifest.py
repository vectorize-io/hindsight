"""Validate the plugin manifest, settings defaults, and Claude Code schema parity.

The README tells users to read the Claude Code plugin's settings reference for
everything except four documented values. That is only safe while the two
`settings.json` files really do share a key set — otherwise a key added to one
plugin silently has no documentation (or worse, wrong documentation) in the
other. `TestSettingsSchemaParityWithClaudeCode` pins that claim.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

INTEGRATION_ROOT = Path(__file__).resolve().parent.parent
CLAUDE_CODE_ROOT = INTEGRATION_ROOT.parent / "claude-code"

# The only settings whose *values* may differ between the two plugins. Each is
# called out in devin-cli/README.md; see that file before adding to this set.
INTENTIONAL_VALUE_DIVERGENCES = {
    "bankId",  # devin_cli vs claude_code — separate default bank
    "bankMission",  # names the host agent
    "retainContext",  # tags retained docs with the source agent
    "apiPort",  # 9078 vs 9077 — side-by-side local daemons
}


# Same idea for lib/config.py, which is the *real* settings schema —
# settings.json only carries the subset worth showing a user.
INTENTIONAL_DEFAULTS_DIVERGENCES = {"apiPort", "agentName", "retainContext"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_config_module(root: Path, name: str):
    """Import a plugin's lib/config.py under a private name, without shadowing.

    `lib.config` is already importable as the devin-cli one via conftest's
    sys.path entry; loading claude-code's under the same name would poison
    every other test in the session.
    """
    path = root / "scripts" / "lib" / "config.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestPluginManifest:
    def test_plugin_json_is_valid_and_complete(self):
        manifest = _load(INTEGRATION_ROOT / ".devin-plugin" / "plugin.json")
        for field in ("name", "description", "version"):
            assert manifest.get(field), f"plugin.json is missing {field}"
        assert manifest["name"] == "hindsight-memory"

    def test_version_is_semver(self):
        manifest = _load(INTEGRATION_ROOT / ".devin-plugin" / "plugin.json")
        assert re.fullmatch(r"\d+\.\d+\.\d+", manifest["version"]), (
            "release-integration.sh bumps this with a semver regex; a "
            "non-semver value breaks `release-integration.sh devin-cli patch`"
        )

    def test_manifest_version_matches_latest_changelog_entry(self):
        """A release bumps plugin.json; the changelog must not lag behind it."""
        manifest = _load(INTEGRATION_ROOT / ".devin-plugin" / "plugin.json")
        changelog = (INTEGRATION_ROOT / "CHANGELOG.md").read_text()
        headings = re.findall(r"^## (\d+\.\d+\.\d+)$", changelog, flags=re.MULTILINE)
        assert headings, "CHANGELOG.md has no versioned section headings"
        assert headings[0] == manifest["version"], (
            f"plugin.json is v{manifest['version']} but the newest CHANGELOG entry is v{headings[0]}"
        )


class TestSettingsDefaults:
    def test_settings_json_is_valid(self):
        settings = _load(INTEGRATION_ROOT / "settings.json")
        assert isinstance(settings, dict) and settings

    def test_default_port_does_not_collide_with_claude_code(self):
        settings = _load(INTEGRATION_ROOT / "settings.json")
        cc_settings = _load(CLAUDE_CODE_ROOT / "settings.json")
        assert settings["apiPort"] != cc_settings["apiPort"], (
            "a shared default port makes the two plugins kill each other's auto-managed daemon"
        )

    def test_settings_json_carries_no_version_key(self):
        """release-integration.sh picks its bump target by manifest precedence.

        `.devin-plugin/plugin.json` is checked before `settings.json`, so a
        stray `"version"` here would be dead weight that silently goes stale.
        """
        settings = _load(INTEGRATION_ROOT / "settings.json")
        assert "version" not in settings


class TestSettingsSchemaParityWithClaudeCode:
    """The README defers to the Claude Code settings reference — enforce that."""

    def test_key_sets_are_identical(self):
        settings = _load(INTEGRATION_ROOT / "settings.json")
        cc_settings = _load(CLAUDE_CODE_ROOT / "settings.json")
        missing = sorted(set(cc_settings) - set(settings))
        extra = sorted(set(settings) - set(cc_settings))
        assert not missing, f"settings absent from devin-cli but documented for claude-code: {missing}"
        assert not extra, (
            f"devin-cli settings with no claude-code counterpart (and therefore "
            f"no documentation, since the README defers): {extra}"
        )

    def test_only_documented_values_diverge(self):
        settings = _load(INTEGRATION_ROOT / "settings.json")
        cc_settings = _load(CLAUDE_CODE_ROOT / "settings.json")
        diverged = {key for key in set(settings) & set(cc_settings) if settings[key] != cc_settings[key]}
        undocumented = sorted(diverged - INTENTIONAL_VALUE_DIVERGENCES)
        assert not undocumented, (
            f"defaults diverge from the Claude Code plugin without being documented in README.md: {undocumented}"
        )


class TestConfigModuleParityWithClaudeCode:
    """`lib/config.py` is the real schema — settings.json is only its visible subset.

    The README's "same config schema, same HINDSIGHT_* env vars" claim rests on
    these two structures matching. A key added to the Claude Code plugin and
    not here means a documented setting that silently does nothing on Devin
    CLI; an env var that maps to a different config key means the documented
    variable writes to the wrong place.
    """

    def test_defaults_key_sets_are_identical(self):
        cc = _load_config_module(CLAUDE_CODE_ROOT, "_cc_config_parity")
        dv = _load_config_module(INTEGRATION_ROOT, "_dv_config_parity")
        missing = sorted(set(cc.DEFAULTS) - set(dv.DEFAULTS))
        extra = sorted(set(dv.DEFAULTS) - set(cc.DEFAULTS))
        assert not missing, f"config keys the Claude Code plugin has and this one lacks: {missing}"
        assert not extra, f"config keys with no Claude Code counterpart: {extra}"

    def test_only_documented_defaults_diverge(self):
        cc = _load_config_module(CLAUDE_CODE_ROOT, "_cc_config_parity")
        dv = _load_config_module(INTEGRATION_ROOT, "_dv_config_parity")
        diverged = {key for key in set(cc.DEFAULTS) & set(dv.DEFAULTS) if cc.DEFAULTS[key] != dv.DEFAULTS[key]}
        undocumented = sorted(diverged - INTENTIONAL_DEFAULTS_DIVERGENCES)
        assert not undocumented, f"lib/config.py defaults diverge without documentation: {undocumented}"

    def test_env_overrides_map_to_the_same_config_keys(self):
        cc = _load_config_module(CLAUDE_CODE_ROOT, "_cc_config_parity")
        dv = _load_config_module(INTEGRATION_ROOT, "_dv_config_parity")
        assert set(cc.ENV_OVERRIDES) == set(dv.ENV_OVERRIDES), (
            "the set of supported HINDSIGHT_* env vars must match the Claude "
            "Code plugin's, since the README defers to its documentation"
        )
        mismatched = {
            var: (cc.ENV_OVERRIDES[var][0], dv.ENV_OVERRIDES[var][0])
            for var in cc.ENV_OVERRIDES
            if cc.ENV_OVERRIDES[var][0] != dv.ENV_OVERRIDES[var][0]
        }
        assert not mismatched, f"env vars targeting different config keys: {mismatched}"


class TestMcpRequirementPin:
    """`mcp` must carry an upper bound.

    mcp 2.0 removed `mcp.server.fastmcp`, which scripts/mcp_server.py imports.
    An unbounded `mcp>=1.0.0` therefore resolves, on a fresh install, to a
    release the server cannot start under — and run_mcp.sh only re-pips when
    `import mcp` fails, which succeeds under 2.x, so the venv is never
    repaired. This is upstream issue #3026 against the Claude Code plugin;
    this plugin inherited the same unbounded spec.
    """

    def test_mcp_is_capped_below_the_release_that_moved_fastmcp(self):
        requirements = (INTEGRATION_ROOT / "requirements.txt").read_text(encoding="utf-8")
        specs = [line.strip() for line in requirements.splitlines() if line.strip() and not line.startswith("#")]
        mcp_specs = [s for s in specs if re.match(r"^mcp\b", s)]
        assert mcp_specs, f"no mcp requirement found in {specs}"
        assert any("<2" in s for s in mcp_specs), (
            f"mcp must be pinned below 2.0 while mcp_server.py imports mcp.server.fastmcp; got {mcp_specs}"
        )

    def test_the_pinned_import_path_is_the_one_mcp_server_uses(self):
        """Guards the *reason* for the pin, so removing the import frees it."""
        source = (INTEGRATION_ROOT / "scripts" / "mcp_server.py").read_text(encoding="utf-8")
        assert "mcp.server.fastmcp" in source, (
            "mcp_server.py no longer imports mcp.server.fastmcp — re-evaluate the <2 pin in requirements.txt"
        )
