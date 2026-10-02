"""Cursor Marketplace submission checklist, applied to this plugin.

Grok Bot installs plugins from the Cursor Marketplace, pinned to the commit the
marketplace approved, so a manifest mistake ships until the next review cycle.
These tests encode the checklist from https://cursor.com/docs/reference/plugins.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

PLUGIN_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PLUGIN_DIR.parent.parent
MCP_JSON = PLUGIN_DIR / "mcp.json"
KEBAB = re.compile(r"^[a-z0-9][a-z0-9.-]*$")


class VariablesSchema(BaseModel):
    # Property names are whatever the plugin declares, so this stays a dynamic mapping.
    properties: dict[str, Any] = {}


class PluginManifest(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str
    description: str
    version: str
    homepage: str
    logo: str
    skills: str
    mcp_servers: str = Field(alias="mcpServers")
    variables: VariablesSchema = VariablesSchema()


class McpServer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    headers: dict[str, str] | None = None


class McpConfig(BaseModel):
    # Keyed by server id, which the plugin chooses.
    mcp_servers: dict[str, McpServer] = Field(alias="mcpServers")


class MarketplaceEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    source: str
    version: str


class Marketplace(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    plugins: list[MarketplaceEntry]


def _plugin() -> PluginManifest:
    return PluginManifest.model_validate_json((PLUGIN_DIR / ".cursor-plugin" / "plugin.json").read_text())


def _mcp() -> McpConfig:
    return McpConfig.model_validate_json(MCP_JSON.read_text())


def _marketplace() -> Marketplace:
    return Marketplace.model_validate_json((REPO_ROOT / ".cursor-plugin" / "marketplace.json").read_text())


def _assert_relative_inside(base: Path, rel: str) -> Path:
    assert not rel.startswith("/"), f"absolute path: {rel}"
    assert ".." not in Path(rel).parts, f"path escapes the plugin: {rel}"
    target = (base / rel).resolve()
    assert target.exists(), f"missing: {rel}"
    return target


def test_name_is_kebab_case_and_described() -> None:
    plugin = _plugin()
    assert KEBAB.match(plugin.name)
    assert plugin.description.strip()


def test_component_paths_are_relative_and_exist() -> None:
    plugin = _plugin()
    for rel in (plugin.skills, plugin.mcp_servers, plugin.logo):
        _assert_relative_inside(PLUGIN_DIR, rel)


def test_logo_is_a_square_512_svg() -> None:
    svg = (PLUGIN_DIR / _plugin().logo).read_text()
    assert 'viewBox="0 0 512 512"' in svg


def test_mcp_server_is_remote_and_uses_oauth() -> None:
    # Grok Bot runs the OAuth flow itself against a remote MCP URL (the same flow Mem0
    # uses), and Hindsight's /mcp supports OAuth discovery and dynamic client
    # registration. So the plugin carries no credential, header or ${VAR} to configure.
    servers = _mcp().mcp_servers
    assert list(servers) == ["hindsight"]
    assert servers["hindsight"].url == "https://api.hindsight.vectorize.io/mcp"
    assert servers["hindsight"].headers is None
    assert "${" not in MCP_JSON.read_text()


def test_every_mcp_variable_is_declared() -> None:
    used = set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", MCP_JSON.read_text()))
    declared = set(_plugin().variables.properties)
    assert used <= declared, f"undeclared variables: {used - declared}"


def test_marketplace_lists_this_plugin() -> None:
    plugin = _plugin()
    marketplace = _marketplace()
    assert KEBAB.match(marketplace.name)
    names = [entry.name for entry in marketplace.plugins]
    assert len(names) == len(set(names)), "plugin names must be unique"
    entry = next(entry for entry in marketplace.plugins if entry.name == plugin.name)
    assert _assert_relative_inside(REPO_ROOT, entry.source) == PLUGIN_DIR
    assert entry.version == plugin.version


def test_readme_documents_configuration() -> None:
    readme = (PLUGIN_DIR / "README.md").read_text()
    assert "grok-bot::shared" in readme
    assert "OAuth" in readme


def test_homepage_is_the_live_docs_page() -> None:
    # The OSS docs deploy to hindsight.vectorize.io (docusaurus.config `url`).
    # docs.hindsight.vectorize.io is the separate Hindsight Cloud docs site, where this route
    # is a 404. The marketplace listing's Website link is this field, pinned to the reviewed
    # commit, so a wrong domain survives until the next review.
    assert _plugin().homepage == f"https://hindsight.vectorize.io/sdks/integrations/{PLUGIN_DIR.name}"
    assert (REPO_ROOT / "hindsight-docs" / "docs-integrations" / f"{PLUGIN_DIR.name}.md").exists()
