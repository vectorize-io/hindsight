#!/usr/bin/env python3
"""Validate this Agent Plugin against the Agent Plugins 1.0.0 required-field contract.

Config-only plugin (no runtime code to unit-test), so the meaningful check is that the
manifests parse and satisfy the spec's structural requirements. Runs as a script
(`python3 validate.py`) and is imported by CI as a pytest.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent
SPEC = "1.0.0"
PLUGIN_SCHEMA = f"https://agent-plugins.org/schemas/{SPEC}/plugin.schema.json"
MCP_SCHEMA = f"https://agent-plugins.org/schemas/{SPEC}/mcp.schema.json"

# https://agent-plugins.org/schemas/1.0.0/plugin.schema.json — name pattern
NAME_RE = re.compile(r"^(?!.*(?:--|\.\.))[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")
AUTHOR_KEYS = {"name", "email", "url"}
SERVER_TYPES = {"stdio", "streamable-http", "sse"}


def _load(name: str) -> dict:
    return json.loads((PLUGIN_DIR / name).read_text())


def validate_plugin_manifest() -> None:
    m = _load("plugin.json")
    assert m.get("$schema") == PLUGIN_SCHEMA, f"plugin.json $schema must be {PLUGIN_SCHEMA}"
    name = m.get("name")
    assert isinstance(name, str) and 1 <= len(name) <= 64, "name must be a 1-64 char string"
    assert NAME_RE.match(name), f"name {name!r} violates the Agent Plugins name pattern"
    author = m.get("author")
    if author is not None:
        assert isinstance(author, dict), "author must be an object"
        assert set(author).issubset(AUTHOR_KEYS), f"author allows only {AUTHOR_KEYS}"
    extensions = m.get("extensions")
    if extensions is not None:
        assert isinstance(extensions, dict), "extensions must be an object"
        for key in extensions:
            assert "." in key, f"extension namespace {key!r} must be reverse-domain"


def validate_mcp_manifest() -> None:
    m = _load("mcp.json")
    assert m.get("$schema") == MCP_SCHEMA, f"mcp.json $schema must be {MCP_SCHEMA}"
    servers = m.get("mcpServers")
    assert isinstance(servers, dict) and servers, "mcp.json needs a non-empty mcpServers object"
    for server_name, server in servers.items():
        stype = server.get("type")
        assert stype in SERVER_TYPES, f"{server_name}: type must be one of {SERVER_TYPES}"
        if stype == "stdio":
            assert server.get("command"), f"{server_name}: stdio server requires command"
        else:  # streamable-http | sse
            assert server.get("url"), f"{server_name}: {stype} server requires url"
        headers = server.get("headers", {})
        assert all(isinstance(v, str) for v in headers.values()), (
            f"{server_name}: header values must be strings"
        )


def validate_skills() -> None:
    skills_dir = PLUGIN_DIR / "skills"
    assert skills_dir.is_dir(), "skills/ directory is missing"
    skill_files = list(skills_dir.glob("*/SKILL.md"))
    assert skill_files, "expected at least one skills/<name>/SKILL.md"
    for skill in skill_files:
        text = skill.read_text()
        assert text.startswith("---"), f"{skill} must open with YAML frontmatter"
        assert "name:" in text and "description:" in text, (
            f"{skill} frontmatter needs name and description"
        )


def test_plugin_manifest() -> None:
    validate_plugin_manifest()


def test_mcp_manifest() -> None:
    validate_mcp_manifest()


def test_skills() -> None:
    validate_skills()


def main() -> int:
    for check in (validate_plugin_manifest, validate_mcp_manifest, validate_skills):
        check()
        print(f"ok: {check.__name__}")
    print("Agent Plugin is valid against Agent Plugins 1.0.0 required fields.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
