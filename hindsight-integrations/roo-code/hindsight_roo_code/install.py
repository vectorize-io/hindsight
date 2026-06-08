"""Install logic for the Roo Code Hindsight integration.

Writes two files into a Roo Code config directory:
  1. ``mcp.json``                — registers the Hindsight MCP server
  2. ``rules/hindsight-memory.md`` — rules injected into every Roo system prompt

The rules file is shipped as package data and read via ``importlib.resources``
so it resolves correctly whether the package is installed as a wheel or run
from a source checkout.
"""

import json
import sys
from importlib import resources
from pathlib import Path

DEFAULT_API_URL = "https://api.hindsight.vectorize.io"
MCP_TIMEOUT_SECONDS = 30
RULES_FILENAME = "hindsight-memory.md"


def rules_text() -> str:
    """Return the bundled rules-file content."""
    resource = resources.files("hindsight_roo_code").joinpath("rules", RULES_FILENAME)
    return resource.read_text(encoding="utf-8")


def get_roo_dir(project_dir: Path, global_install: bool) -> Path:
    if global_install:
        return Path.home() / ".roo"
    return project_dir / ".roo"


def build_mcp_entry(api_url: str) -> dict:
    return {
        "type": "streamable-http",
        "url": f"{api_url.rstrip('/')}/mcp",
        "timeout": MCP_TIMEOUT_SECONDS,
        "alwaysAllow": ["recall", "retain"],
    }


def install_mcp(roo_dir: Path, api_url: str) -> None:
    mcp_path = roo_dir / "mcp.json"
    roo_dir.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if mcp_path.exists():
        try:
            existing = json.loads(mcp_path.read_text())
        except json.JSONDecodeError as e:
            print(f"Warning: could not parse {mcp_path}: {e}. Overwriting.", file=sys.stderr)

    servers: dict = existing.get("mcpServers", {})
    servers["hindsight"] = build_mcp_entry(api_url)
    existing["mcpServers"] = servers

    mcp_path.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"MCP config written: {mcp_path}")


def install_rules(roo_dir: Path) -> None:
    rules_dir = roo_dir / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    dest = rules_dir / RULES_FILENAME
    dest.write_text(rules_text(), encoding="utf-8")
    print(f"Rules file written: {dest}")


def run_install(api_url: str, project_dir: Path, global_install: bool) -> None:
    """Run the full install: write the MCP config and the rules file."""
    roo_dir = get_roo_dir(project_dir.resolve(), global_install)

    print("Installing Hindsight memory for Roo Code...")
    print(f"  API URL : {api_url}")
    print(f"  Roo dir : {roo_dir}")
    print()

    install_mcp(roo_dir, api_url)
    install_rules(roo_dir)

    print()
    print("Done. Restart Roo Code for the changes to take effect.")
    print()
    print("To verify, open Roo Code and check:")
    print("  Settings → MCP Servers → hindsight (should show as connected)")
