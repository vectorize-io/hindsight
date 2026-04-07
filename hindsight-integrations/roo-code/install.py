#!/usr/bin/env python3
"""Install Hindsight memory for Roo Code.

Writes two files:
  1. .roo/mcp.json  — registers the Hindsight MCP server
  2. .roo/rules/hindsight-memory.md — rules injected into every Roo system prompt

Usage:
    python install.py
    python install.py --api-url http://localhost:8888
    python install.py --project-dir /path/to/project
    python install.py --global   # write to ~/.roo/ instead of ./.roo/
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RULES_SRC = SCRIPT_DIR / "rules" / "hindsight-memory.md"

DEFAULT_API_URL = "http://localhost:8888"
MCP_TIMEOUT_MS = 10000


def get_roo_dir(project_dir: Path, global_install: bool) -> Path:
    if global_install:
        return Path.home() / ".roo"
    return project_dir / ".roo"


def build_mcp_entry(api_url: str) -> dict:
    return {
        "url": f"{api_url.rstrip('/')}/mcp",
        "timeout": MCP_TIMEOUT_MS,
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
    dest = rules_dir / "hindsight-memory.md"
    shutil.copy2(RULES_SRC, dest)
    print(f"Rules file written: {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install Hindsight memory integration for Roo Code."
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("HINDSIGHT_API_URL", DEFAULT_API_URL),
        help=f"Hindsight API base URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--project-dir",
        default=".",
        help="Project directory to install into (default: current directory)",
    )
    parser.add_argument(
        "--global",
        dest="global_install",
        action="store_true",
        help="Install globally to ~/.roo/ instead of the project directory",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    roo_dir = get_roo_dir(project_dir, args.global_install)

    print(f"Installing Hindsight memory for Roo Code...")
    print(f"  API URL : {args.api_url}")
    print(f"  Roo dir : {roo_dir}")
    print()

    install_mcp(roo_dir, args.api_url)
    install_rules(roo_dir)

    print()
    print("Done. Restart Roo Code for the changes to take effect.")
    print()
    print("To verify, open Roo Code and check:")
    print("  Settings → MCP Servers → hindsight (should show as connected)")


if __name__ == "__main__":
    main()
