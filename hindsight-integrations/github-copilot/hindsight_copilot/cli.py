"""CLI for the Hindsight GitHub Copilot integration.

``hindsight-copilot init`` wires the Hindsight MCP server into VS Code's
``.vscode/mcp.json`` and writes a recall/retain rule into
``.github/copilot-instructions.md``. Copilot's agent mode then exposes
``recall``/``retain``/``reflect`` and (via the rule) uses them automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import __version__
from .config import USER_CONFIG_FILE, CopilotConfig, load_config
from .instructions import RULE_TEXT, clear_rule, default_instructions_path, write_rule
from .instructions import is_installed as rule_installed
from .mcp_config import (
    McpResult,
    apply_to_mcp,
    build_http_server,
    default_mcp_path,
    remove_from_mcp,
    render_snippet,
)
from .mcp_config import is_installed as server_installed


@dataclass
class InstallOutcome:
    mcp: McpResult
    instructions_path: Path


def build_install(config: CopilotConfig, mcp_path: Path, instructions_path: Path) -> InstallOutcome:
    """Apply the MCP server entry and the recall/retain rule (the testable core)."""
    server = build_http_server(config.hindsight_api_url, config.hindsight_api_token, config.bank_id)
    mcp = apply_to_mcp(mcp_path, server)
    write_rule(instructions_path)
    return InstallOutcome(mcp=mcp, instructions_path=instructions_path)


def _resolve_config(args: argparse.Namespace) -> CopilotConfig:
    cfg = load_config(config_file=_user_config_path(args))
    if args.api_url:
        cfg.hindsight_api_url = args.api_url
    if args.api_token:
        cfg.hindsight_api_token = args.api_token
    if args.bank_id:
        cfg.bank_id = args.bank_id
    return cfg


def _user_config_path(args: argparse.Namespace) -> Path:
    return Path(args.user_config_path) if args.user_config_path else USER_CONFIG_FILE


def _mcp_path(args: argparse.Namespace) -> Path:
    return Path(args.mcp_path) if args.mcp_path else default_mcp_path()


def _instructions_path(args: argparse.Namespace) -> Path:
    return Path(args.instructions_path) if args.instructions_path else default_instructions_path()


def _scaffold_user_config(cfg: CopilotConfig, path: Path) -> None:
    if path.is_file():
        return
    data = {"hindsightApiUrl": cfg.hindsight_api_url, "bankId": cfg.bank_id}
    if cfg.hindsight_api_token:
        data["hindsightApiToken"] = cfg.hindsight_api_token
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def cmd_init(args: argparse.Namespace) -> None:
    cfg = _resolve_config(args)
    mcp_path = _mcp_path(args)
    instructions_path = _instructions_path(args)
    server = build_http_server(cfg.hindsight_api_url, cfg.hindsight_api_token, cfg.bank_id)

    if args.print_only:
        print("Add this to your .vscode/mcp.json:\n")
        print(render_snippet(server))
        print("\nAnd add this rule to .github/copilot-instructions.md:\n")
        print(RULE_TEXT)
        return

    print("Setting up Hindsight for GitHub Copilot ...")
    _scaffold_user_config(cfg, _user_config_path(args))
    outcome = build_install(cfg, mcp_path, instructions_path)

    if outcome.mcp.action == "manual":
        print(f"  Your {outcome.mcp.path} has comments, so I won't rewrite it.")
        print("  Add this `servers` entry yourself:\n")
        print(render_snippet(server))
    else:
        verb = {"created": "Created", "merged": "Updated", "unchanged": "Already configured in"}[outcome.mcp.action]
        print(f"  {verb} {outcome.mcp.path} (MCP server: hindsight -> bank '{cfg.bank_id}')")
    print(f"  Wrote recall/retain rule to {outcome.instructions_path}")
    print("\nDone. Reload VS Code, open Copilot Chat in agent mode, and the")
    print("hindsight MCP tools (recall/retain/reflect) are available + used automatically.")


def cmd_status(args: argparse.Namespace) -> None:
    mcp_path = _mcp_path(args)
    instructions_path = _instructions_path(args)
    print(f"MCP server in {mcp_path}: {'installed' if server_installed(mcp_path) else 'not installed'}")
    print(
        f"Recall/retain rule in {instructions_path}: {'installed' if rule_installed(instructions_path) else 'not installed'}"
    )


def cmd_uninstall(args: argparse.Namespace) -> None:
    mcp_path = _mcp_path(args)
    instructions_path = _instructions_path(args)
    result = remove_from_mcp(mcp_path)
    if result.action == "manual":
        print(f"  {mcp_path} has comments — remove the `hindsight` server entry yourself.")
    elif result.action == "removed":
        print(f"  Removed the hindsight MCP server from {mcp_path}")
    else:
        print(f"  No hindsight MCP server found in {mcp_path}")
    clear_rule(instructions_path)
    print(f"  Removed the recall/retain rule from {instructions_path}")


def _add_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mcp-path", default=None, help=".vscode/mcp.json path (default: ./.vscode/mcp.json)")
    parser.add_argument(
        "--instructions-path", default=None, help="copilot-instructions.md path (default: ./.github/...)"
    )
    parser.add_argument("--user-config-path", default=None, help=argparse.SUPPRESS)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hindsight-copilot", description="Hindsight memory for GitHub Copilot (VS Code, via MCP)"
    )
    parser.add_argument("--version", action="version", version=f"hindsight-copilot {__version__}")
    sub = parser.add_subparsers(dest="command")

    init_p = sub.add_parser("init", help="Configure Copilot's MCP server + recall/retain rule")
    init_p.add_argument("--api-url", default=None, help="Hindsight API URL (default: cloud)")
    init_p.add_argument("--api-token", default=None, help="Hindsight API token (for Cloud)")
    init_p.add_argument("--bank-id", default=None, help="Memory bank for the MCP server (default: copilot)")
    init_p.add_argument("--print-only", action="store_true", help="Print the config to add manually; write nothing")
    _add_overrides(init_p)
    init_p.set_defaults(func=cmd_init)

    status_p = sub.add_parser("status", help="Show whether the MCP server + rule are configured")
    _add_overrides(status_p)
    status_p.set_defaults(func=cmd_status)

    uninst_p = sub.add_parser("uninstall", help="Remove the MCP server + rule")
    _add_overrides(uninst_p)
    uninst_p.set_defaults(func=cmd_uninstall)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
