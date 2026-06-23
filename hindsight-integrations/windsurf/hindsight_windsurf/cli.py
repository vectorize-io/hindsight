"""CLI for the Hindsight Windsurf integration.

``hindsight-windsurf init`` wires the Hindsight MCP server into Windsurf's
``~/.codeium/windsurf/mcp_config.json`` and writes an always-on recall/retain
rule into ``.windsurf/rules/hindsight.md``. Cascade then exposes
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
from .config import USER_CONFIG_FILE, WindsurfConfig, load_config
from .mcp_config import (
    McpResult,
    apply_to_mcp,
    build_http_server,
    default_mcp_path,
    remove_from_mcp,
    render_snippet,
)
from .mcp_config import is_installed as server_installed
from .rules import RULE_TEXT, clear_rule, default_rules_path, write_rule
from .rules import is_installed as rule_installed


@dataclass
class InstallOutcome:
    mcp: McpResult
    rules_path: Path


def build_install(config: WindsurfConfig, mcp_path: Path, rules_path: Path) -> InstallOutcome:
    """Apply the MCP server entry and the recall/retain rule (the testable core)."""
    server = build_http_server(config.hindsight_api_url, config.hindsight_api_token, config.bank_id)
    mcp = apply_to_mcp(mcp_path, server)
    write_rule(rules_path)
    return InstallOutcome(mcp=mcp, rules_path=rules_path)


def _resolve_config(args: argparse.Namespace) -> WindsurfConfig:
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


def _rules_path(args: argparse.Namespace) -> Path:
    return Path(args.rules_path) if args.rules_path else default_rules_path()


def _scaffold_user_config(cfg: WindsurfConfig, path: Path) -> None:
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
    rules_path = _rules_path(args)
    server = build_http_server(cfg.hindsight_api_url, cfg.hindsight_api_token, cfg.bank_id)

    if args.print_only:
        print("Add this to your ~/.codeium/windsurf/mcp_config.json:\n")
        print(render_snippet(server))
        print("\nAnd save this rule as .windsurf/rules/hindsight.md:\n")
        print(RULE_TEXT)
        return

    print("Setting up Hindsight for Windsurf ...")
    _scaffold_user_config(cfg, _user_config_path(args))
    outcome = build_install(cfg, mcp_path, rules_path)

    if outcome.mcp.action == "manual":
        print(f"  Your {outcome.mcp.path} isn't plain JSON, so I won't rewrite it.")
        print("  Add this `mcpServers` entry yourself:\n")
        print(render_snippet(server))
    else:
        verb = {"created": "Created", "merged": "Updated", "unchanged": "Already configured in"}[outcome.mcp.action]
        print(f"  {verb} {outcome.mcp.path} (MCP server: hindsight -> bank '{cfg.bank_id}')")
    print(f"  Wrote always-on recall/retain rule to {outcome.rules_path}")
    print("\nDone. Reload Windsurf (or refresh MCP servers in Cascade) and the")
    print("hindsight MCP tools (recall/retain/reflect) are available + used automatically.")


def cmd_status(args: argparse.Namespace) -> None:
    mcp_path = _mcp_path(args)
    rules_path = _rules_path(args)
    print(f"MCP server in {mcp_path}: {'installed' if server_installed(mcp_path) else 'not installed'}")
    print(f"Recall/retain rule in {rules_path}: {'installed' if rule_installed(rules_path) else 'not installed'}")


def cmd_uninstall(args: argparse.Namespace) -> None:
    mcp_path = _mcp_path(args)
    rules_path = _rules_path(args)
    result = remove_from_mcp(mcp_path)
    if result.action == "manual":
        print(f"  {mcp_path} isn't plain JSON — remove the `hindsight` server entry yourself.")
    elif result.action == "removed":
        print(f"  Removed the hindsight MCP server from {mcp_path}")
    else:
        print(f"  No hindsight MCP server found in {mcp_path}")
    clear_rule(rules_path)
    print(f"  Removed the recall/retain rule at {rules_path}")


def _add_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mcp-path", default=None, help="mcp_config.json path (default: ~/.codeium/windsurf/mcp_config.json)"
    )
    parser.add_argument("--rules-path", default=None, help="rule file path (default: ./.windsurf/rules/hindsight.md)")
    parser.add_argument("--user-config-path", default=None, help=argparse.SUPPRESS)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hindsight-windsurf", description="Hindsight memory for Windsurf (Codeium, via MCP)"
    )
    parser.add_argument("--version", action="version", version=f"hindsight-windsurf {__version__}")
    sub = parser.add_subparsers(dest="command")

    init_p = sub.add_parser("init", help="Configure Windsurf's MCP server + recall/retain rule")
    init_p.add_argument("--api-url", default=None, help="Hindsight API URL (default: cloud)")
    init_p.add_argument("--api-token", default=None, help="Hindsight API token (for Cloud)")
    init_p.add_argument("--bank-id", default=None, help="Memory bank for the MCP server (default: windsurf)")
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
