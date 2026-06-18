"""CLI for the Hindsight OpenHands integration.

``hindsight-openhands init`` wires the Hindsight MCP server into OpenHands'
``config.toml`` (``[mcp].shttp_servers``) and writes a recall/retain rule into
the project's ``AGENTS.md``. OpenHands then exposes ``recall``/``retain``/
``reflect`` in the agent and (via the rule) uses them automatically. There is no
background process.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import __version__
from .agents_md import RULE_TEXT, clear_rule, default_agents_md_path, write_rule
from .agents_md import is_installed as rule_installed
from .config import USER_CONFIG_FILE, OpenHandsConfig, load_config
from .openhands_config import (
    ConfigResult,
    apply_to_config,
    build_shttp_server,
    default_config_path,
    remove_from_config,
    render_snippet,
)
from .openhands_config import (
    is_installed as server_installed,
)


@dataclass
class InstallOutcome:
    config: ConfigResult
    agents_md_path: Path


def build_install(config: OpenHandsConfig, config_path: Path, agents_md_path: Path) -> InstallOutcome:
    """Apply the MCP server entry and the recall/retain rule (the testable core)."""
    server = build_shttp_server(config.hindsight_api_url, config.hindsight_api_token, config.bank_id)
    result = apply_to_config(config_path, server)
    write_rule(agents_md_path)
    return InstallOutcome(config=result, agents_md_path=agents_md_path)


def _resolve_config(args: argparse.Namespace) -> OpenHandsConfig:
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


def _scaffold_user_config(cfg: OpenHandsConfig, path: Path) -> None:
    if path.is_file():
        return
    data = {"hindsightApiUrl": cfg.hindsight_api_url, "bankId": cfg.bank_id}
    if cfg.hindsight_api_token:
        data["hindsightApiToken"] = cfg.hindsight_api_token
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _config_path(args: argparse.Namespace) -> Path:
    return Path(args.config_path) if args.config_path else default_config_path()


def _agents_path(args: argparse.Namespace) -> Path:
    return Path(args.agents_path) if args.agents_path else default_agents_md_path()


def cmd_init(args: argparse.Namespace) -> None:
    cfg = _resolve_config(args)
    config_path = _config_path(args)
    agents_path = _agents_path(args)
    server = build_shttp_server(cfg.hindsight_api_url, cfg.hindsight_api_token, cfg.bank_id)

    if args.print_only:
        print("Add this to your OpenHands config.toml:\n")
        print(render_snippet(server))
        print("\nAnd add this rule to your project's AGENTS.md:\n")
        print(RULE_TEXT)
        return

    print("Setting up Hindsight for OpenHands ...")
    _scaffold_user_config(cfg, _user_config_path(args))
    outcome = build_install(cfg, config_path, agents_path)

    if outcome.config.action == "manual":
        print(f"  Couldn't safely edit {outcome.config.path} — add this [mcp] entry yourself:\n")
        print(render_snippet(server))
    else:
        verb = {"created": "Created", "merged": "Updated", "unchanged": "Already configured in"}[outcome.config.action]
        print(f"  {verb} {outcome.config.path} (MCP server -> bank '{cfg.bank_id}')")
    print(f"  Wrote recall/retain rule to {outcome.agents_md_path}")
    print("\nDone. Start OpenHands in this project — the hindsight MCP tools")
    print("(recall/retain/reflect) are available and used automatically.")


def cmd_status(args: argparse.Namespace) -> None:
    cfg = _resolve_config(args)
    server = build_shttp_server(cfg.hindsight_api_url, cfg.hindsight_api_token, cfg.bank_id)
    config_path = _config_path(args)
    agents_path = _agents_path(args)
    print(f"MCP server in {config_path}: {'installed' if server_installed(config_path, server) else 'not installed'}")
    print(f"Recall/retain rule in {agents_path}: {'installed' if rule_installed(agents_path) else 'not installed'}")


def cmd_uninstall(args: argparse.Namespace) -> None:
    cfg = _resolve_config(args)
    server = build_shttp_server(cfg.hindsight_api_url, cfg.hindsight_api_token, cfg.bank_id)
    config_path = _config_path(args)
    agents_path = _agents_path(args)
    result = remove_from_config(config_path, server)
    if result.action == "manual":
        print(f"  Couldn't parse {config_path} — remove the hindsight [mcp] entry yourself.")
    elif result.action == "removed":
        print(f"  Removed the hindsight MCP server from {config_path}")
    else:
        print(f"  No hindsight MCP server found in {config_path}")
    clear_rule(agents_path)
    print(f"  Removed the recall/retain rule from {agents_path}")


def _add_common(parser: argparse.ArgumentParser) -> None:
    # Connection args (all subcommands resolve the same config to match the entry).
    parser.add_argument("--api-url", default=None, help="Hindsight API URL (default: cloud)")
    parser.add_argument("--api-token", default=None, help="Hindsight API token (for Cloud)")
    parser.add_argument("--bank-id", default=None, help="Memory bank for the MCP server (default: openhands)")
    # Path overrides.
    parser.add_argument("--config-path", default=None, help="OpenHands config.toml path (default: ./config.toml)")
    parser.add_argument("--agents-path", default=None, help="AGENTS.md path (default: ./AGENTS.md)")
    parser.add_argument("--user-config-path", default=None, help=argparse.SUPPRESS)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="hindsight-openhands", description="Hindsight memory for OpenHands (via MCP)")
    parser.add_argument("--version", action="version", version=f"hindsight-openhands {__version__}")
    sub = parser.add_subparsers(dest="command")

    init_p = sub.add_parser("init", help="Configure OpenHands' MCP server + recall/retain rule")
    _add_common(init_p)
    init_p.add_argument("--print-only", action="store_true", help="Print the config to add manually; write nothing")
    init_p.set_defaults(func=cmd_init)

    status_p = sub.add_parser("status", help="Show whether the MCP server + rule are configured")
    _add_common(status_p)
    status_p.set_defaults(func=cmd_status)

    uninst_p = sub.add_parser("uninstall", help="Remove the MCP server + rule")
    _add_common(uninst_p)
    uninst_p.set_defaults(func=cmd_uninstall)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
