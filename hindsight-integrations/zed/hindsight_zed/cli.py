"""CLI for the Hindsight Zed integration.

``hindsight-zed init`` wires Zed's MCP ``context_servers`` to the Hindsight MCP
endpoint and writes a recall/retain rule into Zed's global instructions file.
After that, Zed's Agent Panel has ``recall``/``retain``/``reflect`` tools and is
told (via the rule) to use them automatically. There is no background process.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import __version__
from .config import USER_CONFIG_FILE, ZedConfig, load_config
from .rules_file import RULE_TEXT, clear_rule, default_rules_path, write_rule
from .rules_file import is_installed as rule_installed
from .zed_settings import (
    SettingsResult,
    apply_to_settings,
    build_context_server,
    default_settings_path,
    remove_from_settings,
    render_snippet,
)
from .zed_settings import (
    is_installed as server_installed,
)


@dataclass
class InstallOutcome:
    """Result of an ``init``: how the settings file changed and where the rule went."""

    settings: SettingsResult
    rules_path: Path


def build_install(config: ZedConfig, settings_path: Path, rules_path: Path) -> InstallOutcome:
    """Apply the MCP server entry and the recall/retain rule (the testable core)."""
    server = build_context_server(config.hindsight_api_url, config.hindsight_api_token, config.bank_id)
    settings = apply_to_settings(settings_path, server)
    write_rule(rules_path)
    return InstallOutcome(settings=settings, rules_path=rules_path)


def _config_path(args: argparse.Namespace) -> Path:
    return Path(args.config_path) if args.config_path else USER_CONFIG_FILE


def _resolve_config(args: argparse.Namespace) -> ZedConfig:
    """Config from file/env, overridden by any explicitly-passed CLI flags."""
    cfg = load_config(config_file=_config_path(args))
    if args.api_url:
        cfg.hindsight_api_url = args.api_url
    if args.api_token:
        cfg.hindsight_api_token = args.api_token
    if args.bank_id:
        cfg.bank_id = args.bank_id
    return cfg


def _scaffold_config(cfg: ZedConfig, config_path: Path) -> None:
    """Persist the resolved connection settings so re-runs remember them."""
    if config_path.is_file():
        return
    data = {"hindsightApiUrl": cfg.hindsight_api_url, "bankId": cfg.bank_id}
    if cfg.hindsight_api_token:
        data["hindsightApiToken"] = cfg.hindsight_api_token
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def cmd_init(args: argparse.Namespace) -> None:
    cfg = _resolve_config(args)
    settings_path = Path(args.settings_path) if args.settings_path else default_settings_path()
    rules_path = Path(args.rules_path) if args.rules_path else default_rules_path()
    server = build_context_server(cfg.hindsight_api_url, cfg.hindsight_api_token, cfg.bank_id)

    if args.print_only:
        print("Add this to your Zed settings.json:\n")
        print(render_snippet(server))
        print("\nAnd add this rule to ~/.config/zed/AGENTS.md:\n")
        print(RULE_TEXT)
        return

    print("Setting up Hindsight for Zed ...")
    _scaffold_config(cfg, _config_path(args))
    outcome = build_install(cfg, settings_path, rules_path)

    if outcome.settings.action == "manual":
        print(f"  Your {outcome.settings.path} has comments, so I won't rewrite it.")
        print("  Add this `context_servers` entry yourself:\n")
        print(render_snippet(server))
    else:
        verb = {"created": "Created", "merged": "Updated", "unchanged": "Already configured in"}[
            outcome.settings.action
        ]
        print(f"  {verb} {outcome.settings.path} (MCP server: hindsight → bank '{cfg.bank_id}')")
    print(f"  Wrote recall/retain rule to {outcome.rules_path}")

    if shutil.which("npx") is None:
        print("\n  warning: `npx` (Node.js) was not found on PATH. Zed runs the MCP")
        print("  bridge via `npx mcp-remote`, so install Node.js for the server to start.")

    print("\nDone. Restart Zed, open the Agent Panel, and the `hindsight` MCP server")
    print("should show a green dot. Memory recall/retain then happen automatically.")


def cmd_status(args: argparse.Namespace) -> None:
    settings_path = Path(args.settings_path) if args.settings_path else default_settings_path()
    rules_path = Path(args.rules_path) if args.rules_path else default_rules_path()
    print(f"MCP server in {settings_path}: {'installed' if server_installed(settings_path) else 'not installed'}")
    print(f"Recall/retain rule in {rules_path}: {'installed' if rule_installed(rules_path) else 'not installed'}")


def cmd_uninstall(args: argparse.Namespace) -> None:
    settings_path = Path(args.settings_path) if args.settings_path else default_settings_path()
    rules_path = Path(args.rules_path) if args.rules_path else default_rules_path()
    result = remove_from_settings(settings_path)
    if result.action == "manual":
        print(f"  {settings_path} has comments — remove the `hindsight` context_servers entry yourself.")
    elif result.action == "removed":
        print(f"  Removed the hindsight MCP server from {settings_path}")
    else:
        print(f"  No hindsight MCP server found in {settings_path}")
    clear_rule(rules_path)
    print(f"  Removed the recall/retain rule from {rules_path}")


def _add_path_overrides(parser: argparse.ArgumentParser) -> None:
    # Hidden overrides used by tests and advanced setups.
    parser.add_argument("--settings-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rules-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--config-path", default=None, help=argparse.SUPPRESS)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="hindsight-zed", description="Hindsight memory for Zed (via MCP)")
    parser.add_argument("--version", action="version", version=f"hindsight-zed {__version__}")
    sub = parser.add_subparsers(dest="command")

    init_p = sub.add_parser("init", help="Configure Zed's MCP server + recall/retain rule")
    init_p.add_argument("--api-url", default=None, help="Hindsight API URL (default: cloud)")
    init_p.add_argument("--api-token", default=None, help="Hindsight API token (for Cloud)")
    init_p.add_argument("--bank-id", default=None, help="Memory bank for the MCP server (default: zed)")
    init_p.add_argument("--print-only", action="store_true", help="Print the config to add manually; write nothing")
    _add_path_overrides(init_p)
    init_p.set_defaults(func=cmd_init)

    status_p = sub.add_parser("status", help="Show whether the MCP server + rule are configured")
    _add_path_overrides(status_p)
    status_p.set_defaults(func=cmd_status)

    uninst_p = sub.add_parser("uninstall", help="Remove the MCP server + rule")
    _add_path_overrides(uninst_p)
    uninst_p.set_defaults(func=cmd_uninstall)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
