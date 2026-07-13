"""CLI for the Hindsight Devin Desktop integration.

``hindsight-devin-desktop init`` wires the Hindsight MCP server (multi-bank mode)
into Devin Desktop's global ``mcp_config.json`` and writes always-on memory rules:

* a per-project ``.devin/rules/hindsight.md`` (committed) naming this repo's
  project bank plus the user's global bank, and
* a managed block in ``~/.codeium/windsurf/memories/global_rules.md`` naming the
  global bank so it's active in every workspace.

The project bank is derived from the repo's git remote so each project keeps its
own isolated memory while the global bank carries cross-project preferences.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from . import __version__
from .config import USER_CONFIG_FILE, load_config
from .global_rules import (
    GlobalRuleResult,
    clear_global_rule,
    default_global_rules_path,
    render_block,
    write_global_rule,
)
from .global_rules import is_installed as global_rule_installed
from .mcp_config import (
    McpResult,
    apply_to_mcp,
    build_http_server,
    default_mcp_paths,
    remove_from_mcp,
    render_snippet,
)
from .mcp_config import is_installed as server_installed
from .project import project_bank_id
from .rules import clear_rule, default_rules_path, render_rule, write_rule
from .rules import is_installed as rule_installed


@dataclass
class Resolved:
    """Fully resolved connection + bank settings for a run."""

    api_url: str
    api_token: Optional[str]
    global_bank: str
    project_bank: str
    project_source: str


@dataclass
class InstallOutcome:
    mcp: List[McpResult]
    rules_path: Path
    global_rule: GlobalRuleResult


def build_install(
    resolved: Resolved, mcp_paths: List[Path], rules_path: Path, global_rules_path: Path
) -> InstallOutcome:
    """Apply the MCP server entry (to each path) and both rule files (testable core)."""
    server = build_http_server(resolved.api_url, resolved.api_token, resolved.global_bank)
    mcp = [apply_to_mcp(path, server) for path in mcp_paths]
    write_rule(rules_path, resolved.project_bank, resolved.global_bank)
    global_rule = write_global_rule(global_rules_path, resolved.global_bank)
    return InstallOutcome(mcp=mcp, rules_path=rules_path, global_rule=global_rule)


def _user_config_path(args: argparse.Namespace) -> Path:
    return Path(args.user_config_path) if args.user_config_path else USER_CONFIG_FILE


def _mcp_paths(args: argparse.Namespace) -> List[Path]:
    return [Path(args.mcp_path)] if args.mcp_path else default_mcp_paths()


def _rules_path(args: argparse.Namespace) -> Path:
    return Path(args.rules_path) if args.rules_path else default_rules_path()


def _global_rules_path(args: argparse.Namespace) -> Path:
    return Path(args.global_rules_path) if args.global_rules_path else default_global_rules_path()


def _resolve(args: argparse.Namespace) -> Resolved:
    """Config from file/env, CLI overrides, then derive the project bank."""
    cfg = load_config(config_file=_user_config_path(args))
    if getattr(args, "api_url", None):
        cfg.hindsight_api_url = args.api_url
    if getattr(args, "api_token", None):
        cfg.hindsight_api_token = args.api_token
    if getattr(args, "global_bank", None):
        cfg.global_bank = args.global_bank
    if getattr(args, "bank_id", None):
        cfg.project_bank = args.bank_id

    if cfg.project_bank:
        project_bank, source = cfg.project_bank, "explicit (--bank-id / env)"
    else:
        project_dir = Path(args.project_dir) if getattr(args, "project_dir", None) else Path.cwd()
        derived, source = project_bank_id(cfg.global_bank, project_dir)
        if derived is None:
            project_bank = cfg.global_bank
            source = f"{source} — falling back to the global bank; pass --bank-id to set one"
        else:
            project_bank = derived

    return Resolved(
        api_url=cfg.hindsight_api_url,
        api_token=cfg.hindsight_api_token,
        global_bank=cfg.global_bank,
        project_bank=project_bank,
        project_source=source,
    )


def _scaffold_user_config(resolved: Resolved, path: Path) -> None:
    """Persist connection + global bank (NOT the per-repo project bank)."""
    if path.is_file():
        return
    data = {"hindsightApiUrl": resolved.api_url, "globalBank": resolved.global_bank}
    if resolved.api_token:
        data["hindsightApiToken"] = resolved.api_token
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def cmd_init(args: argparse.Namespace) -> None:
    resolved = _resolve(args)
    mcp_paths = _mcp_paths(args)
    rules_path = _rules_path(args)
    global_rules_path = _global_rules_path(args)
    server = build_http_server(resolved.api_url, resolved.api_token, resolved.global_bank)

    if args.print_only:
        print("Add this to your Devin Desktop mcp_config.json:\n")
        print(render_snippet(server))
        print(f"\nAnd save this rule as .devin/rules/hindsight.md (project bank: {resolved.project_bank}):\n")
        print(render_rule(resolved.project_bank, resolved.global_bank))
        print("And add this block to ~/.codeium/windsurf/memories/global_rules.md:\n")
        print(render_block(resolved.global_bank))
        return

    print("Setting up Hindsight for Devin Desktop ...")
    print(f"  Global bank:  {resolved.global_bank}   (shared across all your projects)")
    print(f"  Project bank: {resolved.project_bank}")
    print(f"      from {resolved.project_source}")
    _scaffold_user_config(resolved, _user_config_path(args))
    outcome = build_install(resolved, mcp_paths, rules_path, global_rules_path)

    for result in outcome.mcp:
        if result.action == "manual":
            print(f"  Your {result.path} isn't plain JSON, so I won't rewrite it.")
            print("  Add this `mcpServers` entry yourself:\n")
            print(render_snippet(server))
        else:
            verb = {"created": "Created", "merged": "Updated", "unchanged": "Already configured in"}[result.action]
            print(f"  {verb} {result.path} (hindsight MCP, multi-bank)")
    print(f"  Wrote project memory rule to {outcome.rules_path}  (commit this)")
    g = outcome.global_rule
    print(f"  {g.action.capitalize()} global memory rule in {g.path}")
    if g.over_cap:
        print(f"  warning: {g.path} is now {g.over_cap} chars (Devin's cap is 6000); trim it so nothing is dropped.")

    print("\nNow open Devin Desktop and press Refresh in the MCP panel (or restart) —")
    print("editing mcp_config.json does not hot-reload. The hindsight tools")
    print("(recall/retain/reflect) then load and are used automatically.")


def cmd_status(args: argparse.Namespace) -> None:
    resolved = _resolve(args)
    print(f"Global bank:  {resolved.global_bank}")
    print(f"Project bank: {resolved.project_bank}  ({resolved.project_source})")
    for path in _mcp_paths(args):
        print(f"MCP server in {path}: {'installed' if server_installed(path) else 'not installed'}")
    rules_path = _rules_path(args)
    print(f"Project rule in {rules_path}: {'installed' if rule_installed(rules_path) else 'not installed'}")
    global_rules_path = _global_rules_path(args)
    print(
        f"Global rule in {global_rules_path}: {'installed' if global_rule_installed(global_rules_path) else 'not installed'}"
    )


def cmd_uninstall(args: argparse.Namespace) -> None:
    for path in _mcp_paths(args):
        result = remove_from_mcp(path)
        if result.action == "manual":
            print(f"  {path} isn't plain JSON — remove the `hindsight` server entry yourself.")
        elif result.action == "removed":
            print(f"  Removed the hindsight MCP server from {path}")
        else:
            print(f"  No hindsight MCP server found in {path}")
    rules_path = _rules_path(args)
    clear_rule(rules_path)
    print(f"  Removed the project memory rule at {rules_path}")
    global_rules_path = _global_rules_path(args)
    clear_global_rule(global_rules_path)
    print(f"  Removed the global memory rule block in {global_rules_path}")


def _add_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mcp-path", default=None, help="mcp_config.json path (default: both ~/.codeium locations)")
    parser.add_argument("--rules-path", default=None, help="project rule file (default: ./.devin/rules/hindsight.md)")
    parser.add_argument("--global-rules-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--user-config-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--project-dir", default=None, help=argparse.SUPPRESS)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hindsight-devin-desktop", description="Hindsight memory for Devin Desktop (formerly Windsurf), via MCP"
    )
    parser.add_argument("--version", action="version", version=f"hindsight-devin-desktop {__version__}")
    sub = parser.add_subparsers(dest="command")

    init_p = sub.add_parser("init", help="Configure Devin Desktop's MCP server + memory rules")
    init_p.add_argument("--api-url", default=None, help="Hindsight API URL (default: cloud)")
    init_p.add_argument("--api-token", default=None, help="Hindsight API token (for Cloud)")
    init_p.add_argument("--bank-id", default=None, help="Override the per-project bank (default: derived from git)")
    init_p.add_argument("--global-bank", default=None, help="Cross-project bank (default: devin-desktop)")
    init_p.add_argument("--print-only", action="store_true", help="Print the config to add manually; write nothing")
    _add_overrides(init_p)
    init_p.set_defaults(func=cmd_init)

    status_p = sub.add_parser("status", help="Show resolved banks + whether the server/rules are configured")
    status_p.add_argument("--global-bank", default=None, help="Cross-project bank (default: devin-desktop)")
    status_p.add_argument("--bank-id", default=None, help="Override the per-project bank (default: derived from git)")
    _add_overrides(status_p)
    status_p.set_defaults(func=cmd_status)

    uninst_p = sub.add_parser("uninstall", help="Remove the MCP server + memory rules")
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
