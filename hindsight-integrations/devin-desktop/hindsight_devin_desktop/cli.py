"""CLI for the Hindsight Devin Desktop integration.

``hindsight-devin-desktop init`` wires the Hindsight MCP server (multi-bank mode)
into **both** agents Devin Desktop ships:

* **Cascade** — MCP in ``~/.codeium/windsurf/mcp_config.json`` (``serverUrl``),
  always-on rules in ``.devin/rules/hindsight.md`` (per-project) +
  ``~/.codeium/windsurf/memories/global_rules.md`` (global).
* **Devin Local** — MCP in ``~/.config/devin/config.json`` (``url``/``transport``/
  ``headers``, plus an auto-approve permission), always-on rules in ``AGENTS.md``
  (repo root, per-project) + ``~/.config/devin/AGENTS.md`` (global).

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

from . import __version__, cascade_hooks, devin_local
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
from .rules import clear_rule, default_rules_path, render_rule, render_rule_text, write_rule
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
class Paths:
    """Every file the integration manages, across both agents."""

    mcp: List[Path]  # Cascade mcp_config.json (both documented locations)
    rules: Path  # Cascade per-project rule (.devin/rules/hindsight.md)
    global_rules: Path  # Cascade global rule (global_rules.md)
    cascade_hooks: Path  # Cascade hooks.json (visibility banner)
    devin_config: Path  # Devin Local config.json (MCP + permissions)
    devin_global_agents: Path  # Devin Local global AGENTS.md
    devin_project_agents: Path  # Devin Local per-project AGENTS.md (repo root)


@dataclass
class InstallOutcome:
    mcp: List[McpResult]
    rules_path: Path
    global_rule: GlobalRuleResult
    cascade_banner: cascade_hooks.BannerResult
    devin_config: devin_local.ConfigResult
    devin_global_action: str
    devin_global_path: Path
    devin_project_action: str
    devin_project_path: Path


def build_install(
    resolved: Resolved,
    paths: Paths,
    recall_hook: bool = True,
    retain_hook: bool = True,
    cascade_banner: bool = True,
    local_only: bool = False,
) -> InstallOutcome:
    """Wire both agents (the testable core).

    ``local_only`` = no shared global bank: everything (project facts + the
    user's preferences) goes to the single project bank; the global rule files
    are removed rather than written.
    """
    # ``rule_global`` is None in local-only, so the rules route everything to the
    # project bank. The MCP server keeps the global bank as its X-Bank-Id prefix
    # (the hooks derive the project bank from it).
    rule_global = None if local_only else resolved.global_bank

    # Cascade
    server = build_http_server(resolved.api_url, resolved.api_token, resolved.global_bank)
    mcp = [apply_to_mcp(path, server) for path in paths.mcp]
    write_rule(paths.rules, resolved.project_bank, rule_global)
    if local_only:
        clear_global_rule(paths.global_rules)
        global_rule = GlobalRuleResult("removed", paths.global_rules)
    else:
        global_rule = write_global_rule(paths.global_rules, resolved.global_bank)
    banner = (
        cascade_hooks.apply_banner(paths.cascade_hooks)
        if cascade_banner
        else cascade_hooks.remove_banner(paths.cascade_hooks)
    )

    # Devin Local
    dl_server = devin_local.build_http_server(resolved.api_url, resolved.api_token, resolved.global_bank)
    devin_config = devin_local.apply_to_config(
        paths.devin_config, dl_server, recall_hook=recall_hook, retain_hook=retain_hook, local_only=local_only
    )
    if local_only:
        devin_local.clear_agents(paths.devin_global_agents)
        dl_global = "removed"
    else:
        dl_global = devin_local.write_global_agents(paths.devin_global_agents, resolved.global_bank)
    dl_project = devin_local.write_project_agents(paths.devin_project_agents, resolved.project_bank, rule_global)

    return InstallOutcome(
        mcp=mcp,
        rules_path=paths.rules,
        global_rule=global_rule,
        cascade_banner=banner,
        devin_config=devin_config,
        devin_global_action=dl_global,
        devin_global_path=paths.devin_global_agents,
        devin_project_action=dl_project,
        devin_project_path=paths.devin_project_agents,
    )


def _user_config_path(args: argparse.Namespace) -> Path:
    return Path(args.user_config_path) if args.user_config_path else USER_CONFIG_FILE


def _paths(args: argparse.Namespace) -> Paths:
    mcp = [Path(args.mcp_path)] if args.mcp_path else default_mcp_paths()
    return Paths(
        mcp=mcp,
        rules=Path(args.rules_path) if args.rules_path else default_rules_path(),
        global_rules=Path(args.global_rules_path) if args.global_rules_path else default_global_rules_path(),
        cascade_hooks=(
            Path(args.cascade_hooks_path) if args.cascade_hooks_path else cascade_hooks.default_hooks_path()
        ),
        devin_config=Path(args.devin_config_path) if args.devin_config_path else devin_local.default_config_path(),
        devin_global_agents=(
            Path(args.devin_global_agents_path)
            if args.devin_global_agents_path
            else devin_local.default_global_agents_path()
        ),
        devin_project_agents=(
            Path(args.project_agents_path) if args.project_agents_path else devin_local.default_project_agents_path()
        ),
    )


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


_VERB = {"created": "Created", "merged": "Updated", "unchanged": "Already configured in", "removed": "Removed"}


def _print_only(
    resolved: Resolved, server: dict, dl_server: dict, recall_hook: bool, retain_hook: bool, local_only: bool
) -> None:
    rule_global = None if local_only else resolved.global_bank
    print("# Cascade agent\nAdd this to ~/.codeium/windsurf/mcp_config.json:\n")
    print(render_snippet(server))
    print(f"\nSave this rule as .devin/rules/hindsight.md (project bank: {resolved.project_bank}):\n")
    print(render_rule(resolved.project_bank, rule_global))
    if not local_only:
        print("Add this block to ~/.codeium/windsurf/memories/global_rules.md:\n")
        print(render_block(resolved.global_bank))
    print("\n# Devin Local agent\nMerge this into ~/.config/devin/config.json:\n")
    print(
        devin_local.render_snippet(dl_server, recall_hook=recall_hook, retain_hook=retain_hook, local_only=local_only)
    )
    print("\nAdd this to AGENTS.md (repo root)" + ("" if local_only else " and ~/.config/devin/AGENTS.md") + ":\n")
    print(render_rule_text(resolved.project_bank, rule_global))


def cmd_init(args: argparse.Namespace) -> None:
    resolved = _resolve(args)
    paths = _paths(args)
    hooks_on = not getattr(args, "no_hooks", False)
    recall_hook = hooks_on
    retain_hook = hooks_on and not getattr(args, "no_retain_hook", False)
    local_only = getattr(args, "no_global_bank", False)
    server = build_http_server(resolved.api_url, resolved.api_token, resolved.global_bank)
    dl_server = devin_local.build_http_server(resolved.api_url, resolved.api_token, resolved.global_bank)

    if args.print_only:
        _print_only(resolved, server, dl_server, recall_hook, retain_hook, local_only)
        return

    print("Setting up Hindsight for Devin Desktop ...")
    if local_only:
        print("  Mode: local-only (no shared bank — this project's memory stays in its own bank)")
        print(f"  Project bank: {resolved.project_bank}")
    else:
        print(f"  Global bank:  {resolved.global_bank}   (shared across all your projects)")
        print(f"  Project bank: {resolved.project_bank}")
    print(f"      from {resolved.project_source}")
    _scaffold_user_config(resolved, _user_config_path(args))
    outcome = build_install(
        resolved,
        paths,
        recall_hook=recall_hook,
        retain_hook=retain_hook,
        cascade_banner=hooks_on,
        local_only=local_only,
    )

    print("\n  Cascade agent:")
    for result in outcome.mcp:
        if result.action == "manual":
            print(f"    {result.path} isn't plain JSON — add the `mcpServers` entry yourself:\n")
            print(render_snippet(server))
        else:
            print(f"    {_VERB[result.action]} {result.path} (hindsight MCP, multi-bank)")
    print(f"    Wrote project rule to {outcome.rules_path}  (commit this)")
    g = outcome.global_rule
    print(f"    {g.action.capitalize()} global rule in {g.path}")
    if g.over_cap:
        print(f"    warning: {g.path} is now {g.over_cap} chars (Cascade's cap is 6000); trim it.")
    b = outcome.cascade_banner
    if b.action in ("created", "merged"):
        print(f"    Added a visibility banner hook in {b.path} (shows '🧠 Hindsight: <tool> used')")
    elif b.action == "manual":
        print(f"    {b.path} isn't plain JSON — add the post_mcp_tool_use banner hook yourself.")

    print("\n  Devin Local agent:")
    dc = outcome.devin_config
    if dc.action == "manual":
        print(f"    {dc.path} isn't plain JSON — merge this yourself:\n")
        print(devin_local.render_snippet(dl_server, recall_hook=recall_hook, retain_hook=retain_hook))
    else:
        print(f"    {_VERB[dc.action]} {dc.path} (hindsight MCP + auto-approve)")
    print(
        f"    {outcome.devin_project_action.capitalize()} project rule in {outcome.devin_project_path}  (commit this)"
    )
    print(f"    {outcome.devin_global_action.capitalize()} global rule in {outcome.devin_global_path}")
    if dc.action != "manual":
        if recall_hook:
            print("    Added a SessionStart auto-recall hook (memory injected even if the model forgets to recall)")
        if retain_hook:
            print("    Added a Stop retain-nudge hook (prompts a retain pass before the session ends)")

    print("\nActivate the server in whichever agent you use (config isn't hot-reloaded):")
    print("  - Cascade:     open the MCP panel and press Refresh.")
    print("  - Devin Local: open the Devin MCP Marketplace, find `hindsight` under")
    print("                 Installed, and click Connect.")
    print("Memory then loads and is used automatically.")


def cmd_status(args: argparse.Namespace) -> None:
    resolved = _resolve(args)
    paths = _paths(args)
    print(f"Global bank:  {resolved.global_bank}")
    print(f"Project bank: {resolved.project_bank}  ({resolved.project_source})")

    def mark(ok: bool) -> str:
        return "installed" if ok else "not installed"

    print("Cascade:")
    for path in paths.mcp:
        print(f"  MCP server in {path}: {mark(server_installed(path))}")
    print(f"  Project rule in {paths.rules}: {mark(rule_installed(paths.rules))}")
    print(f"  Global rule in {paths.global_rules}: {mark(global_rule_installed(paths.global_rules))}")
    print(f"  Visibility banner in {paths.cascade_hooks}: {mark(cascade_hooks.is_installed(paths.cascade_hooks))}")
    print("Devin Local:")
    print(f"  MCP server in {paths.devin_config}: {mark(devin_local.is_installed(paths.devin_config))}")
    print(
        f"  Project rule in {paths.devin_project_agents}: {mark(devin_local.agents_installed(paths.devin_project_agents))}"
    )
    print(
        f"  Global rule in {paths.devin_global_agents}: {mark(devin_local.agents_installed(paths.devin_global_agents))}"
    )
    print(
        f"  Auto-recall hook in {paths.devin_config}: {mark(devin_local.hook_installed(paths.devin_config, 'recall'))}"
    )
    print(
        f"  Retain-nudge hook in {paths.devin_config}: {mark(devin_local.hook_installed(paths.devin_config, 'retain-nudge'))}"
    )


def cmd_uninstall(args: argparse.Namespace) -> None:
    paths = _paths(args)
    print("Cascade:")
    for path in paths.mcp:
        result = remove_from_mcp(path)
        if result.action == "manual":
            print(f"  {path} isn't plain JSON — remove the `hindsight` entry yourself.")
        elif result.action == "removed":
            print(f"  Removed the hindsight MCP server from {path}")
        else:
            print(f"  No hindsight MCP server found in {path}")
    clear_rule(paths.rules)
    print(f"  Removed the project rule at {paths.rules}")
    clear_global_rule(paths.global_rules)
    print(f"  Removed the global rule block in {paths.global_rules}")
    cascade_hooks.remove_banner(paths.cascade_hooks)
    print(f"  Removed the visibility banner hook from {paths.cascade_hooks}")

    print("Devin Local:")
    dc = devin_local.remove_from_config(paths.devin_config)
    if dc.action == "manual":
        print(f"  {paths.devin_config} isn't plain JSON — remove the `hindsight` entry yourself.")
    elif dc.action == "removed":
        print(f"  Removed the hindsight MCP server + allow-rule from {paths.devin_config}")
    else:
        print(f"  No hindsight MCP server found in {paths.devin_config}")
    devin_local.clear_agents(paths.devin_project_agents)
    print(f"  Removed the project rule block in {paths.devin_project_agents}")
    devin_local.clear_agents(paths.devin_global_agents)
    print(f"  Removed the global rule block in {paths.devin_global_agents}")


def _add_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mcp-path", default=None, help="Cascade mcp_config.json (default: both ~/.codeium locations)")
    parser.add_argument(
        "--rules-path", default=None, help="Cascade project rule (default: ./.devin/rules/hindsight.md)"
    )
    parser.add_argument("--global-rules-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cascade-hooks-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--devin-config-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--devin-global-agents-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--project-agents-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--user-config-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--project-dir", default=None, help=argparse.SUPPRESS)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hindsight-devin-desktop", description="Hindsight memory for Devin Desktop (formerly Windsurf), via MCP"
    )
    parser.add_argument("--version", action="version", version=f"hindsight-devin-desktop {__version__}")
    sub = parser.add_subparsers(dest="command")

    init_p = sub.add_parser("init", help="Configure both Devin Desktop agents' MCP server + memory rules")
    init_p.add_argument("--api-url", default=None, help="Hindsight API URL (default: cloud)")
    init_p.add_argument("--api-token", default=None, help="Hindsight API token (for Cloud)")
    init_p.add_argument("--bank-id", default=None, help="Override the per-project bank (default: derived from git)")
    init_p.add_argument("--global-bank", default=None, help="Cross-project bank (default: devin-desktop)")
    init_p.add_argument("--print-only", action="store_true", help="Print the config to add manually; write nothing")
    init_p.add_argument("--no-hooks", action="store_true", help="Skip both Devin Local hooks (recall + retain-nudge)")
    init_p.add_argument(
        "--no-retain-hook", action="store_true", help="Skip the Stop retain-nudge hook (keep auto-recall)"
    )
    init_p.add_argument(
        "--no-global-bank",
        action="store_true",
        help="Local-only: keep all memory in this project's bank (no shared cross-project preferences)",
    )
    _add_overrides(init_p)
    init_p.set_defaults(func=cmd_init)

    status_p = sub.add_parser("status", help="Show resolved banks + whether both agents are configured")
    status_p.add_argument("--global-bank", default=None, help="Cross-project bank (default: devin-desktop)")
    status_p.add_argument("--bank-id", default=None, help="Override the per-project bank (default: derived from git)")
    _add_overrides(status_p)
    status_p.set_defaults(func=cmd_status)

    uninst_p = sub.add_parser("uninstall", help="Remove the MCP server + memory rules from both agents")
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
