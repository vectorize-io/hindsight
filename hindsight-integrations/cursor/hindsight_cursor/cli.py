"""CLI for installing the Hindsight Cursor plugin into a project."""

import argparse
import json
import shutil
import sys
from pathlib import Path

# Plugin files to copy (relative to the plugin_data directory).
# Preserves directory structure.
_PLUGIN_FILES = [
    ".cursor-plugin/plugin.json",
    "hooks/hooks.json",
    "rules/hindsight-memory.mdc",
    "scripts/lib/__init__.py",
    "scripts/lib/bank.py",
    "scripts/lib/client.py",
    "scripts/lib/config.py",
    "scripts/lib/content.py",
    "scripts/lib/daemon.py",
    "scripts/lib/llm.py",
    "scripts/lib/rules_file.py",
    "scripts/lib/state.py",
    "scripts/session_start.py",
    "scripts/retain.py",
    "settings.json",
    "skills/hindsight-recall/SKILL.md",
]

# Marker used to identify Hindsight's own entries when merging/stripping
# project .cursor/hooks.json. Commands point at this install path.
_HOOK_MARKER = ".cursor-plugin/hindsight-memory"

# Artefacts the sessionStart hook writes into the workspace. Kept in sync with
# scripts/lib/rules_file.py by hand — the CLI deliberately does not import the
# plugin scripts, which ship as data rather than as an importable package.
_SESSION_RULES_RELPATH = Path(".cursor") / "rules" / "hindsight-session.mdc"
_GITIGNORE_COMMENT = "# Added by hindsight-cursor plugin"
_GITIGNORE_LINE = "/.cursor/rules/hindsight-session.mdc"

_USER_CONFIG_DIR = Path.home() / ".hindsight"
_USER_CONFIG_FILE = _USER_CONFIG_DIR / "cursor.json"


def _plugin_data_dir() -> Path:
    """Return the path to the bundled plugin data shipped with this package."""
    return Path(__file__).resolve().parent / "plugin_data"


def _copy_plugin(dest: Path) -> None:
    """Copy plugin files into *dest*, creating directories as needed.

    Aborts if anything is missing from the bundled payload. A partial copy is
    never useful — ``session_start.py`` imports every ``lib/`` module, so a
    single absent file turns each hook invocation into a silent ImportError.
    Reporting "Done!" over an install that copied nothing is how #3864 stayed
    invisible for so long.
    """
    src_root = _plugin_data_dir()
    missing = [rel for rel in _PLUGIN_FILES if not (src_root / rel).exists()]
    if missing:
        print(
            f"Error: bundled plugin payload is incomplete ({len(missing)} of "
            f"{len(_PLUGIN_FILES)} files missing under {src_root}):",
            file=sys.stderr,
        )
        for rel in missing:
            print(f"  - {rel}", file=sys.stderr)
        print(
            "This build is broken. Install the published package "
            "(`pip install hindsight-cursor`) rather than a source checkout.",
            file=sys.stderr,
        )
        sys.exit(1)

    for rel in _PLUGIN_FILES:
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_root / rel, dst)


def _scaffold_config(api_url: str | None, api_token: str | None, bank_id: str) -> None:
    """Create ~/.hindsight/cursor.json if it does not already exist."""
    if _USER_CONFIG_FILE.exists():
        print(f"  Config already exists at {_USER_CONFIG_FILE}, skipping.")
        return

    config: dict = {"bankId": bank_id}
    if api_url:
        config["hindsightApiUrl"] = api_url
    if api_token:
        config["hindsightApiToken"] = api_token

    _USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _USER_CONFIG_FILE.write_text(json.dumps(config, indent=2) + "\n")
    print(f"  Created {_USER_CONFIG_FILE}")


def _setup_mcp(project: Path, api_url: str | None, api_token: str | None, bank_id: str) -> None:
    """Write .cursor/mcp.json to connect Cursor to Hindsight's MCP endpoint.

    Uses the single-bank MCP endpoint so that recall/retain/reflect tools
    are scoped to the configured bank without requiring a bank_id parameter.
    """
    if not api_url:
        return

    mcp_dir = project / ".cursor"
    mcp_file = mcp_dir / "mcp.json"

    # Build the MCP server URL — single-bank endpoint
    mcp_url = f"{api_url.rstrip('/')}/mcp/{bank_id}/"

    # Build the server config
    server_config: dict = {"url": mcp_url}
    if api_token:
        server_config["headers"] = {"Authorization": f"Bearer {api_token}"}

    # Merge with existing mcp.json if present
    existing: dict = {}
    if mcp_file.exists():
        try:
            existing = json.loads(mcp_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    servers = existing.get("mcpServers", {})
    servers["hindsight"] = server_config
    existing["mcpServers"] = servers

    mcp_dir.mkdir(parents=True, exist_ok=True)
    mcp_file.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"  MCP config written to {mcp_file}")


def _hindsight_hook_entry(definition: dict) -> bool:
    """True if a hooks.json entry belongs to this plugin install."""
    return _HOOK_MARKER in json.dumps(definition)


def _hook_interpreter() -> str:
    """Interpreter name to invoke the hook scripts with.

    Windows ships ``python.exe`` and has no ``python3`` on PATH, so a
    hardcoded ``python3`` makes every hook a silent no-op there. Cursor runs
    hooks through the shell, so a bare name (resolved off PATH) is what we
    want — not ``sys.executable``, which points at whatever interpreter
    happened to run ``init`` and may live in a throwaway virtualenv.
    """
    return "python" if sys.platform == "win32" else "python3"


def _project_hooks_block() -> dict:
    """Hooks Cursor loads from the project ``.cursor/hooks.json``.

    Paths are workspace-relative so they work without ``CURSOR_PLUGIN_ROOT``
    (which is only set for packages installed under ``~/.cursor/plugins``).

    ``retain.py`` is registered on both ``stop`` and ``sessionEnd``:

    * ``stop`` fires each time an agent loop ends — the granularity that
      periodic auto-retain wants.
    * ``sessionEnd`` fires once when the conversation ends, and carries the
      same ``transcript_path``. It is the final flush.

    Without the flush, every conversation shorter than ``retainEveryNTurns``
    is lost outright: at the default of 10, a seven-turn chat fires ``stop``
    seven times, the turn gate rejects all seven, and nothing is ever stored.

    Both events land in the same script; ``retain.py`` skips a run with no
    new messages since the last retain, so the overlap cannot double-store.
    """
    python = _hook_interpreter()
    return {
        "version": 1,
        "hooks": {
            "sessionStart": [
                {
                    "command": f"{python} {_HOOK_MARKER}/scripts/session_start.py",
                    "timeout": 15,
                }
            ],
            "stop": [
                {
                    "command": f"{python} {_HOOK_MARKER}/scripts/retain.py",
                    "timeout": 15,
                }
            ],
            "sessionEnd": [
                {
                    "command": f"{python} {_HOOK_MARKER}/scripts/retain.py",
                    "timeout": 15,
                }
            ],
        },
    }


def _setup_hooks(project: Path) -> None:
    """Write/merge project ``.cursor/hooks.json`` so Cursor registers the hooks.

    Dropping files under ``.cursor-plugin/`` alone does not register hooks with
    the IDE; Cursor only loads ``.cursor/hooks.json`` (workspace) or
    ``~/.cursor/hooks.json`` (user). Idempotent: existing Hindsight entries are
    replaced, other hooks are preserved.
    """
    cursor_dir = project / ".cursor"
    hooks_file = cursor_dir / "hooks.json"
    hooks_block = _project_hooks_block()

    existing: dict = {}
    if hooks_file.exists():
        try:
            existing = json.loads(hooks_file.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}

    existing.setdefault("version", hooks_block.get("version", 1))
    existing.setdefault("hooks", {})

    for event, definitions in hooks_block.get("hooks", {}).items():
        bucket = [d for d in existing["hooks"].get(event, []) if not _hindsight_hook_entry(d)]
        bucket.extend(definitions)
        existing["hooks"][event] = bucket

    cursor_dir.mkdir(parents=True, exist_ok=True)
    hooks_file.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"  Hooks registered in {hooks_file}")


def _remove_hooks(project: Path) -> None:
    """Strip this plugin's entries from project ``.cursor/hooks.json``."""
    hooks_file = project / ".cursor" / "hooks.json"
    if not hooks_file.exists():
        return
    try:
        data = json.loads(hooks_file.read_text())
    except (json.JSONDecodeError, OSError):
        return

    hooks = data.get("hooks", {})
    changed = False
    for event, definitions in list(hooks.items()):
        kept = [d for d in definitions if not _hindsight_hook_entry(d)]
        if len(kept) != len(definitions):
            changed = True
        if kept:
            hooks[event] = kept
        else:
            del hooks[event]

    if not changed:
        return

    if hooks:
        data["hooks"] = hooks
        hooks_file.write_text(json.dumps(data, indent=2) + "\n")
        print(f"  Removed hindsight hooks from {hooks_file}")
    else:
        other_keys = {k for k in data if k not in ("version", "hooks")}
        if other_keys:
            data["hooks"] = {}
            hooks_file.write_text(json.dumps(data, indent=2) + "\n")
            print(f"  Removed hindsight hooks from {hooks_file}")
        else:
            hooks_file.unlink()
            print(f"  Removed {hooks_file}")


def _remove_session_rules(project: Path) -> None:
    """Delete the generated session rules file and its .gitignore entry.

    ``sessionStart`` regenerates the rules file on every run, so leaving it
    behind after ``uninstall`` would keep injecting a dead session's memories
    into every agent turn — the file is ``alwaysApply: true``.
    """
    rules_file = project / _SESSION_RULES_RELPATH
    if rules_file.exists():
        try:
            rules_file.unlink()
            print(f"  Removed {rules_file}")
        except OSError:
            pass

    gitignore = project / ".gitignore"
    if not gitignore.exists():
        return
    try:
        lines = gitignore.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    kept = [
        line for line in lines if line.strip() != _GITIGNORE_LINE and not line.strip().startswith(_GITIGNORE_COMMENT)
    ]
    if len(kept) == len(lines):
        return

    # Drop a trailing blank run left behind by removing the block.
    while kept and not kept[-1].strip():
        kept.pop()

    try:
        gitignore.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
        print(f"  Removed hindsight entry from {gitignore}")
    except OSError:
        pass


def cmd_init(args: argparse.Namespace) -> None:
    """Install the Hindsight plugin into a Cursor project."""
    project = Path(args.project).resolve()
    if not project.is_dir():
        print(f"Error: {project} is not a directory.", file=sys.stderr)
        sys.exit(1)

    dest = project / ".cursor-plugin" / "hindsight-memory"
    if dest.exists() and not args.force:
        print(f"Plugin already installed at {dest}.")
        print("  Use --force to overwrite.")
        return

    print(f"Installing Hindsight plugin into {dest} ...")
    _copy_plugin(dest)
    print("  Plugin files copied.")

    # Register hooks with Cursor IDE (project .cursor/hooks.json).
    # The bundled hooks/hooks.json alone is not enough — Cursor only loads
    # workspace/user hooks.json, not files under .cursor-plugin/.
    _setup_hooks(project)

    _scaffold_config(args.api_url, args.api_token, args.bank_id)

    # Set up MCP for on-demand recall/retain/reflect tools
    if not args.no_mcp:
        _setup_mcp(project, args.api_url, args.api_token, args.bank_id)

    print()
    print("Done! Fully quit and reopen Cursor to activate the plugin.")


def cmd_uninstall(args: argparse.Namespace) -> None:
    """Remove the Hindsight plugin from a Cursor project."""
    project = Path(args.project).resolve()
    dest = project / ".cursor-plugin" / "hindsight-memory"
    if not dest.exists():
        print("Plugin not found — nothing to remove.")
        return
    shutil.rmtree(dest)
    print(f"Removed {dest}")

    # Clean up MCP config
    mcp_file = project / ".cursor" / "mcp.json"
    if mcp_file.exists():
        try:
            mcp_config = json.loads(mcp_file.read_text())
            servers = mcp_config.get("mcpServers", {})
            if "hindsight" in servers:
                del servers["hindsight"]
                if servers:
                    mcp_config["mcpServers"] = servers
                    mcp_file.write_text(json.dumps(mcp_config, indent=2) + "\n")
                    print(f"  Removed hindsight from {mcp_file}")
                else:
                    mcp_file.unlink()
                    print(f"  Removed {mcp_file}")
        except (json.JSONDecodeError, OSError):
            pass

    _remove_hooks(project)
    _remove_session_rules(project)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hindsight-cursor",
        description="Hindsight memory plugin for Cursor",
    )
    sub = parser.add_subparsers(dest="command")

    # -- init --
    init_p = sub.add_parser("init", help="Install the plugin into a Cursor project")
    init_p.add_argument(
        "project",
        nargs="?",
        default=".",
        help="Path to the Cursor project (default: current directory)",
    )
    init_p.add_argument("--force", action="store_true", help="Overwrite existing installation")
    init_p.add_argument("--api-url", default=None, help="Hindsight API URL (e.g. https://api.hindsight.vectorize.io)")
    init_p.add_argument("--api-token", default=None, help="Hindsight API token")
    init_p.add_argument("--bank-id", default="cursor", help="Memory bank ID (default: cursor)")
    init_p.add_argument("--no-mcp", action="store_true", help="Skip MCP integration setup")
    init_p.set_defaults(func=cmd_init)

    # -- uninstall --
    uninst_p = sub.add_parser("uninstall", help="Remove the plugin from a Cursor project")
    uninst_p.add_argument(
        "project",
        nargs="?",
        default=".",
        help="Path to the Cursor project (default: current directory)",
    )
    uninst_p.set_defaults(func=cmd_uninstall)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
