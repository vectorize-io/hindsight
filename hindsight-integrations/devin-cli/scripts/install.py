#!/usr/bin/env python3
"""Register this plugin's hooks and MCP server with Devin CLI.

Devin CLI's plugin system (as of CLI 3000.3.22) can install a plugin's skills,
rules, and subagents, and *tries* to wire up a plugin's hooks.json and
mcp_config.json too — but hook and MCP commands referencing bundled scripts
have no portable way to find their own install location: there's no
`${DEVIN_PLUGIN_ROOT}`-style substitution (unlike Claude Code's
`${CLAUDE_PLUGIN_ROOT}`), and relative paths in a plugin's own hooks.json /
mcp_config.json resolve against the *project* working directory, not the
plugin's install directory. (Verified empirically against CLI 3000.3.22 —
recheck this if retesting against a newer release fixes it, since at that
point this script becomes unnecessary and installation can move to a plain
`devin plugins install`.)

So instead of relying on that, this script writes absolute-path hook and MCP
entries directly into Devin CLI's documented, non-beta config locations:
  - ~/.config/devin/config.json          ("hooks" key)
  - ~/.config/devin/mcp_config.json      ("mcpServers" key)

Run this once after installing the plugin (or invoke the `setup` skill and
let the agent run it for you — see skills/setup/SKILL.md, which reports this
script's absolute path via the skill tool's own "Base directory" output).

Idempotent: safe to re-run after moving, upgrading, or reinstalling the
plugin. Entries for other tools are preserved; entries written by *any*
version of this script — including ones at a previous install path, which a
version bump always produces — are replaced rather than duplicated. See
`_OURS_RE` for how they are recognised.
"""

import contextlib
import json
import os
import re
import shlex
import sys
import tempfile

PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PLUGIN_ROOT, "scripts")

HOOK_SCRIPTS = ("session_start.py", "recall.py", "retain.py", "session_end.py")

# Recognises the exact command shape `_hook_command()` writes, at *any* install
# path — deliberately not just the current SCRIPTS_DIR. Devin CLI caches
# plugins under `.../plugins/cache/<hash>/hindsight-memory/<version>/`, so every
# version bump relocates the tree. Matching only the current directory would
# leave the previous version's entries in place, and since that path no longer
# exists, each lifecycle event would spawn a failing process for the life of
# the install. The backreference keeps this tight: both halves of the `||`
# fallback must name the same file, so a user's own hook that merely happens to
# run a script called `recall.py` is not swept up.
#
# The path is optionally single-quoted because `_hook_command()` runs it through
# shlex.quote(), which leaves ordinary paths bare and quotes only those needing it.
_OURS_RE = re.compile(
    r"^python3 (?P<path>'?.*[/\\]scripts[/\\](?:"
    + "|".join(re.escape(name) for name in HOOK_SCRIPTS)
    + r")'?) \|\| python (?P=path)$"
)

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".config", "devin")
if sys.platform == "win32":
    CONFIG_DIR = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "devin")

CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")
MCP_CONFIG_PATH = os.path.join(CONFIG_DIR, "mcp_config.json")


def _script(name: str) -> str:
    return os.path.join(SCRIPTS_DIR, name)


def _hook_command(script_name: str) -> str:
    # Devin CLI runs a hook's `command` through a shell, and the plugin cache path
    # is not ours to choose — a `$`, backtick, or quote anywhere in it would be
    # expanded or misparsed, breaking every lifecycle hook. shlex.quote() makes the
    # path inert; it returns ordinary paths unchanged, so the common case stays
    # readable in the user's config file.
    path = shlex.quote(_script(script_name))
    return f"python3 {path} || python {path}"


def _is_our_command(hook) -> bool:
    """True if a single hook object's command was written by this script.

    Type-checked at every level because every level is user-authored JSON, and
    the answer for a shape we do not recognise is "not ours" — which preserves
    it, the safe direction for a file we do not own.
    """
    return isinstance(hook, dict) and isinstance(hook.get("command"), str) and bool(_OURS_RE.match(hook["command"]))


def _is_ours(hook_entry) -> bool:
    """True if a hooks.json entry contains a command written by this script.

    Matches entries written by any version of this plugin, not only the one
    currently installed — see `_OURS_RE`.
    """
    if not isinstance(hook_entry, dict):
        return False
    hooks = hook_entry.get("hooks")
    return isinstance(hooks, list) and any(_is_our_command(h) for h in hooks)


# Returned by _strip_our_hooks() for an entry that should be dropped entirely.
# A distinct object rather than None, because `None` is itself a hook entry a
# user's config can contain, and returning it for "drop this" would delete it.
DROP_ENTRY = object()


def _strip_our_hooks(hook_entry):
    """Drop this plugin's commands from one hooks.json entry, keeping the rest.

    Returns the entry to keep, or DROP_ENTRY when nothing of it survives.

    An entry holds a *list* of commands under a single matcher, and nothing
    stops a user from putting ours in the same list as their own. Reinstalling
    used to drop any entry that mentioned us at all, which deleted those
    third-party hooks silently and permanently — the file is the user's, and
    they never asked us to edit anything but our own lines. Entries we do not
    recognise come back untouched.
    """
    if not _is_ours(hook_entry):
        return hook_entry
    # _is_ours() returning True guarantees both isinstance checks below.
    remaining = [h for h in hook_entry["hooks"] if not _is_our_command(h)]
    if not remaining:
        return DROP_ENTRY
    # Rebuilt rather than mutated: `previous` is the caller's parsed config, and
    # a run that ends up not writing must leave it as it found it.
    return {**hook_entry, "hooks": remaining}


def build_hooks() -> dict:
    return {
        "SessionStart": [{"hooks": [{"type": "command", "command": _hook_command("session_start.py"), "timeout": 5}]}],
        "UserPromptSubmit": [{"hooks": [{"type": "command", "command": _hook_command("recall.py"), "timeout": 12}]}],
        "Stop": [{"hooks": [{"type": "command", "command": _hook_command("retain.py"), "timeout": 15}]}],
        "SessionEnd": [{"hooks": [{"type": "command", "command": _hook_command("session_end.py"), "timeout": 10}]}],
    }


def _load_json(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f) or {}
    # UnicodeDecodeError subclasses ValueError, so invalid UTF-8 in an existing
    # config would otherwise abort the installer with a raw traceback.
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        print(f"Warning: {path} is not valid JSON ({e}); refusing to overwrite it.", file=sys.stderr)
        sys.exit(1)


def _write_json(path: str, data: dict) -> None:
    """Write `data` to `path` atomically.

    Staged under a per-process name, for the same reason lib/state.py's
    write_state() is: a fixed `<path>.tmp` lets a second installer os.replace()
    the file the first is still filling, and both write the shared Devin CLI
    config. Two `install.py` runs are rarer than two hooks, but they land on a
    file the user did not ask us to corrupt — and a run that dies mid-write
    leaves the fixed name behind as debris either way.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, path)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def install_hooks() -> None:
    config = _load_json(CONFIG_PATH)
    existing_hooks = config.get("hooks")
    new_hooks = build_hooks()

    # A config.json holding `{"hooks": null}` is valid JSON the user may well
    # have written, and dict(None) raises TypeError — so the installer died on
    # an unhandled traceback rather than doing its job. Reported and replaced,
    # the same way install_mcp_server() handles a non-object "mcpServers":
    # there is nothing to merge into, and refusing outright would leave the
    # plugin uninstallable with no way forward.
    if existing_hooks is not None and not isinstance(existing_hooks, dict):
        print(
            f"Warning: {CONFIG_PATH} has a non-object 'hooks'; replacing it with this plugin's entries.",
            file=sys.stderr,
        )
        existing_hooks = {}
    existing_hooks = existing_hooks or {}

    merged = dict(existing_hooks)
    for event, entries in new_hooks.items():
        # Per-event too: `{"hooks": {"SessionStart": null}}` is the same class
        # of shape one level down, and iterating it raises just as hard.
        previous = existing_hooks.get(event)
        if not isinstance(previous, list):
            previous = []
        kept = [stripped for stripped in map(_strip_our_hooks, previous) if stripped is not DROP_ENTRY]
        merged[event] = kept + entries

    config["hooks"] = merged
    _write_json(CONFIG_PATH, config)
    print(f"Hooks registered in {CONFIG_PATH}")


def _is_our_mcp_server(entry) -> bool:
    """True if an mcpServers entry points at this plugin's launcher.

    Path-insensitive on purpose, like _is_ours(): an upgrade installs from a new
    directory, and the entry it is replacing is still its own.
    """
    if not isinstance(entry, dict):
        return False
    args = entry.get("args")
    if not isinstance(args, list):
        return False
    return any(isinstance(a, str) and os.path.basename(a) == "run_mcp.sh" for a in args)


def install_mcp_server(force: bool = False) -> bool:
    """Register the MCP server. Returns False if it declined to write.

    `force` overwrites a "hindsight" entry this plugin did not write.
    """
    mcp_config = _load_json(MCP_CONFIG_PATH)
    servers = mcp_config.get("mcpServers")
    # An mcp_config.json whose `mcpServers` is not an object is user-authored
    # JSON we cannot merge into. Starting fresh would delete whatever is there,
    # so say so and leave the file alone.
    if servers is not None and not isinstance(servers, dict):
        print(
            f"Warning: {MCP_CONFIG_PATH} has a non-object 'mcpServers'; "
            f"not registering the MCP server. Fix or remove it and re-run.",
            file=sys.stderr,
        )
        return False
    servers = dict(servers or {})

    # Replacing our own entry is the upgrade path and happens silently.
    # Replacing someone else's is destroying configuration this installer does
    # not own — "hindsight" is a plausible name for a hand-configured server —
    # and a warning printed mid-install is not consent: it scrolls past, and
    # the value it names is already gone. Declining leaves the user a choice.
    existing = servers.get("hindsight")
    if existing is not None and not _is_our_mcp_server(existing) and not force:
        print(
            f"Refusing to replace the existing 'hindsight' MCP server in {MCP_CONFIG_PATH}, "
            f"which this plugin did not write:\n"
            f"  {json.dumps(existing)}\n"
            f"Rename it, or re-run this script with --force to replace it.",
            file=sys.stderr,
        )
        return False

    servers["hindsight"] = {
        "command": "bash",
        "args": [_script("run_mcp.sh")],
    }
    mcp_config["mcpServers"] = servers
    _write_json(MCP_CONFIG_PATH, mcp_config)
    print(f"MCP server registered in {MCP_CONFIG_PATH}")
    return True


def main():
    force = "--force" in sys.argv[1:]
    install_hooks()
    registered = install_mcp_server(force=force)
    print()
    print("hindsight-memory is set up. Start a new Devin CLI session (or")
    print("resume one — hooks/MCP servers are loaded at session start) for")
    print("it to take effect. Configure it at ~/.hindsight/devin-cli.json —")
    print("see README.md for the full settings reference.")
    if not registered:
        # Hooks are what carry recall and retain, so the plugin still works
        # without the MCP server — but its tools are absent, which is worth
        # more than a line that already scrolled past above.
        print()
        print("Note: the MCP server was NOT registered (see the warning above).")
        print("Recall and retain still work; the hindsight MCP tools do not.")


if __name__ == "__main__":
    main()
