#!/usr/bin/env python3
"""Register hindsight-memory hooks into ~/.claude/settings.json.

Claude Code's plugin installer does not currently merge hooks.json into
settings.json automatically. Run this script once after installing the plugin:

    python3 setup_hooks.py

Or via the /hindsight:setup skill inside a Claude Code session.
"""

import json
import os
import sys

SETTINGS_PATH = os.path.expanduser("~/.claude/settings.json")
USER_CONFIG_PATH = os.path.expanduser("~/.hindsight/claude-code.json")

# Default hook timeout for UserPromptSubmit (auto-recall).
# Raised from 12s to 45s — the hooks.json shipped with the plugin also
# uses this floor, but setup_hooks.py can tighten it further when the
# user has set a lower requestTimeoutSeconds (see build_hooks).
HOOK_RECALL_TIMEOUT_DEFAULT = 45
HOOK_RECALL_TIMEOUT_MIN = 30


def _read_request_timeout() -> int | None:
    """Return user-configured requestTimeoutSeconds, or None if unavailable."""
    if not os.path.isfile(USER_CONFIG_PATH):
        return None
    try:
        with open(USER_CONFIG_PATH) as f:
            cfg = json.load(f)
        value = cfg.get("requestTimeoutSeconds")
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _recall_timeout() -> int:
    """Compute the recall hook timeout.

    When the user has set requestTimeoutSeconds, the hook timeout is that
    value plus a 15s margin so the hook process is never killed before the
    MCP request it wraps has a chance to complete (see #2854).

    Otherwise a safe default is used.
    """
    req_timeout = _read_request_timeout()
    if req_timeout is not None:
        return max(req_timeout + 15, HOOK_RECALL_TIMEOUT_MIN)
    return HOOK_RECALL_TIMEOUT_DEFAULT


def find_plugin_root() -> str:
    """Locate the installed hindsight-memory plugin cache directory."""
    cache_base = os.path.expanduser("~/.claude/plugins/cache/hindsight/hindsight-memory")
    if not os.path.isdir(cache_base):
        raise RuntimeError(
            "hindsight-memory plugin not found. "
            "Run /plugin install hindsight-memory inside Claude Code first."
        )
    versions = sorted(os.listdir(cache_base), reverse=True)
    if not versions:
        raise RuntimeError(f"No versions found in {cache_base}")
    return os.path.join(cache_base, versions[0])


def build_hooks(plugin_root: str) -> dict:
    recall_timeout = _recall_timeout()
    return {
        "UserPromptSubmit": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f'python3 "{plugin_root}/scripts/recall.py"',
                        "timeout": recall_timeout,
                    }
                ]
            }
        ],
        "Stop": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f'python3 "{plugin_root}/scripts/retain.py"',
                        "timeout": 15,
                        "async": True,
                    }
                ]
            }
        ],
        "SessionStart": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f'python3 "{plugin_root}/scripts/session_start.py"',
                        "timeout": 5,
                    }
                ]
            }
        ],
        "SessionEnd": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f'python3 "{plugin_root}/scripts/session_end.py"',
                        "timeout": 10,
                    }
                ]
            }
        ],
    }


def main():
    try:
        plugin_root = find_plugin_root()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(SETTINGS_PATH):
        print(f"Error: {SETTINGS_PATH} not found. Is Claude Code installed?", file=sys.stderr)
        sys.exit(1)

    with open(SETTINGS_PATH) as f:
        settings = json.load(f)

    existing_hooks = settings.get("hooks", {})
    if existing_hooks:
        print("Existing hooks found — merging (hindsight-memory hooks take precedence).")

    new_hooks = build_hooks(plugin_root)
    merged = {**existing_hooks, **new_hooks}
    settings["hooks"] = merged
    settings.setdefault("env", {})["CLAUDE_PLUGIN_ROOT"] = plugin_root

    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)

    print(f"hindsight-memory hooks registered successfully.")
    print(f"Plugin root: {plugin_root}")
    print(f"Restart Claude Code for changes to take effect.")


if __name__ == "__main__":
    main()
