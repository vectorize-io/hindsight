"""Cascade visibility banner via a ``post_mcp_tool_use`` hook.

Cascade renders a hook's stdout in its UI when ``show_output: true``. We register
a ``post_mcp_tool_use`` command hook that prints ``🧠 Hindsight: <tool> used`` so
every recall/retain is visibly obvious. Cascade has no per-hook matcher, so the
hook fires for all MCP tools and filters to the hindsight server in-script (see
:func:`hindsight_devin_desktop.hook.cmd_banner`).

Hooks live in ``~/.codeium/windsurf/hooks.json`` (same path on every OS for the
Devin Desktop / ex-Windsurf user tier). The file is shared with the user's own
hooks, so we manage only our single entry — identified by our command marker —
and never clobber theirs. We only rewrite the file when it parses as strict JSON.

(Devin Local uses a different, richer hook system — see
:mod:`hindsight_devin_desktop.devin_local`.)
"""

from __future__ import annotations

import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

EVENT = "post_mcp_tool_use"
HOOK_MODULE = "hindsight_devin_desktop.hook"
_MARKER = f"{HOOK_MODULE} banner"


def default_hooks_path() -> Path:
    """Cascade's user hooks file (``~/.codeium/windsurf/hooks.json``)."""
    return Path.home() / ".codeium" / "windsurf" / "hooks.json"


def banner_command() -> str:
    return f"{shlex.quote(sys.executable)} -m {HOOK_MODULE} banner"


def _our_entry() -> dict[str, Any]:
    cmd = banner_command()
    # `command` = bash (macOS/Linux), `powershell` = Windows; same invocation works in both.
    return {"command": cmd, "powershell": cmd, "show_output": True}


def _is_ours(entry: Any) -> bool:
    return isinstance(entry, dict) and _MARKER in (entry.get("command", "") + entry.get("powershell", ""))


@dataclass
class BannerResult:
    action: str  # created / merged / unchanged / removed / manual
    path: Path


def _load_strict(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def apply_banner(path: Path) -> BannerResult:
    """Add our ``post_mcp_tool_use`` banner hook to ``hooks.json`` (idempotent)."""
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"hooks": {EVENT: [_our_entry()]}}, indent=2) + "\n", encoding="utf-8")
        return BannerResult("created", path)

    data = _load_strict(path)
    if data is None:
        return BannerResult("manual", path)

    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        hooks = {}
    entries = hooks.get(EVENT)
    if not isinstance(entries, list):
        entries = []
    if any(_is_ours(e) for e in entries):
        return BannerResult("unchanged", path)
    entries.append(_our_entry())
    hooks[EVENT] = entries
    data["hooks"] = hooks
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return BannerResult("merged", path)


def remove_banner(path: Path) -> BannerResult:
    """Remove our banner hook from ``hooks.json``, preserving the user's own."""
    data = _load_strict(path)
    if data is None:
        return BannerResult("manual" if path.is_file() else "unchanged", path)

    hooks = data.get("hooks")
    entries = hooks.get(EVENT) if isinstance(hooks, dict) else None
    if not isinstance(entries, list) or not any(_is_ours(e) for e in entries):
        return BannerResult("unchanged", path)

    kept = [e for e in entries if not _is_ours(e)]
    if kept:
        hooks[EVENT] = kept
    else:
        hooks.pop(EVENT, None)
    if not hooks:
        data.pop("hooks", None)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return BannerResult("removed", path)


def is_installed(path: Path) -> bool:
    """Whether our banner hook is present in ``hooks.json``."""
    data = _load_strict(path)
    hooks = data.get("hooks") if data else None
    entries = hooks.get(EVENT) if isinstance(hooks, dict) else None
    return isinstance(entries, list) and any(_is_ours(e) for e in entries)
