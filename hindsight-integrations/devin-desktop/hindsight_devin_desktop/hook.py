"""Devin Local hooks — deterministic recall (SessionStart) and retain (Stop).

Two subcommands, both driven by Devin's hook system:

* ``recall`` (``SessionStart``): reads the connection from Devin Local's
  ``config.json``, derives the project bank from ``DEVIN_PROJECT_DIR``, recalls a
  baseline of project + user memory, and prints it as ``additionalContext`` —
  which Devin injects into the agent's context before the model acts. It
  **always reports status** (loaded / empty / unavailable) so memory use is never
  silent, and never exits non-zero (never breaks a session).

* ``retain-nudge`` (``Stop``): before the agent stops, returns
  ``{"decision":"block","reason":...}`` to force one retain pass — the model
  decides *what* is durable and calls the ``retain`` tool. Loop-guarded via
  ``stop_hook_active``. This makes retain deterministically *triggered* (Devin's
  hooks can't hand a transcript to a script, so the model must author the
  content).

Config-only and dependency-free (stdlib ``urllib`` for the MCP recall).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

from . import devin_local
from .project import project_bank_id

RECALL_QUERY = "Key architecture, decisions, conventions, and the user's preferences and coding style for this work."
MAX_TOKENS = 1024


def _extract_connection(config_data: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return ``(url, token, global_bank)`` from a Devin config's hindsight entry."""
    server = (config_data.get("mcpServers") or {}).get(devin_local.SERVER_NAME) or {}
    url = server.get("url")
    headers = server.get("headers") or {}
    auth = headers.get("Authorization") or ""
    token = auth[len("Bearer ") :].strip() if auth.startswith("Bearer ") else None
    return url, token, headers.get("X-Bank-Id")


def _resolve() -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Read config + env → ``(url, token, global_bank, project_bank)`` (any may be None)."""
    config = devin_local.default_config_path()
    if not config.is_file():
        return None, None, None, None
    try:
        data = json.loads(config.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, None, None, None
    url, token, global_bank = _extract_connection(data)
    if not global_bank:
        return url, token, None, None
    project_bank, _ = project_bank_id(global_bank, Path(os.environ.get("DEVIN_PROJECT_DIR") or "."))
    return url, token, global_bank, project_bank


def _mcp(url: str, token: Optional[str], payload: dict, session: Optional[str]) -> Tuple[Optional[str], Optional[dict]]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json, text/event-stream")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if session:
        req.add_header("Mcp-Session-Id", session)
    with urllib.request.urlopen(req, timeout=8) as r:
        body = r.read().decode().strip()
        ctype = r.headers.get("Content-Type") or ""
        data = None
        if body:
            if "text/event-stream" in ctype:
                data = json.loads("".join(ln[5:].strip() for ln in body.splitlines() if ln.startswith("data:")))
            else:
                data = json.loads(body)
        return r.headers.get("Mcp-Session-Id"), data


def _recall(url: str, token: Optional[str], bank: str) -> List[str]:
    """Recall from one bank via MCP; return memory texts. Raises on transport error."""
    sid, _ = _mcp(
        url,
        token,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "hook", "version": "0"},
            },
        },
        None,
    )
    try:
        _mcp(url, token, {"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)
    except Exception:
        pass
    _, res = _mcp(
        url,
        token,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "recall",
                "arguments": {"query": RECALL_QUERY, "bank_id": bank, "max_tokens": MAX_TOKENS},
            },
        },
        sid,
    )
    if not res or "result" not in res:
        return []
    text = "\n".join(c.get("text", "") for c in res["result"].get("content", []))
    try:
        results = json.loads(text).get("results", [])
        return [m.get("text", "") for m in results if m.get("text")]
    except (json.JSONDecodeError, AttributeError):
        return []


def _context_loaded(project_bank: str, project_mems: List[str], global_bank: str, global_mems: List[str]) -> str:
    n, m = len(project_mems), len(global_mems)
    parts = [
        f"🧠 Hindsight memory loaded ({n} project, {m} user). "
        "Briefly tell the user you recalled from Hindsight, then use what's relevant:"
    ]
    if project_mems:
        parts.append(f"\nThis project ({project_bank}):")
        parts += [f"- {x}" for x in project_mems]
    if global_mems:
        parts.append(f"\nAbout the user ({global_bank}):")
        parts += [f"- {x}" for x in global_mems]
    return "\n".join(parts)


def _context_empty() -> str:
    return "🧠 Hindsight: no stored memory for this project or user yet. Tell the user memory is empty so far."


def _context_error(reason: str) -> str:
    return f"⚠️ Hindsight: could not load memory this session ({reason}). Tell the user memory is unavailable right now."


def _emit_context(additional_context: str) -> None:
    sys.stdout.write(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": devin_local.RECALL_EVENT,
                    "additionalContext": additional_context,
                }
            }
        )
    )


def cmd_recall() -> int:
    url, token, global_bank, project_bank = _resolve()
    if not url or not global_bank:
        return 0  # not configured for Devin Local — nothing to report
    try:
        global_mems = _recall(url, token, global_bank)
        project_mems = _recall(url, token, project_bank) if project_bank and project_bank != global_bank else []
    except Exception as e:  # unreachable / timeout / bad response → report, never silent
        _emit_context(_context_error(type(e).__name__))
        return 0
    if project_mems or global_mems:
        _emit_context(_context_loaded(project_bank or global_bank, project_mems, global_bank, global_mems))
    else:
        _emit_context(_context_empty())
    return 0


def cmd_retain_nudge(event: dict) -> int:
    if event.get("stop_hook_active"):
        return 0  # we already nudged this turn — allow the stop, don't loop
    _, _, global_bank, project_bank = _resolve()
    if not global_bank:
        return 0
    reason = (
        "Before you stop: if you learned anything durable this session, retain it now via the hindsight "
        f'`retain` tool — PROJECT facts (architecture, decisions, conventions) with bank_id "{project_bank}", '
        f'USER facts (preferences, style, identity) with bank_id "{global_bank}". Briefly tell the user what '
        "you saved. If nothing is worth remembering, just say so and stop."
    )
    sys.stdout.write(json.dumps({"decision": "block", "reason": reason}))
    return 0


def cmd_banner(event: dict) -> int:
    """Cascade ``post_mcp_tool_use`` banner: print a visible line for hindsight tools.

    Cascade has no per-hook matcher, so this fires for every MCP tool; we filter
    to the hindsight server in-script and stay silent otherwise.
    """
    info = event.get("tool_info") or {}
    if info.get("mcp_server_name") == devin_local.SERVER_NAME:
        tool = info.get("mcp_tool_name") or "memory"
        sys.stdout.write(f"🧠 Hindsight: {tool} used")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    event: dict = {}
    try:
        if not sys.stdin.isatty():
            raw = sys.stdin.read()
            if raw.strip():
                event = json.loads(raw)
    except Exception:
        event = {}
    if not isinstance(event, dict):
        event = {}
    cmd = argv[0] if argv else ""
    try:
        if cmd == "recall":
            return cmd_recall()
        if cmd == "retain-nudge":
            return cmd_retain_nudge(event)
        if cmd == "banner":
            return cmd_banner(event)
    except Exception:
        # Last resort only — never break a session.
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
