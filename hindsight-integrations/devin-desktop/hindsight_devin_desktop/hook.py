"""Devin Local ``SessionStart`` hook — deterministic auto-recall.

Devin runs this at the start of every session. It reads the connection details
from Devin Local's ``config.json`` (the same ``mcpServers.hindsight`` entry
``init`` wrote), derives the project bank from ``DEVIN_PROJECT_DIR``, recalls a
baseline of project + user memory from Hindsight, and prints it as
``additionalContext`` — which Devin injects into the agent's context **before the
model acts**. So memory is loaded whether or not the model remembers to call
``recall``.

Contract (Devin CLI hooks): event JSON arrives on stdin; we return
``{"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ...}}``
on stdout. The hook must never break a session, so *any* error → exit 0 with no
output. Config-only and dependency-free (stdlib ``urllib`` for the MCP call).
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

# Broad session-start prime; per-turn relevance still comes from the model's own
# recall calls guided by the rule.
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
    """Recall from one bank via MCP; return memory texts (best-effort, may be empty)."""
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


def _format_context(project_bank: str, project_mems: List[str], global_bank: str, global_mems: List[str]) -> str:
    parts = ["Relevant long-term memory (Hindsight) — use what applies, ignore the rest:"]
    if project_mems:
        parts.append(f"\nThis project ({project_bank}):")
        parts += [f"- {m}" for m in project_mems]
    if global_mems:
        parts.append(f"\nAbout the user ({global_bank}):")
        parts += [f"- {m}" for m in global_mems]
    return "\n".join(parts)


def _emit(additional_context: str) -> None:
    sys.stdout.write(
        json.dumps(
            {"hookSpecificOutput": {"hookEventName": devin_local.HOOK_EVENT, "additionalContext": additional_context}}
        )
    )


def cmd_recall() -> int:
    config = devin_local.default_config_path()
    if not config.is_file():
        return 0
    try:
        data = json.loads(config.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    url, token, global_bank = _extract_connection(data)
    if not url or not global_bank:
        return 0

    project_dir = Path(os.environ.get("DEVIN_PROJECT_DIR") or ".")
    project_bank, _ = project_bank_id(global_bank, project_dir)

    global_mems = _recall(url, token, global_bank)
    project_mems = _recall(url, token, project_bank) if project_bank and project_bank != global_bank else []
    if not project_mems and not global_mems:
        return 0
    _emit(_format_context(project_bank or global_bank, project_mems, global_bank, global_mems))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Consume Devin's event JSON on stdin (we don't need it for SessionStart recall).
    try:
        if not sys.stdin.isatty():
            sys.stdin.read()
    except Exception:
        pass
    if argv and argv[0] == "recall":
        try:
            return cmd_recall()
        except Exception:
            # Never break a session — swallow everything.
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
