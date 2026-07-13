"""Wire Hindsight into the **Devin Local** agent (distinct from Cascade).

Devin Desktop runs two agents with *separate* configuration:

* **Cascade** (legacy) — MCP in ``~/.codeium/windsurf/mcp_config.json`` (field
  ``serverUrl``), rules in ``.devin/rules/`` + ``…/memories/global_rules.md``
  (see :mod:`hindsight_devin_desktop.mcp_config` / ``rules`` / ``global_rules``).
* **Devin Local** (the successor agent, shared with the Devin CLI) — this module.

For Devin Local:

* MCP servers live in ``~/.config/devin/config.json`` under ``mcpServers``, but
  the remote-server schema differs from Cascade: ``url`` + ``transport: "http"``
  + ``headers`` (not ``serverUrl``). We preserve any other keys (e.g. ``version``).
* Devin Local **prompts before every MCP tool call** by default, so we pre-seed a
  ``permissions.allow`` entry (``mcp__hindsight__*``) so recall/retain run
  automatically.
* Always-on instructions use **``AGENTS.md``** files (plain Markdown, no
  frontmatter) — Devin Local does *not* read Cascade's ``.devin/rules/``. Global:
  ``~/.config/devin/AGENTS.md``; per-project: the repo-root ``AGENTS.md``. We
  manage only a fenced block in each (see :mod:`managed_block`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from . import managed_block
from .mcp_config import mcp_endpoint_url
from .rules import global_rule_body, render_rule_text

SERVER_NAME = "hindsight"
# Devin Local permission pattern that auto-approves Hindsight's MCP tools.
ALLOW_PATTERN = "mcp__hindsight__*"


def default_config_path() -> Path:
    """Devin Local's user config (``~/.config/devin/config.json``)."""
    return Path.home() / ".config" / "devin" / "config.json"


def default_global_agents_path() -> Path:
    """Devin Local's global always-on instructions (``~/.config/devin/AGENTS.md``)."""
    return Path.home() / ".config" / "devin" / "AGENTS.md"


def default_project_agents_path() -> Path:
    """The repo-root ``AGENTS.md`` (per-project always-on instructions)."""
    return Path.cwd() / "AGENTS.md"


def build_http_server(api_url: str, api_token: Optional[str], default_bank: Optional[str]) -> dict[str, Any]:
    """Build the Devin Local ``mcpServers.hindsight`` entry.

    Multi-bank Streamable-HTTP endpoint via ``url`` + ``transport: "http"``, with
    a Bearer auth header when a token is set and an ``X-Bank-Id`` header naming
    ``default_bank`` as the fallback bank for calls that omit ``bank_id``.
    """
    server: dict[str, Any] = {"url": mcp_endpoint_url(api_url), "transport": "http"}
    headers: dict[str, str] = {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    if default_bank:
        headers["X-Bank-Id"] = default_bank
    if headers:
        server["headers"] = headers
    return server


def render_snippet(server: dict[str, Any]) -> str:
    """Render the config snippet a user can paste into ``config.json``."""
    return json.dumps(
        {"mcpServers": {SERVER_NAME: server}, "permissions": {"allow": [ALLOW_PATTERN]}},
        indent=2,
    )


@dataclass
class ConfigResult:
    """Outcome of editing ``config.json``.

    ``action`` is ``created``/``merged``/``unchanged``/``removed``/``manual``
    (file isn't strict JSON — ``snippet`` holds what to paste).
    """

    action: str
    path: Path
    snippet: Optional[str] = None


def _load_strict(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _allow_present(data: dict[str, Any]) -> bool:
    perms = data.get("permissions")
    allow = perms.get("allow") if isinstance(perms, dict) else None
    return isinstance(allow, list) and ALLOW_PATTERN in allow


def _ensure_allow(data: dict[str, Any]) -> None:
    perms = data.get("permissions")
    if not isinstance(perms, dict):
        perms = {}
    allow = perms.get("allow")
    if not isinstance(allow, list):
        allow = []
    if ALLOW_PATTERN not in allow:
        allow.append(ALLOW_PATTERN)
    perms["allow"] = allow
    data["permissions"] = perms


def _remove_allow(data: dict[str, Any]) -> None:
    perms = data.get("permissions")
    if not isinstance(perms, dict):
        return
    allow = perms.get("allow")
    if isinstance(allow, list) and ALLOW_PATTERN in allow:
        allow = [p for p in allow if p != ALLOW_PATTERN]
        if allow:
            perms["allow"] = allow
        else:
            perms.pop("allow", None)
    if not perms:
        data.pop("permissions", None)


def apply_to_config(path: Path, server: dict[str, Any]) -> ConfigResult:
    """Add/update ``mcpServers.hindsight`` + the allow-rule in ``config.json``.

    Preserves all other keys (e.g. ``version``). Adds ``permissions.allow`` for
    ``mcp__hindsight__*`` so the tools don't prompt on every call.
    """
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {"mcpServers": {SERVER_NAME: server}}
        _ensure_allow(data)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return ConfigResult("created", path)

    data = _load_strict(path)
    if data is None:
        return ConfigResult("manual", path, snippet=render_snippet(server))

    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    if servers.get(SERVER_NAME) == server and _allow_present(data):
        return ConfigResult("unchanged", path)
    servers[SERVER_NAME] = server
    data["mcpServers"] = servers
    _ensure_allow(data)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return ConfigResult("merged", path)


def remove_from_config(path: Path) -> ConfigResult:
    """Remove ``mcpServers.hindsight`` + our allow-rule from ``config.json``."""
    data = _load_strict(path)
    if data is None:
        return ConfigResult("manual" if path.is_file() else "unchanged", path)

    servers = data.get("mcpServers")
    present = isinstance(servers, dict) and SERVER_NAME in servers
    if not present and not _allow_present(data):
        return ConfigResult("unchanged", path)

    if present:
        del servers[SERVER_NAME]
        if servers:
            data["mcpServers"] = servers
        else:
            data.pop("mcpServers", None)
    _remove_allow(data)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return ConfigResult("removed", path)


def is_installed(path: Path) -> bool:
    """Whether our server is present in Devin Local's ``config.json``."""
    data = _load_strict(path)
    if data is None:
        return False
    servers = data.get("mcpServers")
    return isinstance(servers, dict) and SERVER_NAME in servers


# ── AGENTS.md always-on instructions (global + per-project) ─────────────────


def write_global_agents(path: Path, global_bank: str) -> str:
    """Write/replace our block in the global ``AGENTS.md``; returns the action."""
    action, _ = managed_block.upsert(path, global_rule_body(global_bank))
    return action


def write_project_agents(path: Path, project_bank: str, global_bank: str) -> str:
    """Write/replace our block in the repo-root ``AGENTS.md``; returns the action."""
    action, _ = managed_block.upsert(path, render_rule_text(project_bank, global_bank))
    return action


def clear_agents(path: Path) -> None:
    """Remove our managed block from an ``AGENTS.md`` (delete file if empty)."""
    managed_block.clear(path)


def agents_installed(path: Path) -> bool:
    """Whether our managed block is present in an ``AGENTS.md``."""
    return managed_block.has(path)
