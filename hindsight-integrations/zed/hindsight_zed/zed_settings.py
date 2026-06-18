"""Wire Hindsight into Zed's MCP ``context_servers`` block.

Zed has no native HTTP-MCP transport yet, so we connect to Hindsight's HTTP MCP
endpoint through the ``mcp-remote`` stdio bridge (run via ``npx``). The server
is registered under ``context_servers.hindsight`` in Zed's ``settings.json``.

Zed's ``settings.json`` is JSONC (it allows comments and trailing commas), which
the stdlib JSON parser can't round-trip without dropping the user's comments. So
we only edit the file in place when it parses cleanly as strict JSON; otherwise
we return the exact snippet for the user to paste, never risking their config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

SERVER_NAME = "hindsight"


def default_settings_path() -> Path:
    """Zed's user ``settings.json`` (``~/.config/zed`` on macOS and Linux)."""
    return Path.home() / ".config" / "zed" / "settings.json"


def mcp_endpoint_url(api_url: str, bank_id: str) -> str:
    """The Hindsight MCP endpoint for a bank (bank is the last path segment)."""
    return f"{api_url.rstrip('/')}/mcp/{bank_id}/"


def build_context_server(api_url: str, api_token: Optional[str], bank_id: str) -> dict[str, Any]:
    """Build the ``context_servers.hindsight`` entry for Zed's settings.

    Returns the Zed settings JSON object for the server: an ``mcp-remote`` bridge
    to the Hindsight MCP endpoint, with a Bearer auth header when a token is set
    (omitted for an open self-hosted server).
    """
    args = ["-y", "mcp-remote", mcp_endpoint_url(api_url, bank_id)]
    if api_token:
        args += ["--header", f"Authorization: Bearer {api_token}"]
    return {"source": "custom", "command": "npx", "args": args}


def render_snippet(server: dict[str, Any]) -> str:
    """Render the settings snippet the user can paste into ``settings.json``."""
    return json.dumps({"context_servers": {SERVER_NAME: server}}, indent=2)


@dataclass
class SettingsResult:
    """Outcome of editing Zed's settings file.

    ``action`` is one of ``created`` (new file written), ``merged`` (our entry
    written into existing JSON), ``removed`` (our entry deleted), ``unchanged``
    (nothing to do), or ``manual`` (file is JSONC we won't rewrite — ``snippet``
    holds what to paste).
    """

    action: str
    path: Path
    snippet: Optional[str] = None


def _load_strict(path: Path) -> Optional[dict[str, Any]]:
    """Parse ``path`` as strict JSON, or return ``None`` if absent/not strict."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def apply_to_settings(path: Path, server: dict[str, Any]) -> SettingsResult:
    """Add/update ``context_servers.hindsight`` in Zed's settings at ``path``."""
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"context_servers": {SERVER_NAME: server}}, indent=2) + "\n", encoding="utf-8")
        return SettingsResult("created", path)

    data = _load_strict(path)
    if data is None:
        # JSONC (comments/trailing commas) or unreadable — don't risk a rewrite.
        return SettingsResult("manual", path, snippet=render_snippet(server))

    servers = data.get("context_servers")
    if not isinstance(servers, dict):
        servers = {}
    if servers.get(SERVER_NAME) == server:
        return SettingsResult("unchanged", path)
    servers[SERVER_NAME] = server
    data["context_servers"] = servers
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return SettingsResult("merged", path)


def remove_from_settings(path: Path) -> SettingsResult:
    """Remove ``context_servers.hindsight`` from Zed's settings at ``path``."""
    data = _load_strict(path)
    if data is None:
        if path.is_file():
            return SettingsResult("manual", path)
        return SettingsResult("unchanged", path)

    servers = data.get("context_servers")
    if not isinstance(servers, dict) or SERVER_NAME not in servers:
        return SettingsResult("unchanged", path)
    del servers[SERVER_NAME]
    if servers:
        data["context_servers"] = servers
    else:
        data.pop("context_servers", None)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return SettingsResult("removed", path)


def is_installed(path: Path) -> bool:
    """Whether our context server is present in Zed's settings at ``path``."""
    data = _load_strict(path)
    if data is None:
        return False
    servers = data.get("context_servers")
    return isinstance(servers, dict) and SERVER_NAME in servers
