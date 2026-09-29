"""Wire Hindsight into OpenHands' MCP config (``[mcp].shttp_servers``).

OpenHands natively supports Streamable-HTTP MCP servers, so the Hindsight MCP
endpoint connects directly — no bridge needed. The server is registered in
OpenHands' ``config.toml``::

    [mcp]
    shttp_servers = [
        {url = "https://api.hindsight.vectorize.io/mcp/<bank>/", api_key = "hsk_..."}
    ]

``config.toml`` is hand-edited TOML (comments, formatting), so we round-trip it
with ``tomlkit`` to preserve the user's file. If it can't be parsed, we return
the exact snippet to paste rather than risk clobbering it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import tomlkit


def default_config_path() -> Path:
    """OpenHands reads ``config.toml`` from the working directory by default."""
    return Path.cwd() / "config.toml"


def mcp_endpoint_url(api_url: str, bank_id: str) -> str:
    """The Hindsight MCP endpoint for a bank (bank is the last path segment)."""
    return f"{api_url.rstrip('/')}/mcp/{bank_id}/"


def build_shttp_server(api_url: str, api_token: Optional[str], bank_id: str) -> dict[str, str]:
    """Build the ``shttp_servers`` entry: ``{url, api_key?}``.

    The api_key is omitted for an open self-hosted server (no token).
    """
    entry: dict[str, str] = {"url": mcp_endpoint_url(api_url, bank_id)}
    if api_token:
        entry["api_key"] = api_token
    return entry


def render_snippet(server: dict[str, str]) -> str:
    """Render the ``[mcp]`` snippet a user can paste into ``config.toml``."""
    doc = tomlkit.document()
    mcp = tomlkit.table()
    arr = tomlkit.array()
    arr.append(server)
    arr.multiline(True)
    mcp["shttp_servers"] = arr
    doc["mcp"] = mcp
    return tomlkit.dumps(doc)


@dataclass
class ConfigResult:
    """Outcome of editing ``config.toml``.

    ``action`` is one of ``created``, ``merged``, ``unchanged``, ``removed``, or
    ``manual`` (file couldn't be parsed safely — ``snippet`` holds what to paste).
    """

    action: str
    path: Path
    snippet: Optional[str] = None


def _entry_url(entry: Any) -> Optional[str]:
    """A shttp_servers item is either a bare URL string or a table with ``url``."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("url")
    return None


def _load_doc(path: Path):
    """Parse ``config.toml`` with tomlkit; return ``None`` if it can't be parsed."""
    if not path.is_file():
        return tomlkit.document()
    try:
        return tomlkit.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def apply_to_config(path: Path, server: dict[str, str]) -> ConfigResult:
    """Add/update the Hindsight entry in ``[mcp].shttp_servers`` at ``path``."""
    existed = path.is_file()
    doc = _load_doc(path)
    if doc is None:
        return ConfigResult("manual", path, snippet=render_snippet(server))

    mcp = doc.get("mcp")
    if mcp is None:
        mcp = tomlkit.table()
        doc["mcp"] = mcp
    servers = mcp.get("shttp_servers")
    if servers is None:
        servers = tomlkit.array()
        servers.multiline(True)
        mcp["shttp_servers"] = servers

    target = server["url"]
    for i, entry in enumerate(servers):
        if _entry_url(entry) == target:
            if isinstance(entry, dict) and dict(entry) == server:
                return ConfigResult("unchanged", path)
            servers[i] = server
            path.write_text(tomlkit.dumps(doc), encoding="utf-8")
            return ConfigResult("merged", path)

    servers.append(server)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomlkit.dumps(doc), encoding="utf-8")
    return ConfigResult("merged" if existed else "created", path)


def remove_from_config(path: Path, server: dict[str, str]) -> ConfigResult:
    """Remove the Hindsight entry (matched by URL) from ``[mcp].shttp_servers``."""
    if not path.is_file():
        return ConfigResult("unchanged", path)
    doc = _load_doc(path)
    if doc is None:
        return ConfigResult("manual", path)

    mcp = doc.get("mcp")
    servers = mcp.get("shttp_servers") if mcp is not None else None
    if not servers:
        return ConfigResult("unchanged", path)

    target = server["url"]
    kept = [e for e in servers if _entry_url(e) != target]
    if len(kept) == len(servers):
        return ConfigResult("unchanged", path)

    if kept:
        arr = tomlkit.array()
        arr.multiline(True)
        for e in kept:
            arr.append(e)
        mcp["shttp_servers"] = arr
    else:
        del mcp["shttp_servers"]
        if not mcp:
            del doc["mcp"]
    path.write_text(tomlkit.dumps(doc), encoding="utf-8")
    return ConfigResult("removed", path)


def is_installed(path: Path, server: dict[str, str]) -> bool:
    """Whether the Hindsight entry (matched by URL) is present at ``path``."""
    doc = _load_doc(path)
    if doc is None or not path.is_file():
        return False
    mcp = doc.get("mcp")
    servers = mcp.get("shttp_servers") if mcp is not None else None
    if not servers:
        return False
    return any(_entry_url(e) == server["url"] for e in servers)
