"""Hindsight REST API client.

Communicates with a Hindsight server via HTTP using the Python stdlib so the
hook scripts have zero third-party dependencies.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from .config import find_settings_path

DEFAULT_TIMEOUT = 15  # seconds


def _plugin_version() -> str:
    """Read the plugin version from settings.json (single source of truth)."""
    path = find_settings_path()
    if path is None:
        return "0.0.0"
    try:
        return json.loads(path.read_text()).get("version", "0.0.0")
    except (OSError, ValueError):
        return "0.0.0"


# Sent on every request so self-hosted deployments behind Cloudflare (or any
# reverse proxy with UA-based bot filtering) don't block the stdlib default
# "Python-urllib/X.Y", which trips Cloudflare error 1010.
USER_AGENT = f"hindsight-cline/{_plugin_version()}"


def _validate_api_url(url: str) -> str:
    """Validate and normalize the API URL. Reject non-HTTP schemes."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Hindsight API URL must use http or https, got: {parsed.scheme!r}")
    if not parsed.hostname:
        raise ValueError(f"Hindsight API URL has no hostname: {url!r}")
    return url.rstrip("/")


class HindsightClient:
    """HTTP client for the Hindsight API."""

    def __init__(self, api_url: str, api_token: Optional[str] = None):
        self.api_url = _validate_api_url(api_url)
        self.api_token = api_token

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def _request(
        self, method: str, path: str, body: Optional[dict[str, Any]] = None, timeout: int = DEFAULT_TIMEOUT
    ) -> dict[str, Any]:
        # Returns the server's parsed JSON response — an open-ended payload, so a dict is correct here.
        url = f"{self.api_url}{path}"
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode()
            except Exception:
                pass
            raise RuntimeError(f"HTTP {e.code} from {url}: {body_text}") from e

    def health_check(self, timeout: int = 2) -> bool:
        """Single-shot reachability check (GET /health). Fast — runs inside hooks."""
        try:
            url = f"{self.api_url}/health"
            req = urllib.request.Request(url, headers=self._headers(), method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def recall(
        self,
        bank_id: str,
        query: str,
        max_tokens: int = 1024,
        budget: str = "mid",
        types: Optional[list] = None,
        timeout: int = 10,
    ) -> dict[str, Any]:
        """Recall memories from a bank. Returns the raw API response dict with 'results'."""
        path = f"/v1/default/banks/{urllib.parse.quote(bank_id, safe='')}/memories/recall"
        body = {"query": query, "max_tokens": max_tokens}
        if budget:
            body["budget"] = budget
        if types:
            body["types"] = types
        return self._request("POST", path, body, timeout=timeout)

    def retain(
        self,
        bank_id: str,
        content: str,
        document_id: str = "conversation",
        context: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list] = None,
        timeout: int = 15,
    ) -> dict[str, Any]:
        """Retain content into a bank's memory.

        Posts with async=true so the server processes in the background. The
        context field helps Hindsight cluster memories by provenance.
        """
        path = f"/v1/default/banks/{urllib.parse.quote(bank_id, safe='')}/memories"
        item = {
            "content": content,
            "document_id": document_id,
            "metadata": metadata or {},
        }
        if context:
            item["context"] = context
        if tags:
            item["tags"] = tags
        body = {"items": [item], "async": True}
        return self._request("POST", path, body, timeout=timeout)

    def set_bank_mission(
        self, bank_id: str, mission: str, retain_mission: Optional[str] = None, timeout: int = 15
    ) -> dict[str, Any]:
        """Set the mission/persona for a bank via PATCH /banks/{id}/config."""
        path = f"/v1/default/banks/{urllib.parse.quote(bank_id, safe='')}/config"
        updates = {"reflect_mission": mission}
        if retain_mission:
            updates["retain_mission"] = retain_mission
        return self._request("PATCH", path, {"updates": updates}, timeout=timeout)
