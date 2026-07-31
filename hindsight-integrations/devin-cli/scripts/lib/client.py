"""Hindsight REST API client.

Communicates with a Hindsight server via HTTP. Port of the Claude Code
plugin's client.js/client.py, adapted for Python stdlib. Identical wire
protocol — an external Hindsight server can serve both integrations.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

DEFAULT_TIMEOUT = 15  # seconds
HEALTH_CHECK_RETRIES = 3
HEALTH_CHECK_DELAY = 2  # seconds


def _plugin_version() -> str:
    """Read the plugin version from plugin.json (single source of truth)."""
    manifest = Path(__file__).resolve().parents[2] / ".devin-plugin" / "plugin.json"
    try:
        data = json.loads(manifest.read_text())
    except (OSError, ValueError):
        return "0.0.0"
    # A plugin.json holding valid JSON that is not an object would raise
    # AttributeError from .get() — uncaught, at module import, so every hook
    # that imports this client dies before it can make a single request.
    if not isinstance(data, dict):
        return "0.0.0"
    version = data.get("version")
    return version if isinstance(version, str) else "0.0.0"


# Sent on every request so self-hosted deployments behind Cloudflare (or any
# reverse proxy with UA-based bot filtering) don't block the stdlib default
# "Python-urllib/X.Y", which trips Cloudflare error 1010.
USER_AGENT = f"hindsight-devin-cli/{_plugin_version()}"


def _validate_api_url(url: str) -> str:
    """Validate and normalize the API URL. Reject non-HTTP schemes."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Hindsight API URL must use http or https, got: {parsed.scheme!r}")
    if not parsed.hostname:
        raise ValueError(f"Hindsight API URL has no hostname: {url!r}")
    # request() appends the endpoint path by string concatenation, so a query
    # or fragment here would swallow it: "https://h/api?tenant=x" + "/v1/..."
    # sends the whole route into the query string and the request silently
    # lands somewhere else. Rejected rather than stripped — dropping a
    # ?tenant= the user deliberately configured would be worse than failing.
    if parsed.query or parsed.fragment:
        raise ValueError(f"Hindsight API URL must not include a query or fragment: {url!r}")
    return url.rstrip("/")


def _validate_timeout_override(value) -> Optional[int]:
    """Coerce `requestTimeoutSeconds` to a usable timeout, or drop it.

    This one setting escapes lib/config.py's central type check: that check
    reads the expected type from DEFAULTS, and this default is None, which
    carries no type. It is used structurally all the same — it becomes
    urllib's `timeout` — so a string from the settings file reaches urlopen and
    raises TypeError inside a hook, and a zero or negative value makes every
    request fail instantly.

    Dropped rather than raised on: this is an optional tuning knob, and the
    caller's own per-request timeout is a working value to fall back to.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0:
        return int(value)
    return None


class HindsightClient:
    """HTTP client for the Hindsight API."""

    def __init__(
        self,
        api_url: str,
        api_token: Optional[str] = None,
        request_timeout_override: Optional[int] = None,
    ):
        self.api_url = _validate_api_url(api_url)
        self.api_token = api_token
        self.request_timeout_override = _validate_timeout_override(request_timeout_override)

    def _resolve_timeout(self, timeout: int) -> int:
        """Return the override if configured, otherwise the caller's timeout."""
        return self.request_timeout_override if self.request_timeout_override is not None else timeout

    def _headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def request(self, method: str, path: str, body: Optional[dict] = None, timeout: int = DEFAULT_TIMEOUT) -> dict:
        timeout = self._resolve_timeout(timeout)
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

    def health_check(self, timeout: int = 5) -> bool:
        """Check if the Hindsight server is reachable.

        Retries up to 3 times with 2s delay between attempts.
        """
        import time

        for attempt in range(1, HEALTH_CHECK_RETRIES + 1):
            try:
                url = f"{self.api_url}/health"
                req = urllib.request.Request(url, headers=self._headers(), method="GET")
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
            if attempt < HEALTH_CHECK_RETRIES:
                time.sleep(HEALTH_CHECK_DELAY)
        return False

    def recall(
        self,
        bank_id: str,
        query: str,
        max_tokens: int = 1024,
        budget: str = "mid",
        types: Optional[list] = None,
        tags: Optional[list] = None,
        tags_match: Optional[str] = None,
        tag_groups: Optional[object] = None,
        timeout: int = 10,
    ) -> dict:
        """Recall memories from a bank.

        Returns the raw API response dict with 'results' list.
        """
        path = f"/v1/default/banks/{urllib.parse.quote(bank_id, safe='')}/memories/recall"
        body = {
            "query": query,
            "max_tokens": max_tokens,
        }
        if budget:
            body["budget"] = budget
        if types:
            body["types"] = types
        if tags:
            body["tags"] = tags
        if tags_match:
            body["tags_match"] = tags_match
        if tag_groups:
            body["tag_groups"] = tag_groups
        return self.request("POST", path, body, timeout=timeout)

    def retain(
        self,
        bank_id: str,
        content: str,
        document_id: str = "conversation",
        context: Optional[str] = None,
        metadata: Optional[dict] = None,
        tags: Optional[list] = None,
        timeout: int = 15,
    ) -> dict:
        """Retain content into a bank's memory.

        Posts with async=true so the server processes in the background.
        The context field helps Hindsight cluster memories by provenance
        (e.g. "devin-cli" vs manual retains).
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
        body = {
            "items": [item],
            "async": True,
        }
        return self.request("POST", path, body, timeout=timeout)

    def set_bank_mission(
        self, bank_id: str, mission: str, retain_mission: Optional[str] = None, timeout: int = 15
    ) -> dict:
        """Set the mission/persona for a bank.

        Uses PATCH /banks/{id}/config with reflect_mission and retain_mission.
        """
        path = f"/v1/default/banks/{urllib.parse.quote(bank_id, safe='')}/config"
        updates = {"reflect_mission": mission}
        if retain_mission:
            updates["retain_mission"] = retain_mission
        return self.request("PATCH", path, {"updates": updates}, timeout=timeout)
