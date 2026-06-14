"""Hindsight REST API client (stdlib HTTP).

A thin client over the Hindsight HTTP API — recall, retain, and bank mission —
sharing the shape used by the other editor integrations.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

DEFAULT_TIMEOUT = 15  # seconds
HEALTH_CHECK_RETRIES = 3
HEALTH_CHECK_DELAY = 2  # seconds


class HindsightHTTPError(RuntimeError):
    """An HTTP error response from the Hindsight API, carrying the status code.

    Lets callers distinguish auth failures (401/403) from transient errors so
    they can be surfaced more loudly.
    """

    def __init__(self, status_code: int, url: str, body: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code} from {url}: {body}")


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

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def _request(self, method: str, path: str, body: Optional[dict] = None, timeout: int = DEFAULT_TIMEOUT) -> dict:
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
            raise HindsightHTTPError(e.code, url, body_text) from e

    def health_check(self, timeout: int = 5) -> bool:
        """Return True if the Hindsight server is reachable (retries a few times)."""
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
        timeout: int = 10,
    ) -> dict:
        """Recall memories from a bank. Returns the raw API response dict."""
        path = f"/v1/default/banks/{urllib.parse.quote(bank_id, safe='')}/memories/recall"
        body: dict = {"query": query, "max_tokens": max_tokens}
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
        metadata: Optional[dict] = None,
        tags: Optional[list] = None,
        timeout: int = 15,
    ) -> dict:
        """Retain content into a bank's memory (async server-side processing)."""
        path = f"/v1/default/banks/{urllib.parse.quote(bank_id, safe='')}/memories"
        item: dict = {"content": content, "document_id": document_id, "metadata": metadata or {}}
        if context:
            item["context"] = context
        if tags:
            item["tags"] = tags
        return self._request("POST", path, {"items": [item], "async": True}, timeout=timeout)

    def set_bank_mission(
        self, bank_id: str, mission: str, retain_mission: Optional[str] = None, timeout: int = 15
    ) -> dict:
        """Set the reflect/retain mission for a bank."""
        path = f"/v1/default/banks/{urllib.parse.quote(bank_id, safe='')}/config"
        updates: dict = {"reflect_mission": mission}
        if retain_mission:
            updates["retain_mission"] = retain_mission
        return self._request("PATCH", path, {"updates": updates}, timeout=timeout)
