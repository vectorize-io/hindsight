"""Hindsight REST API client.

Communicates with a Hindsight server via HTTP. Mirrors the HTTP mode of the
Openclaw HindsightClient (client.js), adapted for Python stdlib.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

DEFAULT_TIMEOUT = 15  # seconds
HEALTH_CHECK_RETRIES = 3
HEALTH_CHECK_DELAY = 2  # seconds


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
            raise RuntimeError(f"HTTP {e.code} from {url}: {body_text}") from e

    def health_check(self, timeout: int = 5) -> bool:
        """Check if the Hindsight server is reachable.

        Mirrors Openclaw's checkExternalApiHealth: retries up to 3 times
        with 2s delay between attempts.
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
        """Retain content into a bank's memory.

        Posts with async=true so the server processes in the background.
        The context field helps Hindsight cluster memories by provenance
        (e.g. "claude-code" vs manual retains).
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
        return self._request("POST", path, body, timeout=timeout)

    def set_bank_mission(
        self, bank_id: str, mission: str, retain_mission: Optional[str] = None, timeout: int = 15
    ) -> dict:
        """Set the mission/persona for a bank.

        Uses PATCH /banks/{id}/config with reflect_mission and retain_mission.
        The old PUT /banks/{id} with 'mission' field is deprecated in v0.4.19.
        """
        path = f"/v1/default/banks/{urllib.parse.quote(bank_id, safe='')}/config"
        updates = {"reflect_mission": mission}
        if retain_mission:
            updates["retain_mission"] = retain_mission
        return self._request("PATCH", path, {"updates": updates}, timeout=timeout)
