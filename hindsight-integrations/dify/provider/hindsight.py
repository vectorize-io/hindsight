"""Hindsight provider for Dify plugin system.

Validates API URL + optional API key by hitting Hindsight's /health endpoint.
"""

from __future__ import annotations

from typing import Any

import requests
from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError


class HindsightProvider(ToolProvider):
    """Tool provider for Hindsight memory."""

    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        api_url = (credentials.get("api_url") or "").rstrip("/")
        if not api_url:
            raise ToolProviderCredentialValidationError("API URL is required.")

        api_key = credentials.get("api_key") or ""
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        try:
            resp = requests.get(f"{api_url}/health", headers=headers, timeout=10)
        except requests.RequestException as e:
            raise ToolProviderCredentialValidationError(f"Could not reach Hindsight at {api_url}: {e}") from e

        if resp.status_code == 401 or resp.status_code == 403:
            raise ToolProviderCredentialValidationError(
                "Hindsight rejected the API key. Check the value and try again."
            )
        if resp.status_code >= 400:
            raise ToolProviderCredentialValidationError(f"Hindsight health check failed (HTTP {resp.status_code}).")
