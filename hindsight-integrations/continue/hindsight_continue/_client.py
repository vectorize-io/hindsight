"""Shared Hindsight client resolution logic."""

from __future__ import annotations

import os
from importlib import metadata
from typing import Any

from hindsight_client import Hindsight

from .config import DEFAULT_HINDSIGHT_API_URL, HINDSIGHT_API_KEY_ENV, get_config

try:
    _VERSION = metadata.version("hindsight-continue")
except metadata.PackageNotFoundError:
    _VERSION = "0.0.0"
_USER_AGENT = f"hindsight-continue/{_VERSION}"


def resolve_client(
    client: Hindsight | None = None,
    hindsight_api_url: str | None = None,
    api_key: str | None = None,
) -> Hindsight:
    """Resolve a Hindsight client from explicit args or global config.

    Falls back to the configured/default API URL and the ``HINDSIGHT_API_KEY``
    env var so the adapter works with nothing but the env var set. The API key
    is optional at construction time — a missing key only fails when a recall
    is actually made.
    """
    if client is not None:
        return client

    config = get_config()
    url = hindsight_api_url or config.hindsight_api_url or DEFAULT_HINDSIGHT_API_URL
    key = api_key or config.api_key or os.environ.get(HINDSIGHT_API_KEY_ENV)

    kwargs: dict[str, Any] = {"base_url": url, "timeout": 30.0, "user_agent": _USER_AGENT}
    if key:
        kwargs["api_key"] = key
    return Hindsight(**kwargs)
