"""Shared Hindsight client resolution logic."""

from __future__ import annotations

import os
from typing import Any

from hindsight_client import Hindsight

from ._version import __version__
from .config import DEFAULT_HINDSIGHT_API_URL, HINDSIGHT_API_KEY_ENV, get_config

_USER_AGENT = f"hindsight-claude-agent-sdk/{__version__}"


def resolve_client(
    client: Hindsight | None,
    hindsight_api_url: str | None,
    api_key: str | None,
) -> Hindsight:
    """Resolve a Hindsight client from explicit args or global config.

    Falls back to the default API URL and the ``HINDSIGHT_API_KEY`` env var
    when neither an explicit argument nor a prior ``configure()`` call supplied
    them, so the tools and hooks work with nothing but the env var set.
    Self-hosted users override the URL. The API key is optional at construction
    time — a missing key only fails when a call is actually made.
    """
    if client is not None:
        return client

    config = get_config()
    url = hindsight_api_url or (config.hindsight_api_url if config else DEFAULT_HINDSIGHT_API_URL)
    # Read HINDSIGHT_API_KEY directly so the no-configure() path still honours
    # the env var — the base Hindsight client doesn't fall back to it on its own.
    key = api_key or (config.api_key if config else None) or os.environ.get(HINDSIGHT_API_KEY_ENV)

    kwargs: dict[str, Any] = {"base_url": url, "timeout": 30.0, "user_agent": _USER_AGENT}
    if key:
        kwargs["api_key"] = key
    return Hindsight(**kwargs)
