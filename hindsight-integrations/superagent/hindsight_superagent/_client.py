"""Shared client resolution logic for Hindsight and Superagent."""

from __future__ import annotations

import importlib.metadata
from typing import Any

from hindsight_client import Hindsight
from safety_agent import SafetyClient, create_client

from .config import get_config
from .errors import HindsightError

try:
    _VERSION = importlib.metadata.version("hindsight-superagent")
except importlib.metadata.PackageNotFoundError:
    _VERSION = "0.0.0"

_USER_AGENT = f"hindsight-superagent/{_VERSION}"


def resolve_hindsight_client(
    client: Hindsight | None,
    hindsight_api_url: str | None,
    api_key: str | None,
) -> Hindsight:
    """Resolve a Hindsight client from explicit args or global config."""
    if client is not None:
        return client

    config = get_config()
    url = hindsight_api_url or (config.hindsight_api_url if config else None)
    key = api_key or (config.api_key if config else None)

    if url is None:
        raise HindsightError(
            "No Hindsight API URL configured. Pass client= or hindsight_api_url=, or call configure() first."
        )

    kwargs: dict[str, Any] = {"base_url": url, "timeout": 30.0, "user_agent": _USER_AGENT}
    if key:
        kwargs["api_key"] = key
    return Hindsight(**kwargs)


def resolve_safety_client(
    safety_client: SafetyClient | None,
    superagent_api_key: str | None,
) -> SafetyClient:
    """Resolve a Superagent SafetyClient from explicit args or global config."""
    if safety_client is not None:
        return safety_client

    config = get_config()
    key = superagent_api_key or (config.superagent_api_key if config else None)

    kwargs: dict[str, Any] = {}
    if key:
        kwargs["api_key"] = key
    return create_client(**kwargs)
