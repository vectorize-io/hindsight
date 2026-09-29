"""Shared Hindsight client resolution logic."""

from __future__ import annotations

import os
from importlib import metadata
from typing import Any

from hindsight_client import Hindsight

from .config import DEFAULT_HINDSIGHT_API_URL, HINDSIGHT_API_KEY_ENV, AiderConfig

try:
    _VERSION = metadata.version("hindsight-aider")
except metadata.PackageNotFoundError:
    _VERSION = "0.0.0"
_USER_AGENT = f"hindsight-aider/{_VERSION}"


def resolve_client(config: AiderConfig) -> Hindsight:
    """Build a Hindsight client from config, falling back to ``HINDSIGHT_API_KEY``."""
    url = config.hindsight_api_url or DEFAULT_HINDSIGHT_API_URL
    key = config.hindsight_api_token or os.environ.get(HINDSIGHT_API_KEY_ENV)
    kwargs: dict[str, Any] = {"base_url": url, "timeout": 30.0, "user_agent": _USER_AGENT}
    if key:
        kwargs["api_key"] = key
    return Hindsight(**kwargs)
