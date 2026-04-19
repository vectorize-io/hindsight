"""Shared Hindsight client resolution logic."""

from importlib import metadata
from typing import Any, Optional

from hindsight_client import Hindsight

from .config import get_config
from .errors import HindsightError

try:
    _VERSION = metadata.version("hindsight-langgraph")
except metadata.PackageNotFoundError:
    _VERSION = "0.0.0"
_USER_AGENT = f"hindsight-langgraph/{_VERSION}"


def resolve_client(
    client: Optional[Hindsight],
    hindsight_api_url: Optional[str],
    api_key: Optional[str],
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
