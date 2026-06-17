"""Shared Hindsight client resolution logic."""

import os
from importlib import metadata
from typing import Any, Optional

from hindsight_client import Hindsight

from .config import DEFAULT_HINDSIGHT_API_URL, HINDSIGHT_API_KEY_ENV, get_config

try:
    _VERSION = metadata.version("hindsight-agent-framework")
except metadata.PackageNotFoundError:
    _VERSION = "0.0.0"
_USER_AGENT = f"hindsight-agent-framework/{_VERSION}"

# Request timeout (seconds). The hindsight-client SDK takes a single timeout at
# construction and exposes no per-call timeout, so this one bound applies to
# every recall/retain/bank operation. The provider guards all calls and degrades
# gracefully, so this is a backstop, not a tight per-op budget.
TIMEOUT_DEFAULT = 30.0


def resolve_client(
    client: Optional[Hindsight],
    hindsight_api_url: Optional[str],
    api_key: Optional[str],
) -> Hindsight:
    """Resolve a Hindsight client from explicit args or global config.

    Falls back to the default API URL and the ``HINDSIGHT_API_KEY`` env var
    when neither an explicit argument nor a prior ``configure()`` call supplied
    them, so the provider works with nothing but the env var set. Self-hosted
    users override the URL. The API key is optional at construction time — a
    missing key only fails when a call is actually made.
    """
    if client is not None:
        return client

    config = get_config()
    url = hindsight_api_url or (config.hindsight_api_url if config else DEFAULT_HINDSIGHT_API_URL)
    # Read HINDSIGHT_API_KEY directly so the no-configure() path still honours
    # the env var — the base Hindsight client doesn't fall back to it on its own.
    key = api_key or (config.api_key if config else None) or os.environ.get(HINDSIGHT_API_KEY_ENV)

    kwargs: dict[str, Any] = {"base_url": url, "timeout": TIMEOUT_DEFAULT, "user_agent": _USER_AGENT}
    if key:
        kwargs["api_key"] = key
    return Hindsight(**kwargs)
