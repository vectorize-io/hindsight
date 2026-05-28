"""Shared client resolution logic for Hindsight and Superagent."""

from __future__ import annotations

import importlib.metadata
import os
from typing import Any

from hindsight_client import Hindsight
from safety_agent import SafetyClient, create_client

from .config import DEFAULT_HINDSIGHT_API_URL, HINDSIGHT_API_KEY_ENV, SUPERAGENT_API_KEY_ENV, get_config
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
    url = hindsight_api_url or (config.hindsight_api_url if config else DEFAULT_HINDSIGHT_API_URL)
    # Read HINDSIGHT_API_KEY directly so the constructor-only path (no prior
    # configure() call) still honours the env var.  Without this read the
    # base Hindsight client also doesn't fall back to the env, so a key in
    # the environment would be silently dropped.
    key = api_key or (config.api_key if config else None) or os.environ.get(HINDSIGHT_API_KEY_ENV)

    kwargs: dict[str, Any] = {"base_url": url, "timeout": 120.0, "user_agent": _USER_AGENT}
    if key:
        kwargs["api_key"] = key
    return Hindsight(**kwargs)


def snapshot_safety_config(
    superagent_api_key: str | None,
    enable_fallback: bool | None,
    fallback_timeout: float | None,
) -> dict[str, Any]:
    """Resolve effective safety-client settings against current global config + env.

    Returns a dict suitable for passing to `safety_agent.create_client()`.  Used
    by the middleware at construction time so a later `configure()` call can't
    silently change what the lazy-constructed SafetyClient will look like.
    Reading is allowed at this point; building the client is deferred until
    first guard/redact call.
    """
    config = get_config()
    key = (
        superagent_api_key or (config.superagent_api_key if config else None) or os.environ.get(SUPERAGENT_API_KEY_ENV)
    )
    resolved_fallback = (
        enable_fallback if enable_fallback is not None else (config.enable_fallback if config else False)
    )
    resolved_timeout = (
        fallback_timeout if fallback_timeout is not None else (config.fallback_timeout if config else 5.0)
    )
    return {"api_key": key, "enable_fallback": resolved_fallback, "fallback_timeout": resolved_timeout}


def build_safety_client(snapshot: dict[str, Any]) -> SafetyClient:
    """Build a SafetyClient from a previously-resolved snapshot.

    Raises HindsightError if the snapshot is missing the Superagent API key —
    this is the call site that finally requires it, so lazy construction with
    every safety hook disabled never trips the error.
    """
    if not snapshot.get("api_key"):
        raise HindsightError(
            "No Superagent API key configured. Pass superagent_api_key=, set SUPERAGENT_API_KEY env var, "
            "or call configure(superagent_api_key=...) first. Get a key at https://www.superagent.sh"
        )
    return create_client(**snapshot)


def resolve_safety_client(
    safety_client: SafetyClient | None,
    superagent_api_key: str | None,
    enable_fallback: bool | None = None,
    fallback_timeout: float | None = None,
) -> SafetyClient:
    """Resolve a Superagent SafetyClient from explicit args or global config.

    Kept for backwards compatibility — combines snapshot + build into one call.
    New callers should use the two-step API to control when global config and
    env-var resolution happens.
    """
    if safety_client is not None:
        return safety_client
    return build_safety_client(snapshot_safety_config(superagent_api_key, enable_fallback, fallback_timeout))
