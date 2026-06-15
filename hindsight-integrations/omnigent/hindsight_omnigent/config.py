"""Global configuration for the Hindsight-Omnigent integration.

Omnigent invokes ``type: function`` tools as plain Python callables and passes
them **no session context** (see ``omnigent/tools/local_callable.py`` —
``invoke`` does ``del ctx``). So, unlike session-aware integrations, the
Hindsight bank can't be derived per call; it comes from this module-level
config (or the matching environment variables). The natural model is one bank
per Omnigent agent process: set it once via :func:`configure` or env vars.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from hindsight_client import Hindsight

DEFAULT_HINDSIGHT_API_URL = "https://api.hindsight.vectorize.io"

HINDSIGHT_API_URL_ENV = "HINDSIGHT_API_URL"
HINDSIGHT_API_KEY_ENV = "HINDSIGHT_API_KEY"
HINDSIGHT_BANK_ID_ENV = "HINDSIGHT_BANK_ID"

Budget = Literal["low", "mid", "high"]
TagsMatch = Literal["any", "all", "any_strict", "all_strict"]


@dataclass
class HindsightOmnigentConfig:
    """Connection and default settings for the Omnigent integration.

    Attributes:
        hindsight_api_url: URL of the Hindsight API server.
        api_key: API key for Hindsight authentication.
        bank_id: Hindsight memory bank the tool callables read/write.
        budget: Default recall/reflect budget level (low/mid/high).
        max_tokens: Default maximum tokens for recall results.
        tags: Default tags applied when storing memories.
        recall_tags: Default tags to filter when searching memories.
        recall_tags_match: Tag matching mode (any/all/any_strict/all_strict).
        client: Pre-built Hindsight client (overrides url/key when set).
    """

    hindsight_api_url: str = DEFAULT_HINDSIGHT_API_URL
    api_key: str | None = None
    bank_id: str | None = None
    budget: Budget = "mid"
    max_tokens: int = 4096
    tags: list[str] | None = None
    recall_tags: list[str] | None = None
    recall_tags_match: TagsMatch = "any"
    client: Hindsight | None = None


_global_config: HindsightOmnigentConfig | None = None


def configure(
    hindsight_api_url: str | None = None,
    api_key: str | None = None,
    bank_id: str | None = None,
    budget: Budget = "mid",
    max_tokens: int = 4096,
    tags: list[str] | None = None,
    recall_tags: list[str] | None = None,
    recall_tags_match: TagsMatch = "any",
    client: Hindsight | None = None,
) -> HindsightOmnigentConfig:
    """Configure the Hindsight connection and default settings.

    Call this once at import time of the module that hosts your tool callables,
    so they resolve the same connection and bank on every invocation.

    Args:
        hindsight_api_url: Hindsight API URL. Falls back to ``HINDSIGHT_API_URL``
            env var, then to Hindsight Cloud.
        api_key: API key. Falls back to ``HINDSIGHT_API_KEY`` env var.
        bank_id: Memory bank to read/write. Falls back to ``HINDSIGHT_BANK_ID``.
        budget: Default recall/reflect budget (low/mid/high).
        max_tokens: Default max tokens for recall.
        tags: Default tags for retain operations.
        recall_tags: Default tags to filter recall/search.
        recall_tags_match: Tag matching mode.
        client: Pre-built Hindsight client (overrides url/key when set).

    Returns:
        The configured HindsightOmnigentConfig.
    """
    global _global_config

    resolved_url = hindsight_api_url or os.environ.get(HINDSIGHT_API_URL_ENV) or DEFAULT_HINDSIGHT_API_URL
    resolved_key = api_key or os.environ.get(HINDSIGHT_API_KEY_ENV)
    resolved_bank = bank_id or os.environ.get(HINDSIGHT_BANK_ID_ENV)

    _global_config = HindsightOmnigentConfig(
        hindsight_api_url=resolved_url,
        api_key=resolved_key,
        bank_id=resolved_bank,
        budget=budget,
        max_tokens=max_tokens,
        tags=tags,
        recall_tags=recall_tags,
        recall_tags_match=recall_tags_match,
        client=client,
    )

    return _global_config


def get_config() -> HindsightOmnigentConfig:
    """Return the active config, creating one from env vars on first use.

    Tool callables call this with no prior :func:`configure`, so an agent can be
    wired up entirely through ``HINDSIGHT_*`` environment variables in its
    Omnigent ``os_env`` block.
    """
    global _global_config
    if _global_config is None:
        _global_config = configure()
    return _global_config


def reset_config() -> None:
    """Reset global configuration to None."""
    global _global_config
    _global_config = None
