"""Global configuration for the Hindsight-Continue context-provider adapter.

The adapter is an HTTP endpoint that Continue's built-in ``http`` context
provider calls. Settings resolve from explicit :func:`configure` arguments,
falling back to environment variables so the server runs with nothing but
``HINDSIGHT_API_KEY`` and ``HINDSIGHT_CONTINUE_BANK_ID`` set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

DEFAULT_HINDSIGHT_API_URL = "https://api.hindsight.vectorize.io"
HINDSIGHT_API_KEY_ENV = "HINDSIGHT_API_KEY"
HINDSIGHT_API_URL_ENV = "HINDSIGHT_API_URL"
HINDSIGHT_BANK_ID_ENV = "HINDSIGHT_CONTINUE_BANK_ID"
HINDSIGHT_HOST_ENV = "HINDSIGHT_CONTINUE_HOST"
HINDSIGHT_PORT_ENV = "HINDSIGHT_CONTINUE_PORT"

DEFAULT_BUDGET: Literal["low", "mid", "high"] = "mid"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_RECALL_TAGS_MATCH: Literal["any", "all", "any_strict", "all_strict"] = "any"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8123
DEFAULT_ITEM_NAME = "Hindsight Memory"
DEFAULT_PREAMBLE = (
    "Relevant long-term memory recalled from Hindsight for this project (use what's relevant, ignore the rest):"
)

Budget = Literal["low", "mid", "high"]
TagsMatch = Literal["any", "all", "any_strict", "all_strict"]


@dataclass
class HindsightContinueConfig:
    """Connection and default settings for the Continue adapter.

    Attributes:
        hindsight_api_url: URL of the Hindsight API server.
        api_key: API key for Hindsight authentication.
        bank_id: Memory bank recalled against (Continue's request may override
            it via ``options.bankId``).
        budget: Recall budget level (low/mid/high).
        max_tokens: Maximum tokens for recall results.
        recall_types: Fact types to filter (world, experience, opinion, observation).
        recall_tags: Tags to filter recalled memories.
        recall_tags_match: Tag matching mode (any/all/any_strict/all_strict).
        item_name: Title shown for the injected context item in Continue.
        preamble: Text prepended to the recalled memory block.
        host: Interface the adapter HTTP server binds to.
        port: Port the adapter HTTP server listens on.
    """

    hindsight_api_url: str = DEFAULT_HINDSIGHT_API_URL
    api_key: str | None = None
    bank_id: str | None = None
    budget: Budget = DEFAULT_BUDGET
    max_tokens: int = DEFAULT_MAX_TOKENS
    recall_types: list[str] | None = None
    recall_tags: list[str] | None = None
    recall_tags_match: TagsMatch = DEFAULT_RECALL_TAGS_MATCH
    item_name: str = DEFAULT_ITEM_NAME
    preamble: str = DEFAULT_PREAMBLE
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT


_global_config: HindsightContinueConfig | None = None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def configure(
    hindsight_api_url: str | None = None,
    api_key: str | None = None,
    bank_id: str | None = None,
    budget: Budget = DEFAULT_BUDGET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    recall_types: list[str] | None = None,
    recall_tags: list[str] | None = None,
    recall_tags_match: TagsMatch = DEFAULT_RECALL_TAGS_MATCH,
    item_name: str = DEFAULT_ITEM_NAME,
    preamble: str = DEFAULT_PREAMBLE,
    host: str | None = None,
    port: int | None = None,
) -> HindsightContinueConfig:
    """Configure the adapter, falling back to environment variables.

    Args:
        hindsight_api_url: Hindsight API URL (falls back to ``HINDSIGHT_API_URL``
            then the production default).
        api_key: API key (falls back to ``HINDSIGHT_API_KEY``).
        bank_id: Default bank id (falls back to ``HINDSIGHT_CONTINUE_BANK_ID``).
        budget: Recall budget (low/mid/high).
        max_tokens: Max tokens for recall.
        recall_types: Fact types to filter.
        recall_tags: Tags to filter recalled memories.
        recall_tags_match: Tag matching mode.
        item_name: Title for the injected Continue context item.
        preamble: Text prepended to the recalled memory block.
        host: Adapter bind host (falls back to ``HINDSIGHT_CONTINUE_HOST``).
        port: Adapter listen port (falls back to ``HINDSIGHT_CONTINUE_PORT``).

    Returns:
        The configured :class:`HindsightContinueConfig`.
    """
    global _global_config

    resolved_url = hindsight_api_url or os.environ.get(HINDSIGHT_API_URL_ENV) or DEFAULT_HINDSIGHT_API_URL
    resolved_key = api_key or os.environ.get(HINDSIGHT_API_KEY_ENV)
    resolved_bank = bank_id or os.environ.get(HINDSIGHT_BANK_ID_ENV)
    resolved_host = host or os.environ.get(HINDSIGHT_HOST_ENV) or DEFAULT_HOST
    resolved_port = port if port is not None else _env_int(HINDSIGHT_PORT_ENV, DEFAULT_PORT)

    _global_config = HindsightContinueConfig(
        hindsight_api_url=resolved_url,
        api_key=resolved_key,
        bank_id=resolved_bank,
        budget=budget,
        max_tokens=max_tokens,
        recall_types=recall_types,
        recall_tags=recall_tags,
        recall_tags_match=recall_tags_match,
        item_name=item_name,
        preamble=preamble,
        host=resolved_host,
        port=resolved_port,
    )

    return _global_config


def get_config() -> HindsightContinueConfig:
    """Get the current global configuration, creating a default one if needed."""
    if _global_config is None:
        return configure()
    return _global_config


def reset_config() -> None:
    """Reset global configuration to None."""
    global _global_config
    _global_config = None
