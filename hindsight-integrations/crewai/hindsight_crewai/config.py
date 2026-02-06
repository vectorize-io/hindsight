"""Global configuration for Hindsight-CrewAI integration."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_HINDSIGHT_API_URL = "https://api.hindsight.vectorize.io"
HINDSIGHT_API_KEY_ENV = "HINDSIGHT_API_KEY"


@dataclass
class HindsightCrewAIConfig:
    """Connection and default settings for the CrewAI integration.

    Attributes:
        hindsight_api_url: URL of the Hindsight API server.
        api_key: API key for Hindsight authentication.
        budget: Default recall budget level (low/mid/high).
        max_tokens: Default maximum tokens for recall results.
        tags: Default tags applied when storing memories.
        recall_tags: Default tags to filter when searching memories.
        recall_tags_match: Tag matching mode (any/all/any_strict/all_strict).
        verbose: Enable verbose logging.
    """

    hindsight_api_url: str = DEFAULT_HINDSIGHT_API_URL
    api_key: str | None = None
    budget: str = "mid"
    max_tokens: int = 4096
    tags: list[str] | None = None
    recall_tags: list[str] | None = None
    recall_tags_match: str = "any"
    verbose: bool = False


_global_config: HindsightCrewAIConfig | None = None


def configure(
    hindsight_api_url: str | None = None,
    api_key: str | None = None,
    budget: str = "mid",
    max_tokens: int = 4096,
    tags: list[str] | None = None,
    recall_tags: list[str] | None = None,
    recall_tags_match: str = "any",
    verbose: bool = False,
) -> HindsightCrewAIConfig:
    """Configure Hindsight connection and default settings.

    Args:
        hindsight_api_url: Hindsight API URL (default: production).
        api_key: API key. Falls back to HINDSIGHT_API_KEY env var.
        budget: Default recall budget (low/mid/high).
        max_tokens: Default max tokens for recall.
        tags: Default tags for retain operations.
        recall_tags: Default tags to filter recall/search.
        recall_tags_match: Tag matching mode.
        verbose: Enable verbose logging.

    Returns:
        The configured HindsightCrewAIConfig.
    """
    global _global_config

    resolved_url = hindsight_api_url or DEFAULT_HINDSIGHT_API_URL
    resolved_key = api_key or os.environ.get(HINDSIGHT_API_KEY_ENV)

    _global_config = HindsightCrewAIConfig(
        hindsight_api_url=resolved_url,
        api_key=resolved_key,
        budget=budget,
        max_tokens=max_tokens,
        tags=tags,
        recall_tags=recall_tags,
        recall_tags_match=recall_tags_match,
        verbose=verbose,
    )

    return _global_config


def get_config() -> HindsightCrewAIConfig | None:
    """Get the current global configuration."""
    return _global_config


def reset_config() -> None:
    """Reset global configuration to None."""
    global _global_config
    _global_config = None
