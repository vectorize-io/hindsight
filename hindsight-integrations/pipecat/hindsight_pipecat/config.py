"""Global configuration for Hindsight-Pipecat integration."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_HINDSIGHT_API_URL = "https://api.hindsight.vectorize.io"
HINDSIGHT_API_KEY_ENV = "HINDSIGHT_API_KEY"


@dataclass
class HindsightPipecatConfig:
    """Connection and default settings for the Pipecat integration.

    Attributes:
        hindsight_api_url: URL of the Hindsight API server.
        api_key: API key for Hindsight authentication.
        recall_budget: Default recall budget level (low/mid/high).
        recall_max_tokens: Default maximum tokens for recall results.
    """

    hindsight_api_url: str = DEFAULT_HINDSIGHT_API_URL
    api_key: str | None = None
    recall_budget: str = "mid"
    recall_max_tokens: int = 4096


_global_config: HindsightPipecatConfig | None = None


def configure(
    hindsight_api_url: str | None = None,
    api_key: str | None = None,
    recall_budget: str = "mid",
    recall_max_tokens: int = 4096,
) -> HindsightPipecatConfig:
    """Configure Hindsight connection and default settings.

    Args:
        hindsight_api_url: Hindsight API URL (default: production).
        api_key: API key. Falls back to HINDSIGHT_API_KEY env var.
        recall_budget: Default recall budget (low/mid/high).
        recall_max_tokens: Default max tokens for recall.

    Returns:
        The configured HindsightPipecatConfig.
    """
    global _global_config

    resolved_url = hindsight_api_url or DEFAULT_HINDSIGHT_API_URL
    resolved_key = api_key or os.environ.get(HINDSIGHT_API_KEY_ENV)

    _global_config = HindsightPipecatConfig(
        hindsight_api_url=resolved_url,
        api_key=resolved_key,
        recall_budget=recall_budget,
        recall_max_tokens=recall_max_tokens,
    )

    return _global_config


def get_config() -> HindsightPipecatConfig | None:
    """Get the current global configuration."""
    return _global_config


def reset_config() -> None:
    """Reset global configuration to None."""
    global _global_config
    _global_config = None
