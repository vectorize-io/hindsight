"""Global configuration for Hindsight-Google-ADK integration."""

import os
from dataclasses import dataclass
from typing import Optional

DEFAULT_HINDSIGHT_API_URL = "https://api.hindsight.vectorize.io"
HINDSIGHT_API_KEY_ENV = "HINDSIGHT_API_KEY"
DEFAULT_BANK_ID_TEMPLATE = "{app_name}::{user_id}"


@dataclass
class HindsightAdkConfig:
    """Connection and default settings for the Google ADK integration.

    Attributes:
        hindsight_api_url: URL of the Hindsight API server.
        api_key: API key for Hindsight authentication.
        budget: Default recall budget level (low/mid/high).
        max_tokens: Default maximum tokens for recall results.
        tags: Default tags applied when storing memories.
        recall_tags: Default tags to filter when searching memories.
        recall_tags_match: Tag matching mode (any/all/any_strict/all_strict).
        context: Source label for retain operations (default: "google-adk").
        mission: Bank mission for fact extraction context.
        bank_id_template: Format string used to derive a bank id from
            ``app_name`` and ``user_id`` (default: ``"{app_name}::{user_id}"``).
        verbose: Enable verbose logging.
    """

    hindsight_api_url: str = DEFAULT_HINDSIGHT_API_URL
    api_key: Optional[str] = None
    budget: str = "mid"
    max_tokens: int = 4096
    tags: Optional[list[str]] = None
    recall_tags: Optional[list[str]] = None
    recall_tags_match: str = "any"
    context: str = "google-adk"
    mission: Optional[str] = None
    bank_id_template: str = DEFAULT_BANK_ID_TEMPLATE
    verbose: bool = False


_global_config: Optional[HindsightAdkConfig] = None


def configure(
    hindsight_api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    budget: str = "mid",
    max_tokens: int = 4096,
    tags: Optional[list[str]] = None,
    recall_tags: Optional[list[str]] = None,
    recall_tags_match: str = "any",
    context: str = "google-adk",
    mission: Optional[str] = None,
    bank_id_template: str = DEFAULT_BANK_ID_TEMPLATE,
    verbose: bool = False,
) -> HindsightAdkConfig:
    """Configure Hindsight connection and default settings.

    Args:
        hindsight_api_url: Hindsight API URL (default: production cloud).
        api_key: API key. Falls back to ``HINDSIGHT_API_KEY`` env var.
        budget: Default recall budget (low/mid/high).
        max_tokens: Default max tokens for recall results.
        tags: Default tags for retain operations.
        recall_tags: Default tags to filter recall/search.
        recall_tags_match: Tag matching mode.
        context: Source label for retain operations.
        mission: Bank mission for fact extraction context.
        bank_id_template: Format string for deriving bank ids from
            ``app_name`` and ``user_id``.
        verbose: Enable verbose logging.

    Returns:
        The configured ``HindsightAdkConfig``.
    """
    global _global_config

    resolved_url = hindsight_api_url or DEFAULT_HINDSIGHT_API_URL
    resolved_key = api_key or os.environ.get(HINDSIGHT_API_KEY_ENV)

    _global_config = HindsightAdkConfig(
        hindsight_api_url=resolved_url,
        api_key=resolved_key,
        budget=budget,
        max_tokens=max_tokens,
        tags=tags,
        recall_tags=recall_tags,
        recall_tags_match=recall_tags_match,
        context=context,
        mission=mission,
        bank_id_template=bank_id_template,
        verbose=verbose,
    )

    return _global_config


def get_config() -> Optional[HindsightAdkConfig]:
    """Get the current global configuration."""
    return _global_config


def reset_config() -> None:
    """Reset global configuration to None."""
    global _global_config
    _global_config = None
