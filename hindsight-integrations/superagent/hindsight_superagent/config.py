"""Global configuration for Hindsight-Superagent integration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

DEFAULT_HINDSIGHT_API_URL = "https://api.hindsight.vectorize.io"
HINDSIGHT_API_KEY_ENV = "HINDSIGHT_API_KEY"
SUPERAGENT_API_KEY_ENV = "SUPERAGENT_API_KEY"

DEFAULT_BUDGET: Literal["low", "mid", "high"] = "mid"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_RECALL_TAGS_MATCH: Literal["any", "all", "any_strict", "all_strict"] = "any"

Budget = Literal["low", "mid", "high"]
TagsMatch = Literal["any", "all", "any_strict", "all_strict"]


@dataclass
class HindsightSuperagentConfig:
    """Connection and default settings for the Superagent safety integration."""

    hindsight_api_url: str = DEFAULT_HINDSIGHT_API_URL
    api_key: str | None = None
    superagent_api_key: str | None = None
    budget: Budget = DEFAULT_BUDGET
    max_tokens: int = DEFAULT_MAX_TOKENS
    tags: list[str] | None = None
    recall_tags: list[str] | None = None
    recall_tags_match: TagsMatch = DEFAULT_RECALL_TAGS_MATCH
    guard_model: str | None = None
    redact_model: str | None = None
    redact_entities: list[str] | None = None
    redact_rewrite: bool = False
    enable_guard_on_retain: bool = True
    enable_guard_on_recall: bool = True
    enable_guard_on_reflect: bool = True
    enable_redact_on_retain: bool = True
    enable_fallback: bool = False
    fallback_timeout: float = 5.0
    verbose: bool = False


_global_config: HindsightSuperagentConfig | None = None


def configure(
    hindsight_api_url: str | None = None,
    api_key: str | None = None,
    superagent_api_key: str | None = None,
    budget: Budget = DEFAULT_BUDGET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    tags: list[str] | None = None,
    recall_tags: list[str] | None = None,
    recall_tags_match: TagsMatch = DEFAULT_RECALL_TAGS_MATCH,
    guard_model: str | None = None,
    redact_model: str | None = None,
    redact_entities: list[str] | None = None,
    redact_rewrite: bool = False,
    enable_guard_on_retain: bool = True,
    enable_guard_on_recall: bool = True,
    enable_guard_on_reflect: bool = True,
    enable_redact_on_retain: bool = True,
    enable_fallback: bool = False,
    fallback_timeout: float = 5.0,
    verbose: bool = False,
) -> HindsightSuperagentConfig:
    """Configure Hindsight + Superagent connection and default settings."""
    global _global_config
    resolved_url = hindsight_api_url or DEFAULT_HINDSIGHT_API_URL
    resolved_key = api_key or os.environ.get(HINDSIGHT_API_KEY_ENV)
    resolved_sa_key = superagent_api_key or os.environ.get(SUPERAGENT_API_KEY_ENV)

    _global_config = HindsightSuperagentConfig(
        hindsight_api_url=resolved_url,
        api_key=resolved_key,
        superagent_api_key=resolved_sa_key,
        budget=budget,
        max_tokens=max_tokens,
        tags=tags,
        recall_tags=recall_tags,
        recall_tags_match=recall_tags_match,
        guard_model=guard_model,
        redact_model=redact_model,
        redact_entities=redact_entities,
        redact_rewrite=redact_rewrite,
        enable_guard_on_retain=enable_guard_on_retain,
        enable_guard_on_recall=enable_guard_on_recall,
        enable_guard_on_reflect=enable_guard_on_reflect,
        enable_redact_on_retain=enable_redact_on_retain,
        enable_fallback=enable_fallback,
        fallback_timeout=fallback_timeout,
        verbose=verbose,
    )
    return _global_config


def get_config() -> HindsightSuperagentConfig | None:
    """Get the current global configuration."""
    return _global_config


def reset_config() -> None:
    """Reset global configuration to None."""
    global _global_config
    _global_config = None
