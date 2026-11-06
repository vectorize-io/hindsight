"""Which LLM providers need an API key.

A leaf module on purpose. ``config.py`` needs this answer while it builds
``HindsightConfig``, and ``llm_wrapper`` needs the built config at import time to
size its process-wide semaphores; keeping the fact here is what stops those two
from importing each other in a cycle. It sits outside ``engine/`` because importing
anything under that package first runs ``engine/__init__``, which loads the whole
engine (and torch, via sentence-transformers) — seconds of work that config parsing,
and so ``hindsight-api --help``, must not pay.
"""

from __future__ import annotations

import os

_PROVIDERS_WITHOUT_API_KEY = frozenset(
    {
        "ollama",
        "lmstudio",
        "llamacpp",
        "openai-codex",
        "chatgpt",
        "claude-code",
        "cursor",
        "github-copilot",
        "mock",
        "none",
        "vertexai",
        "litellm",
        "litellmrouter",
        "bedrock",
        "nous",
        "xai-oauth",
    }
)


def is_custom_llm_provider(provider: str) -> bool:
    """Whether ``provider`` names an operator's own LLMInterface class (``package.module:ClassName``).

    No built-in provider name contains a colon, so the colon alone tells them apart.
    """
    return ":" in provider


def requires_api_key(provider: str) -> bool:
    """Return True if the given provider requires an API key to operate."""
    # A custom provider decides for itself; it can refuse a missing key in verify_connection.
    if is_custom_llm_provider(provider):
        return False
    provider = provider.lower()
    # Workload Identity Federation: the Anthropic SDK trades a platform identity token
    # for short-lived access tokens itself, so no static key is needed (issue #5502).
    if provider == "anthropic" and os.getenv("ANTHROPIC_FEDERATION_RULE_ID"):
        return False
    return provider not in _PROVIDERS_WITHOUT_API_KEY
