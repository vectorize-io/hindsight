"""Which LLM providers need an API key.

A leaf module on purpose. ``config.py`` needs this answer while it builds
``HindsightConfig``, and ``llm_wrapper`` needs the built config at import time to
size its process-wide semaphores; keeping the fact here is what stops those two
from importing each other in a cycle.
"""

from __future__ import annotations

_PROVIDERS_WITHOUT_API_KEY = frozenset(
    {
        "ollama",
        "lmstudio",
        "llamacpp",
        "openai-codex",
        "claude-code",
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


def requires_api_key(provider: str) -> bool:
    """Return True if the given provider requires an API key to operate."""
    return provider.lower() not in _PROVIDERS_WITHOUT_API_KEY
