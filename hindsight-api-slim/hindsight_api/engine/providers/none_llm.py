"""
No-op LLM provider for chunk-only storage mode.

When the LLM provider is set to "none", the system operates without any LLM dependency.
Retain uses chunks mode (no fact extraction), and reflect/consolidation are disabled.
This provider acts as a safety net — if any code path unexpectedly tries to call the LLM,
it raises a clear error instead of a confusing connection failure.
"""

import logging
from typing import Any

from ..llm_interface import LLMInterface
from ..response_models import LLMToolCallResult

logger = logging.getLogger(__name__)


class LLMNotAvailableError(Exception):
    """Raised when an operation requires an LLM but the provider is set to 'none'."""

    pass


class NoneLLM(LLMInterface):
    """
    No-op LLM provider that rejects all LLM calls.

    Used when HINDSIGHT_API_LLM_PROVIDER=none to run Hindsight as a chunk store
    with semantic search but without LLM-based features (fact extraction, reflect,
    consolidation).
    """

    async def verify_connection(self) -> None:
        """No-op — no LLM connection to verify."""
        logger.debug("NoneLLM: no LLM connection to verify (provider=none)")

    async def call(
        self,
        messages: list[dict[str, str]],
        response_format: Any | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        scope: str = "memory",
        max_retries: int = 10,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        skip_validation: bool = False,
        strict_schema: bool = False,
        return_usage: bool = False,
    ) -> Any:
        """Raise LLMNotAvailableError — no LLM is configured."""
        raise LLMNotAvailableError(
            "LLM provider is set to 'none'. This operation requires an LLM. "
            "Set HINDSIGHT_API_LLM_PROVIDER to a real provider (e.g., openai, anthropic, gemini)."
        )

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        scope: str = "tools",
        max_retries: int = 5,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMToolCallResult:
        """Raise LLMNotAvailableError — no LLM is configured."""
        raise LLMNotAvailableError(
            "LLM provider is set to 'none'. This operation requires an LLM. "
            "Set HINDSIGHT_API_LLM_PROVIDER to a real provider (e.g., openai, anthropic, gemini)."
        )

    async def cleanup(self) -> None:
        """No-op — nothing to clean up."""
        pass
