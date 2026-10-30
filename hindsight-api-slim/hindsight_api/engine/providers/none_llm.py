"""
No-op LLM provider for chunk-only storage mode.

When the LLM provider is set to "none", the system operates without any LLM dependency.
Retain uses chunks mode (no fact extraction), and reflect/consolidation are disabled.
This provider acts as a safety net — if any code path unexpectedly tries to call the LLM,
it raises a clear error instead of a confusing connection failure.
"""

import logging
from contextlib import AbstractAsyncContextManager
from typing import Any, Callable

from ..llm_interface import LLMInterface, ProviderUnavailableError
from ..response_models import LLMCallResult

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

    def supports_vision(self) -> bool:
        """False, definitively: there is no model here to look at an image."""
        return False

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
        attempt_context: Callable[[], AbstractAsyncContextManager[None]] | None = None,
    ) -> LLMCallResult:
        """Raise LLMNotAvailableError — no LLM is configured."""
        raise LLMNotAvailableError(
            "LLM provider is set to 'none'. This operation requires an LLM. "
            "Set HINDSIGHT_API_LLM_PROVIDER to a real provider (e.g., openai, anthropic, gemini)."
        )

    async def cleanup(self) -> None:
        """No-op — nothing to clean up."""
        pass


class UnavailableLLM(NoneLLM):
    """Stand-in for a provider whose sign-in was missing when it was built.

    Every call and verification raises the original error, so a failover chain
    moves on to its next member and the error says which login to run. Unlike
    ``NoneLLM`` it is not "no LLM": its provider name stays the real one, so
    nothing treats it as chunk-only mode.
    """

    def __init__(self, error: ProviderUnavailableError, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._error = error

    async def verify_connection(self) -> None:
        raise self._error

    def supports_vision(self) -> None:
        # Unknown, not False: the real provider's answer is what matters once it is back.
        return None

    async def call(self, *args: Any, **kwargs: Any) -> LLMCallResult:
        raise self._error
