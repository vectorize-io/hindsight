"""
Abstract interface for LLM providers.

This module defines the interface that all LLM providers must implement,
enabling support for multiple LLM backends (OpenAI, Anthropic, Gemini, Codex, etc.)
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from .response_models import LLMToolCallResult


class LLMInterface(ABC):
    """
    Abstract interface for LLM providers.

    All LLM provider implementations must inherit from this class and implement
    the required methods.
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        reasoning_effort: str = "low",
        **kwargs: Any,
    ):
        """
        Initialize LLM provider.

        Args:
            provider: Provider name (e.g., "openai", "codex", "anthropic", "gemini").
            api_key: API key or authentication token.
            base_url: Base URL for the API.
            model: Model name.
            reasoning_effort: Reasoning effort level for supported providers.
            **kwargs: Additional provider-specific parameters.
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.reasoning_effort = reasoning_effort

    @abstractmethod
    async def verify_connection(self) -> None:
        """
        Verify that the LLM provider is configured correctly by making a simple test call.

        Raises:
            RuntimeError: If the connection test fails.
        """
        pass

    @abstractmethod
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
        cached_prefix: str | None = None,
    ) -> Any:
        """
        Make an LLM API call with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            response_format: Optional Pydantic model for structured output.
            max_completion_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0-2.0).
            scope: Scope identifier for tracking.
            max_retries: Maximum retry attempts.
            initial_backoff: Initial backoff time in seconds.
            max_backoff: Maximum backoff time in seconds.
            skip_validation: Return raw JSON without Pydantic validation.
            strict_schema: Grammar-enforce structured output via json_schema strict
                (OpenAI-compatible, LiteLLM) instead of the soft json_object path. Gemini
                enforces its response_schema natively; providers without a strict mode ignore it.
            return_usage: If True, return tuple (result, TokenUsage) instead of just result.
            cached_prefix: Opaque handle from ``get_or_create_cached_prefix`` for the
                cacheable system prefix, or None. Providers without explicit prompt
                caching ignore it (and the wrapper only forwards it when set).

        Returns:
            If return_usage=False: Parsed response if response_format is provided, otherwise text content.
            If return_usage=True: Tuple of (result, TokenUsage) with token counts.

        Raises:
            OutputTooLongError: If output exceeds token limits.
            Exception: Re-raises API errors after retries exhausted.
        """
        pass

    @abstractmethod
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
        cached_prefix: str | None = None,
    ) -> LLMToolCallResult:
        """
        Make an LLM API call with tool/function calling support.

        Args:
            messages: List of message dicts. Can include tool results with role='tool'.
            tools: List of tool definitions in OpenAI format.
            max_completion_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0-2.0).
            scope: Scope identifier for tracking.
            max_retries: Maximum retry attempts.
            initial_backoff: Initial backoff time in seconds.
            max_backoff: Maximum backoff time in seconds.
            tool_choice: How to choose tools - "auto", "none", "required", or specific function.

        Returns:
            LLMToolCallResult with content and/or tool_calls.
        """
        pass

    async def supports_batch_api(self) -> bool:
        """
        Check if this provider supports batch API operations.

        Returns:
            True if provider supports submit_batch/get_batch_status/retrieve_batch_results
        """
        return False

    # ── Prompt prefix caching (optional, per-provider) ─────────────────────────

    def supports_prompt_caching(self) -> bool:
        """Whether this provider can cache a reusable prompt prefix.

        Default False. Providers that return True must implement
        ``get_or_create_cached_prefix`` and honour the ``cached_prefix`` argument
        of ``call`` / ``call_with_tools``.
        """
        return False

    async def get_or_create_cached_prefix(
        self,
        *,
        system_instruction: str,
        response_schema: Any | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str | None:
        """Cache a reusable prompt prefix and return an opaque handle, or None.

        The engine has already decided WHAT is cacheable: it puts the stable,
        bank-agnostic instructions in ``system_instruction`` (plus ``tools``) and
        keeps all per-request / per-bank data (documents, facts, the bank mission)
        in the user message. A provider only chooses HOW to cache that prefix:

        - Explicit-cache providers (e.g. Gemini ``CachedContent``): create the
          cache, return its handle; the engine passes the handle back via
          ``call(cached_prefix=...)`` and the provider then drops the prefix from
          the request, billing it at the cached rate.
        - Automatic-cache providers (e.g. OpenAI): no handle needed — caching is
          transparent as long as the prefix is a stable leading block, which it
          already is. They can keep this default (return None) and still benefit.
        - Inline-marker providers (e.g. Anthropic ``cache_control``): mark the
          prefix block inside ``call`` instead; may also keep this default.

        Returns None when caching is disabled/unsupported or the prefix is too
        small; callers MUST fall back to an uncached call in that case.
        """
        return None

    async def submit_batch(
        self,
        requests: list[dict[str, Any]],
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
    ) -> dict[str, Any]:
        """
        Submit a batch of requests to the provider's batch API.

        Args:
            requests: List of request dicts in JSONL format (custom_id, method, url, body)
            endpoint: API endpoint for the batch (e.g., "/v1/chat/completions")
            completion_window: Completion window (e.g., "24h")

        Returns:
            Dict with batch metadata: {"batch_id": str, "status": str, ...}

        Raises:
            NotImplementedError: If provider doesn't support batch API
        """
        raise NotImplementedError(f"Batch API not supported for provider: {self.provider}")

    async def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        """
        Get the status of a batch job.

        Args:
            batch_id: Batch identifier returned from submit_batch

        Returns:
            Dict with status info: {"batch_id": str, "status": str, "completed_at": str, ...}

        Raises:
            NotImplementedError: If provider doesn't support batch API
        """
        raise NotImplementedError(f"Batch API not supported for provider: {self.provider}")

    async def retrieve_batch_results(self, batch_id: str) -> list[dict[str, Any]]:
        """
        Retrieve completed batch results.

        Args:
            batch_id: Batch identifier returned from submit_batch

        Returns:
            List of result dicts (one per request, matched by custom_id)

        Raises:
            NotImplementedError: If provider doesn't support batch API
        """
        raise NotImplementedError(f"Batch API not supported for provider: {self.provider}")

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources (close connections, etc.)."""
        pass


class OutputTooLongError(Exception):
    """
    Bridge exception raised when LLM output exceeds token limits.

    This wraps provider-specific errors (e.g., OpenAI's LengthFinishReasonError)
    to allow callers to handle output length issues without depending on
    provider-specific implementations.
    """

    pass


class ProviderRateLimitResetError(Exception):
    """Raised when an upstream provider says quota will reopen at a known time."""

    def __init__(self, retry_at: datetime, message: str = "") -> None:
        self.retry_at = retry_at
        super().__init__(message)
