"""
Abstract interface for LLM providers.

This module defines the interface that all LLM providers must implement,
enabling support for multiple LLM backends (OpenAI, Anthropic, Gemini, Codex, etc.)
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Callable, Self

from .response_models import LLMToolCallResult

logger = logging.getLogger(__name__)


class LLMToolChoiceMode(StrEnum):
    """Canonical tool-selection modes shared by every LLM provider."""

    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"
    NAMED = "named"


@dataclass(frozen=True, slots=True)
class LLMToolChoice:
    """Typed internal tool selection serialized only at provider boundaries."""

    mode: LLMToolChoiceMode
    function_name: str | None = None

    def __post_init__(self) -> None:
        if self.mode is LLMToolChoiceMode.NAMED:
            if self.function_name is None or not self.function_name or self.function_name != self.function_name.strip():
                raise ValueError("Named tool choice requires a non-empty canonical function name")
        elif self.function_name is not None:
            raise ValueError(f"Tool choice mode {self.mode.value!r} cannot include a function name")

    @classmethod
    def named(cls, function_name: str) -> Self:
        return cls(mode=LLMToolChoiceMode.NAMED, function_name=function_name)

    @property
    def selected_function_name(self) -> str:
        if self.function_name is None:
            raise ValueError("Tool choice does not select a named function")
        return self.function_name


LLM_TOOL_CHOICE_AUTO = LLMToolChoice(mode=LLMToolChoiceMode.AUTO)
LLM_TOOL_CHOICE_NONE = LLMToolChoice(mode=LLMToolChoiceMode.NONE)
LLM_TOOL_CHOICE_REQUIRED = LLMToolChoice(mode=LLMToolChoiceMode.REQUIRED)


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
        reasoning_effort: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ):
        """
        Initialize LLM provider.

        Args:
            provider: Provider name (e.g., "openai", "codex", "anthropic", "gemini").
            api_key: API key or authentication token.
            base_url: Base URL for the API.
            model: Model name.
            reasoning_effort: Reasoning effort level, or None when the operator
                configured none — in which case no provider sends the parameter and
                every model runs at its own default effort.
            timeout: Per-request timeout in seconds, already resolved by the caller
                from the per-operation/global config (``consolidation_llm_timeout``
                falling back to ``llm_timeout``, etc.). ``None`` means unconfigured
                and lets each provider apply its own default.
            **kwargs: Additional provider-specific parameters.
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        # None means "the operator said nothing", and nothing is what gets sent: no
        # provider may invent a level. Hindsight used to resolve unset to "low" here and
        # ship it to whichever lanes their capability check happened to accept, which
        # made the setting both invisible (a configured value could be silently dropped —
        # issue #3449) and presumptuous (an unconfigured one was still transmitted).
        # An empty string is an unset environment variable, not a level.
        self.reasoning_effort: str | None = reasoning_effort or None
        # Every provider gets the resolved per-request timeout, even the ones that
        # do not use it: a provider that silently drops it (issue #3898 — Codex
        # never had the attribute at all, so a runaway response was read until the
        # backend gave up) is indistinguishable from one that honours it.
        self.timeout: float | None = timeout

    def _warn_reasoning_effort_unsupported(self) -> None:
        """Report, once at startup, that this provider cannot honour a configured effort.

        Providers with no reasoning knob to turn call this from ``__init__``. Silence is
        what made issue #3449 expensive: the variable is set, documented and visible in
        the environment, so every signal the operator has says it is in force. A setting
        this provider cannot act on has to say so out loud.
        """
        if self.reasoning_effort is None:
            return
        logger.warning(
            f"reasoning_effort={self.reasoning_effort!r} is ignored: the {self.provider} provider "
            f"has no reasoning-effort control. Remove the setting or switch provider to apply it."
        )

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
        attempt_context: Callable[[], AbstractAsyncContextManager[None]] | None = None,
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
            attempt_context: Factory for an async context manager holding the shared
                concurrency permits. Passed only when the provider declares
                ``supports_attempt_scoped_concurrency()``; the provider must enter it
                around each individual upstream request so retry backoff never
                occupies a permit.

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
        tool_choice: LLMToolChoice = LLM_TOOL_CHOICE_AUTO,
        cached_prefix: str | None = None,
        cached_prefix_message_count: int = 0,
        attempt_context: Callable[[], AbstractAsyncContextManager[None]] | None = None,
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
            tool_choice: Canonical tool-selection policy.
            attempt_context: Factory for an async context manager holding the shared
                concurrency permits — see ``call``.

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

    @property
    def batch_account_key(self) -> str:
        """Stable, non-secret identifier for the provider *account* this impl calls.

        A batch id only exists on the account that created it, so crash recovery
        has to resolve the exact configured member again. ``provider`` cannot do
        that on its own: two members of the same provider on different
        credentials are indistinguishable by name, so reordering them (or
        inserting a new one in front) would route a resume through the wrong
        account and poll a batch id it has never seen — issue #3671.

        The key is the endpoint identity (provider + base URL) plus a truncated
        SHA-256 of the credential: enough to tell two accounts apart, never the
        credential itself, since this is persisted in ``async_operations``
        metadata and quoted back in operator-facing errors. The model is
        deliberately excluded — a batch belongs to the account, not to the model
        a request happened to name — so retargeting a member's model still
        resumes. Rotating its API key does not: the key changes and recovery
        fails loudly naming both sides, which is the safe direction to fail in.

        A provider that rotates its credential in place (a refreshed OAuth/JWT
        token rather than an operator edit) must override this with a key built
        from the stable account identity — none of the batch-capable providers
        does so today.
        """
        credential = hashlib.sha256((self.api_key or "").encode("utf-8")).hexdigest()[:12]
        return f"{self.provider}|{self.base_url or ''}|{credential}"

    def supports_attempt_scoped_concurrency(self) -> bool:
        """Whether retries can acquire concurrency permits per upstream attempt."""
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

    # ── Step-by-step incremental prompt caching (optional) ─────────────────────
    #
    # For agentic loops (reflect) the dominant cost is the conversation prefix
    # re-sent every turn, not the static system prefix. Providers that can cache
    # a *growing* prefix implement these: the caller rolls one cache per step
    # (each covering the previous step's full input), passes its handle plus the
    # message count it covers to ``call_with_tools`` so only the new turns are
    # sent fresh, and tears the caches down when the loop ends. Default no-ops so
    # non-supporting providers transparently run uncached.

    def supports_incremental_prompt_cache(self) -> bool:
        """Whether this provider can cache a growing multi-turn conversation prefix."""
        return False

    async def create_incremental_cache(
        self,
        *,
        session_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> str | None:
        """Cache ``system + tools + messages`` and return an opaque handle, or None.

        The handle is passed back to ``call_with_tools(cached_prefix=...,
        cached_prefix_message_count=len(messages))``. Caches are grouped under
        ``session_id`` for teardown via ``delete_cache_session``. Returns None
        when caching is unavailable or the prefix is too small — caller falls
        back to an uncached call.
        """
        return None

    async def delete_cached_prefix(self, name: str) -> None:
        """Best-effort delete of a single cache handle (a superseded step)."""
        return None

    async def delete_cache_session(self, session_id: str) -> None:
        """Best-effort teardown of every cache created under ``session_id``."""
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


class ProviderContentPolicyError(RuntimeError):
    """Raised when a provider refuses a request on content-policy (AUP) grounds.

    A refusal is a deterministic function of the content, not a transient fault:
    the same prompt earns the same refusal on every attempt. Retrying it — inside
    the provider's transport loop or by re-running the whole worker task — burns
    identical calls for a guaranteed-identical failure (issue #3690), so both
    layers treat this as permanent: the provider raises it without retrying, and
    ``_is_non_retryable_task_error`` marks the operation failed on first sight.

    A ``RuntimeError`` subclass so existing ``except RuntimeError`` handlers
    around LLM calls keep behaving as they did before the class existed.
    """


class ProviderRateLimitResetError(Exception):
    """Raised when an upstream provider says quota will reopen at a known time."""

    def __init__(self, retry_at: datetime, message: str = "") -> None:
        self.retry_at = retry_at
        super().__init__(message)
