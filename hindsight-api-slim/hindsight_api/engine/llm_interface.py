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

from .response_models import LLMCallResult

logger = logging.getLogger(__name__)


class LLMToolChoiceMode(StrEnum):
    """What a tool-calling turn may call: one function of the model's choosing (``auto`` and
    ``required`` both mean exactly one, see ``LLMProvider.call_with_tools``), or a named one."""

    AUTO = "auto"
    REQUIRED = "required"
    NAMED = "named"


@dataclass(frozen=True, slots=True)
class LLMToolChoice:
    """A turn's tool selection; it shapes the turn's response schema and is never sent to a provider."""

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


@dataclass(frozen=True, slots=True)
class PromptCachePrefix:
    """The reusable prompt prefix a caller wants cached, passed to ``LLMProvider.call``.

    The caller says WHAT is cacheable; the provider that serves the call builds
    its own handle from it. In a multi-LLM chain that is whichever member runs,
    so a fallback member caches against its own account instead of being handed
    the primary's handle, or none at all (#5123). Callers used to resolve the
    handle themselves via ``llm_config._provider_impl``, which in a chain is the
    primary only.
    """

    system_instruction: str
    # Part of the cache key only; ``None`` keeps one cache across schema variants.
    response_schema: Any | None = None


LLM_TOOL_CHOICE_AUTO = LLMToolChoice(mode=LLMToolChoiceMode.AUTO)
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
        cached_prefix: str | None = None,
        attempt_context: Callable[[], AbstractAsyncContextManager[None]] | None = None,
    ) -> LLMCallResult:
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
            cached_prefix: Opaque handle from ``get_or_create_cached_prefix`` for the
                cacheable system prefix, or None. Providers without explicit prompt
                caching ignore it (and the wrapper only forwards it when set).
            attempt_context: Factory for an async context manager holding the shared
                concurrency permits. Passed only when the provider declares
                ``supports_attempt_scoped_concurrency()``; the provider must enter it
                around each individual upstream request so retry backoff never
                occupies a permit.

        Returns:

        Raises:
            OutputTooLongError: If output exceeds token limits.
            Exception: Re-raises API errors after retries exhausted.
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

    def supports_request_overrides(self) -> bool:
        """Whether a call can be sent with the credentials an LLM request extension chose.

        A provider that returns True must apply ``current_llm_request_overrides()``
        (from ``hindsight_api.extensions.llm_request``) to every request it sends: the
        ``api_key`` in place of its own, and the ``headers`` added. One that cannot —
        OAuth/CLI-session providers, a router whose keys live per deployment — keeps
        False, and an overridden call to it is refused instead of going out on the
        server's credentials.
        """
        return False

    def supports_vision(self) -> bool | None:
        """Whether this provider can accept image parts in a user message.

        Three-valued on purpose. ``True``/``False`` mean the provider knows;
        ``None`` — the default — means it cannot tell, which is the honest answer
        for a gateway (LiteLLM, Ollama, LM Studio, an OpenAI-compatible proxy)
        that will happily forward whatever model name it is given.

        Retain treats ``None`` as "refuse": an item carrying images is rejected
        rather than sent to a model that would ignore them, because silently
        dropping an image is exactly the lossy behaviour inline images exist to
        remove. Operators running a vision model behind a gateway say so with
        ``HINDSIGHT_API_LLM_VISION=true``, which overrides this.
        """
        return None

    # ── Prompt prefix caching (optional, per-provider) ─────────────────────────

    def supports_prompt_caching(self) -> bool:
        """Whether this provider can cache a reusable prompt prefix.

        Default False. Providers that return True must implement
        ``get_or_create_cached_prefix`` and honour the ``cached_prefix`` argument
        of ``call``.
        """
        return False

    async def get_or_create_cached_prefix(
        self,
        *,
        system_instruction: str,
        response_schema: Any | None = None,
    ) -> str | None:
        """Cache a reusable prompt prefix and return an opaque handle, or None.

        The engine has already decided WHAT is cacheable: it puts the stable,
        bank-agnostic instructions in ``system_instruction`` and
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


class ProviderAuthError(RuntimeError):
    """Raised when the provider rejected the request's credentials (HTTP 401/403).

    A rejection is a property of the configured credential, not of the moment or
    the content: every replay sends the same key and earns the same rejection
    (issue #5644). The providers already refuse to retry it; this type carries
    that permanence through ``extract_facts_from_text`` to the worker, so the
    retain fails once instead of replaying every chunk through the retry schedule.
    """


class ProviderUnavailableError(RuntimeError):
    """Raised while building a provider whose sign-in is missing or revoked.

    It means "this member cannot serve until someone logs in again", not "the
    configuration is wrong": ``LLMProvider`` turns it into a member that fails
    every call, so a failover chain moves on to its next member and the server
    still starts (issue #6095). Any other construction error stays fatal.
    """


class ProviderRateLimitResetError(Exception):
    """Raised when an upstream provider says quota will reopen at a known time."""

    def __init__(self, retry_at: datetime, message: str = "") -> None:
        self.retry_at = retry_at
        super().__init__(message)
