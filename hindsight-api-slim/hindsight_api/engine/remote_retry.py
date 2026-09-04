"""Bounded retry/backoff for remote model APIs (embeddings and rerankers).

Extracted from ``embeddings.py`` so the rerankers can share it rather than grow a
fourth private copy: TEI had its own loop, the LiteLLM embedding backends had this
one, and Gemini/Cohere/ZeroEntropy each nearly grew another (#4103, #4134). The
policy is deliberately provider-agnostic — it classifies on HTTP status and
transport-level exception shape, never on an SDK type — so a new backend wires in
by passing a policy rather than by re-deriving what "transient" means.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")


# 4xx codes that describe a transient condition rather than a bad request.
# Everything else in the 4xx range (auth, validation, not-found) is a client-side
# problem that retrying cannot fix.
_TRANSIENT_STATUS_CODES = frozenset({408, 409, 425, 429})

# Exception class names treated as transient when the exception carries no HTTP
# status code. Matching by name keeps this module free of a hard litellm/openai
# import (litellm is imported lazily, and only by the providers that need it).
_TRANSIENT_EXCEPTION_NAMES = frozenset(
    {
        "APIConnectionError",
        "APIError",
        "APITimeoutError",
        "ConnectionError",
        "InternalServerError",
        "RateLimitError",
        "ServiceUnavailableError",
        "Timeout",
        "TimeoutError",
    }
)


@dataclass(frozen=True)
class RetryPolicy:
    """
    Bounded retry policy for a remote model API (embeddings, rerank, ...).

    Recall embeds its query and reranks its candidates inline on the request path,
    so a single upstream 5xx would otherwise become a user-visible recall failure.
    Retries are bounded two ways at once:

    * ``max_retries`` caps the number of extra attempts (0 disables retrying).
    * ``budget_seconds`` caps the wall-clock time a single logical call may
      *waste* on retries — failed attempts plus backoff sleeps. Successful work
      never counts against it, so a large multi-batch call is not penalised for the
      batches that worked, while the worst-case added latency stays bounded.

    The pairing matters: attempts alone cannot bound latency (an upstream that
    fails slowly turns 5 attempts into minutes), and a budget alone cannot stop a
    fast-failing upstream from being hammered.
    """

    # Defaults for a provider constructed without an explicit policy (tests, direct
    # instantiation). Production paths pass the resolved HINDSIGHT_API_* values; these
    # mirror the embedding defaults the policy shipped with.
    max_retries: int = 4
    initial_backoff: float = 0.5
    max_backoff: float = 4.0
    budget_seconds: float = 15.0

    def new_budget(self) -> "RetryBudget":
        """Start a fresh retry budget, scoped to one logical embedding call."""
        return RetryBudget(self.budget_seconds)


class RetryBudget:
    """Mutable remaining-retry-time counter shared across the batches of one call.

    The batches of one call may go out concurrently (see ``Embeddings._encode_batched``),
    so the counter is touched from several threads at once; without the lock a
    read-modify-write race would under-charge the budget and let retries run past it.
    """

    __slots__ = ("remaining", "_lock")

    def __init__(self, seconds: float):
        self.remaining = max(0.0, seconds)
        self._lock = threading.Lock()

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0.0

    def spend(self, seconds: float) -> None:
        with self._lock:
            self.remaining = max(0.0, self.remaining - max(0.0, seconds))


def status_code_of(exc: BaseException) -> int | None:
    """Best-effort HTTP status extraction across httpx, openai, litellm and google.genai errors."""
    candidates = (
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
        # google.genai.errors.APIError carries the status on `code`, and leaves
        # `response` as None on the paths that raise from a parsed error body.
        getattr(exc, "code", None),
    )
    for candidate in candidates:
        if isinstance(candidate, bool):
            continue
        if isinstance(candidate, str) and candidate.isdigit():
            candidate = int(candidate)
        # Range-checked because `code` is a common attribute name that is not
        # always an HTTP status (SystemExit.code, OSError subclasses).
        if isinstance(candidate, int) and 100 <= candidate <= 599:
            return candidate
    return None


def is_transient_remote_error(exc: BaseException) -> bool:
    """
    Return True when ``exc`` is worth retrying.

    A status code, when present, is authoritative: 5xx and the transient 4xx set
    are retryable, every other 4xx (401/403 auth, 400/422 validation, 404) is
    permanent and must fail fast. Without a status code we fall back to
    transport-level exception types and known SDK exception names.
    """
    status = status_code_of(exc)
    if status is not None:
        return status >= 500 or status in _TRANSIENT_STATUS_CODES
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return type(exc).__name__ in _TRANSIENT_EXCEPTION_NAMES


def _retry_delay_for(
    exc: BaseException,
    *,
    attempt: int,
    attempts: int,
    policy: RetryPolicy,
    budget: RetryBudget,
    provider: str,
) -> float | None:
    """
    Decide whether ``exc`` should be retried and how long to wait first.

    Returns the sleep duration, or None when the exception must propagate. Logs
    the decision and charges the sleep to ``budget``. Upstream error text is
    truncated (matching the LLM providers) so a verbose provider payload cannot
    flood the log.
    """
    if not is_transient_remote_error(exc):
        return None

    status = status_code_of(exc)
    status_label = f"HTTP {status}" if status is not None else type(exc).__name__
    detail = str(exc)[:200]

    if attempt >= attempts - 1:
        logger.error(f"{provider} call failed after {attempts} attempt(s) ({status_label}): {detail}")
        return None

    if budget.exhausted:
        logger.error(
            f"{provider} call failed on attempt {attempt + 1}/{attempts} ({status_label}) and the "
            f"{policy.budget_seconds:.1f}s retry budget is exhausted, giving up: {detail}"
        )
        return None

    backoff = min(policy.initial_backoff * (2**attempt), policy.max_backoff)
    jitter = backoff * 0.2 * (2 * (time.time() % 1) - 1)
    sleep_for = min(max(0.0, backoff + jitter), budget.remaining)
    budget.spend(sleep_for)
    logger.warning(
        f"{provider} call failed (attempt {attempt + 1}/{attempts}, {status_label}), "
        f"retrying in {sleep_for:.2f}s ({budget.remaining:.1f}s of retry budget left): {detail}"
    )
    return sleep_for


def call_with_retry(
    call: Callable[[], T],
    *,
    policy: RetryPolicy,
    budget: RetryBudget,
    provider: str,
) -> T:
    """Run a blocking remote call, retrying transient upstream failures."""
    attempts = policy.max_retries + 1
    for attempt in range(attempts):
        started = time.monotonic()
        try:
            return call()
        except Exception as exc:
            budget.spend(time.monotonic() - started)
            delay = _retry_delay_for(
                exc, attempt=attempt, attempts=attempts, policy=policy, budget=budget, provider=provider
            )
            if delay is None:
                raise
            time.sleep(delay)
    raise RuntimeError("unreachable: retry loop exited without returning or raising")


async def acall_with_retry(
    call: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    budget: RetryBudget,
    provider: str,
) -> T:
    """Async twin of :func:`call_with_retry`, sharing its policy and classification."""
    attempts = policy.max_retries + 1
    for attempt in range(attempts):
        started = time.monotonic()
        try:
            return await call()
        except Exception as exc:
            budget.spend(time.monotonic() - started)
            delay = _retry_delay_for(
                exc, attempt=attempt, attempts=attempts, policy=policy, budget=budget, provider=provider
            )
            if delay is None:
                raise
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable: retry loop exited without returning or raising")
