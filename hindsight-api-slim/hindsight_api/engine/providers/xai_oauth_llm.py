"""
SuperGrok subscription LLM provider (``xai-oauth``).

Serves Hindsight's LLM lanes from a flat-rate consumer SuperGrok subscription
authenticated by a user-consented device-code OAuth grant — no API key, no
per-token metering — joining the engine's existing subscription-provider
category (``claude-code``, ``openai-codex``, ``nous``). For API-key access to
the same endpoint, use ``provider: openai`` with an ``api.x.ai`` base URL; the
credential is what this provider adds, not the endpoint.

What it talks to
----------------
``https://api.x.ai/v1``, xAI's published API, spoken plainly: an
``Authorization: Bearer`` header carrying the OAuth access token, the standard
content headers, and the cache-affinity hint below. No client-identity or
version headers are sent.

Why not ``OpenAICompatibleLLM``
-------------------------------
The wire format is OpenAI-shaped, but the credential is a rotating OAuth token
with proactive/reactive refresh against a shared on-disk store, and xAI's 403
responses need their own classification (a spending-limit stop is not a
credential failure). Neither fits the OpenAI SDK client the compatible provider
is built on, so the transport is a hand-written ``httpx.AsyncClient`` in the
``codex_llm`` mould.

Logging
-------
Bodies, credentials, and header values are never logged by default: byte
counts, the model name, status codes and durations only. The one gated
exception is ``HINDSIGHT_API_XAI_OAUTH_DEBUG_HEADERS`` (default off), which
logs a small allowlist of non-credential response headers (``via``,
``x-request-id``, ``cf-ray``, ``server``, ``date``) on a non-2xx reply only,
for diagnosing an otherwise untraceable upstream failure. It never fires on a
2xx reply, never logs the request's own headers (which carry the bearer
token), and never logs a body.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import AbstractAsyncContextManager, nullcontext, suppress
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field

from hindsight_api.config import DEFAULT_LLM_TIMEOUT, ENV_LLM_TIMEOUT
from hindsight_api.engine.cache_affinity import XAI_CONV_ID_HEADER, cache_affinity_id
from hindsight_api.engine.llm_interface import LLM_TOOL_CHOICE_AUTO, LLMInterface, LLMToolChoice, LLMToolChoiceMode
from hindsight_api.engine.llm_trace import LLMResponseUsage, stash_response_usage
from hindsight_api.engine.providers.xai_oauth_auth import (
    DEFAULT_REFRESH_SKEW_SECONDS,
    LOGIN_COMMAND,
    XaiOAuthError,
    XaiOAuthLoginRequiredError,
    XaiOAuthManager,
    XaiOAuthRefreshError,
)
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, TokenUsage
from hindsight_api.engine.structured_output import provider_json_schema, strict_json_schema
from hindsight_api.metrics import get_metrics_collector
from hindsight_api.worker.stage import set_stage

logger = logging.getLogger(__name__)

__all__ = [
    "XaiOAuthEntitlementError",
    "XaiOAuthError",
    "XaiOAuthLLM",
    "XaiOAuthLoginRequiredError",
    "XaiOAuthQuotaExhaustedError",
    "XaiOAuthRefreshError",
]

#: Provider-specific base URL override. It wins over the generic
#: ``HINDSIGHT_API_LLM_BASE_URL`` that deployments often set for an unrelated
#: proxy — the same "more specific beats more general" rule the rest of
#: Hindsight's config hierarchy follows.
ENV_BASE_URL = "HINDSIGHT_API_XAI_OAUTH_BASE_URL"

DEFAULT_BASE_URL = "https://api.x.ai/v1"

#: Body marker xAI returns when the account's spending limit stopped the call.
SPENDING_LIMIT_CODE = "personal-team-blocked:spending-limit"

#: ``reasoning_effort`` value that means "no explicit effort". xAI rejects it
#: as a value, so this lane expresses it by omitting the field entirely.
REASONING_EFFORT_NONE = "none"

#: Longest upstream error message carried into an exception. The full body is
#: never echoed (it can be arbitrarily large and is not ours to relay); the
#: recognized error fields are, truncated, because a permanent 4xx that
#: reports only a byte count cannot be diagnosed.
MAX_ERROR_DETAIL_CHARS = 200

#: Debug-only response-header logging on a non-2xx reply (default off). See
#: the module docstring's Logging section for the exact carve-out.
ENV_DEBUG_HEADERS = "HINDSIGHT_API_XAI_OAUTH_DEBUG_HEADERS"

#: Response headers safe to log verbatim under ``ENV_DEBUG_HEADERS``: routing
#: and diagnostic metadata that names no credential and carries no request
#: data. Never authorization, never cookies, never the body.
_DEBUG_HEADER_ALLOWLIST = ("via", "x-request-id", "cf-ray", "server", "date")


class XaiOAuthQuotaExhaustedError(RuntimeError):
    """Raised when xAI stops the call at the account's spending limit.

    The credential is healthy — this is a quota state, so it never triggers a
    token refresh or quarantine.
    """


class XaiOAuthEntitlementError(RuntimeError):
    """Raised when xAI refuses API access to an otherwise valid OAuth grant.

    Also not a credential failure: refreshing or re-logging in does not change
    an entitlement decision, so neither is attempted.
    """


class _PromptTokensDetails(BaseModel):
    cached_tokens: int = 0


class _CompletionTokensDetails(BaseModel):
    reasoning_tokens: int = 0


class _ChatUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: _PromptTokensDetails | None = None
    completion_tokens_details: _CompletionTokensDetails | None = None


class _ChatToolCallFunction(BaseModel):
    name: str = ""
    arguments: str = ""


class _ChatToolCall(BaseModel):
    id: str = ""
    function: _ChatToolCallFunction = Field(default_factory=_ChatToolCallFunction)


class _ChatMessage(BaseModel):
    content: str | None = None
    tool_calls: list[_ChatToolCall] | None = None


class _ChatChoice(BaseModel):
    message: _ChatMessage | None = None
    finish_reason: str | None = None


class _ChatCompletion(BaseModel):
    """The upstream's OpenAI-shaped completion response.

    Parsed at the boundary so the rest of the provider works on typed
    attributes instead of ``dict.get`` chains. Unknown fields are ignored
    (Pydantic's default), so upstream additions do not break the lane.
    """

    choices: list[_ChatChoice] = Field(default_factory=list)
    usage: _ChatUsage | None = None


@dataclass(frozen=True, slots=True)
class _TokenCounts:
    """Normalized token counts for one response.

    ``output_tokens`` is visible-only, matching the rest of the engine's
    providers (see ``openai_compatible_llm._usage_from_openai_response``,
    which folds reasoning into ``completion_tokens`` and then subtracts it
    back out): reasoning is surfaced separately in ``thoughts_tokens`` rather
    than double-counted inside ``output_tokens``/``total_tokens``.

    xAI does not reliably follow the OpenAI o1/o3 convention of folding
    ``reasoning_tokens`` into ``completion_tokens``. A widely used
    OpenAI-compatible client library that talks to the same upstream API
    documents needing to detect and correct for exactly this, folding only
    when ``total_tokens == prompt_tokens + completion_tokens +
    reasoning_tokens`` -- the signature of the unfolded shape, where
    ``completion_tokens`` is already visible-only. Subtracting
    ``reasoning_tokens`` from ``completion_tokens`` unconditionally is
    therefore unsafe: on the unfolded shape it silently clamps a real,
    non-empty completion down to 0 visible output. ``total_tokens`` carries
    the same value (prompt + visible + reasoning) under both shapes, so
    ``output_tokens`` is derived from it instead of from the ambiguous
    ``completion_tokens`` field.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int
    thoughts_tokens: int


@dataclass(frozen=True, slots=True)
class _UpstreamReply:
    """One raw HTTP reply from the upstream. Body text is never logged."""

    status_code: int
    body_text: str
    retry_after: float | None = None


@dataclass(frozen=True, slots=True)
class _CompletionContent:
    """Message text plus the finish reason that came with it."""

    text: str
    finish_reason: str | None


class _UpstreamStatusError(RuntimeError):
    """An upstream reply the caller cannot use, and whether retrying may help.

    Covers both non-2xx statuses and success responses whose shape is unusable
    (no choices, empty content) — from a retry loop's point of view those are
    the same decision, and the message carries the distinction. ``status_code``
    is the actual HTTP status when this came from one (None for a 2xx reply
    with an unusable body/shape); the retry loop reads it to decide whether a
    retryable failure also warrants dropping the pinned connection.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        retry_after: float | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after
        self.status_code = status_code


def _token_counts(usage: _ChatUsage | None) -> _TokenCounts:
    """Build normalized counts from the upstream usage block.

    A response that omits ``prompt_tokens_details`` reads 0 cached tokens — an
    instrument gap, not an error.
    """
    if usage is None:
        return _TokenCounts(input_tokens=0, output_tokens=0, total_tokens=0, cached_tokens=0, thoughts_tokens=0)

    cached = usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details else 0
    thoughts = usage.completion_tokens_details.reasoning_tokens if usage.completion_tokens_details else 0

    if not thoughts:
        return _TokenCounts(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cached_tokens=cached,
            thoughts_tokens=0,
        )

    # total_tokens reads as prompt + visible + reasoning under both the
    # OpenAI-folded shape (completion_tokens already includes reasoning) and
    # xAI's unfolded shape (completion_tokens is visible-only) -- see the
    # class docstring. Deriving output from total sidesteps ever needing to
    # know which shape this particular reply used.
    total = max(0, usage.total_tokens - thoughts)
    output = max(0, total - usage.prompt_tokens)
    return _TokenCounts(
        input_tokens=usage.prompt_tokens,
        output_tokens=output,
        total_tokens=total,
        cached_tokens=cached,
        thoughts_tokens=thoughts,
    )


def _strip_code_fence(content: str) -> str:
    """Unwrap a markdown-fenced JSON payload, if the model produced one."""
    if "```json" in content:
        return content.split("```json")[1].split("```")[0].strip()
    if "```" in content:
        return content.split("```")[1].split("```")[0].strip()
    return content


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Read ``Retry-After`` as delta-seconds, or None.

    Only the delta-seconds form is honored; the HTTP-date form falls back to
    the caller's backoff schedule (not enforced here).
    """
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        seconds = float(raw.strip())
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def _actual_host(base_url: str) -> str:
    """Return the host a request to ``base_url`` actually targets.

    Error messages that name a fixed host go stale the moment a deployment
    points ``base_url`` at something else -- a misconfigured proxy, a
    regional mirror -- and then assert a system that was never contacted,
    sending diagnosis to the wrong place (measured in production: a
    misrouted ``base_url`` returning 502s, with every error message still
    blaming ``api.x.ai``). Falls back to the raw ``base_url`` string when it
    doesn't parse into a URL with a host, so the message still carries some
    signal rather than a hardcoded assumption. ``urlsplit().hostname`` never
    includes userinfo (``user:pass@host``) or the path/query, so a credential
    accidentally embedded in a misconfigured URL cannot leak into a log or
    error message even though ``base_url`` should never carry one.
    """
    try:
        hostname = urlsplit(base_url).hostname
    except ValueError:
        return base_url
    return hostname or base_url


def _debug_headers_enabled() -> bool:
    return os.getenv(ENV_DEBUG_HEADERS, "false").lower() == "true"


def _log_non_2xx_response_headers(response: httpx.Response) -> None:
    """Log status + an allowlisted subset of response headers, non-2xx only.

    Gated behind ``ENV_DEBUG_HEADERS`` (default off): the module's logging
    discipline is to never surface body or header content that could carry
    anything sensitive, and this is the one deliberate, allowlisted carve-out
    -- for a non-2xx reply that would otherwise leave no trace beyond a
    status code and a byte count. Never called for a 2xx reply, and never
    logs the request's own headers (which carry the bearer token) or a body,
    flag or no flag.
    """
    if not _debug_headers_enabled():
        return
    present = {name: response.headers[name] for name in _DEBUG_HEADER_ALLOWLIST if name in response.headers}
    logger.info("xai-oauth non-2xx reply: status=%s headers=%s", response.status_code, present)


class XaiOAuthLLM(LLMInterface):
    """LLM provider backed by a SuperGrok subscription via an xAI OAuth grant."""

    def __init__(
        self,
        provider: str,
        api_key: str,  # Ignored: the credential is the OAuth grant in the token store.
        base_url: str,
        model: str,
        reasoning_effort: str | None = None,
        timeout: float | None = None,
        auth_manager: XaiOAuthManager | None = None,
        **kwargs: Any,
    ):
        """Initialize the provider.

        The credential is not read here: a store that needs a login should fail
        the first call with the login remediation rather than the process
        start, so an unrelated lane can still serve.
        """
        super().__init__(provider, api_key, base_url, model, reasoning_effort, **kwargs)

        self.base_url = (os.environ.get(ENV_BASE_URL, "").strip() or self.base_url or DEFAULT_BASE_URL).rstrip("/")

        # Honour the engine-resolved per-operation timeout; fall back to the
        # same global default the OpenAI-compatible providers use.
        self.timeout = timeout if timeout is not None else float(os.getenv(ENV_LLM_TIMEOUT, str(DEFAULT_LLM_TIMEOUT)))

        self._auth = auth_manager or XaiOAuthManager()
        self._auth_lock = asyncio.Lock()
        self._client_lock = asyncio.Lock()
        self._client = httpx.AsyncClient(timeout=self.timeout)
        # In-flight request count per client object, and the future that fires
        # when a retired client's last request lands. Both keyed by the client
        # itself, because a recycle can leave more than one alive at a time.
        self._inflight: dict[Any, int] = {}
        self._drained: dict[Any, asyncio.Future[None]] = {}
        self._draining: set[asyncio.Task[None]] = set()

        logger.info("xai-oauth provider initialized: model=%s base_url=%s", self.model, self.base_url)

    # ------------------------------------------------------------------
    # Credential
    # ------------------------------------------------------------------

    def _admission_ttl(self) -> float:
        """Minimum token life required to admit a request.

        Both bars apply: refresh early enough that a long call cannot straddle
        expiry (the skew), and never admit a request the token cannot outlive
        (the request timeout). Passing the timeout alone would silently defeat
        the skew on short-timeout lanes.
        """
        return max(DEFAULT_REFRESH_SKEW_SECONDS, self.timeout)

    async def _access_token(self) -> str:
        """Return a token good for at least :meth:`_admission_ttl` seconds."""
        return await asyncio.to_thread(self._auth.get_access_token, self._admission_ttl())

    async def _refresh_after_rejection(self, rejected: str) -> str:
        """Refresh once after the API rejected ``rejected``.

        The asyncio lock keeps concurrent coroutines to one refresh attempt;
        the manager's own store lock and recheck cover threads and sibling
        provider instances.
        """
        async with self._auth_lock:
            return await asyncio.to_thread(
                lambda: self._auth.refresh(reason="reactive (HTTP 401 from api.x.ai)", rejected_token=rejected)
            )

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    async def _post(self, body: dict[str, Any], token: str, conv_id: str | None) -> _UpstreamReply:
        """POST one chat completion.

        Header values never reach the log, with one gated exception: a
        non-2xx reply's allowlisted diagnostic headers (see
        ``_log_non_2xx_response_headers``) under ``ENV_DEBUG_HEADERS``
        (default off). The request's own headers -- which carry the bearer
        token -- are never logged either way.
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }
        if conv_id:
            headers[XAI_CONV_ID_HEADER] = conv_id

        # Pin the client for the whole request: a concurrent recycle swaps
        # self._client, and this request must both finish on the connection it
        # started on and be counted against that client, not its replacement.
        client = self._client
        self._inflight[client] = self._inflight.get(client, 0) + 1
        try:
            response = await client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
        finally:
            self._release_client(client)

        if not (200 <= response.status_code < 300):
            _log_non_2xx_response_headers(response)
        return _UpstreamReply(
            status_code=response.status_code,
            body_text=response.text,
            retry_after=_retry_after_seconds(response),
        )

    def _new_client(self) -> httpx.AsyncClient:
        """Build a replacement pooled client. A seam tests override directly."""
        return httpx.AsyncClient(timeout=self.timeout)

    async def _recycle_client(self) -> None:
        """Drop the shared client's pooled connections after a retryable >=500.

        Retrying on the connection that just produced a 5xx can keep landing
        on the same misbehaving backend replica: measured 20/20 sticky 502s
        in production from the one process-lifetime client reusing its one
        pooled connection across every retry attempt, while requests that
        dialed a fresh connection routed around the same event. Swapping in a
        new client before the next attempt forces it onto a new connection.

        Chosen over sending a request-side ``Connection: close`` header:
        that only hints the *server* to close after replying and depends on
        the misbehaving backend cooperating, which is not something a test
        can assert deterministically. Replacing the client object is -- the
        retry test asserts the post-recycle attempt lands on a different
        client instance, not just on a plausible-sounding header.

        Scoped to >=500 by the caller only: a 4xx is a request-shaped problem
        the connection did not cause, and 408/429 are timeout/rate-limit
        signals, not evidence the connection itself is bad.

        The stale client is closed once its last in-flight request lands, not
        immediately. This provider instance is shared by every concurrent call
        on its lane, so closing on the spot would yank the connection out from
        under siblings mid-response: measured, they surface ``httpx.ReadError``
        and retry, which re-sends completions the upstream had already accepted
        and can fail outright a sibling that was on its final attempt. One
        backend's 5xx must not cost the other requests in flight.
        """
        async with self._client_lock:
            stale = self._client
            self._client = self._new_client()
            busy = bool(self._inflight.get(stale))
            if busy:
                waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
                self._drained[stale] = waiter

        if not busy:
            await stale.aclose()
            return

        task = asyncio.create_task(self._close_when_drained(stale, waiter))
        self._draining.add(task)
        task.add_done_callback(self._draining.discard)

    def _release_client(self, client: Any) -> None:
        """Drop one in-flight count, waking a retired client's closer at zero."""
        remaining = self._inflight.get(client, 1) - 1
        if remaining > 0:
            self._inflight[client] = remaining
            return
        self._inflight.pop(client, None)
        waiter = self._drained.get(client)
        if waiter is not None and not waiter.done():
            waiter.set_result(None)

    async def _close_when_drained(self, stale: Any, waiter: asyncio.Future[None]) -> None:
        """Close a retired client once nothing is using it any more."""
        try:
            await waiter
        except asyncio.CancelledError:
            # cleanup() cancels the wait so shutdown does not block on a
            # sibling; the close below still has to happen, so this is
            # swallowed deliberately rather than propagated.
            pass
        finally:
            self._drained.pop(stale, None)
            self._inflight.pop(stale, None)
            with suppress(Exception):
                await stale.aclose()

    def _classify_forbidden(self, reply: _UpstreamReply) -> Exception:
        """Turn a 403 into one of three classifications, none of them credential failures.

        A spending-limit stop is a quota state; a 403 that names no error code
        is the entitlement shape (an OAuth grant xAI has not enabled for API
        access); a 403 that names some other code is a request-shaped refusal
        and is reported with that code. No branch refreshes or quarantines the
        token.
        """
        code = _error_code(reply.body_text)
        if code == SPENDING_LIMIT_CODE or SPENDING_LIMIT_CODE in reply.body_text:
            return XaiOAuthQuotaExhaustedError(
                "xAI stopped the request at the account spending limit "
                f"({SPENDING_LIMIT_CODE}). The credential is fine; raise or wait out the limit."
            )
        if not code:
            return XaiOAuthEntitlementError(
                "xAI refused API access to this OAuth grant (HTTP 403). xAI may restrict OAuth API access by "
                "subscription tier; verify the account tier or use an api.x.ai API key provider instead "
                "(provider: openai with an api.x.ai base URL)."
            )
        return _UpstreamStatusError(
            f"xAI refused the request with HTTP 403 (code={code})",
            retryable=False,
        )

    async def _request_completion(self, body: dict[str, Any], messages: list[dict[str, Any]]) -> _ChatCompletion:
        """One logical upstream call: token admission, 401 recovery, 403 branch.

        The 401 retry lives here rather than in the callers' retry loops because
        it is auth recovery, not backoff: it happens exactly once and does not
        consume a normal-retry budget slot. A second 401 is terminal — the
        credential is genuinely rejected and looping cannot fix it.
        """
        token = await self._access_token()
        conv_id = cache_affinity_id(messages)

        reply = await self._post(body, token, conv_id)
        if reply.status_code == 401:
            logger.warning("xai-oauth upstream rejected the access token (401); refreshing once and retrying")
            token = await self._refresh_after_rejection(token)
            reply = await self._post(body, token, conv_id)
            if reply.status_code == 401:
                raise XaiOAuthLoginRequiredError(
                    "api.x.ai rejected the access token even after a refresh. "
                    f"Run the xai-oauth login on the host that owns the token store: {LOGIN_COMMAND}"
                )

        if reply.status_code == 403:
            raise self._classify_forbidden(reply)

        if reply.status_code >= 400:
            # Retry 408/429 and 5xx; other 4xx are request-shaped problems that
            # will fail identically on every attempt.
            retryable = reply.status_code in (408, 429) or reply.status_code >= 500
            detail = _error_detail(reply.body_text)
            raise _UpstreamStatusError(
                f"{_actual_host(self.base_url)} returned HTTP {reply.status_code} ({len(reply.body_text)} bytes)"
                + (f": {detail}" if detail else ""),
                retryable=retryable,
                retry_after=reply.retry_after,
                status_code=reply.status_code,
            )

        try:
            return _ChatCompletion.model_validate_json(reply.body_text)
        except ValueError as exc:
            raise _UpstreamStatusError(
                f"{_actual_host(self.base_url)} returned an unparseable success body ({len(reply.body_text)} bytes)",
                retryable=True,
            ) from exc

    def _build_body(
        self,
        messages: list[dict[str, Any]],
        max_completion_tokens: int | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        """Assemble the OpenAI-shaped request body common to both call paths."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
        }
        # BEHAVIOUR CHANGE, deliberate: reaching api.x.ai as provider=openai
        # only sends reasoning_effort when _supports_reasoning_model()
        # matches — a model-name heuristic (gpt-5/o1/o3/deepseek) that
        # "grok-4.5" does not match, so the member-configured effort never
        # reached xAI on that path. Sending it here is the first time that
        # setting takes effect on this lane.
        #
        # "none" is the exception, and it has to be dropped rather than
        # forwarded: it is a real value elsewhere in Hindsight (the documented
        # way to make an OpenAI reasoning model accept function tools), but xAI
        # answers it with HTTP 400 "This model does not support
        # `reasoning_effort` value `none`" — a non-retryable status, so a
        # deployment carrying that setting would fail every call on this lane.
        # Omitting the field is what "no explicit effort" means on this API.
        if self.reasoning_effort and self.reasoning_effort.strip().lower() != REASONING_EFFORT_NONE:
            body["reasoning_effort"] = self.reasoning_effort
        if max_completion_tokens is not None:
            body["max_tokens"] = max_completion_tokens
        if temperature is not None:
            body["temperature"] = temperature
        return body

    def _backoff_seconds(self, error: Exception, attempt: int, initial_backoff: float, max_backoff: float) -> float:
        """Seconds to wait before the next attempt, honoring ``Retry-After``."""
        scheduled = min(initial_backoff * (2**attempt), max_backoff)
        retry_after = getattr(error, "retry_after", None)
        if isinstance(retry_after, (int, float)):
            return min(max(float(retry_after), scheduled), max_backoff)
        return scheduled

    # ------------------------------------------------------------------
    # LLMInterface
    # ------------------------------------------------------------------

    async def verify_connection(self) -> None:
        """Verify the lane with one tiny completion."""
        try:
            logger.info("Verifying xai-oauth: model=%s", self.model)
            await self.call(
                messages=[{"role": "user", "content": "Say 'ok'"}],
                max_completion_tokens=16,
                max_retries=2,
                initial_backoff=0.5,
                max_backoff=2.0,
                scope="verification",
            )
            logger.info("xai-oauth verified: %s", self.model)
        except XaiOAuthQuotaExhaustedError as e:
            # An exhausted subscription is not a misconfiguration — the lane is
            # wired correctly and serves again when the limit resets, so it must
            # not block startup (the same allowance the Codex provider makes).
            logger.warning("xai-oauth quota exhausted for %s, continuing startup: %s", self.model, e)
        except Exception as e:
            # Match on the status this carries, not on "429" appearing in the
            # message: the message also states the body's byte count and now
            # the upstream's error text, so a substring test lets an unrelated
            # failure ("HTTP 500 (429 bytes)") pass itself off as a rate limit
            # and wave a genuinely broken lane through startup. Every upstream
            # status reaches here as an _UpstreamStatusError, so nothing that
            # was recognised before stops being recognised.
            if isinstance(e, _UpstreamStatusError) and e.status_code == 429:
                logger.warning("xai-oauth rate limited for %s, continuing startup: %s", self.model, e)
                return
            raise RuntimeError(f"xai-oauth connection verification failed for {self.model}: {e}") from e

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
        attempt_context: Callable[[], AbstractAsyncContextManager[None]] | None = None,
    ) -> Any:
        """Make a non-streaming completion call with retry logic.

        Non-streaming is deliberate for the first implementation: the engine
        buffers every response anyway, and a streaming path would lose
        ``prompt_tokens_details``, so cached-token accounting would read low.
        """
        start_time = time.time()
        body = self._build_body(list(messages), max_completion_tokens, temperature)

        if response_format is not None and hasattr(response_format, "model_json_schema"):
            schema = strict_json_schema(response_format) if strict_schema else provider_json_schema(response_format)
            if strict_schema:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "strict": True, "schema": schema},
                }
            else:
                schema_msg = (
                    "\n\nYou must respond with valid JSON matching this schema:\n"
                    f"{json.dumps(schema, indent=2, ensure_ascii=False)}"
                )
                first = body["messages"][0] if body["messages"] else None
                if isinstance(first, dict) and isinstance(first.get("content"), str):
                    first["content"] = f"{first['content']}{schema_msg}"
                else:
                    body["messages"].append({"role": "system", "content": schema_msg.strip()})
                body["response_format"] = {"type": "json_object"}

        last_exception: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                async with attempt_context() if attempt_context is not None else nullcontext():
                    set_stage(f"llm.xai_oauth.{scope}.attempt={attempt + 1}/{max_retries + 1}")
                    completion = await self._request_completion(body, body["messages"])

                counts = _token_counts(completion.usage)
                # Stash before parse/validate, which can still fail locally even
                # though the provider counted these tokens.
                stash_response_usage(
                    LLMResponseUsage(
                        input_tokens=counts.input_tokens,
                        output_tokens=counts.output_tokens,
                        cached_tokens=counts.cached_tokens,
                    )
                )

                completion_content = self._content_of(completion, scope)
                content = completion_content.text

                if response_format is not None:
                    try:
                        json_data = json.loads(_strip_code_fence(content))
                    except json.JSONDecodeError as json_err:
                        logger.warning(
                            "xai-oauth JSON parse error (attempt %d/%d, scope=%s, %d chars): %s",
                            attempt + 1,
                            max_retries + 1,
                            scope,
                            len(content),
                            json_err,
                        )
                        if attempt < max_retries:
                            await asyncio.sleep(min(initial_backoff * (2**attempt), max_backoff))
                            last_exception = json_err
                            continue
                        raise
                    result = json_data if skip_validation else response_format.model_validate(json_data)
                else:
                    result = content

                duration = time.time() - start_time
                self._record_success(scope=scope, duration=duration, counts=counts)
                self._record_span(
                    scope=scope,
                    messages=body["messages"],
                    response_content=result,
                    counts=counts,
                    duration=duration,
                    finish_reason=completion_content.finish_reason,
                )

                if return_usage:
                    return result, TokenUsage(
                        input_tokens=counts.input_tokens,
                        output_tokens=counts.output_tokens,
                        total_tokens=counts.total_tokens,
                        cached_tokens=counts.cached_tokens,
                        thoughts_tokens=counts.thoughts_tokens,
                    )
                return result

            # XaiOAuthRefreshError is retried alongside the transport errors: a
            # credential-side blip (a network error reaching auth.x.ai, a 5xx
            # from the token endpoint, the anti-spin gap not yet elapsed) is as
            # transient as the same blip against the API, and leaving it out
            # meant one unlucky refresh failed the whole operation with no
            # retry while an identical failure one hop later got ten. The
            # states a retry cannot fix raise XaiOAuthLoginRequiredError, which
            # is deliberately absent here and stays fatal.
            except (_UpstreamStatusError, httpx.RequestError, XaiOAuthRefreshError) as e:
                last_exception = e
                retryable = e.retryable if isinstance(e, _UpstreamStatusError) else True
                if retryable and attempt < max_retries:
                    if isinstance(e, _UpstreamStatusError) and e.status_code is not None and e.status_code >= 500:
                        await self._recycle_client()
                    logger.warning(
                        "xai-oauth call error (attempt %d/%d, scope=%s): %s",
                        attempt + 1,
                        max_retries + 1,
                        scope,
                        e,
                    )
                    await asyncio.sleep(self._backoff_seconds(e, attempt, initial_backoff, max_backoff))
                    continue
                logger.error("xai-oauth call failed (scope=%s, attempt %d): %s", scope, attempt + 1, e)
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("xai-oauth call failed after all retries with no exception captured")

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
        attempt_context: Callable[[], AbstractAsyncContextManager[None]] | None = None,
    ) -> LLMToolCallResult:
        """Make a non-streaming tool-calling completion.

        One round of the agentic loop the caller drives: tool calls are returned
        as proposed, never executed here.
        """
        start_time = time.time()
        body = self._build_body(list(messages), max_completion_tokens, temperature)

        if tool_choice.mode is LLMToolChoiceMode.NAMED:
            forced_name = tool_choice.selected_function_name
            filtered = [tool for tool in tools if tool.get("function", {}).get("name") == forced_name]
            if len(filtered) != 1:
                raise ValueError(
                    f"Named tool_choice must reference exactly one declared tool; "
                    f"found {len(filtered)} definitions for {forced_name!r}"
                )
            body["tools"] = filtered
            body["tool_choice"] = LLMToolChoiceMode.REQUIRED.value
        else:
            body["tools"] = tools
            if tool_choice.mode is not LLMToolChoiceMode.AUTO:
                body["tool_choice"] = tool_choice.mode.value

        last_exception: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                async with attempt_context() if attempt_context is not None else nullcontext():
                    set_stage(f"llm.xai_oauth.tools.attempt={attempt + 1}/{max_retries + 1}")
                    completion = await self._request_completion(body, body["messages"])

                counts = _token_counts(completion.usage)
                choice = completion.choices[0] if completion.choices else None
                message = choice.message if choice is not None else None
                content = message.content if message is not None else None
                tool_calls: list[LLMToolCall] = []
                for raw_call in (message.tool_calls or []) if message is not None else []:
                    try:
                        arguments = json.loads(raw_call.function.arguments) if raw_call.function.arguments else {}
                    except json.JSONDecodeError:
                        # Keep the raw payload so the caller can see what the
                        # model emitted rather than a silently empty tool call.
                        arguments = {"_raw": raw_call.function.arguments}
                    tool_calls.append(LLMToolCall(id=raw_call.id, name=raw_call.function.name, arguments=arguments))

                duration = time.time() - start_time
                finish_reason = choice.finish_reason if choice is not None else None
                self._record_success(scope=scope, duration=duration, counts=counts)
                self._record_span(
                    scope=scope,
                    messages=body["messages"],
                    response_content=content,
                    counts=counts,
                    duration=duration,
                    finish_reason=finish_reason,
                    tool_calls=tool_calls,
                )

                return LLMToolCallResult(
                    content=content,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                    input_tokens=counts.input_tokens,
                    output_tokens=counts.output_tokens,
                    cached_tokens=counts.cached_tokens,
                    thoughts_tokens=counts.thoughts_tokens,
                )

            except (_UpstreamStatusError, httpx.RequestError, XaiOAuthRefreshError) as e:
                last_exception = e
                retryable = e.retryable if isinstance(e, _UpstreamStatusError) else True
                if retryable and attempt < max_retries:
                    if isinstance(e, _UpstreamStatusError) and e.status_code is not None and e.status_code >= 500:
                        await self._recycle_client()
                    logger.warning(
                        "xai-oauth tool call error (attempt %d/%d, scope=%s): %s",
                        attempt + 1,
                        max_retries + 1,
                        scope,
                        e,
                    )
                    await asyncio.sleep(self._backoff_seconds(e, attempt, initial_backoff, max_backoff))
                    continue
                logger.error("xai-oauth tool call failed (scope=%s, attempt %d): %s", scope, attempt + 1, e)
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("xai-oauth tool call failed after all retries with no exception captured")

    async def cleanup(self) -> None:
        """Close every HTTP client and the credential manager's own client.

        Clients still draining from a recycle are closed too: their waiter is
        cancelled so shutdown does not block on a request that may never land,
        and :meth:`_close_when_drained` closes them on the way out.
        """
        draining = list(self._draining)
        for task in draining:
            task.cancel()
        for task in draining:
            with suppress(asyncio.CancelledError):
                await task
        await self._client.aclose()
        self._auth.close()

    def supports_attempt_scoped_concurrency(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _content_of(self, completion: _ChatCompletion, scope: str) -> _CompletionContent:
        """Extract message content, turning shape problems into retryable errors."""
        if not completion.choices:
            raise _UpstreamStatusError(
                f"{_actual_host(self.base_url)} returned no choices (model={self.model}, scope={scope})",
                retryable=True,
            )
        choice = completion.choices[0]
        content = choice.message.content if choice.message is not None else None
        if not content:
            raise _UpstreamStatusError(
                f"{_actual_host(self.base_url)} returned empty message content (model={self.model}, scope={scope}, "
                f"finish_reason={choice.finish_reason})",
                retryable=True,
            )
        return _CompletionContent(text=content, finish_reason=choice.finish_reason)

    def _record_success(self, *, scope: str, duration: float, counts: _TokenCounts) -> None:
        get_metrics_collector().record_llm_call(
            provider=self.provider,
            model=self.model,
            scope=scope,
            duration=duration,
            input_tokens=counts.input_tokens,
            output_tokens=counts.output_tokens,
            success=True,
            cached_input_tokens=counts.cached_tokens,
            thoughts_tokens=counts.thoughts_tokens,
        )
        if duration > 10.0:
            logger.info(
                "slow llm call: scope=%s, model=%s/%s, input_tokens=%d, output_tokens=%d, cached_tokens=%d, time=%.3fs",
                scope,
                self.provider,
                self.model,
                counts.input_tokens,
                counts.output_tokens,
                counts.cached_tokens,
                duration,
            )

    def _record_span(
        self,
        *,
        scope: str,
        messages: list[dict[str, Any]],
        response_content: Any,
        counts: _TokenCounts,
        duration: float,
        finish_reason: str | None,
        tool_calls: list[LLMToolCall] | None = None,
    ) -> None:
        """Record the GenAI span. Best-effort, but never silently swallowed."""
        try:
            from hindsight_api.tracing import _serialize_for_span, get_span_recorder

            get_span_recorder().record_llm_call(
                provider=self.provider,
                model=self.model,
                scope=scope,
                messages=messages,
                response_content=_serialize_for_span(response_content),
                input_tokens=counts.input_tokens,
                output_tokens=counts.output_tokens,
                duration=duration,
                finish_reason=finish_reason,
                error=None,
                cached_tokens=counts.cached_tokens,
                tool_calls=(
                    [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls]
                    if tool_calls
                    else None
                ),
            )
        except Exception as span_error:
            # Tracing stays best-effort, but instrumentation bugs that would
            # otherwise silently erase spans have to be visible.
            logger.debug("xai-oauth span recording failed: %s", span_error, exc_info=True)


def _error_code(body_text: str) -> str:
    """Pull an error code out of an upstream error body, or return ''.

    Handles the ``{"error": {"code": ...}}``, ``{"error": "..."}`` and
    ``{"code": ...}`` shapes; anything else reads as no code.
    """
    try:
        payload = json.loads(body_text)
    except (json.JSONDecodeError, ValueError):
        return ""
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        if isinstance(code, str) and code:
            return code
    if isinstance(error, str) and error:
        return error
    code = payload.get("code")
    return code if isinstance(code, str) else ""


def _scalar(value: Any) -> str:
    """Render a JSON scalar for an error message, or '' for anything else."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return ""
    return str(value).strip()


def _error_detail(body_text: str) -> str:
    """Summarize an upstream error body from its recognized fields only.

    Returns ``''`` for a body that is not a JSON object or that names neither
    an error code nor an error message.

    Why this exists: a non-retryable 4xx reported as nothing but a status and a
    byte count is undiagnosable — the operator cannot tell a rejected parameter
    from a bad model name without reproducing the call by hand. The whole body
    still never travels (it is unbounded, and relaying it wholesale is how
    prompt text ends up in logs); only ``code`` and the error message do, capped
    at :data:`MAX_ERROR_DETAIL_CHARS`.
    """
    try:
        payload = json.loads(body_text)
    except (json.JSONDecodeError, ValueError):
        return ""
    if not isinstance(payload, dict):
        return ""

    error = payload.get("error")
    code = _scalar(payload.get("code"))
    message = ""
    if isinstance(error, dict):
        code = _scalar(error.get("code")) or code
        message = _scalar(error.get("message"))
    else:
        message = _scalar(error)

    parts = [part for part in (code, message) if part]
    if len(parts) == 2 and parts[0] == parts[1]:
        parts.pop()
    detail = ": ".join(parts)
    if len(detail) > MAX_ERROR_DETAIL_CHARS:
        detail = detail[:MAX_ERROR_DETAIL_CHARS].rstrip() + "..."
    return detail
