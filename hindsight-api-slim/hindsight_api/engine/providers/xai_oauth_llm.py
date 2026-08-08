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
Bodies, credentials, and header values are never logged: byte counts, the model
name, status codes and durations only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable

import httpx
from pydantic import BaseModel, Field

from hindsight_api.config import DEFAULT_LLM_TIMEOUT, ENV_LLM_TIMEOUT
from hindsight_api.engine.llm_interface import LLM_TOOL_CHOICE_AUTO, LLMInterface, LLMToolChoice, LLMToolChoiceMode
from hindsight_api.engine.llm_trace import LLMResponseUsage, current_trace_context, stash_response_usage
from hindsight_api.engine.providers.xai_oauth_auth import (
    DEFAULT_REFRESH_SKEW_SECONDS,
    LOGIN_COMMAND,
    XaiOAuthError,
    XaiOAuthLoginRequiredError,
    XaiOAuthManager,
)
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, TokenUsage
from hindsight_api.engine.structured_output import strict_json_schema
from hindsight_api.metrics import get_metrics_collector
from hindsight_api.worker.stage import set_stage

logger = logging.getLogger(__name__)

__all__ = [
    "XaiOAuthEntitlementError",
    "XaiOAuthError",
    "XaiOAuthLLM",
    "XaiOAuthLoginRequiredError",
    "XaiOAuthQuotaExhaustedError",
]

#: Provider-specific base URL override. It wins over the generic
#: ``HINDSIGHT_API_LLM_BASE_URL`` that deployments often set for an unrelated
#: proxy — the same "more specific beats more general" rule the rest of
#: Hindsight's config hierarchy follows.
ENV_BASE_URL = "HINDSIGHT_API_XAI_OAUTH_BASE_URL"

DEFAULT_BASE_URL = "https://api.x.ai/v1"

#: Request header xAI uses to route calls of one conversation to the same
#: cache-holding backend.
CONV_ID_HEADER = "x-grok-conv-id"

#: Body marker xAI returns when the account's spending limit stopped the call.
SPENDING_LIMIT_CODE = "personal-team-blocked:spending-limit"


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
    the same decision, and the message carries the distinction.
    """

    def __init__(self, message: str, *, retryable: bool, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


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


def _conversation_affinity_id(messages: list[dict[str, Any]]) -> str | None:
    """Stable ``x-grok-conv-id`` so the upstream's prompt cache can hit.

    xAI stores prompt-cache entries per backend server and routes requests
    carrying the same ``x-grok-conv-id`` to the same server, so without the
    header each call of a multi-turn loop can land on a cache-cold replica.

    The id comes from the operation's trace id when there is one (every LLM call
    of a single retain/reflect/consolidation run shares it, which is exactly the
    grouping the cache wants), and otherwise from a hash of the first message —
    the system prompt, byte-identical across the calls of one loop. It is hashed
    rather than sent raw so no internal identifier leaves the process, and is
    always 32 lowercase hex characters.

    Fail-open: any shape or serialization problem returns None and the request
    goes upstream without the header.
    """
    context = current_trace_context()
    if context is not None and context.trace_id:
        return hashlib.sha256(str(context.trace_id).encode("utf-8")).hexdigest()[:32]
    if not isinstance(messages, list) or not messages or not isinstance(messages[0], dict):
        return None
    try:
        first = json.dumps(messages[0], sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(first.encode("utf-8")).hexdigest()[:32]


class XaiOAuthLLM(LLMInterface):
    """LLM provider backed by a SuperGrok subscription via an xAI OAuth grant."""

    def __init__(
        self,
        provider: str,
        api_key: str,  # Ignored: the credential is the OAuth grant in the token store.
        base_url: str,
        model: str,
        reasoning_effort: str = "low",
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
        self._client = httpx.AsyncClient(timeout=self.timeout)

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
        """POST one chat completion. Header values never reach the log."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }
        if conv_id:
            headers[CONV_ID_HEADER] = conv_id

        response = await self._client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
        return _UpstreamReply(
            status_code=response.status_code,
            body_text=response.text,
            retry_after=_retry_after_seconds(response),
        )

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
        conv_id = _conversation_affinity_id(messages)

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
            raise _UpstreamStatusError(
                f"api.x.ai returned HTTP {reply.status_code} ({len(reply.body_text)} bytes)",
                retryable=retryable,
                retry_after=reply.retry_after,
            )

        try:
            return _ChatCompletion.model_validate_json(reply.body_text)
        except ValueError as exc:
            raise _UpstreamStatusError(
                f"api.x.ai returned an unparseable success body ({len(reply.body_text)} bytes)",
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
            # BEHAVIOUR CHANGE, deliberate: reaching api.x.ai as provider=openai
            # only sends reasoning_effort when _supports_reasoning_model()
            # matches — a model-name heuristic (gpt-5/o1/o3/deepseek) that
            # "grok-4.5" does not match, so the member-configured effort never
            # reached xAI on that path. Sending it here is the first time that
            # setting takes effect on this lane.
            "reasoning_effort": self.reasoning_effort,
        }
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
            if "429" in str(e):
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
            schema = strict_json_schema(response_format) if strict_schema else response_format.model_json_schema()
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

            except (_UpstreamStatusError, httpx.RequestError) as e:
                last_exception = e
                retryable = e.retryable if isinstance(e, _UpstreamStatusError) else True
                if retryable and attempt < max_retries:
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

            except (_UpstreamStatusError, httpx.RequestError) as e:
                last_exception = e
                retryable = e.retryable if isinstance(e, _UpstreamStatusError) else True
                if retryable and attempt < max_retries:
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
        """Close the HTTP client and the credential manager's own client."""
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
                f"api.x.ai returned no choices (model={self.model}, scope={scope})",
                retryable=True,
            )
        choice = completion.choices[0]
        content = choice.message.content if choice.message is not None else None
        if not content:
            raise _UpstreamStatusError(
                f"api.x.ai returned empty message content (model={self.model}, scope={scope}, "
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
