"""
LiteLLM LLM provider for universal model support.

This provider enables using 100+ LLM providers via the LiteLLM SDK, including:
- AWS Bedrock (bedrock/anthropic.claude-3-5-sonnet-...)
- Azure OpenAI (azure/gpt-4o)
- Together AI (together_ai/meta-llama/...)
- Any other LiteLLM-supported provider

Uses litellm.acompletion() for async chat completions.
Authentication for cloud providers (e.g., AWS Bedrock via boto3 credential chain)
is handled automatically by LiteLLM.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any

from litellm.exceptions import Timeout as LiteLLMTimeout

from hindsight_api.config import DEFAULT_LLM_TIMEOUT, ENV_LLM_TIMEOUT
from hindsight_api.engine.llm_interface import LLMInterface, OutputTooLongError
from hindsight_api.engine.llm_trace import LLMResponseUsage, stash_response_usage
from hindsight_api.engine.providers.llm_debug import dump_request_on_4xx
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, TokenUsage
from hindsight_api.engine.structured_output import strict_json_schema
from hindsight_api.metrics import get_metrics_collector
from hindsight_api.worker.stage import set_stage

logger = logging.getLogger(__name__)


def _usage_from_litellm_response(response: Any) -> LLMResponseUsage:
    """Extract prompt/completion/cached token counts from a LiteLLM (OpenAI-shaped) usage block."""
    usage = getattr(response, "usage", None)
    if not usage:
        return LLMResponseUsage()
    cached_tokens = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details:
        cached_tokens = getattr(details, "cached_tokens", 0) or 0
    return LLMResponseUsage(
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cached_tokens=cached_tokens,
    )


class LiteLLMLLM(LLMInterface):
    """
    LLM provider using the LiteLLM SDK for universal model support.

    Supports any model accessible via litellm.acompletion(), including AWS Bedrock,
    Azure OpenAI, Together AI, Fireworks AI, and more.

    Model names follow LiteLLM conventions with provider prefixes:
    - bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
    - azure/gpt-4o
    - together_ai/meta-llama/Llama-3-70b-chat-hf
    - fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        reasoning_effort: str = "low",
        timeout: float | None = None,
        extra_body: dict[str, Any] | None = None,
        bedrock_service_tier: str | None = None,
        default_headers: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(provider, api_key, base_url, model, reasoning_effort, **kwargs)
        # ``None`` falls back to HINDSIGHT_API_LLM_TIMEOUT, then DEFAULT_LLM_TIMEOUT — never None,
        # so the hard ``asyncio.wait_for`` backstop in ``call`` is always bounded.
        self.timeout = timeout if timeout is not None else float(os.getenv(ENV_LLM_TIMEOUT, str(DEFAULT_LLM_TIMEOUT)))
        self._litellm: Any = None
        # User-configured extra params merged as top-level kwargs into every
        # completion call so LiteLLM normalizes them per-provider (e.g. maps
        # temperature/top_p/max_tokens across OpenAI, Anthropic, Bedrock, …) and
        # drops any the target model rejects (litellm.drop_params=True below).
        # Sourced from llm_extra_body (env: HINDSIGHT_API_LLM_EXTRA_BODY).
        self._extra_body: dict[str, Any] = extra_body or {}
        # Operator-configured default headers forwarded to litellm.acompletion as
        # ``extra_headers`` (used by deployments routing through proxies / request-
        # tracing middleware). Mirrors the Anthropic provider's default_headers
        # wiring. Sourced from llm_default_headers (env: HINDSIGHT_API_LLM_DEFAULT_HEADERS).
        # Copied so a caller-owned dict can't be mutated through us, and a fresh
        # copy is handed to each call below to avoid cross-request contamination.
        self._default_headers: dict[str, Any] = dict(default_headers or {})
        self.bedrock_service_tier = bedrock_service_tier

        try:
            import litellm

            self._litellm = litellm
            # Suppress LiteLLM's verbose logging
            litellm.suppress_debug_info = True  # type: ignore[assignment]
            # Drop unsupported params instead of raising errors (e.g. tool_choice on some Bedrock models)
            litellm.drop_params = True  # type: ignore[assignment]
            logging.getLogger("LiteLLM").setLevel(logging.WARNING)
            logger.info(f"LiteLLM SDK initialized for model: {self.model}")
        except ImportError as e:
            raise RuntimeError("LiteLLM SDK not installed. Run: uv add litellm or pip install litellm") from e

    async def verify_connection(self) -> None:
        from ...config import get_config

        try:
            test_messages = [{"role": "user", "content": "test"}]
            await self.call(
                messages=test_messages,
                max_completion_tokens=50,
                temperature=get_config().llm_temperature_verification,
                scope="verification",
                max_retries=0,
            )
            logger.info("LiteLLM connection verified successfully")
        except OutputTooLongError:
            # Truncation is fine for verification — it means the connection works
            logger.info("LiteLLM connection verified successfully (response truncated)")
        except Exception as e:
            logger.error(f"LiteLLM connection verification failed: {e}")
            raise RuntimeError(f"Failed to verify LiteLLM connection: {e}") from e

    def _build_common_kwargs(
        self,
        messages: list[dict[str, Any]],
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Build common kwargs for litellm calls."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "timeout": self.timeout,
        }

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = self._cap_max_completion_tokens(max_completion_tokens)
        if temperature is not None:
            kwargs["temperature"] = temperature

        # User-configured extras fill in only where the caller didn't set a value,
        # so explicit per-call params (model, messages, temperature, …) always win.
        for key, value in self._extra_body.items():
            kwargs.setdefault(key, value)

        # Forward operator-configured default headers as ``extra_headers`` so they
        # reach the provider behind LiteLLM (proxies / request-tracing middleware).
        # ``setdefault`` keeps any explicit per-call ``extra_headers`` authoritative;
        # a per-call copy prevents LiteLLM/downstream from mutating the stored dict.
        if self._default_headers:
            kwargs.setdefault("extra_headers", dict(self._default_headers))

        # Bedrock service tier: flex (50% cheaper), priority, or reserved
        if self.model.startswith("bedrock/") and self.bedrock_service_tier is not None:
            kwargs["service_tier"] = self.bedrock_service_tier

        return kwargs

    # ── per-model output-tokens cap (shared with Router subclass) ────────────
    # Hindsight's defaults (e.g. retain_max_completion_tokens=64000) target
    # high-capacity models. When a configured deployment supports fewer
    # completion tokens (e.g. gpt-4.1-nano caps at 32768), the call would
    # otherwise be rejected. Cap pre-emptively using LiteLLM's per-model
    # registry so things work out of the box across the supported model set.

    def _cap_max_completion_tokens(self, value: int) -> int:
        cap = self._get_model_output_cap()
        if cap and value > cap:
            logger.debug("capping max_completion_tokens %d -> %d for model %s", value, cap, self.model)
            return cap
        return value

    def _get_model_output_cap(self) -> int | None:
        """Return the configured model's max output tokens, per LiteLLM's registry."""
        try:
            cap = self._litellm.get_max_tokens(self.model)
            return int(cap) if cap else None
        except Exception:
            return None

    # ── hooks for Router-style subclasses ────────────────────────────────────
    # The retry+parse loop in call() / call_with_tools() is shared by every
    # LiteLLM-backed provider. Subclasses override the small surface below to
    # swap the completion fn (direct vs Router) and rename the deployment that
    # actually answered the request.

    @property
    def _stage_label(self) -> str:
        """Stage breadcrumb label — overridden by subclasses (e.g. ``litellmrouter``)."""
        return "litellm"

    async def _acompletion(self, **kwargs: Any) -> Any:
        """Issue a chat completion. Subclasses override to route via ``litellm.Router``."""
        return await self._litellm.acompletion(**kwargs)

    def _resolve_completion_model(self, response: Any) -> str:
        """
        Return the model name to record in metrics/tracing.

        For Router-backed providers this can differ from ``self.model`` — the Router
        may pick a different deployment than the primary. Default: ``self.model``.
        """
        return self.model

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
        start_time = time.time()

        call_kwargs = self._build_common_kwargs(messages, max_completion_tokens, temperature)

        # Add JSON schema response format if provided
        if response_format is not None and hasattr(response_format, "model_json_schema"):
            schema = strict_json_schema(response_format) if strict_schema else response_format.model_json_schema()
            call_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__ if hasattr(response_format, "__name__") else "response",
                    "schema": schema,
                    "strict": strict_schema,
                },
            }

        last_exception = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                set_stage(f"llm.{self._stage_label}.{scope}.attempt={attempt + 1}/{max_retries + 1}")
            try:
                response = await asyncio.wait_for(
                    self._acompletion(**call_kwargs),
                    timeout=self.timeout,
                )
                # Stash usage before the length check and parse/validate below,
                # which may raise locally even though the provider charged for
                # these tokens (#2387).
                stash_response_usage(_usage_from_litellm_response(response))

                content = response.choices[0].message.content or ""
                finish_reason = response.choices[0].finish_reason
                model_name = self._resolve_completion_model(response)

                # Check for length-limited output
                if finish_reason == "length":
                    raise OutputTooLongError("LiteLLM response was truncated due to token limit")

                if response_format is not None:
                    # Strip markdown code fences if present
                    clean_content = content
                    if "```json" in content:
                        clean_content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        clean_content = content.split("```")[1].split("```")[0].strip()

                    try:
                        json_data = json.loads(clean_content)
                    except json.JSONDecodeError:
                        json_data = json.loads(content)

                    if skip_validation:
                        result = json_data
                    else:
                        result = response_format.model_validate(json_data)
                else:
                    result = content

                # Extract usage
                response_usage = _usage_from_litellm_response(response)
                input_tokens = response_usage.input_tokens
                output_tokens = response_usage.output_tokens
                total_tokens = input_tokens + output_tokens

                # Record metrics
                duration = time.time() - start_time
                metrics = get_metrics_collector()
                metrics.record_llm_call(
                    provider=self.provider,
                    model=model_name,
                    scope=scope,
                    duration=duration,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=True,
                )

                # Record trace span
                from hindsight_api.tracing import _serialize_for_span, get_span_recorder

                span_recorder = get_span_recorder()
                span_recorder.record_llm_call(
                    provider=self.provider,
                    model=model_name,
                    scope=scope,
                    messages=messages,
                    response_content=_serialize_for_span(result),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration=duration,
                    finish_reason=finish_reason,
                    error=None,
                )

                if duration > 10.0:
                    logger.info(
                        f"slow llm call: scope={scope}, model={self.provider}/{model_name}, "
                        f"input_tokens={input_tokens}, output_tokens={output_tokens}, "
                        f"time={duration:.3f}s"
                    )

                if return_usage:
                    token_usage = TokenUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                    )
                    return result, token_usage
                return result

            except OutputTooLongError:
                raise

            except json.JSONDecodeError as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning("LiteLLM returned invalid JSON, retrying...")
                    backoff = min(initial_backoff * (2**attempt), max_backoff)
                    await asyncio.sleep(backoff)
                    continue
                else:
                    logger.error(f"LiteLLM returned invalid JSON after {max_retries + 1} attempts")
                    raise

            except (TimeoutError, asyncio.TimeoutError, LiteLLMTimeout) as e:
                # litellm/httpx don't always honor their own ``timeout=`` (e.g. a connection held
                # open with no token progress), so ``wait_for`` is the hard cap that cancels a hung
                # call regardless — otherwise one straggler pins a worker slot and stalls its gather.
                last_exception = e
                exc_name = type(e).__name__
                if attempt < max_retries:
                    logger.warning(
                        f"LiteLLM call exceeded timeout={self.timeout}s ({exc_name}, scope={scope}), retrying..."
                    )
                    backoff = min(initial_backoff * (2**attempt), max_backoff)
                    await asyncio.sleep(backoff)
                    continue
                logger.error(
                    f"LiteLLM call timed out after {self.timeout}s on {attempt + 1} attempts "
                    f"({exc_name}, scope={scope})"
                )
                raise

            except Exception as e:
                error_str = str(e).lower()
                # Fast fail on auth errors
                if "401" in error_str or "403" in error_str or "unauthorized" in error_str:
                    logger.error(f"LiteLLM auth error, not retrying: {e}")
                    raise

                # Diagnostic dump (opt-in) of the exact request behind any 4xx.
                dump_request_on_4xx(scope=scope, provider=self.provider, model=self.model, err=e, request=call_kwargs)

                last_exception = e
                if attempt < max_retries:
                    # Retry on rate limits, connection errors, server errors
                    is_retryable = any(
                        keyword in error_str
                        for keyword in ("rate", "limit", "timeout", "connection", "500", "502", "503", "529")
                    )
                    if is_retryable:
                        backoff = min(initial_backoff * (2**attempt), max_backoff)
                        jitter = backoff * 0.2 * (2 * (time.time() % 1) - 1)
                        await asyncio.sleep(backoff + jitter)
                        continue

                logger.error(f"LiteLLM API error after {attempt + 1} attempts: {e}")
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("LiteLLM call failed after all retries")

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
        start_time = time.time()

        call_kwargs = self._build_common_kwargs(messages, max_completion_tokens, temperature)
        call_kwargs["tools"] = tools
        call_kwargs["tool_choice"] = tool_choice

        last_exception = None
        for attempt in range(max_retries + 1):
            if attempt > 0:
                set_stage(f"llm.{self._stage_label}.tools.attempt={attempt + 1}/{max_retries + 1}")
            try:
                response = await asyncio.wait_for(
                    self._acompletion(**call_kwargs),
                    timeout=self.timeout,
                )
                # Stash usage before the tool-call argument parse below, which
                # can raise json.JSONDecodeError locally even though the provider
                # already billed for these tokens; without this the error trace
                # records 0/0 tokens (#2387). Mirrors call() and the anthropic/
                # gemini call_with_tools paths so the litellm tool path (and the
                # LiteLLMRouterLLM subclass that inherits this method) completes
                # the #2396 usage-on-error coverage.
                stash_response_usage(_usage_from_litellm_response(response))

                message = response.choices[0].message
                content = message.content
                finish_reason = response.choices[0].finish_reason
                model_name = self._resolve_completion_model(response)

                # Extract tool calls
                tool_calls: list[LLMToolCall] = []
                if message.tool_calls:
                    for tc in message.tool_calls:
                        arguments = tc.function.arguments
                        if isinstance(arguments, str):
                            arguments = json.loads(arguments)
                        tool_calls.append(
                            LLMToolCall(
                                id=tc.id,
                                name=tc.function.name,
                                arguments=arguments,
                            )
                        )

                # Extract usage
                input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(response.usage, "completion_tokens", 0) or 0

                # Record metrics
                duration = time.time() - start_time
                metrics = get_metrics_collector()
                metrics.record_llm_call(
                    provider=self.provider,
                    model=model_name,
                    scope=scope,
                    duration=duration,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=True,
                )

                # Record trace span
                from hindsight_api.tracing import get_span_recorder

                span_recorder = get_span_recorder()
                tool_calls_dict = (
                    [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls]
                    if tool_calls
                    else None
                )
                span_recorder.record_llm_call(
                    provider=self.provider,
                    model=model_name,
                    scope=scope,
                    messages=messages,
                    response_content=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration=duration,
                    finish_reason=finish_reason,
                    error=None,
                    tool_calls=tool_calls_dict,
                )

                return LLMToolCallResult(
                    content=content,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason or ("tool_calls" if tool_calls else "stop"),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            except (TimeoutError, asyncio.TimeoutError, LiteLLMTimeout) as e:
                # See ``call`` — hard cap so a hung completion cannot block
                # forever and pin a worker slot / concurrency permit.
                last_exception = e
                exc_name = type(e).__name__
                if attempt < max_retries:
                    logger.warning(
                        f"LiteLLM tool call exceeded timeout={self.timeout}s ({exc_name}, scope={scope}), retrying..."
                    )
                    await asyncio.sleep(min(initial_backoff * (2**attempt), max_backoff))
                    continue
                logger.error(
                    f"LiteLLM tool call timed out after {self.timeout}s on {attempt + 1} attempts "
                    f"({exc_name}, scope={scope})"
                )
                raise

            except Exception as e:
                error_str = str(e).lower()
                if "401" in error_str or "403" in error_str or "unauthorized" in error_str:
                    raise

                # Diagnostic dump (opt-in) of the exact request behind any 4xx.
                dump_request_on_4xx(scope=scope, provider=self.provider, model=self.model, err=e, request=call_kwargs)

                last_exception = e
                if attempt < max_retries:
                    is_retryable = any(
                        keyword in error_str
                        for keyword in ("rate", "limit", "timeout", "connection", "500", "502", "503", "529")
                    )
                    if is_retryable:
                        await asyncio.sleep(min(initial_backoff * (2**attempt), max_backoff))
                        continue

                logger.error(f"LiteLLM tool call error after {attempt + 1} attempts: {e}")
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("LiteLLM tool call failed after all retries")

    async def cleanup(self) -> None:
        """Clean up resources."""
        pass
