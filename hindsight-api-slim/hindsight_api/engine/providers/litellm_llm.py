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
import time
from typing import Any

from hindsight_api.engine.llm_interface import LLMInterface, OutputTooLongError
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, TokenUsage
from hindsight_api.metrics import get_metrics_collector
from hindsight_api.worker.stage import set_stage

logger = logging.getLogger(__name__)


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
        timeout: float = 300.0,
        **kwargs: Any,
    ):
        super().__init__(provider, api_key, base_url, model, reasoning_effort, **kwargs)
        self.timeout = timeout
        self._litellm: Any = None

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
        try:
            test_messages = [{"role": "user", "content": "test"}]
            await self.call(
                messages=test_messages,
                max_completion_tokens=50,
                temperature=0.0,
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
            kwargs["max_completion_tokens"] = max_completion_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature

        return kwargs

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
            schema = response_format.model_json_schema()
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
                set_stage(f"llm.litellm.{scope}.attempt={attempt + 1}/{max_retries + 1}")
            try:
                response = await self._litellm.acompletion(**call_kwargs)

                content = response.choices[0].message.content or ""
                finish_reason = response.choices[0].finish_reason

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
                input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
                total_tokens = input_tokens + output_tokens

                # Record metrics
                duration = time.time() - start_time
                metrics = get_metrics_collector()
                metrics.record_llm_call(
                    provider=self.provider,
                    model=self.model,
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
                    model=self.model,
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
                        f"slow llm call: scope={scope}, model={self.provider}/{self.model}, "
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

            except Exception as e:
                error_str = str(e).lower()
                # Fast fail on auth errors
                if "401" in error_str or "403" in error_str or "unauthorized" in error_str:
                    logger.error(f"LiteLLM auth error, not retrying: {e}")
                    raise

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
                set_stage(f"llm.litellm.tools.attempt={attempt + 1}/{max_retries + 1}")
            try:
                response = await self._litellm.acompletion(**call_kwargs)

                message = response.choices[0].message
                content = message.content
                finish_reason = response.choices[0].finish_reason

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
                    model=self.model,
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
                    model=self.model,
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

            except Exception as e:
                error_str = str(e).lower()
                if "401" in error_str or "403" in error_str or "unauthorized" in error_str:
                    raise

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
