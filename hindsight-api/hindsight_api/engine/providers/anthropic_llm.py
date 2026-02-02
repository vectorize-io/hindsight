"""
Anthropic LLM provider using the Anthropic Python SDK.

This provider enables using Claude models from Anthropic with support for:
- Structured JSON output
- Tool/function calling with proper format conversion
- Extended thinking mode
- Retry logic with exponential backoff
"""

import asyncio
import json
import logging
import time
from typing import Any

from hindsight_api.engine.llm_interface import LLMInterface, OutputTooLongError
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, TokenUsage
from hindsight_api.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class AnthropicLLM(LLMInterface):
    """
    LLM provider using Anthropic's Claude models.

    Supports structured output, tool calling, and extended thinking mode.
    Handles format conversion between OpenAI-style messages and Anthropic's format.
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
        """
        Initialize Anthropic LLM provider.

        Args:
            provider: Provider name (should be "anthropic").
            api_key: Anthropic API key.
            base_url: Base URL for the API (optional, uses Anthropic default if empty).
            model: Model name (e.g., "claude-sonnet-4-20250514").
            reasoning_effort: Reasoning effort level (not used by Anthropic).
            timeout: Request timeout in seconds.
            **kwargs: Additional provider-specific parameters.
        """
        super().__init__(provider, api_key, base_url, model, reasoning_effort, **kwargs)

        if not self.api_key:
            raise ValueError("API key is required for Anthropic provider")

        # Import and initialize Anthropic client
        try:
            from anthropic import AsyncAnthropic

            client_kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            if timeout:
                client_kwargs["timeout"] = timeout

            self._client = AsyncAnthropic(**client_kwargs)
            logger.info(f"Anthropic client initialized for model: {self.model}")
        except ImportError as e:
            raise RuntimeError("Anthropic SDK not installed. Run: uv add anthropic or pip install anthropic") from e

    async def verify_connection(self) -> None:
        """
        Verify that the Anthropic provider is configured correctly by making a simple test call.

        Raises:
            RuntimeError: If the connection test fails.
        """
        try:
            test_messages = [{"role": "user", "content": "test"}]
            await self.call(
                messages=test_messages,
                max_completion_tokens=10,
                temperature=0.0,
                scope="test",
                max_retries=0,
            )
            logger.info("Anthropic connection verified successfully")
        except Exception as e:
            logger.error(f"Anthropic connection verification failed: {e}")
            raise RuntimeError(f"Failed to verify Anthropic connection: {e}") from e

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
            strict_schema: Use strict JSON schema enforcement (not supported by Anthropic).
            return_usage: If True, return tuple (result, TokenUsage) instead of just result.

        Returns:
            If return_usage=False: Parsed response if response_format is provided, otherwise text content.
            If return_usage=True: Tuple of (result, TokenUsage) with token counts.

        Raises:
            OutputTooLongError: If output exceeds token limits.
            Exception: Re-raises API errors after retries exhausted.
        """
        from anthropic import APIConnectionError, APIStatusError, RateLimitError

        start_time = time.time()

        # Convert OpenAI-style messages to Anthropic format
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if system_prompt:
                    system_prompt += "\n\n" + content
                else:
                    system_prompt = content
            else:
                anthropic_messages.append({"role": role, "content": content})

        # Add JSON schema instruction if response_format is provided
        if response_format is not None and hasattr(response_format, "model_json_schema"):
            schema = response_format.model_json_schema()
            schema_msg = f"\n\nYou must respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
            if system_prompt:
                system_prompt += schema_msg
            else:
                system_prompt = schema_msg

        # Prepare parameters
        call_params: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_completion_tokens if max_completion_tokens is not None else 4096,
        }

        if system_prompt:
            call_params["system"] = system_prompt

        if temperature is not None:
            call_params["temperature"] = temperature

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                response = await self._client.messages.create(**call_params)

                # Anthropic response content is a list of blocks
                content = ""
                for block in response.content:
                    if block.type == "text":
                        content += block.text

                if response_format is not None:
                    # Models may wrap JSON in markdown code blocks
                    clean_content = content
                    if "```json" in content:
                        clean_content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        clean_content = content.split("```")[1].split("```")[0].strip()

                    try:
                        json_data = json.loads(clean_content)
                    except json.JSONDecodeError:
                        # Fallback to parsing raw content if markdown stripping failed
                        json_data = json.loads(content)

                    if skip_validation:
                        result = json_data
                    else:
                        result = response_format.model_validate(json_data)
                else:
                    result = content

                # Record metrics and log slow calls
                duration = time.time() - start_time
                input_tokens = response.usage.input_tokens or 0 if response.usage else 0
                output_tokens = response.usage.output_tokens or 0 if response.usage else 0
                total_tokens = input_tokens + output_tokens

                # Record LLM metrics
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

                # Log slow calls
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

            except json.JSONDecodeError as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning("Anthropic returned invalid JSON, retrying...")
                    backoff = min(initial_backoff * (2**attempt), max_backoff)
                    await asyncio.sleep(backoff)
                    continue
                else:
                    logger.error(f"Anthropic returned invalid JSON after {max_retries + 1} attempts")
                    raise

            except (APIConnectionError, RateLimitError, APIStatusError) as e:
                # Fast fail on 401/403
                if isinstance(e, APIStatusError) and e.status_code in (401, 403):
                    logger.error(f"Anthropic auth error (HTTP {e.status_code}), not retrying: {str(e)}")
                    raise

                last_exception = e
                if attempt < max_retries:
                    # Check if it's a rate limit or server error
                    should_retry = isinstance(e, (APIConnectionError, RateLimitError)) or (
                        isinstance(e, APIStatusError) and e.status_code >= 500
                    )

                    if should_retry:
                        backoff = min(initial_backoff * (2**attempt), max_backoff)
                        jitter = backoff * 0.2 * (2 * (time.time() % 1) - 1)
                        await asyncio.sleep(backoff + jitter)
                        continue

                logger.error(f"Anthropic API error after {max_retries + 1} attempts: {str(e)}")
                raise

            except Exception as e:
                logger.error(f"Unexpected error during Anthropic call: {type(e).__name__}: {str(e)}")
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Anthropic call failed after all retries")

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
        from anthropic import APIConnectionError, APIStatusError

        start_time = time.time()

        # Convert OpenAI tool format to Anthropic format
        anthropic_tools = []
        for tool in tools:
            func = tool.get("function", {})
            anthropic_tools.append(
                {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                }
            )

        # Convert messages - handle tool results
        system_prompt = None
        anthropic_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_prompt = (system_prompt + "\n\n" + content) if system_prompt else content
            elif role == "tool":
                # Anthropic uses tool_result blocks
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": msg.get("tool_call_id", ""), "content": content}
                        ],
                    }
                )
            elif role == "assistant" and msg.get("tool_calls"):
                # Convert assistant tool calls
                tool_use_blocks = []
                for tc in msg["tool_calls"]:
                    tool_use_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "input": json.loads(tc.get("function", {}).get("arguments", "{}")),
                        }
                    )
                anthropic_messages.append({"role": "assistant", "content": tool_use_blocks})
            else:
                anthropic_messages.append({"role": role, "content": content})

        call_params: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "tools": anthropic_tools,
            "max_tokens": max_completion_tokens or 4096,
        }
        if system_prompt:
            call_params["system"] = system_prompt

        if temperature is not None:
            call_params["temperature"] = temperature

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                response = await self._client.messages.create(**call_params)

                # Extract content and tool calls
                content_parts = []
                tool_calls: list[LLMToolCall] = []

                for block in response.content:
                    if block.type == "text":
                        content_parts.append(block.text)
                    elif block.type == "tool_use":
                        tool_calls.append(LLMToolCall(id=block.id, name=block.name, arguments=block.input or {}))

                content = "".join(content_parts) if content_parts else None
                finish_reason = "tool_calls" if tool_calls else "stop"

                # Extract token usage
                input_tokens = response.usage.input_tokens or 0
                output_tokens = response.usage.output_tokens or 0

                # Record metrics
                metrics = get_metrics_collector()
                metrics.record_llm_call(
                    provider=self.provider,
                    model=self.model,
                    scope=scope,
                    duration=time.time() - start_time,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=True,
                )

                return LLMToolCallResult(
                    content=content,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            except (APIConnectionError, APIStatusError) as e:
                if isinstance(e, APIStatusError) and e.status_code in (401, 403):
                    raise
                last_exception = e
                if attempt < max_retries:
                    await asyncio.sleep(min(initial_backoff * (2**attempt), max_backoff))
                    continue
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Anthropic tool call failed")

    async def cleanup(self) -> None:
        """Clean up resources (close Anthropic client connections)."""
        if hasattr(self, "_client") and self._client:
            await self._client.close()
