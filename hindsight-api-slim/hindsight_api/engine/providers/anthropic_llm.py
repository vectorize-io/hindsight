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

from hindsight_api.engine.llm_interface import LLMInterface
from hindsight_api.engine.llm_trace import LLMResponseUsage, stash_response_usage
from hindsight_api.engine.providers.llm_debug import dump_request_on_4xx
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, TokenUsage
from hindsight_api.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


def _usage_from_anthropic_response(response: Any) -> LLMResponseUsage:
    """Extract input/output/cached token counts from an Anthropic usage block."""
    usage = getattr(response, "usage", None)
    if not usage:
        return LLMResponseUsage()
    return LLMResponseUsage(
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        cached_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
    )


_EPHEMERAL_CACHE = {"type": "ephemeral"}


def _cached_system_blocks(system_prompt: str) -> list[dict[str, Any]]:
    """Render the system prompt as a block list with a cache_control marker.

    Anthropic prompt caching is a prefix match: marking the (single) system
    block caches tools + system together. The system prompt is stable per
    scope — fact extraction reuses it across every chunk, reflect and
    consolidation keep their stable instructions there — so repeat calls read
    it at ~10% of the base input price. Markers below the model's minimum
    cacheable prefix are silently ignored (no write premium), so marking is
    safe unconditionally. This is the "inline-marker provider" strategy that
    ``LLMInterface.get_or_create_cached_prefix`` documents for Anthropic.
    """
    return [{"type": "text", "text": system_prompt, "cache_control": _EPHEMERAL_CACHE}]


def _mark_last_message_for_caching(messages: list[dict[str, Any]]) -> None:
    """Add a cache_control marker to the final content block, in place.

    Used on the multi-turn (tool-calling) path: the reflect agent loop resends
    the entire growing conversation each iteration, so this request's
    end-marker becomes the next iteration's cache read point. Together with
    the system marker this uses 2 of the 4 allowed breakpoints.
    """
    if not messages:
        return
    last = messages[-1]
    content = last.get("content")
    if isinstance(content, str):
        if content.strip():  # the API rejects empty text blocks
            last["content"] = [{"type": "text", "text": content, "cache_control": _EPHEMERAL_CACHE}]
    elif isinstance(content, list) and content and isinstance(content[-1], dict):
        content[-1]["cache_control"] = _EPHEMERAL_CACHE


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
        default_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
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
            default_headers: Optional custom headers passed as ``default_headers`` to
                the Anthropic SDK client. Used by operators routing through proxies
                or request-tracing middleware. Sourced from ``llm_default_headers`` in
                ``HindsightConfig`` (env: ``HINDSIGHT_API_LLM_DEFAULT_HEADERS``).
            extra_body: Extra request-body params (e.g. ``{"temperature": 0.2,
                "top_p": 0.9, "top_k": 40}``) passed via the Anthropic SDK's
                ``extra_body`` so they merge into the JSON sent to the Messages API.
                Sourced from ``llm_extra_body`` (env: ``HINDSIGHT_API_LLM_EXTRA_BODY``).
            **kwargs: Additional provider-specific parameters.
        """
        super().__init__(provider, api_key, base_url, model, reasoning_effort, **kwargs)

        if not self.api_key:
            raise ValueError("API key is required for Anthropic provider")

        # User-configured extra body params (merged into every Messages API call)
        self._extra_body = extra_body or {}

        # Import and initialize Anthropic client
        try:
            from anthropic import AsyncAnthropic

            # SDK retries disabled — wrapper-level retry loop in ``call`` handles
            # backoff (mirrors ``OpenAICompatibleLLM`` so the two providers behave
            # consistently).
            client_kwargs: dict[str, Any] = {"api_key": self.api_key, "max_retries": 0}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            if timeout:
                client_kwargs["timeout"] = timeout
            if default_headers:
                client_kwargs["default_headers"] = default_headers

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
                scope="verification",
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
            strict_schema: Route structured output through a forced tool_use tool for
                native constrained decoding (issue #1002). When False, falls back to
                schema-in-prompt + JSON parse.
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

        # Structured output: prefer Anthropic-native constrained decoding via a single
        # forced tool_use tool (strict_schema) over text-injecting the schema and
        # parsing the reply. Native constrained decoding guarantees schema-valid JSON,
        # eliminating the invalid-JSON retry storm (issue #1002). When strict_schema is
        # off we keep the text-inject + json.loads fallback for backward compatibility.
        schema = None
        use_forced_tool = False
        _tool_name = "structured_response"
        if response_format is not None and hasattr(response_format, "model_json_schema"):
            schema = response_format.model_json_schema()
            if strict_schema:
                use_forced_tool = True
            else:
                schema_msg = f"\n\nYou must respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2, ensure_ascii=False)}"
                system_prompt = (system_prompt + schema_msg) if system_prompt else schema_msg

        # Prepare parameters
        call_params: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_completion_tokens if max_completion_tokens is not None else 4096,
        }

        if system_prompt:
            # One-shot calls share only the system prompt with each other, so
            # that is the sole cache breakpoint on this path.
            call_params["system"] = _cached_system_blocks(system_prompt)

        if use_forced_tool:
            # Single tool whose input_schema IS the response schema; force the model to
            # emit it via tool_choice so the SDK does constrained decoding for us.
            call_params["tools"] = [
                {"name": _tool_name, "description": "Return the structured response.", "input_schema": schema}
            ]
            call_params["tool_choice"] = {"type": "tool", "name": _tool_name}

        if self._extra_body:
            call_params["extra_body"] = self._extra_body

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                response = await self._client.messages.create(**call_params)
                # Stash usage before parse/validate, which may raise locally
                # even though the provider charged for these tokens (#2387).
                stash_response_usage(_usage_from_anthropic_response(response))

                if use_forced_tool:
                    # Forced tool_use → the validated args are already a dict; no parsing,
                    # no markdown-strip, no JSON-decode retry possible.
                    tool_input = None
                    for block in response.content:
                        if block.type == "tool_use" and block.name == _tool_name:
                            tool_input = block.input or {}
                            break
                    if tool_input is None:
                        # Model ignored the forced tool (rare, e.g. a gateway that drops
                        # tool_choice). Fall back to text parse so we don't hard-fail; the
                        # existing retry loop still covers genuine errors.
                        content = "".join(b.text for b in response.content if b.type == "text")
                        tool_input = json.loads(content)
                    content = json.dumps(tool_input)
                    result = tool_input if skip_validation else response_format.model_validate(tool_input)
                else:
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
                response_usage = _usage_from_anthropic_response(response)
                input_tokens = response_usage.input_tokens
                output_tokens = response_usage.output_tokens
                total_tokens = input_tokens + output_tokens
                cached_tokens = response_usage.cached_tokens

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

                # Record trace span
                from hindsight_api.tracing import _serialize_for_span, get_span_recorder

                finish_reason = response.stop_reason if hasattr(response, "stop_reason") else None
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
                    cached_tokens=cached_tokens,
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
                        cached_tokens=cached_tokens,
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

                # Diagnostic dump (opt-in) of the exact request behind any 4xx.
                dump_request_on_4xx(scope=scope, provider=self.provider, model=self.model, err=e, request=call_params)

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

        # Multi-turn tool loop: cache the stable prefix (tools + system) via
        # the system marker, and the growing conversation via an end-marker
        # that the next iteration reads back.
        _mark_last_message_for_caching(anthropic_messages)

        call_params: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "tools": anthropic_tools,
            "max_tokens": max_completion_tokens or 4096,
        }
        if system_prompt:
            call_params["system"] = _cached_system_blocks(system_prompt)

        if self._extra_body:
            call_params["extra_body"] = self._extra_body

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                response = await self._client.messages.create(**call_params)
                stash_response_usage(_usage_from_anthropic_response(response))

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
                duration = time.time() - start_time
                metrics.record_llm_call(
                    provider=self.provider,
                    model=self.model,
                    scope=scope,
                    duration=duration,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=True,
                )

                # Record OpenTelemetry span
                from hindsight_api.tracing import get_span_recorder

                span_recorder = get_span_recorder()
                # Convert LLMToolCall objects to dicts for span recording
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
                    finish_reason=finish_reason,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            except (APIConnectionError, APIStatusError) as e:
                if isinstance(e, APIStatusError) and e.status_code in (401, 403):
                    raise
                # Diagnostic dump (opt-in) of the exact request behind any 4xx.
                dump_request_on_4xx(scope=scope, provider=self.provider, model=self.model, err=e, request=call_params)
                last_exception = e
                if attempt < max_retries:
                    await asyncio.sleep(min(initial_backoff * (2**attempt), max_backoff))
                    continue
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Anthropic tool call failed")

    # ── Message Batches API (50% token discount) ─────────────────────────────

    _BATCH_TOOL_NAME = "structured_response"

    async def supports_batch_api(self) -> bool:
        """Anthropic supports batch operations via the Message Batches API."""
        return True

    @staticmethod
    def _map_batch_status(processing_status: str) -> str:
        """Map Anthropic ``processing_status`` onto the OpenAI vocabulary.

        The engine's poll loop breaks on "completed" and hard-fails on
        "failed"/"expired"/"cancelled"; anything else keeps polling. Anthropic
        batches only end as "ended" (per-request failures surface in the
        results, mirroring OpenAI's "completed"-with-errors semantics), so
        "ended" maps to "completed" and the non-terminal states pass through.
        """
        return "completed" if processing_status == "ended" else processing_status

    def _translate_batch_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """Translate one OpenAI-shaped request body into Messages API params.

        Mirrors the conversion rules of ``call()``: system messages fold into
        the ``system`` param; ``max_completion_tokens`` becomes ``max_tokens``
        (default 4096); ``temperature`` is dropped (the sync path never sends
        it either — current Claude models reject non-default sampling params);
        an OpenAI ``response_format`` json_schema becomes a single forced
        tool_use tool when strict (native constrained decoding, issue #1002),
        else the schema is injected into the system prompt.

        The system prompt carries the same cache_control marker as the sync
        one-shot path (its sole breakpoint): every request in a retain batch
        shares the fact-extraction system prompt, so the first item's cache
        write serves the remaining items as best-effort reads — and the
        cache-read discount stacks with the 50% batch discount.
        """
        system_prompt: str | None = None
        messages: list[dict[str, Any]] = []
        for msg in body.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = (system_prompt + "\n\n" + content) if system_prompt else content
            else:
                messages.append({"role": role, "content": content})

        params: dict[str, Any] = {
            "model": body.get("model") or self.model,
            "messages": messages,
            "max_tokens": body.get("max_completion_tokens") or 4096,
        }

        json_schema = (body.get("response_format") or {}).get("json_schema") or {}
        schema = json_schema.get("schema")
        if schema is not None:
            if json_schema.get("strict"):
                params["tools"] = [
                    {
                        "name": self._BATCH_TOOL_NAME,
                        "description": "Return the structured response.",
                        "input_schema": schema,
                    }
                ]
                params["tool_choice"] = {"type": "tool", "name": self._BATCH_TOOL_NAME}
            else:
                schema_msg = "\n\nYou must respond with valid JSON matching this schema:\n" + json.dumps(
                    schema, indent=2, ensure_ascii=False
                )
                system_prompt = (system_prompt + schema_msg) if system_prompt else schema_msg

        if system_prompt:
            params["system"] = _cached_system_blocks(system_prompt)

        # Batch params ARE the raw Messages body, so operator-configured extra
        # body params merge directly (the sync path routes them through the
        # SDK's extra_body, which does the same merge server-side).
        if self._extra_body:
            params.update(self._extra_body)

        return params

    def _translate_batch_message(self, message: Any) -> dict[str, Any]:
        """Render an Anthropic Message as the OpenAI response body the engine parses.

        The engine reads ``choices[0].message.content`` (json.loads'ing it when
        a schema was requested) and sums ``usage`` under the OpenAI key names.
        Forced-tool responses carry their JSON in the tool_use block's input,
        so that is re-serialized as the content string.
        """
        content = ""
        tool_input = None
        for block in message.content:
            if block.type == "tool_use" and block.name == self._BATCH_TOOL_NAME:
                tool_input = block.input or {}
            elif block.type == "text":
                content += block.text
        if tool_input is not None:
            content = json.dumps(tool_input, ensure_ascii=False)

        usage = getattr(message, "usage", None)
        input_tokens = (usage.input_tokens or 0) if usage else 0
        output_tokens = (usage.output_tokens or 0) if usage else 0

        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": getattr(message, "stop_reason", None),
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }

    async def submit_batch(
        self,
        requests: list[dict[str, Any]],
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
    ) -> dict[str, Any]:
        """Submit a batch of requests to the Message Batches API.

        Accepts the engine's OpenAI-JSONL-shaped entries. ``endpoint`` and
        ``completion_window`` belong to that shared shape and have no Anthropic
        equivalent (batches always resolve within 24 hours); both are ignored.
        """
        batch_requests = [
            {
                "custom_id": req["custom_id"],
                "params": self._translate_batch_body(req.get("body") or {}),
            }
            for req in requests
        ]

        logger.info(f"Submitting Anthropic message batch with {len(batch_requests)} requests")
        batch = await self._client.messages.batches.create(requests=batch_requests)
        logger.info(f"Anthropic batch submitted: {batch.id}, status={batch.processing_status}")

        return {
            "batch_id": batch.id,
            "status": self._map_batch_status(batch.processing_status),
            "created_at": batch.created_at,
            "request_count": len(batch_requests),
        }

    async def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        """Get batch status in the shape the engine's poll loop expects."""
        batch = await self._client.messages.batches.retrieve(batch_id)

        counts = batch.request_counts
        processing = getattr(counts, "processing", 0) or 0
        succeeded = getattr(counts, "succeeded", 0) or 0
        errored = getattr(counts, "errored", 0) or 0
        canceled = getattr(counts, "canceled", 0) or 0
        expired = getattr(counts, "expired", 0) or 0
        resolved = succeeded + errored + canceled + expired

        result: dict[str, Any] = {
            "batch_id": batch.id,
            "status": self._map_batch_status(batch.processing_status),
            "created_at": batch.created_at,
            "request_counts": {
                "total": processing + resolved,
                "completed": resolved,
                "failed": errored,
            },
        }

        ended_at = getattr(batch, "ended_at", None)
        if ended_at:
            result["completed_at"] = ended_at

        return result

    async def retrieve_batch_results(self, batch_id: str) -> list[dict[str, Any]]:
        """Retrieve completed batch results, translated to the OpenAI shape.

        Succeeded entries become ``{"custom_id", "response": {"body": ...}}``;
        errored/canceled/expired entries become ``{"custom_id", "error": ...}``
        so the engine's per-result error handling applies unchanged.
        """
        batch = await self._client.messages.batches.retrieve(batch_id)
        if batch.processing_status != "ended":
            raise ValueError(f"Batch {batch_id} is not completed yet (status: {batch.processing_status})")

        decoder = await self._client.messages.batches.results(batch_id)
        results: list[dict[str, Any]] = []
        async for entry in decoder:
            outcome = entry.result
            if outcome.type == "succeeded":
                results.append(
                    {
                        "custom_id": entry.custom_id,
                        "response": {"body": self._translate_batch_message(outcome.message)},
                    }
                )
            else:
                error = getattr(outcome, "error", None)
                if error is not None:
                    detail = f"{getattr(error, 'type', 'error')}: {getattr(error, 'message', error)}"
                else:
                    detail = f"batch request {outcome.type}"
                results.append({"custom_id": entry.custom_id, "error": detail})

        logger.info(f"Retrieved {len(results)} results for Anthropic batch {batch_id}")
        return results

    async def cleanup(self) -> None:
        """Clean up resources (close Anthropic client connections)."""
        if hasattr(self, "_client") and self._client:
            await self._client.close()
