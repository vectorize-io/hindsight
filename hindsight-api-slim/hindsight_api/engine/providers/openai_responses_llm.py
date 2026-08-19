"""OpenAI Responses API provider (``client.responses.create`` → ``/v1/responses``).

This provider **always** uses the Responses API — never chat/completions. The
chat/completions path rejects ``reasoning_effort`` combined with function tools on
some reasoning models — gpt-5.6-terra returns HTTP 400 unless ``reasoning_effort``
is exactly ``"none"`` (see #2983). Reflect is a tool-calling search loop, so that
constraint forces the whole reflect operation, including the final synthesis, to
run with reasoning disabled.

The Responses API models the chain-of-thought as a first-class reasoning item, so
reasoning and function tools coexist. This provider issues requests against
``responses.create``, translating the engine's chat-shaped inputs:

- chat messages → Responses ``input`` items (assistant ``tool_calls`` →
  ``function_call`` items, ``role="tool"`` results → ``function_call_output``),
- nested ``{"type":"function","function":{...}}`` tools → flattened
  ``{"type":"function","name":...,"parameters":...}``,
- the flat ``reasoning_effort`` scalar → a ``reasoning={"effort": ...}`` object,
- ``response_format`` → ``text={"format": {...}}``,

and reads ``response.output_text`` plus ``function_call`` items out of
``response.output``.

It is OpenAI-only (unlike the multi-vendor ``OpenAICompatibleLLM``, it carries no
groq/ollama/deepseek special-casing) and stands alone on ``LLMInterface`` so the
"always Responses, never completions" guarantee is structural.

The conversation is replayed statelessly on every turn (``store=False``, no
``previous_response_id``); the model re-derives reasoning each turn rather than
resuming a server-side chain. Server-side reasoning reuse across turns is a
possible future optimization.
"""

import asyncio
import json
import logging
import os
import time
from contextlib import AbstractAsyncContextManager, nullcontext
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse, urlunparse

from openai import APIConnectionError, APIStatusError, AsyncOpenAI

from hindsight_api.config import DEFAULT_LLM_TIMEOUT, ENV_LLM_TIMEOUT
from hindsight_api.engine.bank_attribution import apply_bank_attribution
from hindsight_api.engine.llm_interface import (
    LLM_TOOL_CHOICE_AUTO,
    LLMInterface,
    LLMToolChoice,
    LLMToolChoiceMode,
    OutputTooLongError,
)
from hindsight_api.engine.llm_trace import LLMResponseUsage, stash_response_usage
from hindsight_api.engine.providers.llm_debug import dump_request_on_4xx

# Provider-agnostic pure helpers (text cleanup, quota-defer parsing, json-mode
# hint). These are module-level utilities, not chat/completions behavior.
from hindsight_api.engine.providers.openai_compatible_llm import (
    _ensure_json_word_in_user_message,
    _raise_provider_quota_defer,
    _strip_code_fences,
    _strip_reasoning_tags,
)
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, TokenUsage
from hindsight_api.engine.structured_output import strict_json_schema
from hindsight_api.metrics import get_metrics_collector
from hindsight_api.worker.stage import set_stage

logger = logging.getLogger(__name__)


def _tools_to_responses(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert chat/completions tool schemas to the flattened Responses shape.

    Chat nests the function under a ``function`` key; Responses lifts ``name`` /
    ``description`` / ``parameters`` to the top level. Anything already flattened
    (or a built-in tool) is passed through unchanged.
    """
    converted: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if tool.get("type") == "function" and isinstance(function, dict):
            item: dict[str, Any] = {
                "type": "function",
                "name": function.get("name"),
                "parameters": function.get("parameters", {}),
            }
            if function.get("description") is not None:
                item["description"] = function["description"]
            if "strict" in function:
                item["strict"] = function["strict"]
            converted.append(item)
        else:
            converted.append(tool)
    return converted


def _messages_to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert chat/completions messages to Responses ``input`` items.

    - ``role="tool"`` result → ``function_call_output`` keyed by ``call_id``,
    - assistant ``tool_calls`` → ``function_call`` items (plus a message item when
      the assistant turn also carried text),
    - every other message → a role/content message item.
    """
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")

        if role == "tool":
            content = message.get("content")
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id"),
                    "output": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
                }
            )
            continue

        if role == "assistant" and message.get("tool_calls"):
            text = message.get("content")
            if isinstance(text, str) and text:
                items.append({"role": "assistant", "content": text})
            for tool_call in message["tool_calls"]:
                function = tool_call.get("function", {})
                arguments = function.get("arguments")
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.get("id"),
                        "name": function.get("name"),
                        "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments or {}),
                    }
                )
            continue

        content = message.get("content")
        items.append({"role": role, "content": content if content is not None else ""})
    return items


def _inject_schema_into_input(input_items: list[dict[str, Any]], schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Soft structured-output path: append the JSON schema to the first message.

    A leading ``system`` message gets the schema appended, otherwise it is
    prepended to the first message. Only string-content message items are touched.
    """
    schema_msg = (
        f"\n\nYou must respond with valid JSON matching this schema:\n"
        f"{json.dumps(schema, indent=2, ensure_ascii=False)}"
    )
    items = [dict(item) for item in input_items]
    for item in items:
        if not isinstance(item.get("content"), str):
            continue
        if item.get("role") == "system":
            item["content"] += schema_msg
        else:
            item["content"] = schema_msg + "\n\n" + item["content"]
        return items
    return items


class OpenAIResponsesLLM(LLMInterface):
    """OpenAI provider that talks exclusively to the Responses API.

    OpenAI-only and standalone (does not inherit the multi-vendor
    chat/completions provider), so it never routes through ``chat.completions``.
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        reasoning_effort: str | None = None,
        timeout: float | None = None,
        extra_body: dict[str, Any] | None = None,
        default_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        """Initialize the Responses provider.

        Args:
            provider: Provider name (``"openai-responses"``).
            api_key: OpenAI API key (required).
            base_url: Optional custom endpoint; empty uses the OpenAI default.
            model: Model name (reasoning models — gpt-5.x, o-series — benefit most).
            reasoning_effort: Effort forwarded as ``reasoning={"effort": ...}`` for
                reasoning models. Unlike chat/completions, this is sent alongside
                function tools.
            timeout: Request timeout in seconds (env var or 120s default).
            extra_body: Extra body params merged into every request.
            default_headers: Custom headers passed to the OpenAI SDK client (for
                operators routing through proxies / request-tracing middleware).
            **kwargs: Additional provider-specific parameters (e.g. ``openai_service_tier``).
        """
        super().__init__(provider, api_key, base_url, model, reasoning_effort, **kwargs)

        if not self.api_key:
            raise ValueError(f"API key is required for {self.provider}")

        # OpenAI service tier ("flex" = 50% cheaper / variable latency, "priority",
        # "default"). Applied as the native Responses ``service_tier`` param.
        self.openai_service_tier = kwargs.get("openai_service_tier")
        self._config_extra_body = extra_body or {}
        self.default_headers = default_headers
        self.timeout = timeout or float(os.getenv(ENV_LLM_TIMEOUT, str(DEFAULT_LLM_TIMEOUT)))

        # Manual retries (max_retries=0). Extract query params from base_url so an
        # Azure-style ``?api-version=`` is forwarded as a default query param.
        client_kwargs: dict[str, Any] = {"api_key": self.api_key, "max_retries": 0}
        if self.default_headers:
            client_kwargs["default_headers"] = self.default_headers
        if self.base_url:
            parsed = urlparse(self.base_url)
            if parsed.query:
                clean_url = urlunparse(parsed._replace(query=""))
                client_kwargs["base_url"] = clean_url
                client_kwargs["default_query"] = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                self.base_url = clean_url
            else:
                client_kwargs["base_url"] = self.base_url
        if self.timeout:
            client_kwargs["timeout"] = self.timeout

        self._client = AsyncOpenAI(**client_kwargs)
        logger.info(
            f"OpenAI Responses client initialized: provider={self.provider}, model={self.model}, "
            f"base_url={self.base_url or 'default'}"
        )

    def _supports_reasoning_model(self) -> bool:
        """Whether the model is an OpenAI reasoning model (gpt-5.x, o1, o3)."""
        model_lower = self.model.lower()
        return any(x in model_lower for x in ["gpt-5", "o1", "o3"])

    async def verify_connection(self) -> None:
        """Verify configuration with a minimal Responses call."""
        try:
            logger.info(f"Verifying connection: {self.provider}/{self.model}")
            await self.call(
                messages=[{"role": "user", "content": "Say 'ok'"}],
                max_completion_tokens=512,
                max_retries=2,
                initial_backoff=0.5,
                max_backoff=2.0,
                scope="verification",
            )
            logger.info(f"Connection verified: {self.provider}/{self.model}")
        except Exception as e:
            raise RuntimeError(f"Connection verification failed for {self.provider}/{self.model}: {e}") from e

    async def cleanup(self) -> None:
        """Close the OpenAI client connections."""
        if hasattr(self, "_client") and self._client:
            await self._client.close()

    @staticmethod
    def _extract_usage(response: Any) -> TokenUsage:
        """Read Responses token usage, splitting reasoning out of visible output.

        The provider folds reasoning tokens into ``output_tokens``; the
        ``TokenUsage`` contract treats output as visible-only and surfaces
        reasoning separately, so subtract to avoid double-counting.
        """
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        raw_output = int(getattr(usage, "output_tokens", 0) or 0)
        cached = int(getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0)
        reasoning = int(getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0) or 0)
        visible_output = max(0, raw_output - reasoning)
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=visible_output,
            total_tokens=input_tokens + visible_output,
            cached_tokens=cached,
            thoughts_tokens=reasoning,
        )

    @staticmethod
    def _raise_if_truncated(response: Any) -> None:
        """Map a ``max_output_tokens`` truncation to ``OutputTooLongError``."""
        if getattr(response, "status", None) != "incomplete":
            return
        details = getattr(response, "incomplete_details", None)
        if getattr(details, "reason", None) == "max_output_tokens":
            raise OutputTooLongError(
                "LLM output exceeded token limits. Input may need to be split into smaller chunks."
            )

    def _record_success(
        self,
        *,
        scope: str,
        usage: TokenUsage,
        duration: float,
        span_messages: list[dict[str, Any]],
        response_content: Any,
        finish_reason: str,
        tool_calls_dict: list[dict[str, Any]] | None = None,
    ) -> None:
        """Record metrics + a trace span for a successful call (shared by both paths)."""
        get_metrics_collector().record_llm_call(
            provider=self.provider,
            model=self.model,
            scope=scope,
            duration=duration,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            success=True,
            cached_input_tokens=usage.cached_tokens,
            thoughts_tokens=usage.thoughts_tokens,
        )

        from hindsight_api.tracing import _serialize_for_span, get_span_recorder

        get_span_recorder().record_llm_call(
            provider=self.provider,
            model=self.model,
            scope=scope,
            messages=span_messages,
            response_content=_serialize_for_span(response_content),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            duration=duration,
            finish_reason=finish_reason,
            error=None,
            cached_tokens=usage.cached_tokens,
            tool_calls=tool_calls_dict,
        )

    async def _run_with_retries(
        self,
        params: dict[str, Any],
        *,
        scope: str,
        parse: Any,
        max_retries: int,
        initial_backoff: float,
        max_backoff: float,
        attempt_context: Callable[[], AbstractAsyncContextManager[None]] | None = None,
    ) -> Any:
        """Call ``responses.create`` with retries and hand the response to ``parse``.

        ``parse(response)`` extracts the result; raising ``json.JSONDecodeError``
        signals a bad structured reply and triggers a retry. Connection and
        non-auth 4xx/5xx errors back off and retry; 401/403 fail fast. Provider
        usage is stashed before ``parse`` runs so an error trace can still attach
        real token counts (see #2387).
        """
        last_exception: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                async with attempt_context() if attempt_context is not None else nullcontext():
                    set_stage(f"llm.{self.provider}.{scope}.attempt={attempt + 1}/{max_retries + 1}")
                    response = await self._client.responses.create(**params)
                usage = self._extract_usage(response)
                stash_response_usage(
                    LLMResponseUsage(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        cached_tokens=usage.cached_tokens,
                    )
                )
                return parse(response)

            except json.JSONDecodeError as e:
                last_exception = e
                logger.warning(
                    f"JSON parse error from Responses reply "
                    f"({self.provider}/{self.model}, scope={scope}, attempt {attempt + 1}/{max_retries + 1}): "
                    f"{str(e)[:200]}"
                )
                if attempt < max_retries:
                    await asyncio.sleep(min(initial_backoff * (2**attempt), max_backoff))
                    continue
                raise

            except APIConnectionError as e:
                last_exception = e
                status_code = getattr(e, "status_code", None) or getattr(
                    getattr(e, "response", None), "status_code", None
                )
                logger.warning(
                    f"APIConnectionError ({self.provider}/{self.model}, scope={scope}, HTTP {status_code}, "
                    f"attempt {attempt + 1}/{max_retries + 1}): {str(e)[:200]}"
                )
                if attempt < max_retries:
                    await asyncio.sleep(min(initial_backoff * (2**attempt), max_backoff))
                    continue
                raise

            except APIStatusError as e:
                if e.status_code in (401, 403):
                    logger.error(f"Auth error (HTTP {e.status_code}, {self.provider}/{self.model}), not retrying")
                    raise

                dump_request_on_4xx(scope=scope, provider=self.provider, model=self.model, err=e, request=params)
                _raise_provider_quota_defer(
                    e, provider=self.provider, model=self.model, scope=scope, max_backoff=max_backoff
                )

                last_exception = e
                if attempt < max_retries:
                    logger.warning(
                        f"APIStatusError ({self.provider}/{self.model}, scope={scope}, "
                        f"attempt {attempt + 1}/{max_retries + 1}): HTTP {e.status_code}"
                    )
                    backoff = min(initial_backoff * (2**attempt), max_backoff)
                    jitter = backoff * 0.2 * (2 * (time.time() % 1) - 1)
                    await asyncio.sleep(backoff + jitter)
                    continue
                logger.error(
                    f"API error after {max_retries + 1} attempts "
                    f"({self.provider}/{self.model}, scope={scope}): HTTP {e.status_code}"
                )
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("LLM call failed after all retries with no exception captured")

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
        """Make a Responses API call with retry logic (see ``LLMInterface.call``)."""
        start_time = time.time()
        is_reasoning_model = self._supports_reasoning_model()

        input_items = _messages_to_responses_input(messages)
        params: dict[str, Any] = {"model": self.model, "input": input_items, "store": False}

        if max_completion_tokens is not None:
            # Reasoning models spend part of the budget on hidden reasoning; keep
            # headroom so the visible answer isn't truncated.
            if is_reasoning_model and max_completion_tokens < 16000:
                max_completion_tokens = 16000
            params["max_output_tokens"] = max_completion_tokens
        if temperature is not None and not is_reasoning_model:
            params["temperature"] = temperature
        # Only when the operator configured a level: unset means the model runs at the
        # Responses API's own default effort rather than one Hindsight picked.
        if is_reasoning_model and self.reasoning_effort is not None:
            params["reasoning"] = {"effort": self.reasoning_effort}
        if self.openai_service_tier:
            params["service_tier"] = self.openai_service_tier
        if self._config_extra_body:
            params["extra_body"] = {**self._config_extra_body}

        # Structured output: strict grammar-enforced schema, or the soft
        # schema-in-prompt + json_object fallback.
        if response_format is not None and hasattr(response_format, "model_json_schema"):
            if strict_schema:
                params["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "response",
                        "strict": True,
                        "schema": strict_json_schema(response_format),
                    }
                }
            else:
                schema = response_format.model_json_schema()
                params["input"] = _ensure_json_word_in_user_message(_inject_schema_into_input(input_items, schema))
                params["text"] = {"format": {"type": "json_object"}}

        apply_bank_attribution(params)

        def parse(response: Any) -> Any:
            self._raise_if_truncated(response)
            usage = self._extract_usage(response)
            duration = time.time() - start_time
            content = _strip_reasoning_tags(response.output_text or "")

            if response_format is not None:
                json_data = json.loads(_strip_code_fences(content))
                result: Any = json_data if skip_validation else response_format.model_validate(json_data)
            else:
                result = content

            self._record_success(
                scope=scope,
                usage=usage,
                duration=duration,
                span_messages=params["input"],
                response_content=result,
                finish_reason=getattr(response, "status", "completed"),
            )
            if duration > 10.0:
                logger.info(
                    f"slow llm call: scope={scope}, model={self.provider}/{self.model}, "
                    f"input_tokens={usage.input_tokens}, output_tokens={usage.output_tokens}, time={duration:.3f}s"
                )
            return (result, usage) if return_usage else result

        return await self._run_with_retries(
            params,
            scope=scope,
            parse=parse,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            max_backoff=max_backoff,
            attempt_context=attempt_context,
        )

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
        """Make a Responses API call with tools (see ``LLMInterface.call_with_tools``).

        Unlike chat/completions, ``reasoning`` and ``tools`` are sent together —
        the reason this provider exists.
        """
        start_time = time.time()
        is_reasoning_model = self._supports_reasoning_model()

        responses_tools = _tools_to_responses(tools)

        request_tool_choice: str | dict[str, Any] | None
        if tool_choice.mode is LLMToolChoiceMode.NAMED:
            forced_name = tool_choice.selected_function_name
            filtered = [tool for tool in responses_tools if tool.get("name") == forced_name]
            if len(filtered) != 1:
                raise ValueError(
                    f"Named tool_choice must reference exactly one declared tool; "
                    f"found {len(filtered)} definitions for {forced_name!r}"
                )
            responses_tools = filtered
            request_tool_choice = {"type": "function", "name": forced_name}
        elif tool_choice.mode is LLMToolChoiceMode.AUTO:
            request_tool_choice = None
        else:
            request_tool_choice = tool_choice.mode.value

        params: dict[str, Any] = {
            "model": self.model,
            "input": _messages_to_responses_input(messages),
            "tools": responses_tools,
            "store": False,
        }
        if request_tool_choice is not None:
            params["tool_choice"] = request_tool_choice
        if max_completion_tokens is not None:
            params["max_output_tokens"] = max_completion_tokens
        if temperature is not None and not is_reasoning_model:
            params["temperature"] = temperature
        # Only when the operator configured a level: unset means the model runs at the
        # Responses API's own default effort rather than one Hindsight picked.
        if is_reasoning_model and self.reasoning_effort is not None:
            params["reasoning"] = {"effort": self.reasoning_effort}
        if self.openai_service_tier:
            params["service_tier"] = self.openai_service_tier
        if self._config_extra_body:
            params["extra_body"] = {**self._config_extra_body}

        apply_bank_attribution(params)

        def parse(response: Any) -> LLMToolCallResult:
            self._raise_if_truncated(response)
            usage = self._extract_usage(response)
            duration = time.time() - start_time

            tool_calls: list[LLMToolCall] = []
            for item in getattr(response, "output", None) or []:
                if getattr(item, "type", None) != "function_call":
                    continue
                raw_args = getattr(item, "arguments", "") or ""
                try:
                    arguments = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    arguments = {"_raw": raw_args}
                tool_calls.append(
                    LLMToolCall(id=getattr(item, "call_id", None), name=getattr(item, "name", ""), arguments=arguments)
                )

            content = response.output_text or None
            finish_reason = "tool_calls" if tool_calls else getattr(response, "status", "completed")

            self._record_success(
                scope=scope,
                usage=usage,
                duration=duration,
                span_messages=params["input"],
                response_content=content,
                finish_reason=finish_reason,
                tool_calls_dict=(
                    [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls]
                    if tool_calls
                    else None
                ),
            )

            return LLMToolCallResult(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_tokens=usage.cached_tokens,
                thoughts_tokens=usage.thoughts_tokens,
            )

        return await self._run_with_retries(
            params,
            scope=scope,
            parse=parse,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            max_backoff=max_backoff,
            attempt_context=attempt_context,
        )

    def supports_attempt_scoped_concurrency(self) -> bool:
        return True
