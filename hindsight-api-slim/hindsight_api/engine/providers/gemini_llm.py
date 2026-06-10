"""
Google Gemini/VertexAI LLM provider.

This provider supports both:
1. Gemini API (api.generativeai.google.com) with API key authentication
2. Vertex AI with service account or Application Default Credentials (ADC)
"""

import asyncio
import base64
import io
import json
import logging
import os
import time
from contextvars import ContextVar
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from hindsight_api.engine.llm_interface import LLMInterface, OutputTooLongError
from hindsight_api.engine.llm_wrapper import parse_llm_json
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, TokenUsage
from hindsight_api.metrics import get_metrics_collector
from hindsight_api.worker.stage import set_stage

logger = logging.getLogger(__name__)

# Per-request Gemini safety settings override.
# Set exclusively by ConfiguredLLMProvider.call() / call_with_tools() via token-based
# set/reset, so it is properly scoped to each individual LLM call and never leaks.
_safety_settings_ctx: ContextVar[list | None] = ContextVar("gemini_safety_settings", default=None)


# Vertex AI imports (optional)
try:
    import google.auth
    from google.oauth2 import service_account

    VERTEXAI_AVAILABLE = True
except ImportError:
    VERTEXAI_AVAILABLE = False


def _to_int(value: Any) -> int:
    """Coerce Gemini's optional/string completion counts to int, defaulting to 0."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


class GeminiLLM(LLMInterface):
    """
    LLM provider for Google Gemini and Vertex AI.

    Supports:
    - Gemini API: provider="gemini", requires api_key
    - Vertex AI: provider="vertexai", requires project_id and region, uses ADC or service account
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        reasoning_effort: str = "low",
        **kwargs: Any,
    ):
        """Initialize Gemini/VertexAI LLM provider."""
        super().__init__(provider, api_key, base_url, model, reasoning_effort, **kwargs)

        self._client = None
        self._is_vertexai = self.provider == "vertexai"

        # Safety settings: None means use Gemini's defaults
        self._safety_settings: list | None = kwargs.get("gemini_safety_settings")

        # User-configured extra params merged into the GenerateContentConfig of
        # every call. Gemini's request body nests generation params, so we expose
        # them in the SDK's native config space rather than as a raw body merge:
        # keys must be GenerateContentConfig fields (e.g. temperature, top_p,
        # top_k, max_output_tokens, seed). Sourced from llm_extra_body
        # (env: HINDSIGHT_API_LLM_EXTRA_BODY).
        self._extra_body: dict[str, Any] = kwargs.get("extra_body") or {}

        # Context-cache manager. Lazy-initialized on first cache lookup so
        # nothing happens for models/workloads that never reach it. The instance
        # default here is off (a directly-constructed GeminiLLM doesn't cache); the
        # server-level default is on and flows in via the prompt_cache_enabled kwarg
        # resolved from config in LLMProvider.
        self._cache_manager: Any | None = None
        self._prompt_cache_enabled: bool = bool(kwargs.get("prompt_cache_enabled", False))

        if self._is_vertexai:
            self._init_vertexai(**kwargs)
        else:
            self._init_gemini()

    def _init_gemini(self) -> None:
        """Initialize Gemini API client."""
        if not self.api_key:
            raise ValueError("Gemini provider requires api_key")

        self._client = genai.Client(api_key=self.api_key)
        logger.info(f"Gemini API: model={self.model}")

    def _init_vertexai(self, **kwargs: Any) -> None:
        """Initialize Vertex AI client with project, region, and credentials."""
        # Extract Vertex AI config from kwargs
        project_id = kwargs.get("vertexai_project_id")
        region = kwargs.get("vertexai_region", "us-central1")
        service_account_key = kwargs.get("vertexai_service_account_key")
        credentials = kwargs.get("vertexai_credentials")  # Pre-loaded credentials object

        if not project_id:
            raise ValueError(
                "HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID is required for Vertex AI provider. "
                "Set it to your GCP project ID."
            )

        auth_method = "ADC"

        # Use pre-loaded credentials if provided (passed from LLMProvider)
        if credentials is not None:
            auth_method = "service_account"
        # Otherwise, load explicit service account credentials if path provided
        elif service_account_key:
            if not VERTEXAI_AVAILABLE:
                raise ValueError(
                    "Vertex AI service account auth requires 'google-auth' package. "
                    "Install with: pip install google-auth"
                )
            credentials = service_account.Credentials.from_service_account_file(
                service_account_key,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            auth_method = "service_account"
            logger.info(f"Vertex AI: Using service account key: {service_account_key}")

        # Strip google/ prefix from model name — native SDK uses bare names
        # e.g. "google/gemini-2.0-flash-lite-001" -> "gemini-2.0-flash-lite-001"
        if self.model.startswith("google/"):
            self.model = self.model[len("google/") :]

        # Create Vertex AI client
        client_kwargs: dict[str, Any] = {
            "vertexai": True,
            "project": project_id,
            "location": region,
        }
        if credentials is not None:
            client_kwargs["credentials"] = credentials

        self._client = genai.Client(**client_kwargs)

        logger.info(f"Vertex AI: project={project_id}, region={region}, model={self.model}, auth={auth_method}")

    async def verify_connection(self) -> None:
        """
        Verify that the Gemini/VertexAI provider is configured correctly.

        Raises:
            RuntimeError: If the connection test fails.
        """
        try:
            logger.info(f"Verifying {self.provider.upper()}: model={self.model}...")
            await self.call(
                messages=[{"role": "user", "content": "Say 'ok'"}],
                max_completion_tokens=100,
                max_retries=2,
                initial_backoff=0.5,
                max_backoff=2.0,
                scope="verification",
            )
            logger.info(f"{self.provider.upper()} connection verified successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to verify {self.provider.upper()} connection: {e}") from e

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
    ) -> Any:
        """
        Make a Gemini/VertexAI API call with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            response_format: Optional Pydantic model for structured output.
            max_completion_tokens: Maximum tokens in response (mapped to Gemini's max_output_tokens).
            temperature: Sampling temperature (0.0-2.0).
            scope: Scope identifier for tracking.
            max_retries: Maximum retry attempts.
            initial_backoff: Initial backoff time in seconds.
            max_backoff: Maximum backoff time in seconds.
            skip_validation: Return raw JSON without Pydantic validation.
            strict_schema: Ignored — Gemini always grammar-enforces structured output via its
                native response_schema, so it is strict regardless of this flag.
            return_usage: If True, return tuple (result, TokenUsage).
            cached_prefix: Optional CachedContent resource name (from
                ``GeminiCacheManager.get_or_create``). When set, the
                system_instruction is assumed to live in the cache; this call
                skips resending it and the cached prefix is billed at the
                cached-input rate instead of the standard input rate. The
                response_schema is still sent per-request (it is not cacheable).
                Pass ``None`` to use the
                normal uncached path.

        Returns:
            If return_usage=False: Parsed response if response_format provided, else text.
            If return_usage=True: Tuple of (result, TokenUsage).
        """
        start_time = time.time()

        # Convert OpenAI-style messages to Gemini format. We ALWAYS build
        # system_instruction (even when a cache is in use): the config builder
        # below omits it from the request while the cache carries the prefix, but
        # it must be available so the cached-call-failed safety net can re-send it
        # inline. Whether it's actually sent is decided in _build_generation_config.
        system_instruction = None
        gemini_contents = []
        using_cache = cached_prefix is not None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if system_instruction:
                    system_instruction += "\n\n" + content
                else:
                    system_instruction = content
            elif role == "assistant":
                gemini_contents.append(genai_types.Content(role="model", parts=[genai_types.Part(text=content)]))
            else:
                gemini_contents.append(genai_types.Content(role="user", parts=[genai_types.Part(text=content)]))

        # Add the JSON schema as a textual hint in the system_instruction (matching
        # the normal uncached path). Structured output is still enforced via
        # response_schema regardless; this is just guidance text.
        if response_format is not None and hasattr(response_format, "model_json_schema"):
            schema = response_format.model_json_schema()
            schema_msg = f"\n\nYou must respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2, ensure_ascii=False)}"
            if system_instruction:
                system_instruction += schema_msg
            else:
                system_instruction = schema_msg

        # Apply safety settings: context var (per-request bank override) takes precedence over instance default
        effective_safety_settings = _safety_settings_ctx.get()
        if effective_safety_settings is None:
            effective_safety_settings = self._safety_settings

        # Build generation config. ``cached_content`` and ``system_instruction``
        # are mutually exclusive (the cache IS the prefix; the SDK rejects
        # re-sending it). ``response_schema``/``response_mime_type`` are
        # request-level output constraints — NOT cacheable — so they're set on
        # every structured call, including cached ones where they ride alongside
        # ``cached_content``. Built as a closure so we can rebuild it WITHOUT the
        # cache and retry inline if a stale/invalid CachedContent makes the call fail.
        def _build_generation_config(use_cache: bool) -> "genai_types.GenerateContentConfig | None":
            # Seed with user-configured extra params; explicit settings below win.
            config_kwargs: dict[str, Any] = dict(self._extra_body)
            if use_cache:
                config_kwargs["cached_content"] = cached_prefix
            elif system_instruction:
                config_kwargs["system_instruction"] = system_instruction
            if response_format is not None:
                config_kwargs["response_mime_type"] = "application/json"
                config_kwargs["response_schema"] = response_format
            if temperature is not None:
                config_kwargs["temperature"] = temperature
            # Gemini's equivalent of OpenAI-style max_completion_tokens is max_output_tokens.
            # Without it the model can produce arbitrarily long responses, ignoring the
            # caller's intended cap (e.g. mental_models max_tokens during refresh).
            if max_completion_tokens is not None:
                config_kwargs["max_output_tokens"] = max_completion_tokens
            if effective_safety_settings is not None:
                config_kwargs["safety_settings"] = [
                    genai_types.SafetySetting(category=s["category"], threshold=s["threshold"])
                    for s in effective_safety_settings
                ]
            return genai_types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        cache_active = using_cache
        generation_config = _build_generation_config(cache_active)

        last_exception = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                set_stage(f"llm.gemini.{scope}.attempt={attempt + 1}/{max_retries + 1}")
            try:
                response = await asyncio.wait_for(
                    self._client.aio.models.generate_content(
                        model=self.model,
                        contents=gemini_contents,
                        config=generation_config,
                    ),
                    timeout=90.0,  # Safety net for network hangs; valid slow responses are <90s
                )

                content = response.text

                # Handle empty response
                if content is None:
                    block_reason = None
                    if hasattr(response, "candidates") and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, "finish_reason"):
                            block_reason = candidate.finish_reason

                    if attempt < max_retries:
                        logger.warning(f"Gemini returned empty response (reason: {block_reason}), retrying...")
                        backoff = min(initial_backoff * (2**attempt), max_backoff)
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        raise RuntimeError(f"Gemini returned empty response after {max_retries + 1} attempts")

                # Parse structured output if requested
                if response_format is not None:
                    json_data = parse_llm_json(content)
                    if skip_validation:
                        result = json_data
                    else:
                        result = response_format.model_validate(json_data)
                else:
                    result = content

                # Extract token usage. ``cached_content_token_count`` and
                # ``thoughts_token_count`` are populated on the Gemini 2.5+
                # family; treat missing fields as 0 so older models still
                # record sensible metrics.
                input_tokens = 0
                output_tokens = 0
                cached_input_tokens = 0
                thoughts_tokens = 0
                cached_tokens = 0
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage = response.usage_metadata
                    input_tokens = usage.prompt_token_count or 0
                    output_tokens = usage.candidates_token_count or 0
                    cached_input_tokens = getattr(usage, "cached_content_token_count", 0) or 0
                    thoughts_tokens = getattr(usage, "thoughts_token_count", 0) or 0
                    # Tracing/TokenUsage consume ``cached_tokens``; metrics consume
                    # ``cached_input_tokens`` — same value, two downstream names.
                    cached_tokens = cached_input_tokens

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
                    cached_input_tokens=cached_input_tokens,
                    thoughts_tokens=thoughts_tokens,
                )

                # Record trace span
                from hindsight_api.tracing import get_span_recorder

                finish_reason = None
                if hasattr(response, "candidates") and response.candidates:
                    if hasattr(response.candidates[0], "finish_reason"):
                        finish_reason = str(response.candidates[0].finish_reason)
                span_recorder = get_span_recorder()
                from hindsight_api.tracing import _serialize_for_span

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
                if duration > 10.0 and input_tokens > 0:
                    logger.info(
                        f"slow llm call: scope={scope}, model={self.provider}/{self.model}, "
                        f"input_tokens={input_tokens}, output_tokens={output_tokens}, "
                        f"time={duration:.3f}s"
                    )

                if return_usage:
                    token_usage = TokenUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                        cached_tokens=cached_tokens,
                    )
                    return result, token_usage
                return result

            except json.JSONDecodeError as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning("Gemini returned invalid JSON, retrying...")
                    backoff = min(initial_backoff * (2**attempt), max_backoff)
                    await asyncio.sleep(backoff)
                    continue
                else:
                    logger.error(f"Gemini returned invalid JSON after {max_retries + 1} attempts")
                    raise

            except genai_errors.APIError as e:
                # Fast fail on auth errors - these won't recover with retries
                if e.code in (401, 403):
                    logger.error(f"Gemini auth error (HTTP {e.code}), not retrying: {str(e)}")
                    raise

                # Cached-request safety net: a stale/invalid/expired CachedContent
                # (or an incompatibility like cache + tool_config) surfaces as a 400.
                # Retrying the same cached request can't recover, so on the first
                # such failure drop the cache, invalidate it so later operations
                # recreate it, and retry THIS call inline with the prefix inlined.
                # Caching must never break a request.
                if cache_active and e.code == 400:
                    logger.warning(f"Gemini cached call failed (400); retrying uncached. Reason: {str(e)}")
                    if self._cache_manager is not None and cached_prefix is not None:
                        self._cache_manager.invalidate(cached_prefix)
                    cache_active = False
                    generation_config = _build_generation_config(cache_active)
                    continue

                # Retry on retryable errors (rate limits, server errors, client errors)
                if e.code in (400, 429, 500, 502, 503, 504) or (e.code and e.code >= 500):
                    last_exception = e
                    if attempt < max_retries:
                        backoff = min(initial_backoff * (2**attempt), max_backoff)
                        jitter = backoff * 0.2 * (2 * (time.time() % 1) - 1)
                        await asyncio.sleep(backoff + jitter)
                    else:
                        logger.error(f"Gemini API error after {max_retries + 1} attempts: {str(e)}")
                        raise
                else:
                    logger.error(f"Gemini API error: {type(e).__name__}: {str(e)}")
                    raise

            except Exception as e:
                logger.error(f"Unexpected error during Gemini call: {type(e).__name__}: {str(e)}")
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Gemini call failed after all retries")

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
        cached_prefix: str | None = None,
    ) -> LLMToolCallResult:
        """
        Make a Gemini/VertexAI API call with tool/function calling support.

        Args:
            messages: List of message dicts. Can include tool results with role='tool'.
            tools: List of tool definitions in OpenAI format.
            max_completion_tokens: Maximum tokens (mapped to Gemini's max_output_tokens).
            temperature: Sampling temperature.
            scope: Scope identifier for tracking.
            max_retries: Maximum retry attempts.
            initial_backoff: Initial backoff time in seconds.
            max_backoff: Maximum backoff time in seconds.
            tool_choice: How to choose tools (Gemini uses "auto" only).
            cached_prefix: Optional CachedContent resource name (from
                ``GeminiCacheManager.get_or_create`` with ``tools=...``). When
                set, the system_instruction and tool definitions are assumed
                to live in the cache; this call will skip resending them and
                the cached prefix is billed at the cached-input rate. The
                ``tools`` argument is still required (the caller may pass
                an empty list when the cache holds them) so existing call
                sites don't break.

        Returns:
            LLMToolCallResult with content and/or tool_calls.
        """
        start_time = time.time()
        using_cache = cached_prefix is not None

        # Convert tools to Gemini format. When the cache is in use, the
        # tool definitions are baked into the CachedContent at create time
        # and the SDK rejects re-sending them alongside ``cached_content``.
        gemini_tools = []
        if not using_cache:
            for tool in tools:
                func = tool.get("function", {})
                gemini_tools.append(
                    genai_types.Tool(
                        function_declarations=[
                            genai_types.FunctionDeclaration(
                                name=func.get("name", ""),
                                description=func.get("description", ""),
                                parameters=func.get("parameters"),
                            )
                        ]
                    )
                )

        # Convert messages
        system_instruction = None
        gemini_contents = []
        msg_list = list(messages)
        i = 0
        while i < len(msg_list):
            msg = msg_list[i]
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Always capture system_instruction. _build_tools_config omits it
                # (and tools) from the request while the cache carries the prefix,
                # but it must be available so the cached-call-failed safety net can
                # re-send the prefix + tools inline.
                system_instruction = (system_instruction + "\n\n" + content) if system_instruction else content
                i += 1
            elif role == "tool":
                # Gemini requires ALL tool responses for a given model turn to be grouped
                # into a single Content with multiple FunctionResponse parts.
                # Consecutive role="tool" messages correspond to one model turn's tool calls.
                parts = []
                while i < len(msg_list) and msg_list[i].get("role") == "tool":
                    tool_msg = msg_list[i]
                    tool_content = tool_msg.get("content", "")
                    parts.append(
                        genai_types.Part(
                            function_response=genai_types.FunctionResponse(
                                name=tool_msg.get("name", ""),
                                response={"result": tool_content},
                            )
                        )
                    )
                    i += 1
                gemini_contents.append(genai_types.Content(role="user", parts=parts))
            elif role == "assistant":
                tool_calls_in_msg = msg.get("tool_calls", [])
                if tool_calls_in_msg:
                    # Convert OpenAI-style tool_calls to Gemini function_call parts
                    # This is required for proper multi-turn conversation history
                    parts = []
                    if content:
                        parts.append(genai_types.Part(text=content))
                    for tc in tool_calls_in_msg:
                        fn = tc.get("function", {})
                        fn_name = fn.get("name", "")
                        fn_args_str = fn.get("arguments", "{}")
                        fn_args = parse_llm_json(fn_args_str)
                        thought_signature = tc.get("thought_signature")
                        fc_kwargs: dict[str, Any] = {"name": fn_name, "args": fn_args}
                        part_kwargs: dict[str, Any] = {"function_call": genai_types.FunctionCall(**fc_kwargs)}
                        if thought_signature:
                            part_kwargs["thought_signature"] = base64.b64decode(thought_signature)
                        parts.append(genai_types.Part(**part_kwargs))
                    gemini_contents.append(genai_types.Content(role="model", parts=parts))
                else:
                    gemini_contents.append(genai_types.Content(role="model", parts=[genai_types.Part(text=content)]))
                i += 1
            else:
                gemini_contents.append(genai_types.Content(role="user", parts=[genai_types.Part(text=content)]))
                i += 1

        # Apply safety settings: context var (per-request bank override) takes precedence over instance default
        effective_safety_settings = _safety_settings_ctx.get()
        if effective_safety_settings is None:
            effective_safety_settings = self._safety_settings

        # When using a cached prefix, the SDK rejects re-sending system_instruction
        # or tools alongside ``cached_content`` — the cache IS the prefix.
        # tool_config (mode / allowed_function_names) is a per-request decision and
        # stays out of the cache. Built as a closure so we can rebuild it WITHOUT
        # the cache and retry inline if a stale/invalid cache makes the call fail.
        def _build_tools_config(use_cache: bool) -> "genai_types.GenerateContentConfig":
            # Seed with user-configured extra params; explicit settings below win.
            config_kwargs: dict[str, Any] = dict(self._extra_body)
            if use_cache:
                config_kwargs["cached_content"] = cached_prefix
            else:
                config_kwargs["tools"] = gemini_tools
                if system_instruction:
                    config_kwargs["system_instruction"] = system_instruction
            if temperature is not None:
                config_kwargs["temperature"] = temperature
            # See note in `call`: Gemini's max_output_tokens is the equivalent of
            # OpenAI-style max_completion_tokens.
            if max_completion_tokens is not None:
                config_kwargs["max_output_tokens"] = max_completion_tokens

            # Map OpenAI-style tool_choice to Gemini FunctionCallingConfig
            if tool_choice == "required":
                config_kwargs["tool_config"] = genai_types.ToolConfig(
                    function_calling_config=genai_types.FunctionCallingConfig(
                        mode="ANY",
                    )
                )
            elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                fn_name = tool_choice.get("function", {}).get("name")
                if fn_name:
                    config_kwargs["tool_config"] = genai_types.ToolConfig(
                        function_calling_config=genai_types.FunctionCallingConfig(
                            mode="ANY",
                            allowed_function_names=[fn_name],
                        )
                    )
            elif tool_choice == "none":
                config_kwargs["tool_config"] = genai_types.ToolConfig(
                    function_calling_config=genai_types.FunctionCallingConfig(mode="NONE")
                )
            # "auto" is the default (no tool_config needed)

            if effective_safety_settings is not None:
                config_kwargs["safety_settings"] = [
                    genai_types.SafetySetting(category=s["category"], threshold=s["threshold"])
                    for s in effective_safety_settings
                ]
            return genai_types.GenerateContentConfig(**config_kwargs)

        cache_active = using_cache
        config = _build_tools_config(cache_active)

        last_exception = None
        for attempt in range(max_retries + 1):
            if attempt > 0:
                set_stage(f"llm.gemini.tools.attempt={attempt + 1}/{max_retries + 1}")
            try:
                response = await asyncio.wait_for(
                    self._client.aio.models.generate_content(
                        model=self.model,
                        contents=gemini_contents,
                        config=config,
                    ),
                    timeout=90.0,  # Safety net for network hangs; valid slow responses are <90s
                )

                # Extract content and tool calls
                content = None
                tool_calls: list[LLMToolCall] = []

                if response.candidates and response.candidates[0].content:
                    parts = response.candidates[0].content.parts
                    if parts:
                        for part in parts:
                            if hasattr(part, "text") and part.text:
                                content = part.text
                            if hasattr(part, "function_call") and part.function_call:
                                fc = part.function_call
                                _raw_ts = getattr(part, "thought_signature", None)
                                thought_signature = (
                                    base64.b64encode(_raw_ts).decode("ascii") if isinstance(_raw_ts, bytes) else _raw_ts
                                )
                                tool_calls.append(
                                    LLMToolCall(
                                        id=f"gemini_{len(tool_calls)}",
                                        name=fc.name,
                                        arguments=dict(fc.args) if fc.args else {},
                                        thought_signature=thought_signature,
                                    )
                                )

                finish_reason = "tool_calls" if tool_calls else "stop"

                # Extract token usage. ``cached_content_token_count`` and
                # ``thoughts_token_count`` are populated on the Gemini 2.5+
                # family; absent fields are treated as 0.
                input_tokens = 0
                output_tokens = 0
                cached_input_tokens = 0
                thoughts_tokens = 0
                if response.usage_metadata:
                    input_tokens = response.usage_metadata.prompt_token_count or 0
                    output_tokens = response.usage_metadata.candidates_token_count or 0
                    cached_input_tokens = getattr(response.usage_metadata, "cached_content_token_count", 0) or 0
                    thoughts_tokens = getattr(response.usage_metadata, "thoughts_token_count", 0) or 0

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
                    cached_input_tokens=cached_input_tokens,
                    thoughts_tokens=thoughts_tokens,
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
                    cached_tokens=cached_input_tokens,
                )

                return LLMToolCallResult(
                    content=content,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            except genai_errors.APIError as e:
                # Fast fail on auth errors
                if e.code in (401, 403):
                    logger.error(f"Gemini auth error (HTTP {e.code}), not retrying: {str(e)}")
                    raise

                # Cached-request safety net (see ``call``): a stale/invalid cache or
                # a cache+tool_config conflict surfaces as a 400. Drop the cache,
                # invalidate it for later operations, and retry THIS call inline
                # with the prefix + tools re-sent. Caching must never break a call.
                if cache_active and e.code == 400:
                    logger.warning(f"Gemini cached tool call failed (400); retrying uncached. Reason: {str(e)}")
                    if self._cache_manager is not None and cached_prefix is not None:
                        self._cache_manager.invalidate(cached_prefix)
                    cache_active = False
                    config = _build_tools_config(cache_active)
                    continue

                # Retry on retryable errors
                last_exception = e
                if attempt < max_retries:
                    backoff = min(initial_backoff * (2**attempt), max_backoff)
                    await asyncio.sleep(backoff)
                    continue
                raise

            except Exception as e:
                logger.error(f"Unexpected error during Gemini tool call: {type(e).__name__}: {str(e)}")
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Gemini tool call failed")

    def supports_prompt_caching(self) -> bool:
        """True when explicit Gemini context caching is enabled for this instance.

        Reflects the opt-in flag so callers skip the cache lookup entirely when
        it's off; ``get_or_create_cached_prefix`` also returns None in that case.
        """
        return self._prompt_cache_enabled

    async def get_or_create_cached_prefix(
        self,
        *,
        system_instruction: str,
        response_schema: Any | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str | None:
        """Return a CachedContent resource name for the given prefix, or
        ``None`` if context caching is disabled, the provider doesn't
        support it, or Gemini rejects the create (prefix too small, etc.).

        ``tools`` is the OpenAI-style tools list; pass it when caching a
        prefix that will be used by ``call_with_tools()``. The fingerprint
        includes the tool definitions so a loop that swaps a tool gets a
        fresh cache automatically.

        Callers pass the returned name to ``call(cached_prefix=...)``
        or ``call_with_tools(cached_prefix=...)`` and treat ``None``
        as "cache unavailable — use the normal path". That fallback is
        essential: the system must continue to work if caching is disabled,
        if Gemini's caching API has an outage, or if the prefix is below
        the model's minimum cacheable size.
        """
        if not self._prompt_cache_enabled:
            return None
        if self._client is None:
            return None
        if self._cache_manager is None:
            # Lazy import so the cache module is only loaded when caching
            # is actually used.
            from hindsight_api.engine.providers.gemini_cache import GeminiCacheManager

            self._cache_manager = GeminiCacheManager(self._client)
        return await self._cache_manager.get_or_create(
            model=self.model,
            system_instruction=system_instruction,
            response_schema=response_schema,
            tools=tools,
        )

    # ── Batch API (Gemini API only — not Vertex AI) ─────────────────────────
    #
    # Google's Gemini Batch API gives a flat 50% discount on input + output
    # tokens with a 24h completion SLA (https://ai.google.dev/gemini-api/docs/batch-api).
    # The retain orchestrator and ``fact_extraction`` consumer speak the
    # OpenAI-batch interface contract, so these overrides translate that shape
    # to/from Gemini's file-upload → ``batches.create`` → ``batches.get`` →
    # download flow — nothing downstream changes (same pattern as FireworksLLM).
    #
    # Interface contract preserved (see fact_extraction.py result handling)::
    #     result["response"]["body"]["choices"][0]["message"]["content"]

    async def supports_batch_api(self) -> bool:
        """True for the Gemini API; False for Vertex AI.

        Only ``provider="gemini"`` is supported: it exposes the file-upload
        Batch API used below. Vertex AI's batch path is GCS/BigQuery-backed (no
        file-upload analogue), so it stays unsupported here — the startup
        validation then surfaces a clear error instead of silently falling back
        to synchronous, full-price calls.
        """
        return self.provider == "gemini"

    async def submit_batch(
        self,
        requests: list[dict[str, Any]],
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
    ) -> dict[str, Any]:
        """Submit a batch of (OpenAI-shaped) requests to the Gemini Batch API."""
        if not await self.supports_batch_api():
            raise NotImplementedError(f"Batch API not supported for provider: {self.provider}")

        # endpoint/completion_window are part of the shared LLMInterface batch
        # contract (used by the OpenAI path) but have no analogue on Gemini: the
        # request shape is fixed (generateContent) and the SLA is server-side.
        # Kept for signature compatibility with the shared retain driver.
        logger.info(f"Submitting Gemini batch with {len(requests)} requests")

        jsonl = self._translate_requests(requests)

        # Upload the JSONL as a Gemini file (mime_type must be "jsonl"; a
        # BytesIO has no path for the SDK to infer it from).
        file_obj = io.BytesIO(jsonl.encode("utf-8"))
        uploaded = await self._client.aio.files.upload(
            file=file_obj,
            config=genai_types.UploadFileConfig(mime_type="jsonl", display_name="hindsight-batch-input"),
        )

        batch = await self._client.aio.batches.create(
            model=self.model,
            src=uploaded.name,
            config=genai_types.CreateBatchJobConfig(display_name="hindsight-batch"),
        )

        logger.info(f"Gemini batch submitted: {batch.name}, state={self._state_name(batch.state)}")

        return {
            "batch_id": batch.name,
            "status": self._normalize_state(batch.state),
            "input_file_id": uploaded.name,
            "request_count": len(requests),
        }

    async def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        """Get the status of a Gemini batch job, in the shared status shape."""
        if not await self.supports_batch_api():
            raise NotImplementedError(f"Batch API not supported for provider: {self.provider}")

        batch = await self._client.aio.batches.get(name=batch_id)

        stats = batch.completion_stats
        successful = _to_int(getattr(stats, "successful_count", None)) if stats else 0
        failed = _to_int(getattr(stats, "failed_count", None)) if stats else 0
        incomplete = _to_int(getattr(stats, "incomplete_count", None)) if stats else 0

        result: dict[str, Any] = {
            "batch_id": batch.name,
            "status": self._normalize_state(batch.state),
            "request_counts": {
                "total": successful + failed + incomplete,
                "completed": successful,
                "failed": failed,
            },
        }

        if batch.dest and getattr(batch.dest, "file_name", None):
            result["output_file_id"] = batch.dest.file_name
        if batch.error:
            result["errors"] = self._error_to_dict(batch.error)

        return result

    async def retrieve_batch_results(self, batch_id: str) -> list[dict[str, Any]]:
        """Download and normalize completed Gemini batch results."""
        if not await self.supports_batch_api():
            raise NotImplementedError(f"Batch API not supported for provider: {self.provider}")

        batch = await self._client.aio.batches.get(name=batch_id)

        status = self._normalize_state(batch.state)
        if status != "completed":
            raise ValueError(f"Gemini batch {batch_id} is not completed yet (state: {self._state_name(batch.state)})")

        dest = batch.dest
        if not dest or not getattr(dest, "file_name", None):
            raise ValueError(
                f"Gemini batch {batch_id} completed but reported no output file "
                f"(submit_batch always uses file mode, so this is unexpected)"
            )

        content = await self._client.aio.files.download(file=dest.file_name)
        text = content.decode("utf-8") if isinstance(content, (bytes, bytearray)) else str(content)

        # The output is a JSONL error file plus results merged into one stream;
        # error lines carry an `error` so partial failures surface per key
        # instead of vanishing (JOB_STATE_PARTIALLY_SUCCEEDED maps to completed).
        results: list[dict[str, Any]] = []
        for line in text.strip().split("\n"):
            if line.strip():
                results.append(self._normalize_output_line(json.loads(line)))

        logger.info(f"Retrieved {len(results)} results for Gemini batch {batch_id}")
        return results

    # ----- pure translation/normalization helpers (unit-tested) ----------

    @staticmethod
    def _translate_requests(requests: list[dict[str, Any]]) -> str:
        """OpenAI batch requests -> Gemini batch input JSONL.

        Each output line is ``{"key": <custom_id>, "request": <GenerateContentRequest>}``;
        the model is supplied to ``batches.create`` so it is omitted per-line.
        """
        lines = []
        for req in requests:
            gemini_request = GeminiLLM._openai_body_to_gemini_request(req.get("body") or {})
            lines.append(json.dumps({"key": req.get("custom_id"), "request": gemini_request}, ensure_ascii=False))
        return "\n".join(lines)

    @staticmethod
    def _openai_body_to_gemini_request(body: dict[str, Any]) -> dict[str, Any]:
        """OpenAI chat-completions body -> Gemini ``GenerateContentRequest`` JSON.

        Mirrors the synchronous ``call`` path: system messages become
        ``systemInstruction``; a ``response_format`` json_schema forces JSON
        output (``responseMimeType``), appends the schema as a textual hint, and
        grammar-enforces via ``responseJsonSchema`` when ``strict`` is set.
        """
        system_texts: list[str] = []
        contents: list[dict[str, Any]] = []
        for msg in body.get("messages") or []:
            role = msg.get("role", "user")
            text = msg.get("content", "") or ""
            if role == "system":
                system_texts.append(text)
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": text}]})
            else:
                contents.append({"role": "user", "parts": [{"text": text}]})

        generation_config: dict[str, Any] = {}
        if body.get("temperature") is not None:
            generation_config["temperature"] = body["temperature"]
        if body.get("max_completion_tokens") is not None:
            generation_config["maxOutputTokens"] = body["max_completion_tokens"]

        response_format = body.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            json_schema = response_format.get("json_schema") or {}
            schema = json_schema.get("schema")
            generation_config["responseMimeType"] = "application/json"
            if schema:
                system_texts.append(
                    "You must respond with valid JSON matching this schema:\n" + json.dumps(schema, ensure_ascii=False)
                )
                if json_schema.get("strict"):
                    generation_config["responseJsonSchema"] = schema

        request: dict[str, Any] = {"contents": contents}
        if system_texts:
            request["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_texts)}]}
        if generation_config:
            request["generationConfig"] = generation_config
        return request

    @staticmethod
    def _normalize_output_line(line: dict[str, Any]) -> dict[str, Any]:
        """Gemini batch output line -> OpenAI-batch-output shape.

        Target: ``{"custom_id", "response": {"body": {"choices": [...], "usage": {...}}}, "error"}``
        so the consumer's ``result["response"]["body"]["choices"][0]...`` works and
        it can read ``body["usage"]`` for token accounting (the consumer reports
        zero usage otherwise).
        """
        custom_id = line.get("key") if line.get("key") is not None else line.get("custom_id")
        error = line.get("error")
        if error:
            return {"custom_id": custom_id, "response": None, "error": error}

        response = line.get("response") or {}
        body: dict[str, Any] = {"choices": [{"message": {"content": GeminiLLM._extract_text_from_response(response)}}]}
        usage = GeminiLLM._usage_from_response(response)
        if usage is not None:
            body["usage"] = usage
        return {"custom_id": custom_id, "response": {"body": body}, "error": None}

    @staticmethod
    def _extract_text_from_response(response: dict[str, Any]) -> str:
        """Concatenate the text parts of a (JSON) GenerateContentResponse."""
        candidates = response.get("candidates") or []
        if not candidates:
            return ""
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text"))

    @staticmethod
    def _usage_from_response(response: dict[str, Any]) -> dict[str, Any] | None:
        """Gemini ``usageMetadata`` -> OpenAI-shaped ``usage`` block, or None.

        The batch consumer accumulates token usage from ``body["usage"]`` using
        OpenAI key names, so translate here to keep the output contract uniform
        across providers. Handles both the REST camelCase (downloaded JSONL) and
        snake_case spellings defensively.
        """
        meta = response.get("usageMetadata") or response.get("usage_metadata")
        if not isinstance(meta, dict):
            return None
        prompt = meta.get("promptTokenCount") or meta.get("prompt_token_count") or 0
        completion = meta.get("candidatesTokenCount") or meta.get("candidates_token_count") or 0
        total = meta.get("totalTokenCount") or meta.get("total_token_count") or 0
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}

    @staticmethod
    def _normalize_state(state: Any) -> str:
        """Gemini ``JobState`` -> the retain driver's status strings.

        Unknown / in-flight states map to ``in_progress`` so the driver keeps
        polling; ``PARTIALLY_SUCCEEDED`` maps to ``completed`` (per-line errors
        surface the partial failures during retrieval).
        """
        name = GeminiLLM._state_name(state).upper()
        if name in ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"):
            return "completed"
        if name == "JOB_STATE_FAILED":
            return "failed"
        if name in ("JOB_STATE_CANCELLED", "JOB_STATE_CANCELLING"):
            return "cancelled"
        if name == "JOB_STATE_EXPIRED":
            return "expired"
        return "in_progress"

    @staticmethod
    def _state_name(state: Any) -> str:
        """Extract the bare ``JOB_STATE_*`` name from a JobState enum or string."""
        if state is None:
            return ""
        name = getattr(state, "name", None)
        if name:
            return str(name)
        text = str(state)
        if "." in text:
            text = text.rsplit(".", 1)[-1]
        return text

    @staticmethod
    def _error_to_dict(error: Any) -> dict[str, Any]:
        """Coerce a Gemini JobError into a JSON-serializable dict for logging."""
        if hasattr(error, "model_dump"):
            try:
                return error.model_dump(exclude_none=True)
            except Exception:
                pass
        return {"message": str(error)}

    async def cleanup(self) -> None:
        """Clean up resources (close connections, etc.)."""
        # Gemini client doesn't require explicit cleanup
        pass
