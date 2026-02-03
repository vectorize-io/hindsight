"""
OpenAI-compatible LLM provider supporting OpenAI, Groq, Ollama, and LMStudio.

This provider handles all OpenAI API-compatible models including:
- OpenAI: GPT-4, GPT-4o, GPT-5, o1, o3 (reasoning models)
- Groq: Fast inference with seed control and service tiers
- Ollama: Local models with native streaming API support
- LMStudio: Local models with OpenAI-compatible API

Features:
- Reasoning models with extended thinking (o1, o3, GPT-5 families)
- Strict JSON schema enforcement (OpenAI)
- Provider-specific parameters (Groq seed, service tier)
- Native Ollama streaming for better structured output
- Automatic token limit handling per model family
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, LengthFinishReasonError

from hindsight_api.config import DEFAULT_LLM_TIMEOUT, ENV_LLM_TIMEOUT
from hindsight_api.engine.llm_interface import LLMInterface, OutputTooLongError
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, TokenUsage
from hindsight_api.metrics import get_metrics_collector

logger = logging.getLogger(__name__)

# Seed applied to every Groq request for deterministic behavior
DEFAULT_LLM_SEED = 4242


class OpenAICompatibleLLM(LLMInterface):
    """
    LLM provider for OpenAI-compatible APIs.

    Supports:
    - OpenAI: Standard models (GPT-4, GPT-4o) and reasoning models (o1, o3, GPT-5)
    - Groq: Fast inference with seed control and service tiers
    - Ollama: Local models with native streaming API for better structured output
    - LMStudio: Local models with OpenAI-compatible API
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        reasoning_effort: str = "low",
        timeout: float | None = None,
        groq_service_tier: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize OpenAI-compatible LLM provider.

        Args:
            provider: Provider name ("openai", "groq", "ollama", "lmstudio").
            api_key: API key (optional for ollama/lmstudio).
            base_url: Base URL for the API (uses defaults for groq/ollama/lmstudio if empty).
            model: Model name.
            reasoning_effort: Reasoning effort level for supported models ("low", "medium", "high").
            timeout: Request timeout in seconds (uses env var or 300s default).
            groq_service_tier: Groq service tier ("on_demand", "flex", "auto").
            **kwargs: Additional provider-specific parameters.
        """
        super().__init__(provider, api_key, base_url, model, reasoning_effort, **kwargs)

        # Validate provider
        valid_providers = ["openai", "groq", "ollama", "lmstudio"]
        if self.provider not in valid_providers:
            raise ValueError(f"OpenAICompatibleLLM only supports: {', '.join(valid_providers)}. Got: {self.provider}")

        # Set default base URLs
        if not self.base_url:
            if self.provider == "groq":
                self.base_url = "https://api.groq.com/openai/v1"
            elif self.provider == "ollama":
                self.base_url = "http://localhost:11434/v1"
            elif self.provider == "lmstudio":
                self.base_url = "http://localhost:1234/v1"

        # For ollama/lmstudio, use dummy key if not provided
        if self.provider in ("ollama", "lmstudio") and not self.api_key:
            self.api_key = "local"

        # Validate API key for cloud providers
        if self.provider in ("openai", "groq") and not self.api_key:
            raise ValueError(f"API key is required for {self.provider}")

        # Groq service tier configuration
        self.groq_service_tier = groq_service_tier or os.getenv("HINDSIGHT_API_LLM_GROQ_SERVICE_TIER", "auto")

        # Get timeout config
        self.timeout = timeout or float(os.getenv(ENV_LLM_TIMEOUT, str(DEFAULT_LLM_TIMEOUT)))

        # Create OpenAI client
        client_kwargs: dict[str, Any] = {"api_key": self.api_key, "max_retries": 0}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if self.timeout:
            client_kwargs["timeout"] = self.timeout

        self._client = AsyncOpenAI(**client_kwargs)
        logger.info(
            f"OpenAI-compatible client initialized: provider={self.provider}, model={self.model}, "
            f"base_url={self.base_url or 'default'}"
        )

    async def verify_connection(self) -> None:
        """
        Verify that the provider is configured correctly by making a simple test call.

        Raises:
            RuntimeError: If the connection test fails.
        """
        try:
            logger.info(f"Verifying connection: {self.provider}/{self.model}")
            await self.call(
                messages=[{"role": "user", "content": "Say 'ok'"}],
                max_completion_tokens=100,
                max_retries=2,
                initial_backoff=0.5,
                max_backoff=2.0,
            )
            logger.info(f"Connection verified: {self.provider}/{self.model}")
        except Exception as e:
            raise RuntimeError(f"Connection verification failed for {self.provider}/{self.model}: {e}") from e

    def _supports_reasoning_model(self) -> bool:
        """Check if the current model is a reasoning model (o1, o3, GPT-5, DeepSeek)."""
        model_lower = self.model.lower()
        return any(x in model_lower for x in ["gpt-5", "o1", "o3", "deepseek"])

    def _get_max_reasoning_tokens(self) -> int | None:
        """Get max reasoning tokens for reasoning models."""
        model_lower = self.model.lower()

        # GPT-4 and GPT-4.1 models have different caps
        if any(x in model_lower for x in ["gpt-4.1", "gpt-4-"]):
            return 32000
        elif "gpt-4o" in model_lower:
            return 16384

        return None

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
            strict_schema: Use strict JSON schema enforcement (OpenAI only).
            return_usage: If True, return tuple (result, TokenUsage) instead of just result.

        Returns:
            If return_usage=False: Parsed response if response_format is provided, otherwise text content.
            If return_usage=True: Tuple of (result, TokenUsage) with token counts.

        Raises:
            OutputTooLongError: If output exceeds token limits.
            Exception: Re-raises API errors after retries exhausted.
        """
        # Handle Ollama with native API for structured output (better schema enforcement)
        if self.provider == "ollama" and response_format is not None:
            return await self._call_ollama_native(
                messages=messages,
                response_format=response_format,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                skip_validation=skip_validation,
                scope=scope,
                return_usage=return_usage,
            )

        start_time = time.time()

        # Build call parameters
        call_params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        # Check if model supports reasoning parameter
        is_reasoning_model = self._supports_reasoning_model()

        # Apply model-specific token limits
        if max_completion_tokens is not None:
            max_tokens_cap = self._get_max_reasoning_tokens()
            if max_tokens_cap and max_completion_tokens > max_tokens_cap:
                max_completion_tokens = max_tokens_cap
            # For reasoning models, enforce minimum to ensure space for reasoning + output
            if is_reasoning_model and max_completion_tokens < 16000:
                max_completion_tokens = 16000
            call_params["max_completion_tokens"] = max_completion_tokens

        # Temperature - reasoning models don't support custom temperature
        if temperature is not None and not is_reasoning_model:
            call_params["temperature"] = temperature

        # Set reasoning_effort for reasoning models
        if is_reasoning_model:
            call_params["reasoning_effort"] = self.reasoning_effort

        # Provider-specific parameters
        if self.provider == "groq":
            call_params["seed"] = DEFAULT_LLM_SEED
            extra_body: dict[str, Any] = {}
            # Add service_tier if configured
            if self.groq_service_tier:
                extra_body["service_tier"] = self.groq_service_tier
            # Add reasoning parameters for reasoning models
            if is_reasoning_model:
                extra_body["include_reasoning"] = False
            if extra_body:
                call_params["extra_body"] = extra_body

        # Prepare response format ONCE before retry loop
        if response_format is not None:
            schema = None
            if hasattr(response_format, "model_json_schema"):
                schema = response_format.model_json_schema()

            if strict_schema and schema is not None:
                # Use OpenAI's strict JSON schema enforcement
                call_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "strict": True,
                        "schema": schema,
                    },
                }
            else:
                # Soft enforcement: add schema to prompt and use json_object mode
                if schema is not None:
                    schema_msg = (
                        f"\n\nYou must respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
                    )

                    if call_params["messages"] and call_params["messages"][0].get("role") == "system":
                        first_msg = call_params["messages"][0]
                        if isinstance(first_msg, dict) and isinstance(first_msg.get("content"), str):
                            first_msg["content"] += schema_msg
                    elif call_params["messages"]:
                        first_msg = call_params["messages"][0]
                        if isinstance(first_msg, dict) and isinstance(first_msg.get("content"), str):
                            first_msg["content"] = schema_msg + "\n\n" + first_msg["content"]
                if self.provider not in ("lmstudio", "ollama"):
                    # LM Studio and Ollama don't support json_object response format reliably
                    call_params["response_format"] = {"type": "json_object"}

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                if response_format is not None:
                    response = await self._client.chat.completions.create(**call_params)

                    content = response.choices[0].message.content

                    # Strip reasoning model thinking tags
                    # Supports: <think>, <thinking>, <reasoning>, |startthink|/|endthink|
                    if content:
                        original_len = len(content)
                        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                        content = re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL)
                        content = re.sub(r"<reasoning>.*?</reasoning>", "", content, flags=re.DOTALL)
                        content = re.sub(r"\|startthink\|.*?\|endthink\|", "", content, flags=re.DOTALL)
                        content = content.strip()
                        if len(content) < original_len:
                            logger.debug(f"Stripped {original_len - len(content)} chars of reasoning tokens")

                    # For local models, they may wrap JSON in markdown code blocks
                    if self.provider in ("lmstudio", "ollama"):
                        clean_content = content
                        if "```json" in content:
                            clean_content = content.split("```json")[1].split("```")[0].strip()
                        elif "```" in content:
                            clean_content = content.split("```")[1].split("```")[0].strip()
                        try:
                            json_data = json.loads(clean_content)
                        except json.JSONDecodeError:
                            # Fallback to parsing raw content
                            json_data = json.loads(content)
                    else:
                        # Log raw LLM response for debugging JSON parse issues
                        try:
                            json_data = json.loads(content)
                        except json.JSONDecodeError as json_err:
                            # Truncate content for logging
                            content_preview = content[:500] if content else "<empty>"
                            if content and len(content) > 700:
                                content_preview = f"{content[:500]}...TRUNCATED...{content[-200:]}"
                            logger.warning(
                                f"JSON parse error from LLM response (attempt {attempt + 1}/{max_retries + 1}): {json_err}\n"
                                f"  Model: {self.provider}/{self.model}\n"
                                f"  Content length: {len(content) if content else 0} chars\n"
                                f"  Content preview: {content_preview!r}\n"
                                f"  Finish reason: {response.choices[0].finish_reason if response.choices else 'unknown'}"
                            )
                            # Retry on JSON parse errors
                            if attempt < max_retries:
                                backoff = min(initial_backoff * (2**attempt), max_backoff)
                                await asyncio.sleep(backoff)
                                last_exception = json_err
                                continue
                            else:
                                logger.error(f"JSON parse error after {max_retries + 1} attempts, giving up")
                                raise

                    if skip_validation:
                        result = json_data
                    else:
                        result = response_format.model_validate(json_data)
                else:
                    response = await self._client.chat.completions.create(**call_params)
                    result = response.choices[0].message.content

                # Record token usage metrics
                duration = time.time() - start_time
                usage = response.usage
                input_tokens = usage.prompt_tokens or 0 if usage else 0
                output_tokens = usage.completion_tokens or 0 if usage else 0
                total_tokens = usage.total_tokens or 0 if usage else 0

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
                if duration > 10.0 and usage:
                    ratio = max(1, output_tokens) / max(1, input_tokens)
                    cached_tokens = 0
                    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
                        cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
                    cache_info = f", cached_tokens={cached_tokens}" if cached_tokens > 0 else ""
                    logger.info(
                        f"slow llm call: scope={scope}, model={self.provider}/{self.model}, "
                        f"input_tokens={input_tokens}, output_tokens={output_tokens}, "
                        f"total_tokens={total_tokens}{cache_info}, time={duration:.3f}s, ratio out/in={ratio:.2f}"
                    )

                if return_usage:
                    token_usage = TokenUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                    )
                    return result, token_usage
                return result

            except LengthFinishReasonError as e:
                logger.warning(f"LLM output exceeded token limits: {str(e)}")
                raise OutputTooLongError(
                    "LLM output exceeded token limits. Input may need to be split into smaller chunks."
                ) from e

            except APIConnectionError as e:
                last_exception = e
                status_code = getattr(e, "status_code", None) or getattr(
                    getattr(e, "response", None), "status_code", None
                )
                logger.warning(f"APIConnectionError (HTTP {status_code}), attempt {attempt + 1}: {str(e)[:200]}")
                if attempt < max_retries:
                    backoff = min(initial_backoff * (2**attempt), max_backoff)
                    await asyncio.sleep(backoff)
                    continue
                else:
                    logger.error(f"Connection error after {max_retries + 1} attempts: {str(e)}")
                    raise

            except APIStatusError as e:
                # Fast fail only on 401 (unauthorized) and 403 (forbidden)
                if e.status_code in (401, 403):
                    logger.error(f"Auth error (HTTP {e.status_code}), not retrying: {str(e)}")
                    raise

                # Handle tool_use_failed error - model outputted in tool call format
                if e.status_code == 400 and response_format is not None:
                    try:
                        error_body = e.body if hasattr(e, "body") else {}
                        if isinstance(error_body, dict):
                            error_info: dict[str, Any] = error_body.get("error") or {}
                            if error_info.get("code") == "tool_use_failed":
                                failed_gen = error_info.get("failed_generation", "")
                                if failed_gen:
                                    # Parse tool call format and convert to expected format
                                    tool_call = json.loads(failed_gen)
                                    tool_name = tool_call.get("name", "")
                                    tool_args = tool_call.get("arguments", {})
                                    converted = {"actions": [{"tool": tool_name, **tool_args}]}
                                    if skip_validation:
                                        result = converted
                                    else:
                                        result = response_format.model_validate(converted)

                                    # Record metrics
                                    duration = time.time() - start_time
                                    metrics = get_metrics_collector()
                                    metrics.record_llm_call(
                                        provider=self.provider,
                                        model=self.model,
                                        scope=scope,
                                        duration=duration,
                                        input_tokens=0,
                                        output_tokens=0,
                                        success=True,
                                    )
                                    if return_usage:
                                        return result, TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0)
                                    return result
                    except (json.JSONDecodeError, KeyError, TypeError):
                        pass  # Failed to parse tool_use_failed, continue with normal retry

                last_exception = e
                if attempt < max_retries:
                    backoff = min(initial_backoff * (2**attempt), max_backoff)
                    jitter = backoff * 0.2 * (2 * (time.time() % 1) - 1)
                    sleep_time = backoff + jitter
                    await asyncio.sleep(sleep_time)
                else:
                    logger.error(f"API error after {max_retries + 1} attempts: {str(e)}")
                    raise

            except Exception:
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("LLM call failed after all retries with no exception captured")

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
        start_time = time.time()

        # Build call parameters
        call_params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
        }

        if max_completion_tokens is not None:
            call_params["max_completion_tokens"] = max_completion_tokens
        if temperature is not None:
            call_params["temperature"] = temperature

        # Provider-specific parameters
        if self.provider == "groq":
            call_params["seed"] = DEFAULT_LLM_SEED

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                response = await self._client.chat.completions.create(**call_params)

                message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason

                # Extract tool calls if present
                tool_calls: list[LLMToolCall] = []
                if message.tool_calls:
                    for tc in message.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                        except json.JSONDecodeError:
                            args = {"_raw": tc.function.arguments}
                        tool_calls.append(LLMToolCall(id=tc.id, name=tc.function.name, arguments=args))

                content = message.content

                # Record metrics
                duration = time.time() - start_time
                usage = response.usage
                input_tokens = usage.prompt_tokens or 0 if usage else 0
                output_tokens = usage.completion_tokens or 0 if usage else 0

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

                return LLMToolCallResult(
                    content=content,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            except APIConnectionError as e:
                last_exception = e
                if attempt < max_retries:
                    await asyncio.sleep(min(initial_backoff * (2**attempt), max_backoff))
                    continue
                raise

            except APIStatusError as e:
                if e.status_code in (401, 403):
                    raise
                last_exception = e
                if attempt < max_retries:
                    await asyncio.sleep(min(initial_backoff * (2**attempt), max_backoff))
                    continue
                raise

            except Exception:
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Tool call failed after all retries")

    async def _call_ollama_native(
        self,
        messages: list[dict[str, str]],
        response_format: Any,
        max_completion_tokens: int | None,
        temperature: float | None,
        max_retries: int,
        initial_backoff: float,
        max_backoff: float,
        skip_validation: bool,
        scope: str = "memory",
        return_usage: bool = False,
    ) -> Any:
        """
        Call Ollama using native API with JSON schema enforcement.

        Ollama's native API supports passing a full JSON schema in the 'format' parameter,
        which provides better structured output control than the OpenAI-compatible API.
        """
        start_time = time.time()

        # Get the JSON schema from the Pydantic model
        schema = response_format.model_json_schema() if hasattr(response_format, "model_json_schema") else None

        # Build the base URL for Ollama's native API
        # Default OpenAI-compatible URL is http://localhost:11434/v1
        # Native API is at http://localhost:11434/api/chat
        base_url = self.base_url or "http://localhost:11434/v1"
        if base_url.endswith("/v1"):
            native_url = base_url[:-3] + "/api/chat"
        else:
            native_url = base_url.rstrip("/") + "/api/chat"

        # Build request payload
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        # Add schema as format parameter for structured output
        if schema:
            payload["format"] = schema

        # Add optional parameters with optimized defaults for Ollama
        options: dict[str, Any] = {
            "num_ctx": 16384,  # 16k context window for larger prompts
            "num_batch": 512,  # Optimal batch size for prompt processing
        }
        if max_completion_tokens:
            options["num_predict"] = max_completion_tokens
        if temperature is not None:
            options["temperature"] = temperature
        payload["options"] = options

        last_exception = None

        async with httpx.AsyncClient(timeout=300.0) as client:
            for attempt in range(max_retries + 1):
                try:
                    response = await client.post(native_url, json=payload)
                    response.raise_for_status()

                    result = response.json()
                    content = result.get("message", {}).get("content", "")

                    # Parse JSON response
                    try:
                        json_data = json.loads(content)
                    except json.JSONDecodeError as json_err:
                        content_preview = content[:500] if content else "<empty>"
                        if content and len(content) > 700:
                            content_preview = f"{content[:500]}...TRUNCATED...{content[-200:]}"
                        logger.warning(
                            f"Ollama JSON parse error (attempt {attempt + 1}/{max_retries + 1}): {json_err}\n"
                            f"  Model: ollama/{self.model}\n"
                            f"  Content length: {len(content) if content else 0} chars\n"
                            f"  Content preview: {content_preview!r}"
                        )
                        if attempt < max_retries:
                            backoff = min(initial_backoff * (2**attempt), max_backoff)
                            await asyncio.sleep(backoff)
                            last_exception = json_err
                            continue
                        else:
                            raise

                    # Extract token usage from Ollama response
                    duration = time.time() - start_time
                    input_tokens = result.get("prompt_eval_count", 0) or 0
                    output_tokens = result.get("eval_count", 0) or 0
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

                    # Validate against Pydantic model or return raw JSON
                    if skip_validation:
                        validated_result = json_data
                    else:
                        validated_result = response_format.model_validate(json_data)

                    if return_usage:
                        token_usage = TokenUsage(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            total_tokens=total_tokens,
                        )
                        return validated_result, token_usage
                    return validated_result

                except httpx.HTTPStatusError as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Ollama HTTP error (attempt {attempt + 1}/{max_retries + 1}): {e.response.status_code}"
                        )
                        backoff = min(initial_backoff * (2**attempt), max_backoff)
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        logger.error(f"Ollama HTTP error after {max_retries + 1} attempts: {e}")
                        raise

                except httpx.RequestError as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Ollama connection error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        backoff = min(initial_backoff * (2**attempt), max_backoff)
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        logger.error(f"Ollama connection error after {max_retries + 1} attempts: {e}")
                        raise

                except Exception as e:
                    logger.error(f"Unexpected error during Ollama call: {type(e).__name__}: {e}")
                    raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Ollama call failed after all retries")

    async def cleanup(self) -> None:
        """Clean up resources (close OpenAI client connections)."""
        if hasattr(self, "_client") and self._client:
            await self._client.close()
