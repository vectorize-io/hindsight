"""
LLM wrapper for unified configuration across providers.
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, LengthFinishReasonError

# Vertex AI imports (conditional)
try:
    import google.auth
    from google.oauth2 import service_account

    VERTEXAI_AVAILABLE = True
except ImportError:
    VERTEXAI_AVAILABLE = False

from ..config import (
    DEFAULT_LLM_MAX_CONCURRENT,
    DEFAULT_LLM_TIMEOUT,
    ENV_LLM_GROQ_SERVICE_TIER,
    ENV_LLM_MAX_CONCURRENT,
    ENV_LLM_TIMEOUT,
)
from ..metrics import get_metrics_collector
from .response_models import TokenUsage

# Seed applied to every Groq request for deterministic behavior.
DEFAULT_LLM_SEED = 4242

logger = logging.getLogger(__name__)

# Disable httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Global semaphore to limit concurrent LLM requests across all instances
# Set HINDSIGHT_API_LLM_MAX_CONCURRENT=1 for local LLMs (LM Studio, Ollama)
_llm_max_concurrent = int(os.getenv(ENV_LLM_MAX_CONCURRENT, str(DEFAULT_LLM_MAX_CONCURRENT)))
_global_llm_semaphore = asyncio.Semaphore(_llm_max_concurrent)


class OutputTooLongError(Exception):
    """
    Bridge exception raised when LLM output exceeds token limits.

    This wraps provider-specific errors (e.g., OpenAI's LengthFinishReasonError)
    to allow callers to handle output length issues without depending on
    provider-specific implementations.
    """

    pass


class LLMProvider:
    """
    Unified LLM provider.

    Supports OpenAI, Groq, Ollama (OpenAI-compatible), and Gemini.
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        reasoning_effort: str = "low",
        groq_service_tier: str | None = None,
    ):
        """
        Initialize LLM provider.

        Args:
            provider: Provider name ("openai", "groq", "ollama", "gemini", "anthropic", "lmstudio").
            api_key: API key.
            base_url: Base URL for the API.
            model: Model name.
            reasoning_effort: Reasoning effort level for supported providers.
            groq_service_tier: Groq service tier ("on_demand", "flex", "auto"). Default: None (uses Groq's default).
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.reasoning_effort = reasoning_effort
        # Default to 'auto' for best performance, users can override to 'on_demand' for free tier
        self.groq_service_tier = groq_service_tier or os.getenv(ENV_LLM_GROQ_SERVICE_TIER, "auto")

        # Validate provider
        valid_providers = ["openai", "groq", "ollama", "gemini", "anthropic", "lmstudio", "vertexai", "mock"]
        if self.provider not in valid_providers:
            raise ValueError(f"Invalid LLM provider: {self.provider}. Must be one of: {', '.join(valid_providers)}")

        # Mock provider tracking (for testing)
        self._mock_calls: list[dict] = []
        self._mock_response: Any = None

        # Set default base URLs
        if not self.base_url:
            if self.provider == "groq":
                self.base_url = "https://api.groq.com/openai/v1"
            elif self.provider == "ollama":
                self.base_url = "http://localhost:11434/v1"
            elif self.provider == "lmstudio":
                self.base_url = "http://localhost:1234/v1"

        # Vertex AI config — stored for client creation below
        self._vertexai_project_id: str | None = None
        self._vertexai_region: str | None = None
        self._vertexai_credentials: Any = None

        if self.provider == "vertexai":
            from ..config import get_config

            config = get_config()

            self._vertexai_project_id = config.llm_vertexai_project_id
            if not self._vertexai_project_id:
                raise ValueError(
                    "HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID is required for Vertex AI provider. "
                    "Set it to your GCP project ID."
                )

            self._vertexai_region = config.llm_vertexai_region or "us-central1"
            service_account_key = config.llm_vertexai_service_account_key

            # Load explicit service account credentials if provided
            if service_account_key:
                if not VERTEXAI_AVAILABLE:
                    raise ValueError(
                        "Vertex AI service account auth requires 'google-auth' package. "
                        "Install with: pip install google-auth"
                    )
                self._vertexai_credentials = service_account.Credentials.from_service_account_file(
                    service_account_key,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                logger.info(f"Vertex AI: Using service account key: {service_account_key}")

            # Strip google/ prefix from model name — native SDK uses bare names
            # e.g. "google/gemini-2.0-flash-lite-001" -> "gemini-2.0-flash-lite-001"
            if self.model.startswith("google/"):
                self.model = self.model[len("google/") :]

            logger.info(
                f"Vertex AI: project={self._vertexai_project_id}, region={self._vertexai_region}, "
                f"model={self.model}, auth={'service_account' if service_account_key else 'ADC'}"
            )

        # Validate API key (not needed for ollama, lmstudio, vertexai, or mock)
        if self.provider not in ("ollama", "lmstudio", "vertexai", "mock") and not self.api_key:
            raise ValueError(f"API key not found for {self.provider}")

        # Get timeout config (set HINDSIGHT_API_LLM_TIMEOUT for local LLMs that need longer timeouts)
        self.timeout = float(os.getenv(ENV_LLM_TIMEOUT, str(DEFAULT_LLM_TIMEOUT)))

        # Create client based on provider
        self._client = None
        self._gemini_client = None
        self._anthropic_client = None

        if self.provider == "mock":
            # Mock provider - no client needed
            pass
        elif self.provider == "gemini":
            self._gemini_client = genai.Client(api_key=self.api_key)
        elif self.provider == "anthropic":
            from anthropic import AsyncAnthropic

            # Only pass base_url if it's set (Anthropic uses default URL otherwise)
            anthropic_kwargs = {"api_key": self.api_key}
            if self.base_url:
                anthropic_kwargs["base_url"] = self.base_url
            if self.timeout:
                anthropic_kwargs["timeout"] = self.timeout
            self._anthropic_client = AsyncAnthropic(**anthropic_kwargs)
        elif self.provider == "vertexai":
            # Native genai SDK with Vertex AI — handles ADC automatically,
            # or uses explicit service account credentials if provided
            client_kwargs = {
                "vertexai": True,
                "project": self._vertexai_project_id,
                "location": self._vertexai_region,
            }
            if self._vertexai_credentials is not None:
                client_kwargs["credentials"] = self._vertexai_credentials
            self._gemini_client = genai.Client(**client_kwargs)
        elif self.provider in ("ollama", "lmstudio"):
            # Use dummy key if not provided for local
            api_key = self.api_key or "local"
            client_kwargs = {"api_key": api_key, "base_url": self.base_url, "max_retries": 0}
            if self.timeout:
                client_kwargs["timeout"] = self.timeout
            self._client = AsyncOpenAI(**client_kwargs)
        else:
            # Only pass base_url if it's set (OpenAI uses default URL otherwise)
            client_kwargs = {"api_key": self.api_key, "max_retries": 0}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            if self.timeout:
                client_kwargs["timeout"] = self.timeout
            self._client = AsyncOpenAI(**client_kwargs)

    async def verify_connection(self) -> None:
        """
        Verify that the LLM provider is configured correctly by making a simple test call.

        Raises:
            RuntimeError: If the connection test fails.
        """
        try:
            logger.info(
                f"Verifying LLM: provider={self.provider}, model={self.model}, base_url={self.base_url or 'default'}..."
            )
            await self.call(
                messages=[{"role": "user", "content": "Say 'ok'"}],
                max_completion_tokens=100,
                max_retries=2,
                initial_backoff=0.5,
                max_backoff=2.0,
            )
            # If we get here without exception, the connection is working
            logger.info(f"LLM verified: {self.provider}/{self.model}")
        except Exception as e:
            raise RuntimeError(f"LLM connection verification failed for {self.provider}/{self.model}: {e}") from e

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
            strict_schema: Use strict JSON schema enforcement (OpenAI only). Guarantees all required fields.
            return_usage: If True, return tuple (result, TokenUsage) instead of just result.

        Returns:
            If return_usage=False: Parsed response if response_format is provided, otherwise text content.
            If return_usage=True: Tuple of (result, TokenUsage) with token counts from the LLM call.

        Raises:
            OutputTooLongError: If output exceeds token limits.
            Exception: Re-raises API errors after retries exhausted.
        """
        semaphore_start = time.time()
        async with _global_llm_semaphore:
            semaphore_wait_time = time.time() - semaphore_start
            start_time = time.time()

            # Handle Mock provider (for testing)
            if self.provider == "mock":
                return await self._call_mock(
                    messages,
                    response_format,
                    scope,
                    return_usage,
                )

            # Handle Gemini and Vertex AI providers (both use native genai SDK)
            if self.provider in ("gemini", "vertexai"):
                return await self._call_gemini(
                    messages,
                    response_format,
                    max_retries,
                    initial_backoff,
                    max_backoff,
                    skip_validation,
                    start_time,
                    scope,
                    return_usage,
                    semaphore_wait_time,
                )

            # Handle Anthropic provider separately
            if self.provider == "anthropic":
                return await self._call_anthropic(
                    messages,
                    response_format,
                    max_completion_tokens,
                    max_retries,
                    initial_backoff,
                    max_backoff,
                    skip_validation,
                    start_time,
                    scope,
                    return_usage,
                    semaphore_wait_time,
                )

            # Handle Ollama with native API for structured output (better schema enforcement)
            if self.provider == "ollama" and response_format is not None:
                return await self._call_ollama_native(
                    messages,
                    response_format,
                    max_completion_tokens,
                    temperature,
                    max_retries,
                    initial_backoff,
                    max_backoff,
                    skip_validation,
                    start_time,
                    scope,
                    return_usage,
                    semaphore_wait_time,
                )

            call_params = {
                "model": self.model,
                "messages": messages,
            }

            # Check if model supports reasoning parameter (o1, o3, gpt-5 families)
            model_lower = self.model.lower()
            is_reasoning_model = any(x in model_lower for x in ["gpt-5", "o1", "o3", "deepseek"])

            # For GPT-4 and GPT-4.1 models, cap max_completion_tokens to 32000
            # For GPT-4o models, cap to 16384
            is_gpt4_model = any(x in model_lower for x in ["gpt-4.1", "gpt-4-"])
            is_gpt4o_model = "gpt-4o" in model_lower
            if max_completion_tokens is not None:
                if is_gpt4o_model and max_completion_tokens > 16384:
                    max_completion_tokens = 16384
                elif is_gpt4_model and max_completion_tokens > 32000:
                    max_completion_tokens = 32000
                # For reasoning models, max_completion_tokens includes reasoning + output tokens
                # Enforce minimum of 16000 to ensure enough space for both
                if is_reasoning_model and max_completion_tokens < 16000:
                    max_completion_tokens = 16000
                call_params["max_completion_tokens"] = max_completion_tokens

            # GPT-5/o1/o3 family doesn't support custom temperature (only default 1)
            if temperature is not None and not is_reasoning_model:
                call_params["temperature"] = temperature

            # Set reasoning_effort for reasoning models (OpenAI gpt-5, o1, o3)
            if is_reasoning_model:
                call_params["reasoning_effort"] = self.reasoning_effort

            # Provider-specific parameters
            if self.provider == "groq":
                call_params["seed"] = DEFAULT_LLM_SEED
                extra_body: dict[str, Any] = {}
                # Add service_tier if configured (requires paid plan for flex/auto)
                if self.groq_service_tier:
                    extra_body["service_tier"] = self.groq_service_tier
                # Add reasoning parameters for reasoning models
                if is_reasoning_model:
                    extra_body["include_reasoning"] = False
                if extra_body:
                    call_params["extra_body"] = extra_body

            last_exception = None

            # Prepare response format ONCE before the retry loop
            # (to avoid appending schema to messages on every retry)
            if response_format is not None:
                schema = None
                if hasattr(response_format, "model_json_schema"):
                    schema = response_format.model_json_schema()

                if strict_schema and schema is not None:
                    # Use OpenAI's strict JSON schema enforcement
                    # This guarantees all required fields are returned
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
                        schema_msg = f"\n\nYou must respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"

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
                        # We rely on the schema in the system message instead
                        call_params["response_format"] = {"type": "json_object"}

            for attempt in range(max_retries + 1):
                try:
                    if response_format is not None:
                        response = await self._client.chat.completions.create(**call_params)

                        content = response.choices[0].message.content

                        # Strip reasoning model thinking tags
                        # Supports: <think>, <thinking>, <reasoning>, |startthink|/|endthink|
                        # for reasoning models that embed thinking in their output (e.g., Qwen3, DeepSeek)
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
                                # Truncate content for logging (first 500 and last 200 chars)
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
                                # Retry on JSON parse errors - LLM may return valid JSON on next attempt
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
                        wait_info = f", wait={semaphore_wait_time:.3f}s" if semaphore_wait_time > 0.1 else ""
                        logger.info(
                            f"slow llm call: scope={scope}, model={self.provider}/{self.model}, "
                            f"input_tokens={input_tokens}, output_tokens={output_tokens}, "
                            f"total_tokens={total_tokens}{cache_info}, time={duration:.3f}s{wait_info}, ratio out/in={ratio:.2f}"
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
                    # Fast fail only on 401 (unauthorized) and 403 (forbidden) - these won't recover with retries
                    if e.status_code in (401, 403):
                        logger.error(f"Auth error (HTTP {e.status_code}), not retrying: {str(e)}")
                        raise

                    # Handle tool_use_failed error - model outputted in tool call format
                    # Convert to expected JSON format and continue
                    if e.status_code == 400 and response_format is not None:
                        try:
                            error_body = e.body if hasattr(e, "body") else {}
                            if isinstance(error_body, dict):
                                error_info: dict[str, Any] = error_body.get("error") or {}
                                if error_info.get("code") == "tool_use_failed":
                                    failed_gen = error_info.get("failed_generation", "")
                                    if failed_gen:
                                        # Parse the tool call format and convert to actions format
                                        tool_call = json.loads(failed_gen)
                                        tool_name = tool_call.get("name", "")
                                        tool_args = tool_call.get("arguments", {})
                                        # Convert to actions format: {"actions": [{"tool": "name", ...args}]}
                                        converted = {"actions": [{"tool": tool_name, **tool_args}]}
                                        if skip_validation:
                                            result = converted
                                        else:
                                            result = response_format.model_validate(converted)

                                        # Record metrics for this successful recovery
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
    ) -> "LLMToolCallResult":
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
            tool_choice: How to choose tools - "auto", "none", "required", or {"type": "function", "function": {"name": "..."}}

        Returns:
            LLMToolCallResult with content and/or tool_calls.
        """
        from .response_models import LLMToolCall, LLMToolCallResult

        async with _global_llm_semaphore:
            start_time = time.time()

            # Handle Mock provider
            if self.provider == "mock":
                return await self._call_with_tools_mock(messages, tools, scope)

            # Handle Anthropic separately (uses different tool format)
            if self.provider == "anthropic":
                return await self._call_with_tools_anthropic(
                    messages, tools, max_completion_tokens, max_retries, initial_backoff, max_backoff, start_time, scope
                )

            # Handle Gemini and Vertex AI (convert to Gemini tool format)
            if self.provider in ("gemini", "vertexai"):
                return await self._call_with_tools_gemini(
                    messages, tools, max_retries, initial_backoff, max_backoff, start_time, scope
                )

            # OpenAI-compatible providers (OpenAI, Groq, Ollama, LMStudio)
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

    async def _call_with_tools_mock(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        scope: str,
    ) -> "LLMToolCallResult":
        """Handle mock tool calls for testing."""
        from .response_models import LLMToolCallResult

        call_record = {
            "provider": self.provider,
            "model": self.model,
            "messages": messages,
            "tools": [t.get("function", {}).get("name") for t in tools],
            "scope": scope,
        }
        self._mock_calls.append(call_record)

        if self._mock_response is not None:
            if isinstance(self._mock_response, LLMToolCallResult):
                return self._mock_response
            # Allow setting just tool calls as a list
            if isinstance(self._mock_response, list):
                from .response_models import LLMToolCall

                return LLMToolCallResult(
                    tool_calls=[
                        LLMToolCall(id=f"mock_{i}", name=tc["name"], arguments=tc.get("arguments", {}))
                        for i, tc in enumerate(self._mock_response)
                    ],
                    finish_reason="tool_calls",
                )

        return LLMToolCallResult(content="mock response", finish_reason="stop")

    async def _call_with_tools_anthropic(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_completion_tokens: int | None,
        max_retries: int,
        initial_backoff: float,
        max_backoff: float,
        start_time: float,
        scope: str,
    ) -> "LLMToolCallResult":
        """Handle Anthropic tool calling."""
        from anthropic import APIConnectionError, APIStatusError

        from .response_models import LLMToolCall, LLMToolCallResult

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

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                response = await self._anthropic_client.messages.create(**call_params)

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

    async def _call_with_tools_gemini(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_retries: int,
        initial_backoff: float,
        max_backoff: float,
        start_time: float,
        scope: str,
    ) -> "LLMToolCallResult":
        """Handle Gemini tool calling."""
        from .response_models import LLMToolCall, LLMToolCallResult

        # Convert tools to Gemini format
        gemini_tools = []
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
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = (system_instruction + "\n\n" + content) if system_instruction else content
            elif role == "tool":
                # Gemini uses function_response
                gemini_contents.append(
                    genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part(
                                function_response=genai_types.FunctionResponse(
                                    name=msg.get("name", ""),
                                    response={"result": content},
                                )
                            )
                        ],
                    )
                )
            elif role == "assistant":
                gemini_contents.append(genai_types.Content(role="model", parts=[genai_types.Part(text=content)]))
            else:
                gemini_contents.append(genai_types.Content(role="user", parts=[genai_types.Part(text=content)]))

        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=gemini_tools,
        )

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                response = await self._gemini_client.aio.models.generate_content(
                    model=self.model,
                    contents=gemini_contents,
                    config=config,
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
                                tool_calls.append(
                                    LLMToolCall(
                                        id=f"gemini_{len(tool_calls)}",
                                        name=fc.name,
                                        arguments=dict(fc.args) if fc.args else {},
                                    )
                                )

                finish_reason = "tool_calls" if tool_calls else "stop"

                # Record metrics
                metrics = get_metrics_collector()
                input_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
                output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
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

            except genai_errors.APIError as e:
                if e.code in (401, 403):
                    raise
                last_exception = e
                if attempt < max_retries:
                    await asyncio.sleep(min(initial_backoff * (2**attempt), max_backoff))
                    continue
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Gemini tool call failed")

    async def _call_anthropic(
        self,
        messages: list[dict[str, str]],
        response_format: Any | None,
        max_completion_tokens: int | None,
        max_retries: int,
        initial_backoff: float,
        max_backoff: float,
        skip_validation: bool,
        start_time: float,
        scope: str = "memory",
        return_usage: bool = False,
        semaphore_wait_time: float = 0.0,
    ) -> Any:
        """Handle Anthropic-specific API calls."""
        from anthropic import APIConnectionError, APIStatusError, RateLimitError

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
        call_params = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_completion_tokens if max_completion_tokens is not None else 4096,
        }

        if system_prompt:
            call_params["system"] = system_prompt

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                response = await self._anthropic_client.messages.create(**call_params)

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
                    wait_info = f", wait={semaphore_wait_time:.3f}s" if semaphore_wait_time > 0.1 else ""
                    logger.info(
                        f"slow llm call: scope={scope}, model={self.provider}/{self.model}, "
                        f"input_tokens={input_tokens}, output_tokens={output_tokens}, "
                        f"time={duration:.3f}s{wait_info}"
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
        start_time: float,
        scope: str = "memory",
        return_usage: bool = False,
        semaphore_wait_time: float = 0.0,
    ) -> Any:
        """
        Call Ollama using native API with JSON schema enforcement.

        Ollama's native API supports passing a full JSON schema in the 'format' parameter,
        which provides better structured output control than the OpenAI-compatible API.
        """
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
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        # Add schema as format parameter for structured output
        if schema:
            payload["format"] = schema

        # Add optional parameters with optimized defaults for Ollama
        # Benchmarking shows num_ctx=16384 + num_batch=512 is optimal
        options = {
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
                    # Ollama returns prompt_eval_count (input) and eval_count (output)
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

    async def _call_gemini(
        self,
        messages: list[dict[str, str]],
        response_format: Any | None,
        max_retries: int,
        initial_backoff: float,
        max_backoff: float,
        skip_validation: bool,
        start_time: float,
        scope: str = "memory",
        return_usage: bool = False,
        semaphore_wait_time: float = 0.0,
    ) -> Any:
        """Handle Gemini-specific API calls."""
        # Convert OpenAI-style messages to Gemini format
        system_instruction = None
        gemini_contents = []

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

        # Add JSON schema instruction if response_format is provided
        if response_format is not None and hasattr(response_format, "model_json_schema"):
            schema = response_format.model_json_schema()
            schema_msg = f"\n\nYou must respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
            if system_instruction:
                system_instruction += schema_msg
            else:
                system_instruction = schema_msg

        # Build generation config
        config_kwargs = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if response_format is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_format

        generation_config = genai_types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                response = await self._gemini_client.aio.models.generate_content(
                    model=self.model,
                    contents=gemini_contents,
                    config=generation_config,
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

                if response_format is not None:
                    json_data = json.loads(content)
                    if skip_validation:
                        result = json_data
                    else:
                        result = response_format.model_validate(json_data)
                else:
                    result = content

                # Record metrics and log slow calls
                duration = time.time() - start_time
                input_tokens = 0
                output_tokens = 0
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage = response.usage_metadata
                    input_tokens = usage.prompt_token_count or 0
                    output_tokens = usage.candidates_token_count or 0

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
                if duration > 10.0 and input_tokens > 0:
                    wait_info = f", wait={semaphore_wait_time:.3f}s" if semaphore_wait_time > 0.1 else ""
                    logger.info(
                        f"slow llm call: scope={scope}, model={self.provider}/{self.model}, "
                        f"input_tokens={input_tokens}, output_tokens={output_tokens}, "
                        f"time={duration:.3f}s{wait_info}"
                    )

                if return_usage:
                    token_usage = TokenUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
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
                # Fast fail only on 401 (unauthorized) and 403 (forbidden) - these won't recover with retries
                if e.code in (401, 403):
                    logger.error(f"Gemini auth error (HTTP {e.code}), not retrying: {str(e)}")
                    raise

                # Retry on retryable errors (rate limits, server errors, and other client errors like 400)
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

    async def _call_mock(
        self,
        messages: list[dict[str, str]],
        response_format: Any | None,
        scope: str,
        return_usage: bool,
    ) -> Any:
        """
        Handle mock provider calls for testing.

        Records the call and returns a configurable mock response.
        """
        # Record the call for test verification
        call_record = {
            "provider": self.provider,
            "model": self.model,
            "messages": messages,
            "response_format": response_format.__name__
            if response_format and hasattr(response_format, "__name__")
            else str(response_format),
            "scope": scope,
        }
        self._mock_calls.append(call_record)
        logger.debug(f"Mock LLM call recorded: scope={scope}, model={self.model}")

        # Return mock response
        if self._mock_response is not None:
            result = self._mock_response
        elif response_format is not None:
            # Try to create a minimal valid instance of the response format
            try:
                # For Pydantic models, try to create with minimal valid data
                result = {"mock": True}
            except Exception:
                result = {"mock": True}
        else:
            result = "mock response"

        if return_usage:
            token_usage = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
            return result, token_usage
        return result

    def set_mock_response(self, response: Any) -> None:
        """Set the response to return from mock calls."""
        self._mock_response = response

    def get_mock_calls(self) -> list[dict]:
        """Get the list of recorded mock calls."""
        return self._mock_calls

    def clear_mock_calls(self) -> None:
        """Clear the recorded mock calls."""
        self._mock_calls = []

    async def cleanup(self) -> None:
        """Clean up resources."""
        pass

    @classmethod
    def for_memory(cls) -> "LLMProvider":
        """Create provider for memory operations from environment variables."""
        provider = os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq")
        api_key = os.getenv("HINDSIGHT_API_LLM_API_KEY")
        if not api_key:
            raise ValueError("HINDSIGHT_API_LLM_API_KEY environment variable is required")
        base_url = os.getenv("HINDSIGHT_API_LLM_BASE_URL", "")
        model = os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-120b")

        return cls(provider=provider, api_key=api_key, base_url=base_url, model=model, reasoning_effort="low")

    @classmethod
    def for_answer_generation(cls) -> "LLMProvider":
        """Create provider for answer generation. Falls back to memory config if not set."""
        provider = os.getenv("HINDSIGHT_API_ANSWER_LLM_PROVIDER", os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq"))
        api_key = os.getenv("HINDSIGHT_API_ANSWER_LLM_API_KEY", os.getenv("HINDSIGHT_API_LLM_API_KEY"))
        if not api_key:
            raise ValueError(
                "HINDSIGHT_API_LLM_API_KEY or HINDSIGHT_API_ANSWER_LLM_API_KEY environment variable is required"
            )
        base_url = os.getenv("HINDSIGHT_API_ANSWER_LLM_BASE_URL", os.getenv("HINDSIGHT_API_LLM_BASE_URL", ""))
        model = os.getenv("HINDSIGHT_API_ANSWER_LLM_MODEL", os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-120b"))

        return cls(provider=provider, api_key=api_key, base_url=base_url, model=model, reasoning_effort="high")

    @classmethod
    def for_judge(cls) -> "LLMProvider":
        """Create provider for judge/evaluator operations. Falls back to memory config if not set."""
        provider = os.getenv("HINDSIGHT_API_JUDGE_LLM_PROVIDER", os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq"))
        api_key = os.getenv("HINDSIGHT_API_JUDGE_LLM_API_KEY", os.getenv("HINDSIGHT_API_LLM_API_KEY"))
        if not api_key:
            raise ValueError(
                "HINDSIGHT_API_LLM_API_KEY or HINDSIGHT_API_JUDGE_LLM_API_KEY environment variable is required"
            )
        base_url = os.getenv("HINDSIGHT_API_JUDGE_LLM_BASE_URL", os.getenv("HINDSIGHT_API_LLM_BASE_URL", ""))
        model = os.getenv("HINDSIGHT_API_JUDGE_LLM_MODEL", os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-120b"))

        return cls(provider=provider, api_key=api_key, base_url=base_url, model=model, reasoning_effort="high")


# Backwards compatibility alias
LLMConfig = LLMProvider
