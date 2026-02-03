"""
Google Gemini/VertexAI LLM provider.

This provider supports both:
1. Gemini API (api.generativeai.google.com) with API key authentication
2. Vertex AI with service account or Application Default Credentials (ADC)
"""

import asyncio
import json
import logging
import os
import time
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from hindsight_api.engine.llm_interface import LLMInterface, OutputTooLongError
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, TokenUsage
from hindsight_api.metrics import get_metrics_collector

logger = logging.getLogger(__name__)

# Vertex AI imports (optional)
try:
    import google.auth
    from google.oauth2 import service_account

    VERTEXAI_AVAILABLE = True
except ImportError:
    VERTEXAI_AVAILABLE = False


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
    ) -> Any:
        """
        Make a Gemini/VertexAI API call with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            response_format: Optional Pydantic model for structured output.
            max_completion_tokens: Maximum tokens in response (not supported by Gemini).
            temperature: Sampling temperature (0.0-2.0).
            scope: Scope identifier for tracking.
            max_retries: Maximum retry attempts.
            initial_backoff: Initial backoff time in seconds.
            max_backoff: Maximum backoff time in seconds.
            skip_validation: Return raw JSON without Pydantic validation.
            strict_schema: Use strict JSON schema enforcement (not supported by Gemini).
            return_usage: If True, return tuple (result, TokenUsage).

        Returns:
            If return_usage=False: Parsed response if response_format provided, else text.
            If return_usage=True: Tuple of (result, TokenUsage).
        """
        start_time = time.time()

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
        config_kwargs: dict[str, Any] = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if response_format is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_format
        if temperature is not None:
            config_kwargs["temperature"] = temperature

        generation_config = genai_types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                response = await self._client.aio.models.generate_content(
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

                # Parse structured output if requested
                if response_format is not None:
                    json_data = json.loads(content)
                    if skip_validation:
                        result = json_data
                    else:
                        result = response_format.model_validate(json_data)
                else:
                    result = content

                # Extract token usage
                input_tokens = 0
                output_tokens = 0
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage = response.usage_metadata
                    input_tokens = usage.prompt_token_count or 0
                    output_tokens = usage.candidates_token_count or 0

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
    ) -> LLMToolCallResult:
        """
        Make a Gemini/VertexAI API call with tool/function calling support.

        Args:
            messages: List of message dicts. Can include tool results with role='tool'.
            tools: List of tool definitions in OpenAI format.
            max_completion_tokens: Maximum tokens (not supported by Gemini).
            temperature: Sampling temperature.
            scope: Scope identifier for tracking.
            max_retries: Maximum retry attempts.
            initial_backoff: Initial backoff time in seconds.
            max_backoff: Maximum backoff time in seconds.
            tool_choice: How to choose tools (Gemini uses "auto" only).

        Returns:
            LLMToolCallResult with content and/or tool_calls.
        """
        start_time = time.time()

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

        config_kwargs: dict[str, Any] = {"tools": gemini_tools}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if temperature is not None:
            config_kwargs["temperature"] = temperature

        config = genai_types.GenerateContentConfig(**config_kwargs)

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                response = await self._client.aio.models.generate_content(
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

                # Extract token usage
                input_tokens = 0
                output_tokens = 0
                if response.usage_metadata:
                    input_tokens = response.usage_metadata.prompt_token_count or 0
                    output_tokens = response.usage_metadata.candidates_token_count or 0

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

    async def cleanup(self) -> None:
        """Clean up resources (close connections, etc.)."""
        # Gemini client doesn't require explicit cleanup
        pass
