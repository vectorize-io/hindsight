"""
LLM wrapper for unified configuration across providers.
"""
import os
import time
import asyncio
from typing import Optional, Any, Dict, List
from openai import AsyncOpenAI, RateLimitError, APIError, APIStatusError, LengthFinishReasonError
from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors
import logging

logger = logging.getLogger(__name__)

# Disable httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Global semaphore to limit concurrent LLM requests across all instances
_global_llm_semaphore = asyncio.Semaphore(32)

# Model-specific max output token limits
# These represent the maximum tokens a model can generate in a single response
MODEL_MAX_OUTPUT_TOKENS = {
    # OpenAI models
    "gpt-4o": 16384,
    "gpt-4o-mini": 16384,
    "gpt-4-turbo": 4096,
    "gpt-4-turbo-preview": 4096,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 4096,
    "o1": 100000,
    "o1-mini": 65536,
    "o1-preview": 32768,
    # Groq models
    "llama-3.1-70b-versatile": 32768,
    "llama-3.1-8b-instant": 8192,
    "llama-3.3-70b-versatile": 32768,
    "llama3-70b-8192": 8192,
    "llama3-8b-8192": 8192,
    "mixtral-8x7b-32768": 32768,
    # Gemini models
    "gemini-2.0-flash": 8192,
    "gemini-1.5-pro": 8192,
    "gemini-1.5-flash": 8192,
}

# Conservative default for unknown models
DEFAULT_MAX_OUTPUT_TOKENS = 4096


class OutputTooLongError(Exception):
    """
    Bridge exception raised when LLM output exceeds token limits.

    This wraps provider-specific errors (e.g., OpenAI's LengthFinishReasonError)
    to allow callers to handle output length issues without depending on
    provider-specific implementations.
    """
    pass


class LLMConfig:
    """Configuration for an LLM provider."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
    ):
        """
        Initialize LLM configuration.

        Args:
            provider: Provider name ("openai", "groq", "ollama"). Required.
            api_key: API key. Required.
            base_url: Base URL. Required.
            model: Model name. Required.
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

        # Validate provider
        if self.provider not in ["openai", "groq", "ollama", "gemini"]:
            raise ValueError(
                f"Invalid LLM provider: {self.provider}. Must be 'openai', 'groq', 'ollama', or 'gemini'."
            )

        # Set default base URLs
        if not self.base_url:
            if self.provider == "groq":
                self.base_url = "https://api.groq.com/openai/v1"
            elif self.provider == "ollama":
                self.base_url = "http://localhost:11434/v1"

        # Validate API key (not needed for ollama)
        if self.provider not in ["ollama"] and not self.api_key:
            raise ValueError(
                f"API key not found for {self.provider}"
            )

        # Create client (private - use .call() method instead)
        # Disable automatic retries - we handle retries in the call() method
        if self.provider == "gemini":
            self._gemini_client = genai.Client(api_key=self.api_key)
            self._client = None  # Not used for Gemini
        elif self.provider == "ollama":
            self._client = AsyncOpenAI(api_key="ollama", base_url=self.base_url, max_retries=0)
            self._gemini_client = None
        elif self.base_url:
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)
            self._gemini_client = None
        else:
            self._client = AsyncOpenAI(api_key=self.api_key, max_retries=0)
            self._gemini_client = None

        logger.info(
            f"Initialized LLM: provider={self.provider}, model={self.model}, base_url={self.base_url}"
        )

    @property
    def max_output_tokens(self) -> int:
        """
        Get the max output tokens for the configured model.

        Returns the model-specific limit from MODEL_MAX_OUTPUT_TOKENS,
        or DEFAULT_MAX_OUTPUT_TOKENS if the model is not in the mapping.
        """
        return MODEL_MAX_OUTPUT_TOKENS.get(self.model, DEFAULT_MAX_OUTPUT_TOKENS)

    async def call(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Any] = None,
        scope: str = "memory",
        max_retries: int = 10,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        skip_validation: bool = False,
        **kwargs
    ) -> Any:
        """
        Make an LLM API call with consistent configuration and retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'
            response_format: Optional Pydantic model for structured output
            scope: Scope identifier (e.g., 'memory', 'judge') for future tracking
            max_retries: Maximum number of retry attempts (default: 5)
            initial_backoff: Initial backoff time in seconds (default: 1.0)
            max_backoff: Maximum backoff time in seconds (default: 60.0)
            **kwargs: Additional parameters to pass to the API (temperature, max_tokens, etc.)

        Returns:
            Parsed response if response_format is provided, otherwise the text content

        Raises:
            Exception: Re-raises any API errors after all retries are exhausted
        """
        # Use global semaphore to limit concurrent requests
        async with _global_llm_semaphore:
            start_time = time.time()
            import json

            # Handle Gemini provider separately
            if self.provider == "gemini":
                return await self._call_gemini(messages, response_format, max_retries, initial_backoff, max_backoff, skip_validation, start_time, **kwargs)

            call_params = {
                "model": self.model,
                "messages": messages,
                **kwargs
            }
            if self.provider == "groq":
                call_params["extra_body"] = {
                    "service_tier": "auto",
                    "reasoning_effort": "low",  # Reduce reasoning overhead
                    "include_reasoning": False,  # Disable hidden reasoning tokens
                }

            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    # Use the appropriate response format
                    if response_format is not None:
                        # Use JSON mode instead of strict parse for flexibility with optional fields
                        # This allows the LLM to omit optional fields without validation errors

                        # Add schema to the system message
                        if hasattr(response_format, 'model_json_schema'):
                            schema = response_format.model_json_schema()
                            schema_msg = f"\n\nYou must respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"

                            # Add schema to the system message if present, otherwise prepend as user message
                            if call_params['messages'] and call_params['messages'][0].get('role') == 'system':
                                call_params['messages'][0]['content'] += schema_msg
                            else:
                                # No system message, add schema instruction to first user message
                                if call_params['messages']:
                                    call_params['messages'][0]['content'] = schema_msg + "\n\n" + call_params['messages'][0]['content']

                        call_params['response_format'] = {"type": "json_object"}
                        response = await self._client.chat.completions.create(**call_params)

                        # Parse the JSON response
                        content = response.choices[0].message.content
                        json_data = json.loads(content)

                        # Return raw JSON if skip_validation is True, otherwise validate with Pydantic
                        if skip_validation:
                            result = json_data
                        else:
                            result = response_format.model_validate(json_data)
                    else:
                        # Standard completion and return text content
                        response = await self._client.chat.completions.create(**call_params)
                        result = response.choices[0].message.content

                    # Log call details only if it takes more than 5 seconds
                    duration = time.time() - start_time
                    usage = response.usage
                    if duration > 10.0:
                        ratio = max(1, usage.completion_tokens) / usage.prompt_tokens
                        logger.info(
                            f"slow llm call: model={self.provider}/{self.model}, "
                            f"input_tokens={usage.prompt_tokens}, output_tokens={usage.completion_tokens}, "
                            f"total_tokens={usage.total_tokens}, time={duration:.3f}s, ratio out/in={ratio:.2f}"
                        )

                    return result

                except LengthFinishReasonError as e:
                    # Output exceeded token limits - raise bridge exception for caller to handle
                    logger.warning(f"LLM output exceeded token limits: {str(e)}")
                    raise OutputTooLongError(
                        f"LLM output exceeded token limits. Input may need to be split into smaller chunks."
                    ) from e

                except APIStatusError as e:
                    last_exception = e
                    if attempt < max_retries:
                        # Calculate exponential backoff with jitter
                        backoff = min(initial_backoff * (2 ** attempt), max_backoff)
                        # Add jitter (±20%)
                        jitter = backoff * 0.2 * (2 * (time.time() % 1) - 1)
                        sleep_time = backoff + jitter

                        # Only log if it's a non-retryable error or final attempt
                        # Silent retry for common transient errors like capacity exceeded
                        await asyncio.sleep(sleep_time)
                    else:
                        # Log only on final failed attempt
                        logger.error(f"API error after {max_retries + 1} attempts: {str(e)}")
                        raise

                except Exception as e:
                    logger.error(f"Unexpected error during LLM call: {type(e).__name__}: {str(e)}")
                    raise

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError(f"LLM call failed after all retries with no exception captured")

    async def _call_gemini(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Any],
        max_retries: int,
        initial_backoff: float,
        max_backoff: float,
        skip_validation: bool,
        start_time: float,
        **kwargs
    ) -> Any:
        """Handle Gemini-specific API calls using google-genai SDK."""
        import json

        # Convert OpenAI-style messages to Gemini format
        # Gemini uses 'user' and 'model' roles, and system instructions are separate
        system_instruction = None
        gemini_contents = []

        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            if role == 'system':
                # Accumulate system messages as system instruction
                if system_instruction:
                    system_instruction += "\n\n" + content
                else:
                    system_instruction = content
            elif role == 'assistant':
                gemini_contents.append(genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text=content)]
                ))
            else:  # user or any other role
                gemini_contents.append(genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=content)]
                ))

        # Add JSON schema instruction if response_format is provided
        if response_format is not None and hasattr(response_format, 'model_json_schema'):
            schema = response_format.model_json_schema()
            schema_msg = f"\n\nYou must respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
            if system_instruction:
                system_instruction += schema_msg
            else:
                system_instruction = schema_msg

        # Build generation config
        config_kwargs = {}
        if system_instruction:
            config_kwargs['system_instruction'] = system_instruction
        if 'temperature' in kwargs:
            config_kwargs['temperature'] = kwargs['temperature']
        if 'max_tokens' in kwargs:
            config_kwargs['max_output_tokens'] = kwargs['max_tokens']
        if response_format is not None:
            config_kwargs['response_mime_type'] = 'application/json'

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

                if response_format is not None:
                    # Parse the JSON response
                    json_data = json.loads(content)

                    # Return raw JSON if skip_validation is True, otherwise validate with Pydantic
                    if skip_validation:
                        result = json_data
                    else:
                        result = response_format.model_validate(json_data)
                else:
                    result = content

                # Log call details only if it takes more than 10 seconds
                duration = time.time() - start_time
                if duration > 10.0 and hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = response.usage_metadata
                    logger.info(
                        f"slow llm call: model={self.provider}/{self.model}, "
                        f"input_tokens={usage.prompt_token_count}, output_tokens={usage.candidates_token_count}, "
                        f"time={duration:.3f}s"
                    )

                return result

            except genai_errors.APIError as e:
                # Handle rate limits and server errors with retry
                if e.code in (429, 503, 500):
                    last_exception = e
                    if attempt < max_retries:
                        backoff = min(initial_backoff * (2 ** attempt), max_backoff)
                        jitter = backoff * 0.2 * (2 * (time.time() % 1) - 1)
                        sleep_time = backoff + jitter
                        await asyncio.sleep(sleep_time)
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
        raise RuntimeError(f"Gemini call failed after all retries with no exception captured")

    @classmethod
    def for_memory(cls) -> "LLMConfig":
        """Create configuration for memory operations from environment variables."""
        provider = os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq")
        api_key = os.getenv("HINDSIGHT_API_LLM_API_KEY")
        base_url = os.getenv("HINDSIGHT_API_LLM_BASE_URL")
        model = os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-120b")

        # Set default base URL if not provided
        if not base_url:
            if provider == "groq":
                base_url = "https://api.groq.com/openai/v1"
            elif provider == "ollama":
                base_url = "http://localhost:11434/v1"
            else:
                base_url = ""

        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

    @classmethod
    def for_judge(cls) -> "LLMConfig":
        """
        Create configuration for judge/evaluator operations from environment variables.

        Falls back to memory LLM config if judge-specific config not set.
        """
        # Check if judge-specific config exists, otherwise fall back to memory config
        provider = os.getenv("HINDSIGHT_API_JUDGE_LLM_PROVIDER", os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq"))
        api_key = os.getenv("HINDSIGHT_API_JUDGE_LLM_API_KEY", os.getenv("HINDSIGHT_API_LLM_API_KEY"))
        base_url = os.getenv("HINDSIGHT_API_JUDGE_LLM_BASE_URL", os.getenv("HINDSIGHT_API_LLM_BASE_URL"))
        model = os.getenv("HINDSIGHT_API_JUDGE_LLM_MODEL", os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-120b"))

        # Set default base URL if not provided
        if not base_url:
            if provider == "groq":
                base_url = "https://api.groq.com/openai/v1"
            elif provider == "ollama":
                base_url = "http://localhost:11434/v1"
            else:
                base_url = ""

        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
