"""
LLM wrapper for unified configuration across providers.
"""
import os
import time
import asyncio
from typing import Optional, Any, Dict, List
from openai import AsyncOpenAI, RateLimitError, APIError, APIStatusError, LengthFinishReasonError
import logging

logger = logging.getLogger(__name__)

# Disable httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)


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
        if self.provider not in ["openai", "groq", "ollama"]:
            raise ValueError(
                f"Invalid LLM provider: {self.provider}. Must be 'openai', 'groq', or 'ollama'."
            )

        # Set default base URLs
        if not self.base_url:
            if self.provider == "groq":
                self.base_url = "https://api.groq.com/openai/v1"
            elif self.provider == "ollama":
                self.base_url = "http://localhost:11434/v1"

        # Validate API key (not needed for ollama)
        if self.provider != "ollama" and not self.api_key:
            raise ValueError(
                f"API key not found for {self.provider}"
            )

        # Create client (private - use .call() method instead)
        if self.provider == "ollama":
            self._client = AsyncOpenAI(api_key="ollama", base_url=self.base_url)
        elif self.base_url:
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self._client = AsyncOpenAI(api_key=self.api_key)

        logger.info(
            f"Initialized LLM: provider={self.provider}, model={self.model}, base_url={self.base_url}"
        )

    async def call(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Any] = None,
        scope: str = "memory",
        max_retries: int = 5,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
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
        start_time = time.time()

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
                if response_format is not None:
                    # Use structured output parsing and return .parsed
                    response = await self._client.beta.chat.completions.parse(
                        response_format=response_format,
                        **call_params
                    )
                    result = response.choices[0].message.parsed
                else:
                    # Standard completion and return text content
                    response = await self._client.chat.completions.create(**call_params)
                    result = response.choices[0].message.content

                # Log call details on success
                duration = time.time() - start_time
                usage = response.usage
                ratio = max(1, usage.completion_tokens) / usage.prompt_tokens
                if ratio > 3:
                    raw_content = response.choices[0].message.content
                    raw_len = len(raw_content) if raw_content else 0
                    logger.info(
                        f"model={self.provider}/{self.model}, "
                        f"input_tokens={usage.prompt_tokens}, output_tokens={usage.completion_tokens}, "
                        f"total_tokens={usage.total_tokens}, time={duration:.3f}s, ratio out/in={ratio:.2f}, HIGH RATIO - raw_content_chars={raw_len}, \n\nin={messages}\n\nout={result}\n\nraw={raw_content}\n\n"
                    )
                else:

                    logger.info(
                        f"model={self.provider}/{self.model}, "
                        f"input_tokens={usage.prompt_tokens}, output_tokens={usage.completion_tokens}, "
                        f"total_tokens={usage.total_tokens}, time={duration:.3f}s, ratio out/in={ratio:.2f}, "
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

                    logger.warning(
                        f"LLM error on attempt {attempt + 1}/{max_retries + 1}. "
                        f"Retrying in {sleep_time:.2f}s... Error: {str(e)}"
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    logger.error(f"Non-retryable API error after {max_retries + 1} attempts: {str(e)}, input {messages}")
                    raise

            except Exception as e:
                logger.error(f"Unexpected error during LLM call: {type(e).__name__}: {str(e)}, input {messages}")
                raise

        # This should never be reached, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError(f"LLM call failed after all retries with no exception captured, input {messages}")

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
