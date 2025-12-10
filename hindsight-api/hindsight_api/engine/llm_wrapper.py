"""
LLM wrapper for unified configuration across providers.
"""
import os
import time
import asyncio
from typing import Optional, Any, Dict, List
from openai import AsyncOpenAI, RateLimitError, APIError, APIStatusError, APIConnectionError, LengthFinishReasonError
import logging

# Seed applied to every Groq request for deterministic behavior.
DEFAULT_LLM_SEED = 4242

logger = logging.getLogger(__name__)

# Disable httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Global semaphore to limit concurrent LLM requests across all instances
_global_llm_semaphore = asyncio.Semaphore(32)


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
    Unified LLM provider using OpenAI-compatible API.

    Supports OpenAI, Groq, and Ollama (any OpenAI-compatible endpoint).
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        reasoning_effort: str = "low",
    ):
        """
        Initialize LLM provider.

        Args:
            provider: Provider name ("openai", "groq", "ollama").
            api_key: API key.
            base_url: Base URL for the API.
            model: Model name.
            reasoning_effort: Reasoning effort level for supported providers.
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.reasoning_effort = reasoning_effort

        # Validate provider
        valid_providers = ["openai", "groq", "ollama"]
        if self.provider not in valid_providers:
            raise ValueError(
                f"Invalid LLM provider: {self.provider}. Must be one of: {', '.join(valid_providers)}"
            )

        # Set default base URLs
        if not self.base_url:
            if self.provider == "groq":
                self.base_url = "https://api.groq.com/openai/v1"
            elif self.provider == "ollama":
                self.base_url = "http://localhost:11434/v1"

        # Validate API key (not needed for ollama)
        if self.provider != "ollama" and not self.api_key:
            raise ValueError(f"API key not found for {self.provider}")

        # Create OpenAI-compatible client for all providers
        if self.provider == "ollama":
            self._client = AsyncOpenAI(api_key="ollama", base_url=self.base_url, max_retries=0)
        else:
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)

        logger.info(
            f"Initialized LLM: provider={self.provider}, model={self.model}, base_url={self.base_url}"
        )

    async def call(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Any] = None,
        max_completion_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        scope: str = "memory",
        max_retries: int = 10,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        skip_validation: bool = False,
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

        Returns:
            Parsed response if response_format is provided, otherwise text content.

        Raises:
            OutputTooLongError: If output exceeds token limits.
            Exception: Re-raises API errors after retries exhausted.
        """
        async with _global_llm_semaphore:
            start_time = time.time()
            import json

            call_params = {
                "model": self.model,
                "messages": messages,
            }

            if max_completion_tokens is not None:
                call_params["max_completion_tokens"] = max_completion_tokens
            if temperature is not None:
                call_params["temperature"] = temperature

            # Provider-specific parameters
            if self.provider == "groq":
                call_params["seed"] = DEFAULT_LLM_SEED
                call_params["extra_body"] = {
                    "service_tier": "auto",
                    "reasoning_effort": self.reasoning_effort,
                    "include_reasoning": False,
                }

            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    if response_format is not None:
                        # Add schema to system message for JSON mode
                        if hasattr(response_format, 'model_json_schema'):
                            schema = response_format.model_json_schema()
                            schema_msg = f"\n\nYou must respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"

                            if call_params['messages'] and call_params['messages'][0].get('role') == 'system':
                                call_params['messages'][0]['content'] += schema_msg
                            elif call_params['messages']:
                                call_params['messages'][0]['content'] = schema_msg + "\n\n" + call_params['messages'][0]['content']

                        call_params['response_format'] = {"type": "json_object"}
                        response = await self._client.chat.completions.create(**call_params)

                        content = response.choices[0].message.content
                        json_data = json.loads(content)

                        if skip_validation:
                            result = json_data
                        else:
                            result = response_format.model_validate(json_data)
                    else:
                        response = await self._client.chat.completions.create(**call_params)
                        result = response.choices[0].message.content

                    # Log slow calls
                    duration = time.time() - start_time
                    usage = response.usage
                    if duration > 10.0:
                        ratio = max(1, usage.completion_tokens) / usage.prompt_tokens
                        cached_tokens = 0
                        if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                            cached_tokens = getattr(usage.prompt_tokens_details, 'cached_tokens', 0) or 0
                        cache_info = f", cached_tokens={cached_tokens}" if cached_tokens > 0 else ""
                        logger.info(
                            f"slow llm call: model={self.provider}/{self.model}, "
                            f"input_tokens={usage.prompt_tokens}, output_tokens={usage.completion_tokens}, "
                            f"total_tokens={usage.total_tokens}{cache_info}, time={duration:.3f}s, ratio out/in={ratio:.2f}"
                        )

                    return result

                except LengthFinishReasonError as e:
                    logger.warning(f"LLM output exceeded token limits: {str(e)}")
                    raise OutputTooLongError(
                        f"LLM output exceeded token limits. Input may need to be split into smaller chunks."
                    ) from e

                except APIConnectionError as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Connection error, retrying... (attempt {attempt + 1}/{max_retries + 1})")
                        backoff = min(initial_backoff * (2 ** attempt), max_backoff)
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        logger.error(f"Connection error after {max_retries + 1} attempts: {str(e)}")
                        raise

                except APIStatusError as e:
                    # Fast fail on 4xx client errors (except 429 rate limit and 498 which is treated as server error)
                    if 400 <= e.status_code < 500 and e.status_code not in (429, 498):
                        logger.error(f"Client error (HTTP {e.status_code}), not retrying: {str(e)}")
                        raise

                    last_exception = e
                    if attempt < max_retries:
                        backoff = min(initial_backoff * (2 ** attempt), max_backoff)
                        jitter = backoff * 0.2 * (2 * (time.time() % 1) - 1)
                        sleep_time = backoff + jitter
                        await asyncio.sleep(sleep_time)
                    else:
                        logger.error(f"API error after {max_retries + 1} attempts: {str(e)}")
                        raise

                except Exception as e:
                    logger.error(f"Unexpected error during LLM call: {type(e).__name__}: {str(e)}")
                    raise

            if last_exception:
                raise last_exception
            raise RuntimeError(f"LLM call failed after all retries with no exception captured")

    @classmethod
    def for_memory(cls) -> "LLMProvider":
        """Create provider for memory operations from environment variables."""
        provider = os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq")
        api_key = os.getenv("HINDSIGHT_API_LLM_API_KEY")
        base_url = os.getenv("HINDSIGHT_API_LLM_BASE_URL", "")
        model = os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-120b")

        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort="low"
        )

    @classmethod
    def for_answer_generation(cls) -> "LLMProvider":
        """Create provider for answer generation. Falls back to memory config if not set."""
        provider = os.getenv("HINDSIGHT_API_ANSWER_LLM_PROVIDER", os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq"))
        api_key = os.getenv("HINDSIGHT_API_ANSWER_LLM_API_KEY", os.getenv("HINDSIGHT_API_LLM_API_KEY"))
        base_url = os.getenv("HINDSIGHT_API_ANSWER_LLM_BASE_URL", os.getenv("HINDSIGHT_API_LLM_BASE_URL", ""))
        model = os.getenv("HINDSIGHT_API_ANSWER_LLM_MODEL", os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-120b"))

        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort="high"
        )

    @classmethod
    def for_judge(cls) -> "LLMProvider":
        """Create provider for judge/evaluator operations. Falls back to memory config if not set."""
        provider = os.getenv("HINDSIGHT_API_JUDGE_LLM_PROVIDER", os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq"))
        api_key = os.getenv("HINDSIGHT_API_JUDGE_LLM_API_KEY", os.getenv("HINDSIGHT_API_LLM_API_KEY"))
        base_url = os.getenv("HINDSIGHT_API_JUDGE_LLM_BASE_URL", os.getenv("HINDSIGHT_API_LLM_BASE_URL", ""))
        model = os.getenv("HINDSIGHT_API_JUDGE_LLM_MODEL", os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-120b"))

        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort="high"
        )


# Backwards compatibility alias
LLMConfig = LLMProvider
