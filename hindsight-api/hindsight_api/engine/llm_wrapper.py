"""
LLM wrapper for unified configuration across providers.
"""
import os
import time
import asyncio
from typing import Optional, Any, Dict, List
from openai import AsyncOpenAI, RateLimitError, APIError, APIStatusError, APIConnectionError, LengthFinishReasonError
from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors
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
    ):
        """
        Initialize LLM provider.

        Args:
            provider: Provider name ("openai", "groq", "ollama", "gemini").
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
        valid_providers = ["openai", "groq", "ollama", "gemini"]
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

        # Create client based on provider
        if self.provider == "gemini":
            self._gemini_client = genai.Client(api_key=self.api_key)
            self._client = None
        elif self.provider == "ollama":
            self._client = AsyncOpenAI(api_key="ollama", base_url=self.base_url, max_retries=0)
            self._gemini_client = None
        else:
            # Only pass base_url if it's set (OpenAI uses default URL otherwise)
            client_kwargs = {"api_key": self.api_key, "max_retries": 0}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._client = AsyncOpenAI(**client_kwargs)
            self._gemini_client = None

    async def verify_connection(self) -> None:
        """
        Verify that the LLM provider is configured correctly by making a simple test call.

        Raises:
            RuntimeError: If the connection test fails.
        """
        try:
            logger.info(f"Verifying LLM: provider={self.provider}, model={self.model}, base_url={self.base_url or 'default'}...")
            await self.call(
                messages=[{"role": "user", "content": "Say 'ok'"}],
                max_completion_tokens=10,
                max_retries=2,
                initial_backoff=0.5,
                max_backoff=2.0,
            )
            # If we get here without exception, the connection is working
            logger.info(f"LLM verified: {self.provider}/{self.model}")
        except Exception as e:
            raise RuntimeError(
                f"LLM connection verification failed for {self.provider}/{self.model}: {e}"
            ) from e

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

            # Handle Gemini provider separately
            if self.provider == "gemini":
                return await self._call_gemini(
                    messages, response_format, max_retries, initial_backoff,
                    max_backoff, skip_validation, start_time
                )

            call_params = {
                "model": self.model,
                "messages": messages,
            }

            # Check if model supports reasoning parameter (o1, o3, gpt-5 families)
            model_lower = self.model.lower()
            is_reasoning_model = any(x in model_lower for x in ["gpt-5", "o1", "o3"])

            # For GPT-4 and GPT-4.1 models, cap max_completion_tokens to 32000
            is_gpt4_model = any(x in model_lower for x in ["gpt-4.1", "gpt-4-"])
            if max_completion_tokens is not None:
                if is_gpt4_model and max_completion_tokens > 32000:
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
            if is_reasoning_model and self.provider == "openai":
                call_params["reasoning_effort"] = self.reasoning_effort

            # Provider-specific parameters
            if self.provider == "groq":
                call_params["seed"] = DEFAULT_LLM_SEED
                extra_body = {"service_tier": "auto"}
                # Only add reasoning parameters for reasoning models
                if is_reasoning_model:
                    extra_body["reasoning_effort"] = self.reasoning_effort
                    extra_body["include_reasoning"] = False
                call_params["extra_body"] = extra_body

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
                        status_code = getattr(e, 'status_code', None) or getattr(getattr(e, 'response', None), 'status_code', None)
                        logger.warning(f"Connection error, retrying... (attempt {attempt + 1}/{max_retries + 1}) - status_code={status_code}, message={e}")
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

    async def _call_gemini(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Any],
        max_retries: int,
        initial_backoff: float,
        max_backoff: float,
        skip_validation: bool,
        start_time: float,
    ) -> Any:
        """Handle Gemini-specific API calls."""
        import json

        # Convert OpenAI-style messages to Gemini format
        system_instruction = None
        gemini_contents = []

        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            if role == 'system':
                if system_instruction:
                    system_instruction += "\n\n" + content
                else:
                    system_instruction = content
            elif role == 'assistant':
                gemini_contents.append(genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text=content)]
                ))
            else:
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
        if response_format is not None:
            config_kwargs['response_mime_type'] = 'application/json'
            config_kwargs['response_schema'] = response_format

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
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'finish_reason'):
                            block_reason = candidate.finish_reason

                    if attempt < max_retries:
                        logger.warning(f"Gemini returned empty response (reason: {block_reason}), retrying...")
                        backoff = min(initial_backoff * (2 ** attempt), max_backoff)
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

                # Log slow calls
                duration = time.time() - start_time
                if duration > 10.0 and hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = response.usage_metadata
                    logger.info(
                        f"slow llm call: model={self.provider}/{self.model}, "
                        f"input_tokens={usage.prompt_token_count}, output_tokens={usage.candidates_token_count}, "
                        f"time={duration:.3f}s"
                    )

                return result

            except json.JSONDecodeError as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(f"Gemini returned invalid JSON, retrying...")
                    backoff = min(initial_backoff * (2 ** attempt), max_backoff)
                    await asyncio.sleep(backoff)
                    continue
                else:
                    logger.error(f"Gemini returned invalid JSON after {max_retries + 1} attempts")
                    raise

            except genai_errors.APIError as e:
                # Fast fail on 4xx client errors (except 429 rate limit)
                if e.code and 400 <= e.code < 500 and e.code != 429:
                    logger.error(f"Gemini client error (HTTP {e.code}), not retrying: {str(e)}")
                    raise

                # Retry on 429 and 5xx
                if e.code in (429, 500, 502, 503, 504):
                    last_exception = e
                    if attempt < max_retries:
                        backoff = min(initial_backoff * (2 ** attempt), max_backoff)
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
        raise RuntimeError(f"Gemini call failed after all retries")

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
