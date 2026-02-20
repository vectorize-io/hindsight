"""
LLM wrapper for unified configuration across providers.
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, LengthFinishReasonError

# Vertex AI imports (conditional - for LLMProvider to pass credentials to GeminiLLM)
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


def parse_llm_json(raw: str) -> Any:
    """
    Robustly parse JSON returned by an LLM.

    Handles common LLM output quirks:
    1. Markdown code fences (```json ... ```) — strip them before parsing.
    2. Embedded control characters (\\x00-\\x1f, \\x7f) — replace with space
       and retry if the initial parse fails.

    Args:
        raw: Raw text returned by the LLM.

    Returns:
        Parsed Python object (dict, list, etc.).

    Raises:
        json.JSONDecodeError: If the text cannot be parsed even after cleanup.
    """
    text = raw.strip()

    # Strip markdown code fences (some models wrap JSON in ```json ... ```)
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Some models (e.g. Gemini) embed raw control characters inside JSON
        # string values. Replacing them with a space usually produces valid JSON.
        cleaned = re.sub(r"[\x00-\x1f\x7f]", " ", text)
        return json.loads(cleaned)


_PROVIDERS_WITHOUT_API_KEY = frozenset(
    {
        "ollama",
        "lmstudio",
        "openai-codex",
        "claude-code",
        "mock",
        "vertexai",
    }
)


def requires_api_key(provider: str) -> bool:
    """Return True if the given provider requires an API key to operate."""
    return provider.lower() not in _PROVIDERS_WITHOUT_API_KEY


def create_llm_provider(
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    reasoning_effort: str,
    groq_service_tier: str | None = None,
    openai_service_tier: str | None = None,
    vertexai_project_id: str | None = None,
    vertexai_region: str | None = None,
    vertexai_credentials: Any = None,
) -> Any:  # Returns LLMInterface
    """
    Factory function to create the appropriate LLM provider implementation.

    Args:
        provider: Provider name ("openai", "groq", "ollama", "gemini", "anthropic", etc.).
        api_key: API key (may be None for local providers or OAuth providers).
        base_url: Base URL for the API.
        model: Model name.
        reasoning_effort: Reasoning effort level for supported providers.
        groq_service_tier: Groq service tier (for Groq provider) - "on_demand", "flex", or "auto".
        openai_service_tier: OpenAI service tier (for OpenAI provider) - None (default) or "flex" (50% cheaper).
        vertexai_project_id: Vertex AI project ID (for VertexAI provider).
        vertexai_region: Vertex AI region (for VertexAI provider).
        vertexai_credentials: Vertex AI credentials object (for VertexAI provider).

    Returns:
        LLMInterface implementation for the specified provider.
    """
    from .llm_interface import LLMInterface
    from .providers import (
        AnthropicLLM,
        ClaudeCodeLLM,
        CodexLLM,
        GeminiLLM,
        MockLLM,
        OpenAICompatibleLLM,
    )

    provider_lower = provider.lower()

    if provider_lower == "openai-codex":
        return CodexLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
        )

    elif provider_lower == "claude-code":
        return ClaudeCodeLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
        )

    elif provider_lower == "mock":
        return MockLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
        )

    elif provider_lower in ("gemini", "vertexai"):
        return GeminiLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            vertexai_project_id=vertexai_project_id,
            vertexai_region=vertexai_region,
            vertexai_credentials=vertexai_credentials,
        )

    elif provider_lower == "anthropic":
        return AnthropicLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
        )

    elif provider_lower in ("openai", "groq", "ollama", "lmstudio"):
        return OpenAICompatibleLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            groq_service_tier=groq_service_tier,
            openai_service_tier=openai_service_tier,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


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
        openai_service_tier: str | None = None,
    ):
        """
        Initialize LLM provider.

        Args:
            provider: Provider name ("openai", "groq", "ollama", "gemini", "anthropic", "lmstudio").
            api_key: API key.
            base_url: Base URL for the API.
            model: Model name.
            reasoning_effort: Reasoning effort level for supported providers.
            groq_service_tier: Groq service tier ("on_demand", "flex", "auto") - from config.
            openai_service_tier: OpenAI service tier (None or "flex") - from config.
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.reasoning_effort = reasoning_effort
        # Service tiers from hierarchical config (not env vars)
        self.groq_service_tier = groq_service_tier
        self.openai_service_tier = openai_service_tier

        # Validate provider
        valid_providers = [
            "openai",
            "groq",
            "ollama",
            "gemini",
            "anthropic",
            "lmstudio",
            "vertexai",
            "openai-codex",
            "claude-code",
            "mock",
        ]
        if self.provider not in valid_providers:
            raise ValueError(f"Invalid LLM provider: {self.provider}. Must be one of: {', '.join(valid_providers)}")

        # Set default base URLs
        if not self.base_url:
            if self.provider == "groq":
                self.base_url = "https://api.groq.com/openai/v1"
            elif self.provider == "ollama":
                self.base_url = "http://localhost:11434/v1"
            elif self.provider == "lmstudio":
                self.base_url = "http://localhost:1234/v1"

        # Prepare Vertex AI config (if applicable)
        vertexai_project_id = None
        vertexai_region = None
        vertexai_credentials = None

        if self.provider == "vertexai":
            from ..config import get_config

            config = get_config()

            vertexai_project_id = config.llm_vertexai_project_id
            if not vertexai_project_id:
                raise ValueError(
                    "HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID is required for Vertex AI provider. "
                    "Set it to your GCP project ID."
                )

            vertexai_region = config.llm_vertexai_region or "us-central1"
            service_account_key = config.llm_vertexai_service_account_key

            # Load explicit service account credentials if provided
            if service_account_key:
                if not VERTEXAI_AVAILABLE:
                    raise ValueError(
                        "Vertex AI service account auth requires 'google-auth' package. "
                        "Install with: pip install google-auth"
                    )
                vertexai_credentials = service_account.Credentials.from_service_account_file(
                    service_account_key,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                logger.info(f"Vertex AI: Using service account key: {service_account_key}")

            # Strip google/ prefix from model name — native SDK uses bare names
            if self.model.startswith("google/"):
                self.model = self.model[len("google/") :]

            logger.info(
                f"Vertex AI: project={vertexai_project_id}, region={vertexai_region}, "
                f"model={self.model}, auth={'service_account' if service_account_key else 'ADC'}"
            )

        # Create provider implementation using factory
        self._provider_impl = create_llm_provider(
            provider=self.provider,
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            groq_service_tier=self.groq_service_tier,
            openai_service_tier=self.openai_service_tier,
            vertexai_project_id=vertexai_project_id,
            vertexai_region=vertexai_region,
            vertexai_credentials=vertexai_credentials,
        )

        # Backward compatibility: Keep mock provider properties
        self._mock_calls: list[dict] = []
        self._mock_response: Any = None

    @property
    def _client(self) -> Any:
        """
        Get the OpenAI client for OpenAI-compatible providers.

        This property provides backward compatibility for code that directly accesses
        the _client attribute (e.g., benchmarks, memory_engine).

        Returns:
            AsyncOpenAI client instance for OpenAI-compatible providers, or None for other providers.
        """
        from .providers.openai_compatible_llm import OpenAICompatibleLLM

        if isinstance(self._provider_impl, OpenAICompatibleLLM):
            return self._provider_impl._client
        return None

    @property
    def _gemini_client(self) -> Any:
        """
        Get the Gemini client for Gemini/VertexAI providers.

        This property provides backward compatibility for code that directly accesses
        the _gemini_client attribute.

        Returns:
            genai.Client instance for Gemini/VertexAI providers, or None for other providers.
        """
        from .providers.gemini_llm import GeminiLLM

        if isinstance(self._provider_impl, GeminiLLM):
            return self._provider_impl._client
        return None

    async def verify_connection(self) -> None:
        """
        Verify that the LLM provider is configured correctly by making a simple test call.

        Raises:
            RuntimeError: If the connection test fails.
        """
        await self._provider_impl.verify_connection()

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
        async with _global_llm_semaphore:
            # Delegate to provider implementation
            result = await self._provider_impl.call(
                messages=messages,
                response_format=response_format,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                scope=scope,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                skip_validation=skip_validation,
                strict_schema=strict_schema,
                return_usage=return_usage,
            )

            # Backward compatibility: Update mock call tracking for mock provider
            # This allows existing tests using LLMProvider._mock_calls to continue working
            if self.provider == "mock":
                from .providers.mock_llm import MockLLM

                if isinstance(self._provider_impl, MockLLM):
                    # Sync the mock calls from provider implementation to wrapper
                    self._mock_calls = self._provider_impl.get_mock_calls()

            return result

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
        async with _global_llm_semaphore:
            # Delegate to provider implementation
            result = await self._provider_impl.call_with_tools(
                messages=messages,
                tools=tools,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                scope=scope,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                tool_choice=tool_choice,
            )

            # Backward compatibility: Update mock call tracking for mock provider
            # This allows existing tests using LLMProvider._mock_calls to continue working
            if self.provider == "mock":
                from .providers.mock_llm import MockLLM

                if isinstance(self._provider_impl, MockLLM):
                    # Sync the mock calls from provider implementation to wrapper
                    self._mock_calls = self._provider_impl.get_mock_calls()

            return result

    def set_mock_response(self, response: Any) -> None:
        """Set the response to return from mock calls."""
        # Backward compatibility: Store in both wrapper and provider implementation
        self._mock_response = response
        if self.provider == "mock":
            from .providers.mock_llm import MockLLM

            if isinstance(self._provider_impl, MockLLM):
                self._provider_impl.set_mock_response(response)

    def get_mock_calls(self) -> list[dict]:
        """Get the list of recorded mock calls."""
        # Backward compatibility: Read from provider implementation if mock provider
        if self.provider == "mock":
            from .providers.mock_llm import MockLLM

            if isinstance(self._provider_impl, MockLLM):
                return self._provider_impl.get_mock_calls()
        return self._mock_calls

    def clear_mock_calls(self) -> None:
        """Clear the recorded mock calls."""
        # Backward compatibility: Clear in both wrapper and provider implementation
        self._mock_calls = []
        if self.provider == "mock":
            from .providers.mock_llm import MockLLM

            if isinstance(self._provider_impl, MockLLM):
                self._provider_impl.clear_mock_calls()

    def _load_codex_auth(self) -> tuple[str, str]:
        """
        Load OAuth credentials from ~/.codex/auth.json.

        Returns:
            Tuple of (access_token, account_id).

        Raises:
            FileNotFoundError: If auth file doesn't exist.
            ValueError: If auth file is invalid.
        """
        auth_file = Path.home() / ".codex" / "auth.json"

        if not auth_file.exists():
            raise FileNotFoundError(
                f"Codex auth file not found: {auth_file}\nRun 'codex auth login' to authenticate with ChatGPT Plus/Pro."
            )

        with open(auth_file) as f:
            data = json.load(f)

        # Validate auth structure
        auth_mode = data.get("auth_mode")
        if auth_mode != "chatgpt":
            raise ValueError(f"Expected auth_mode='chatgpt', got: {auth_mode}")

        tokens = data.get("tokens", {})
        access_token = tokens.get("access_token")
        account_id = tokens.get("account_id")

        if not access_token:
            raise ValueError("No access_token found in Codex auth file. Run 'codex auth login' again.")

        return access_token, account_id

    def _verify_claude_code_available(self) -> None:
        """
        Verify that Claude Agent SDK can be imported and is properly configured.

        Raises:
            ImportError: If Claude Agent SDK is not installed.
            RuntimeError: If Claude Code is not authenticated.
        """
        try:
            # Import Claude Agent SDK
            # Reduce Claude Agent SDK logging verbosity
            import logging as sdk_logging

            from claude_agent_sdk import query  # noqa: F401

            sdk_logging.getLogger("claude_agent_sdk").setLevel(sdk_logging.WARNING)
            sdk_logging.getLogger("claude_agent_sdk._internal").setLevel(sdk_logging.WARNING)

            logger.debug("Claude Agent SDK imported successfully")
        except ImportError as e:
            raise ImportError(
                "Claude Agent SDK not installed. Run: uv add claude-agent-sdk or pip install claude-agent-sdk"
            ) from e

        # SDK will automatically check for authentication when first used
        # No need to verify here - let it fail gracefully on first call with helpful error

    async def cleanup(self) -> None:
        """Clean up resources."""
        pass

    @classmethod
    def for_memory(cls) -> "LLMProvider":
        """Create provider for memory operations from environment variables."""
        provider = os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq")
        api_key = os.getenv("HINDSIGHT_API_LLM_API_KEY", "")

        # API key not needed for openai-codex (uses OAuth), claude-code (uses Keychain OAuth),
        # ollama (local), or vertexai (uses GCP service account credentials)
        if not api_key and provider not in ("openai-codex", "claude-code", "ollama", "vertexai"):
            raise ValueError(
                "HINDSIGHT_API_LLM_API_KEY environment variable is required (unless using openai-codex or claude-code)"
            )

        base_url = os.getenv("HINDSIGHT_API_LLM_BASE_URL", "")
        model = os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-120b")

        return cls(provider=provider, api_key=api_key, base_url=base_url, model=model, reasoning_effort="low")

    @classmethod
    def for_answer_generation(cls) -> "LLMProvider":
        """Create provider for answer generation. Falls back to memory config if not set."""
        provider = os.getenv("HINDSIGHT_API_ANSWER_LLM_PROVIDER", os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq"))
        api_key = os.getenv("HINDSIGHT_API_ANSWER_LLM_API_KEY", os.getenv("HINDSIGHT_API_LLM_API_KEY", ""))

        # API key not needed for openai-codex (uses OAuth), claude-code (uses Keychain OAuth),
        # ollama (local), or vertexai (uses GCP service account credentials)
        if not api_key and provider not in ("openai-codex", "claude-code", "ollama", "vertexai"):
            raise ValueError(
                "HINDSIGHT_API_LLM_API_KEY or HINDSIGHT_API_ANSWER_LLM_API_KEY environment variable is required "
                "(unless using openai-codex or claude-code)"
            )

        base_url = os.getenv("HINDSIGHT_API_ANSWER_LLM_BASE_URL", os.getenv("HINDSIGHT_API_LLM_BASE_URL", ""))
        model = os.getenv("HINDSIGHT_API_ANSWER_LLM_MODEL", os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-120b"))

        return cls(provider=provider, api_key=api_key, base_url=base_url, model=model, reasoning_effort="high")

    @classmethod
    def for_judge(cls) -> "LLMProvider":
        """Create provider for judge/evaluator operations. Falls back to memory config if not set."""
        provider = os.getenv("HINDSIGHT_API_JUDGE_LLM_PROVIDER", os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq"))
        api_key = os.getenv("HINDSIGHT_API_JUDGE_LLM_API_KEY", os.getenv("HINDSIGHT_API_LLM_API_KEY", ""))

        # API key not needed for openai-codex (uses OAuth), claude-code (uses Keychain OAuth),
        # ollama (local), or vertexai (uses GCP service account credentials)
        if not api_key and provider not in ("openai-codex", "claude-code", "ollama", "vertexai"):
            raise ValueError(
                "HINDSIGHT_API_LLM_API_KEY or HINDSIGHT_API_JUDGE_LLM_API_KEY environment variable is required "
                "(unless using openai-codex or claude-code)"
            )

        base_url = os.getenv("HINDSIGHT_API_JUDGE_LLM_BASE_URL", os.getenv("HINDSIGHT_API_LLM_BASE_URL", ""))
        model = os.getenv("HINDSIGHT_API_JUDGE_LLM_MODEL", os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-120b"))

        return cls(provider=provider, api_key=api_key, base_url=base_url, model=model, reasoning_effort="high")


# Backwards compatibility alias
LLMConfig = LLMProvider
