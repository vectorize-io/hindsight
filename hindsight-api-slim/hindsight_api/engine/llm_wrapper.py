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


def sanitize_llm_output(text: str | None) -> str | None:
    """
    Sanitize text by removing characters that break downstream systems.

    Removes:
    - ASCII control characters (0x00-0x08, 0x0B-0x0C, 0x0E-0x1F, 0x7F): break
      json.loads and PostgreSQL UTF-8 encoding; tab (0x09), newline (0x0A), and
      carriage return (0x0D) are preserved as they are valid in text and JSON.
    - Unicode surrogates (U+D800-U+DFFF): Invalid in UTF-8, break LLM APIs

    Surrogate characters are used in UTF-16 encoding but cannot be encoded
    in UTF-8. They can appear in Python strings from improperly decoded data
    (e.g., from JavaScript or broken files). Control characters commonly appear
    in LLM output embedded inside JSON string values.
    """
    if text is None:
        return None
    if not text:
        return text
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff]", "", text)


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
        "llamacpp",
        "openai-codex",
        "claude-code",
        "mock",
        "none",
        "vertexai",
        "litellm",
        "litellmrouter",
        "bedrock",
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
    extra_body: dict[str, Any] | None = None,
    default_headers: dict[str, str] | None = None,
    vertexai_project_id: str | None = None,
    vertexai_region: str | None = None,
    vertexai_credentials: Any = None,
    gemini_safety_settings: list | None = None,
    litellmrouter_config: dict[str, Any] | None = None,
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
        extra_body: Extra body params merged into OpenAI-compatible API calls.
        default_headers: Custom headers passed as ``default_headers`` to provider SDK clients
            (used by operators routing through proxies / request-tracing middleware). Currently
            wired into the Anthropic provider; other providers may opt in as needed.
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
        LiteLLMLLM,
        LiteLLMRouterLLM,
        LlamaCppLLM,
        MockLLM,
        NoneLLM,
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

    elif provider_lower == "none":
        return NoneLLM(
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
            gemini_safety_settings=gemini_safety_settings,
        )

    elif provider_lower == "anthropic":
        return AnthropicLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            default_headers=default_headers,
        )

    elif provider_lower == "litellm":
        return LiteLLMLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
        )

    elif provider_lower == "litellmrouter":
        if not litellmrouter_config:
            raise ValueError(
                "Provider 'litellmrouter' requires a config object. "
                "Set HINDSIGHT_API_LLM_LITELLMROUTER_CONFIG (or the per-op variant) "
                "to a JSON object accepted by litellm.Router. "
                "See https://docs.litellm.ai/docs/routing."
            )
        return LiteLLMRouterLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            config=litellmrouter_config,
            reasoning_effort=reasoning_effort,
        )

    elif provider_lower == "bedrock":
        # Bedrock is a first-class alias backed by LiteLLM with auto-prefixed model names
        bedrock_model = model if model.startswith("bedrock/") else f"bedrock/{model}"
        return LiteLLMLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=bedrock_model,
            reasoning_effort=reasoning_effort,
        )

    elif provider_lower == "llamacpp":
        from ..config import get_config

        config = get_config()
        return LlamaCppLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            model_path=config.llamacpp_model_path,
            gpu_layers=config.llamacpp_gpu_layers,
            context_size=config.llamacpp_context_size,
            chat_format=config.llamacpp_chat_format,
            no_grammar=config.llamacpp_no_grammar,
            extra_args=config.llamacpp_extra_args,
        )

    elif provider_lower in (
        "openai",
        "groq",
        "ollama",
        "lmstudio",
        "minimax",
        "deepseek",
        "volcano",
        "openrouter",
        "zai",
    ):
        return OpenAICompatibleLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            groq_service_tier=groq_service_tier,
            openai_service_tier=openai_service_tier,
            extra_body=extra_body,
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
        gemini_safety_settings: list | None = None,
        extra_body: dict[str, Any] | None = None,
        default_headers: dict[str, str] | None = None,
        litellmrouter_config: dict[str, Any] | None = None,
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
            gemini_safety_settings: Safety settings for Gemini/VertexAI providers.
            extra_body: Extra body params merged into OpenAI-compatible API calls.
            default_headers: Custom headers passed as ``default_headers`` to provider SDK clients.
                Used by operators routing through proxies / request-tracing middleware. Falls
                back to ``HindsightConfig.llm_default_headers`` (env: ``HINDSIGHT_API_LLM_DEFAULT_HEADERS``)
                when ``None``.
            litellmrouter_config: Provider-specific config for ``provider="litellmrouter"``.
                JSON object passed verbatim to ``litellm.Router(**config)`` — see
                https://docs.litellm.ai/docs/routing. Ignored unless ``provider == "litellmrouter"``.
                When None and the provider is ``litellmrouter``, falls back to
                ``HindsightConfig.llm_litellmrouter_config``.
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.litellmrouter_config = litellmrouter_config
        # Service tiers from hierarchical config (not env vars)
        self.groq_service_tier = groq_service_tier
        self.openai_service_tier = openai_service_tier
        # Gemini safety settings (instance default; can be overridden per-request via context var)
        self.gemini_safety_settings = gemini_safety_settings
        # Extra body params for OpenAI-compatible providers (e.g. chat_template_kwargs)
        self.extra_body = extra_body
        # Default headers passed to provider SDK clients (e.g. proxy auth, request tracing).
        # Same pattern as ``gemini_safety_settings``: explicit override wins; otherwise read
        # the static server-level default from ``HindsightConfig`` via ``_get_raw_config()``.
        self.default_headers = default_headers
        if self.default_headers is None:
            from ..config import _get_raw_config

            try:
                self.default_headers = _get_raw_config().llm_default_headers
            except Exception:
                pass  # Config may not be initialized in test environments

        # Validate provider
        valid_providers = [
            "openai",
            "groq",
            "ollama",
            "gemini",
            "anthropic",
            "lmstudio",
            "llamacpp",
            "vertexai",
            "openai-codex",
            "claude-code",
            "mock",
            "none",
            "minimax",
            "deepseek",
            "litellm",
            "litellmrouter",
            "bedrock",
            "volcano",
            "openrouter",
            "zai",
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
            elif self.provider == "minimax":
                self.base_url = "https://api.minimax.io/v1"
            elif self.provider == "deepseek":
                self.base_url = "https://api.deepseek.com"
            elif self.provider == "openrouter":
                self.base_url = "https://openrouter.ai/api/v1"
            elif self.provider == "zai":
                self.base_url = "https://api.z.ai/api/coding/paas/v4"

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

        # For Gemini/VertexAI providers: read safety settings from global config if not explicitly provided
        # Use _get_raw_config() to bypass StaticConfigProxy (which blocks configurable fields),
        # since LLMProvider initialization legitimately needs the server-level default.
        if self.provider in ("gemini", "vertexai") and self.gemini_safety_settings is None:
            from ..config import _get_raw_config

            try:
                raw_config = _get_raw_config()
                self.gemini_safety_settings = raw_config.llm_gemini_safety_settings
            except Exception:
                pass  # Config may not be initialized in test environments

        # For litellmrouter: prefer an explicit chain from the caller (per-op
        # construction in MemoryEngine threads the right chain through). If the caller
        # didn't supply one, fall back to the global ``llm_litellmrouter_config`` so
        # ad-hoc constructions (e.g. ``LLMProvider.from_env()``) keep working.
        router_config: dict[str, Any] | None = self.litellmrouter_config
        if self.provider == "litellmrouter" and router_config is None:
            from ..config import _get_raw_config

            try:
                router_config = _get_raw_config().llm_litellmrouter_config
            except Exception:
                router_config = None

        # Create provider implementation using factory
        self._provider_impl = create_llm_provider(
            provider=self.provider,
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            groq_service_tier=self.groq_service_tier,
            openai_service_tier=self.openai_service_tier,
            extra_body=self.extra_body,
            default_headers=self.default_headers,
            vertexai_project_id=vertexai_project_id,
            vertexai_region=vertexai_region,
            vertexai_credentials=vertexai_credentials,
            gemini_safety_settings=self.gemini_safety_settings,
            litellmrouter_config=router_config,
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
        # Stage breadcrumb so the worker log shows which LLM call a task is
        # currently inside; the stage_age field then reveals long JSON-schema
        # retry loops (e.g. a small model that can't satisfy strict_schema).
        # No-op outside a worker context.
        from ..worker.stage import set_stage

        structured = "+structured" if response_format is not None else ""
        set_stage(f"llm.{self.provider}.{scope}{structured}")

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
        from ..worker.stage import set_stage

        set_stage(f"llm.{self.provider}.{scope}+tools")

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

    def set_response_callback(self, fn: Any) -> None:
        """Set a callback invoked on each call() instead of the fixed mock response."""
        if self.provider == "mock":
            from .providers.mock_llm import MockLLM

            if isinstance(self._provider_impl, MockLLM):
                self._provider_impl.set_response_callback(fn)

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

            from claude_agent_sdk import query  # noqa: F401  # type: ignore[unresolved-import]

            sdk_logging.getLogger("claude_agent_sdk").setLevel(sdk_logging.WARNING)
            sdk_logging.getLogger("claude_agent_sdk._internal").setLevel(sdk_logging.WARNING)

            logger.debug("Claude Agent SDK imported successfully")
        except ImportError as e:
            raise ImportError(
                "Claude Agent SDK not installed. Run: uv add claude-agent-sdk or pip install claude-agent-sdk"
            ) from e

        # SDK will automatically check for authentication when first used
        # No need to verify here - let it fail gracefully on first call with helpful error

    def with_config(self, config: Any) -> "ConfiguredLLMProvider":
        """
        Return a configured wrapper for a specific bank operation.

        The wrapper applies per-bank overrides (e.g. Gemini safety settings)
        to every ``call()`` / ``call_with_tools()`` invocation without
        changing the underlying provider or its long-lived client connection.

        Args:
            config: Resolved ``HindsightConfig`` for the current bank/request.

        Returns:
            A ``ConfiguredLLMProvider`` that delegates to this provider with
            the supplied config applied.
        """
        return ConfiguredLLMProvider(self, config.llm_gemini_safety_settings)

    async def cleanup(self) -> None:
        """Clean up resources (e.g. stop llamacpp subprocess)."""
        if self._provider_impl:
            await self._provider_impl.cleanup()

    @classmethod
    def from_env(cls) -> "LLMProvider":
        """Create provider from environment variables using config.py constants."""
        from ..config import (
            DEFAULT_LLM_MODEL,
            DEFAULT_LLM_PROVIDER,
            ENV_LLM_API_KEY,
            ENV_LLM_BASE_URL,
            ENV_LLM_DEFAULT_HEADERS,
            ENV_LLM_EXTRA_BODY,
            ENV_LLM_MODEL,
            ENV_LLM_PROVIDER,
        )

        provider = os.getenv(ENV_LLM_PROVIDER, DEFAULT_LLM_PROVIDER)
        api_key = os.getenv(ENV_LLM_API_KEY, "")

        if not api_key and not requires_api_key(provider):
            pass  # Provider handles its own auth
        elif not api_key:
            raise ValueError(
                f"{ENV_LLM_API_KEY} environment variable is required (unless using openai-codex, claude-code, or litellm)"
            )

        base_url = os.getenv(ENV_LLM_BASE_URL, "")
        model = os.getenv(ENV_LLM_MODEL, DEFAULT_LLM_MODEL)
        extra_body = json.loads(os.getenv(ENV_LLM_EXTRA_BODY, "null"))
        default_headers = json.loads(os.getenv(ENV_LLM_DEFAULT_HEADERS, "null"))

        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort="low",
            extra_body=extra_body,
            default_headers=default_headers,
        )


class ConfiguredLLMProvider:
    """
    Thin wrapper around LLMProvider that applies bank-specific config to every call.

    Obtained via ``LLMProvider.with_config(resolved_config)``.  The wrapper
    sets any provider-specific overrides (currently Gemini safety settings)
    immediately before each call using a ContextVar token, then resets it
    afterwards — so nesting is safe and the configuration cannot leak across
    operations.

    All attribute access falls through to the underlying provider so callers
    that read ``llm.provider``, ``llm.model``, etc. continue to work without
    any changes.
    """

    def __init__(self, provider: "LLMProvider", gemini_safety_settings: list | None) -> None:
        # Use object.__setattr__ to avoid triggering __getattr__
        object.__setattr__(self, "_provider", provider)
        object.__setattr__(self, "_gemini_safety_settings", gemini_safety_settings)

    # ── attribute passthrough ──────────────────────────────────────────────────

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_provider"), name)

    # ── overridden call methods ────────────────────────────────────────────────

    async def call(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        from .providers.gemini_llm import _safety_settings_ctx

        token = _safety_settings_ctx.set(object.__getattribute__(self, "_gemini_safety_settings"))
        try:
            return await object.__getattribute__(self, "_provider").call(messages=messages, **kwargs)
        finally:
            _safety_settings_ctx.reset(token)

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> "LLMToolCallResult":
        from .providers.gemini_llm import _safety_settings_ctx

        token = _safety_settings_ctx.set(object.__getattribute__(self, "_gemini_safety_settings"))
        try:
            return await object.__getattribute__(self, "_provider").call_with_tools(
                messages=messages, tools=tools, **kwargs
            )
        finally:
            _safety_settings_ctx.reset(token)


# Backwards compatibility alias
LLMConfig = LLMProvider
