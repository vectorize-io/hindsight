"""
Tests for Gemini safety settings feature.

Verifies that:
- Safety settings are read from env var and stored on GeminiLLM instances
- Settings are applied to GenerateContentConfig in call() and call_with_tools()
- The context variable override allows per-bank settings at request time
- None (unset) means Gemini's default safety settings are used (no override)
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("google.genai")


SAMPLE_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]


# ─── Config / env var parsing ─────────────────────────────────────────────────


def test_gemini_safety_settings_parsed_from_env():
    """Safety settings JSON from env var is parsed into HindsightConfig."""
    import json

    from hindsight_api.config import ENV_LLM_GEMINI_SAFETY_SETTINGS, HindsightConfig, clear_config_cache

    settings_json = json.dumps(SAMPLE_SAFETY_SETTINGS)
    with patch.dict(os.environ, {ENV_LLM_GEMINI_SAFETY_SETTINGS: settings_json}, clear=False):
        clear_config_cache()
        config = HindsightConfig.from_env()
        assert config.llm_gemini_safety_settings == SAMPLE_SAFETY_SETTINGS
        clear_config_cache()


def test_gemini_safety_settings_default_is_none():
    """When env var is not set, llm_gemini_safety_settings defaults to None."""
    from hindsight_api.config import ENV_LLM_GEMINI_SAFETY_SETTINGS, HindsightConfig, clear_config_cache

    env = {k: v for k, v in os.environ.items() if k != ENV_LLM_GEMINI_SAFETY_SETTINGS}
    with patch.dict(os.environ, env, clear=True):
        clear_config_cache()
        config = HindsightConfig.from_env()
        assert config.llm_gemini_safety_settings is None
        clear_config_cache()


def test_gemini_safety_settings_is_configurable_field():
    """llm_gemini_safety_settings appears in configurable (per-bank) fields."""
    from hindsight_api.config import HindsightConfig

    assert "llm_gemini_safety_settings" in HindsightConfig.get_configurable_fields()


def test_gemini_safety_settings_not_in_credential_fields():
    """llm_gemini_safety_settings is NOT a credential — it is safe to expose via API."""
    from hindsight_api.config import HindsightConfig

    assert "llm_gemini_safety_settings" not in HindsightConfig.get_credential_fields()


# ─── GeminiLLM instance ───────────────────────────────────────────────────────


def _make_gemini_provider(safety_settings=None):
    """Return a GeminiLLM instance with a mocked genai.Client."""
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        from hindsight_api.engine.providers.gemini_llm import GeminiLLM

        provider = GeminiLLM(
            provider="gemini",
            api_key="fake-api-key",
            base_url="",
            model="gemini-2.5-flash",
            gemini_safety_settings=safety_settings,
        )
        # Replace client with a fresh mock so we can inspect calls
        provider._client = MagicMock()
        return provider


def test_gemini_llm_stores_safety_settings():
    """GeminiLLM stores safety settings passed at construction."""
    provider = _make_gemini_provider(safety_settings=SAMPLE_SAFETY_SETTINGS)
    assert provider._safety_settings == SAMPLE_SAFETY_SETTINGS


def test_gemini_llm_no_safety_settings_is_none():
    """GeminiLLM._safety_settings is None when not provided."""
    provider = _make_gemini_provider(safety_settings=None)
    assert provider._safety_settings is None


# ─── call() applies safety settings ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_applies_safety_settings():
    """call() includes safety_settings in GenerateContentConfig when configured."""
    from google.genai import types as genai_types

    provider = _make_gemini_provider(safety_settings=SAMPLE_SAFETY_SETTINGS)

    # Build a fake successful response
    fake_response = MagicMock()
    fake_response.text = "hello"
    fake_response.candidates = [MagicMock(finish_reason="STOP")]
    fake_response.usage_metadata = MagicMock(prompt_token_count=5, candidates_token_count=2)

    provider._client.aio.models.generate_content = AsyncMock(return_value=fake_response)

    await provider.call(
        messages=[{"role": "user", "content": "hi"}],
        scope="test",
    )

    # Inspect the config passed to generate_content
    call_args = provider._client.aio.models.generate_content.call_args
    config_arg = call_args.kwargs.get("config") or call_args.args[0] if call_args.args else None
    # config may be in kwargs or positional; grab from kwargs
    config_arg = call_args.kwargs.get("config")

    assert config_arg is not None, "GenerateContentConfig should have been passed"
    assert hasattr(config_arg, "safety_settings"), "Config should have safety_settings"
    assert config_arg.safety_settings is not None

    categories = [s.category.value if hasattr(s.category, "value") else str(s.category) for s in config_arg.safety_settings]
    assert "HARM_CATEGORY_HARASSMENT" in categories
    assert "HARM_CATEGORY_HATE_SPEECH" in categories
    assert "HARM_CATEGORY_SEXUALLY_EXPLICIT" in categories
    assert "HARM_CATEGORY_DANGEROUS_CONTENT" in categories

    thresholds = [s.threshold.value if hasattr(s.threshold, "value") else str(s.threshold) for s in config_arg.safety_settings]
    assert all(t == "BLOCK_NONE" for t in thresholds)


@pytest.mark.asyncio
async def test_call_no_safety_settings_omits_key():
    """call() does NOT add safety_settings to GenerateContentConfig when none configured."""
    provider = _make_gemini_provider(safety_settings=None)

    fake_response = MagicMock()
    fake_response.text = "hello"
    fake_response.candidates = [MagicMock(finish_reason="STOP")]
    fake_response.usage_metadata = MagicMock(prompt_token_count=5, candidates_token_count=2)

    provider._client.aio.models.generate_content = AsyncMock(return_value=fake_response)

    await provider.call(
        messages=[{"role": "user", "content": "hi"}],
        scope="test",
    )

    call_args = provider._client.aio.models.generate_content.call_args
    config_arg = call_args.kwargs.get("config")

    # When no safety settings, config is either None or lacks safety_settings
    if config_arg is not None:
        assert not hasattr(config_arg, "safety_settings") or config_arg.safety_settings is None


# ─── call_with_tools() applies safety settings ────────────────────────────────


@pytest.mark.asyncio
async def test_call_with_tools_applies_safety_settings():
    """call_with_tools() includes safety_settings in GenerateContentConfig."""
    provider = _make_gemini_provider(safety_settings=SAMPLE_SAFETY_SETTINGS)

    # Build a fake tool-use response (no tool calls, just text)
    fake_part = MagicMock()
    fake_part.text = "answer"
    fake_part.function_call = None

    fake_candidate = MagicMock()
    fake_candidate.content = MagicMock(parts=[fake_part])

    fake_response = MagicMock()
    fake_response.candidates = [fake_candidate]
    fake_response.usage_metadata = MagicMock(prompt_token_count=5, candidates_token_count=3)

    provider._client.aio.models.generate_content = AsyncMock(return_value=fake_response)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]

    await provider.call_with_tools(
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        scope="test",
    )

    call_args = provider._client.aio.models.generate_content.call_args
    config_arg = call_args.kwargs.get("config")

    assert config_arg is not None
    assert config_arg.safety_settings is not None
    categories = [s.category.value if hasattr(s.category, "value") else str(s.category) for s in config_arg.safety_settings]
    assert "HARM_CATEGORY_HARASSMENT" in categories


# ─── with_config() override ───────────────────────────────────────────────────


def _make_llm_provider(safety_settings=None):
    """Return an LLMProvider (wrapping GeminiLLM) with a mocked genai.Client."""
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        from hindsight_api.engine.llm_wrapper import LLMProvider

        provider = LLMProvider(
            provider="gemini",
            api_key="fake-api-key",
            base_url="",
            model="gemini-2.5-flash",
            gemini_safety_settings=safety_settings,
        )
        # Replace the underlying Gemini client with a fresh mock
        provider._provider_impl._client = MagicMock()
        return provider


def _fake_response():
    r = MagicMock()
    r.text = "hello"
    r.candidates = [MagicMock(finish_reason="STOP")]
    r.usage_metadata = MagicMock(prompt_token_count=5, candidates_token_count=2)
    return r


def _make_config(safety_settings):
    """Return a minimal config-like object with llm_gemini_safety_settings."""
    cfg = MagicMock()
    cfg.llm_gemini_safety_settings = safety_settings
    return cfg


@pytest.mark.asyncio
async def test_with_config_overrides_instance_settings():
    """with_config() settings take precedence over the provider instance defaults."""
    instance_settings = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"}]
    override_settings = [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}]

    provider = _make_llm_provider(safety_settings=instance_settings)
    provider._provider_impl._client.aio.models.generate_content = AsyncMock(return_value=_fake_response())

    configured = provider.with_config(_make_config(override_settings))
    await configured.call(messages=[{"role": "user", "content": "hi"}], scope="test")

    config_arg = provider._provider_impl._client.aio.models.generate_content.call_args.kwargs.get("config")
    assert config_arg is not None
    categories = [s.category.value if hasattr(s.category, "value") else str(s.category) for s in config_arg.safety_settings]
    # Should use override_settings (HATE_SPEECH), not instance_settings (HARASSMENT)
    assert "HARM_CATEGORY_HATE_SPEECH" in categories
    assert "HARM_CATEGORY_HARASSMENT" not in categories


@pytest.mark.asyncio
async def test_with_config_none_falls_back_to_instance():
    """When with_config() supplies None, the instance default is used."""
    instance_settings = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}]

    provider = _make_llm_provider(safety_settings=instance_settings)
    provider._provider_impl._client.aio.models.generate_content = AsyncMock(return_value=_fake_response())

    configured = provider.with_config(_make_config(None))
    await configured.call(messages=[{"role": "user", "content": "hi"}], scope="test")

    config_arg = provider._provider_impl._client.aio.models.generate_content.call_args.kwargs.get("config")
    assert config_arg is not None
    categories = [s.category.value if hasattr(s.category, "value") else str(s.category) for s in config_arg.safety_settings]
    assert "HARM_CATEGORY_HARASSMENT" in categories


@pytest.mark.asyncio
async def test_with_config_resets_after_call():
    """The ContextVar is properly reset after a with_config() call (no leakage)."""
    from hindsight_api.engine.providers.gemini_llm import _safety_settings_ctx

    settings = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}]
    provider = _make_llm_provider(safety_settings=None)
    provider._provider_impl._client.aio.models.generate_content = AsyncMock(return_value=_fake_response())

    before = _safety_settings_ctx.get()
    configured = provider.with_config(_make_config(settings))
    await configured.call(messages=[{"role": "user", "content": "hi"}], scope="test")
    after = _safety_settings_ctx.get()

    assert after == before  # ContextVar restored to its original value


# ─── LLMProvider reads safety settings from config ────────────────────────────


def test_llm_provider_reads_safety_settings_from_config():
    """LLMProvider reads llm_gemini_safety_settings from global config for Gemini provider."""
    import json

    from hindsight_api.config import ENV_LLM_GEMINI_SAFETY_SETTINGS, clear_config_cache

    settings_json = json.dumps(SAMPLE_SAFETY_SETTINGS)
    env_overrides = {
        "HINDSIGHT_API_LLM_PROVIDER": "gemini",
        "HINDSIGHT_API_LLM_API_KEY": "fake-key",
        ENV_LLM_GEMINI_SAFETY_SETTINGS: settings_json,
    }

    with patch.dict(os.environ, env_overrides, clear=False):
        clear_config_cache()
        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()
            from hindsight_api.engine.llm_wrapper import LLMProvider

            provider = LLMProvider(
                provider="gemini",
                api_key="fake-key",
                base_url="",
                model="gemini-2.5-flash",
            )

            assert provider.gemini_safety_settings == SAMPLE_SAFETY_SETTINGS

        clear_config_cache()
