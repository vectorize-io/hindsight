"""
Tests for the ``llm_extra_body`` knob (env: ``HINDSIGHT_API_LLM_EXTRA_BODY``).

The same JSON dict of extra request-body params is threaded into every API
provider. Each provider merges it in its own native parameter space:

- OpenAI-compatible / Fireworks: OpenAI SDK ``extra_body`` (already covered
  elsewhere; the wiring predates this change).
- Anthropic: the Anthropic SDK ``extra_body`` kwarg.
- Gemini / VertexAI: seeded into ``GenerateContentConfig`` (the SDK's native
  generation-param space — Gemini nests these in the request body).
- LiteLLM (+ bedrock alias + router): merged as top-level ``acompletion`` kwargs
  so LiteLLM normalizes/drops them per-provider.

These are deterministic unit tests: the SDK client is mocked and we assert the
params actually reach the call.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

EXTRA_BODY = {"temperature": 0.2, "top_p": 0.9}


# ─── config / env parsing ─────────────────────────────────────────────────────


def test_extra_body_parsed_from_env():
    """The JSON env var is parsed into ``HindsightConfig.llm_extra_body``."""
    import json

    from hindsight_api.config import ENV_LLM_EXTRA_BODY, HindsightConfig, clear_config_cache

    with patch.dict(os.environ, {ENV_LLM_EXTRA_BODY: json.dumps(EXTRA_BODY)}, clear=False):
        clear_config_cache()
        config = HindsightConfig.from_env()
        assert config.llm_extra_body == EXTRA_BODY
        clear_config_cache()


def test_extra_body_default_is_none():
    """When the env var is unset, ``llm_extra_body`` defaults to None."""
    from hindsight_api.config import ENV_LLM_EXTRA_BODY, HindsightConfig, clear_config_cache

    env = {k: v for k, v in os.environ.items() if k != ENV_LLM_EXTRA_BODY}
    with patch.dict(os.environ, env, clear=True):
        clear_config_cache()
        config = HindsightConfig.from_env()
        assert config.llm_extra_body is None
        clear_config_cache()


# ─── Anthropic ────────────────────────────────────────────────────────────────


def _make_anthropic_provider(extra_body=None):
    with patch("anthropic.AsyncAnthropic") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        from hindsight_api.engine.providers.anthropic_llm import AnthropicLLM

        provider = AnthropicLLM(
            provider="anthropic",
            api_key="fake-key",
            base_url="",
            model="claude-sonnet-4-20250514",
            extra_body=extra_body,
        )
    provider._client = MagicMock()
    return provider


def _fake_anthropic_response():
    block = MagicMock()
    block.type = "text"
    block.text = "ok"
    resp = MagicMock()
    resp.content = [block]
    resp.usage = MagicMock(input_tokens=5, output_tokens=2, cache_read_input_tokens=0)
    resp.stop_reason = "end_turn"
    return resp


def test_anthropic_stores_extra_body():
    provider = _make_anthropic_provider(extra_body=EXTRA_BODY)
    assert provider._extra_body == EXTRA_BODY


def test_anthropic_empty_extra_body_defaults_to_dict():
    provider = _make_anthropic_provider(extra_body=None)
    assert provider._extra_body == {}


@pytest.mark.asyncio
async def test_anthropic_call_passes_extra_body():
    """``call()`` forwards extra_body via the Anthropic SDK ``extra_body`` kwarg."""
    provider = _make_anthropic_provider(extra_body=EXTRA_BODY)
    provider._client.messages.create = AsyncMock(return_value=_fake_anthropic_response())

    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        await provider.call(messages=[{"role": "user", "content": "hi"}], scope="test", max_retries=0)

    kwargs = provider._client.messages.create.call_args.kwargs
    assert kwargs.get("extra_body") == EXTRA_BODY


@pytest.mark.asyncio
async def test_anthropic_no_extra_body_omits_key():
    """``call()`` does not pass ``extra_body`` when none is configured."""
    provider = _make_anthropic_provider(extra_body=None)
    provider._client.messages.create = AsyncMock(return_value=_fake_anthropic_response())

    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        await provider.call(messages=[{"role": "user", "content": "hi"}], scope="test", max_retries=0)

    assert "extra_body" not in provider._client.messages.create.call_args.kwargs


# ─── Gemini ───────────────────────────────────────────────────────────────────


def _make_gemini_provider(extra_body=None, gemini_service_tier=None):
    pytest.importorskip("google.genai")
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        from hindsight_api.engine.providers.gemini_llm import GeminiLLM

        provider = GeminiLLM(
            provider="gemini",
            api_key="fake-key",
            base_url="",
            model="gemini-2.5-flash",
            extra_body=extra_body,
            gemini_service_tier=gemini_service_tier,
        )
    provider._client = MagicMock()
    return provider


def _fake_gemini_response():
    r = MagicMock()
    r.text = "hello"
    r.candidates = [MagicMock(finish_reason="STOP")]
    r.usage_metadata = MagicMock(prompt_token_count=5, candidates_token_count=2)
    return r


def test_gemini_stores_extra_body():
    provider = _make_gemini_provider(extra_body=EXTRA_BODY)
    assert provider._extra_body == EXTRA_BODY


@pytest.mark.asyncio
async def test_gemini_call_applies_extra_body_to_generation_config():
    """``call()`` seeds extra_body into GenerateContentConfig (temperature/top_p)."""
    provider = _make_gemini_provider(extra_body=EXTRA_BODY)
    provider._client.aio.models.generate_content = AsyncMock(return_value=_fake_gemini_response())

    await provider.call(messages=[{"role": "user", "content": "hi"}], scope="test")

    config_arg = provider._client.aio.models.generate_content.call_args.kwargs.get("config")
    assert config_arg is not None
    assert config_arg.temperature == 0.2
    assert config_arg.top_p == 0.9


@pytest.mark.asyncio
async def test_gemini_explicit_temperature_overrides_extra_body():
    """An explicit per-call temperature wins over the extra_body default."""
    provider = _make_gemini_provider(extra_body={"temperature": 0.2})
    provider._client.aio.models.generate_content = AsyncMock(return_value=_fake_gemini_response())

    await provider.call(messages=[{"role": "user", "content": "hi"}], temperature=0.9, scope="test")

    config_arg = provider._client.aio.models.generate_content.call_args.kwargs.get("config")
    assert config_arg.temperature == 0.9


@pytest.mark.asyncio
async def test_gemini_service_tier_applies_to_http_options_extra_body():
    """The native Gemini service tier flag reaches GenerateContentConfig."""
    provider = _make_gemini_provider(gemini_service_tier="flex")
    provider._client.aio.models.generate_content = AsyncMock(return_value=_fake_gemini_response())

    await provider.call(messages=[{"role": "user", "content": "hi"}], scope="test")

    config_arg = provider._client.aio.models.generate_content.call_args.kwargs.get("config")
    assert config_arg.http_options.extra_body["service_tier"] == "flex"


@pytest.mark.asyncio
async def test_gemini_extra_body_service_tier_takes_precedence():
    """The explicit extra_body escape hatch wins over the native flag."""
    provider = _make_gemini_provider(
        extra_body={"http_options": {"extra_body": {"service_tier": "standard"}}},
        gemini_service_tier="flex",
    )
    provider._client.aio.models.generate_content = AsyncMock(return_value=_fake_gemini_response())

    await provider.call(messages=[{"role": "user", "content": "hi"}], scope="test")

    config_arg = provider._client.aio.models.generate_content.call_args.kwargs.get("config")
    assert config_arg.http_options.extra_body["service_tier"] == "standard"
    assert provider._extra_body["http_options"]["extra_body"]["service_tier"] == "standard"


# ─── LiteLLM ──────────────────────────────────────────────────────────────────


def _make_litellm_provider(extra_body=None):
    pytest.importorskip("litellm")
    from hindsight_api.engine.providers.litellm_llm import LiteLLMLLM

    return LiteLLMLLM(
        provider="litellm",
        api_key="fake-key",
        base_url="",
        model="gpt-4o",
        extra_body=extra_body,
    )


def _fake_litellm_response():
    msg = MagicMock()
    msg.content = "ok"
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=5, completion_tokens=2)
    return resp


def test_litellm_stores_extra_body():
    provider = _make_litellm_provider(extra_body=EXTRA_BODY)
    assert provider._extra_body == EXTRA_BODY


@pytest.mark.asyncio
async def test_litellm_call_merges_extra_body_as_top_level_kwargs():
    """``call()`` merges extra_body into the acompletion kwargs."""
    provider = _make_litellm_provider(extra_body=EXTRA_BODY)
    provider._acompletion = AsyncMock(return_value=_fake_litellm_response())

    with patch("hindsight_api.engine.providers.litellm_llm.get_metrics_collector"):
        await provider.call(messages=[{"role": "user", "content": "hi"}], scope="test", max_retries=0)

    kwargs = provider._acompletion.call_args.kwargs
    assert kwargs.get("temperature") == 0.2
    assert kwargs.get("top_p") == 0.9


@pytest.mark.asyncio
async def test_litellm_explicit_param_wins_over_extra_body():
    """``setdefault`` semantics: an explicit per-call value is not overwritten."""
    provider = _make_litellm_provider(extra_body={"temperature": 0.2})
    provider._acompletion = AsyncMock(return_value=_fake_litellm_response())

    with patch("hindsight_api.engine.providers.litellm_llm.get_metrics_collector"):
        await provider.call(messages=[{"role": "user", "content": "hi"}], temperature=0.9, scope="test", max_retries=0)

    assert provider._acompletion.call_args.kwargs.get("temperature") == 0.9


def test_litellm_router_forwards_extra_body():
    """The Router subclass forwards extra_body through to the shared LiteLLM base."""
    pytest.importorskip("litellm")
    from hindsight_api.engine.providers.litellm_router_llm import LiteLLMRouterLLM

    config = {"model_list": [{"model_name": "m", "litellm_params": {"model": "gpt-4o", "api_key": "x"}}]}
    provider = LiteLLMRouterLLM(
        provider="litellmrouter",
        api_key="",
        base_url="",
        model="m",
        config=config,
        extra_body=EXTRA_BODY,
    )
    assert provider._extra_body == EXTRA_BODY
