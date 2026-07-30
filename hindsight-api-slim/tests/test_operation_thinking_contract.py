"""Contract tests for explicit per-operation thinking controls."""

from dataclasses import replace

from hindsight_api.config import HindsightConfig, LLMMemberConfig, LLMStrategyConfig
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.memory_engine import (
    _LLMCallDefaults,
    _build_llm,
    _operation_extra_body,
)
from hindsight_api.engine.multi_llm import MultiLLMProvider


_NO_CALL_DEFAULTS = _LLMCallDefaults(
    timeout=None, max_retries=None, initial_backoff=None, max_backoff=None
)


# ── _operation_extra_body unit tests ───────────────────────────────────────────


def test_operation_extra_body_sets_thinking_for_openai_compatible_provider():
    configured = {
        "temperature": 0.2,
        "chat_template_kwargs": {"foo": "bar"},
    }

    result = _operation_extra_body(configured, enable_thinking=True, provider="openai")

    assert result == {
        "temperature": 0.2,
        "chat_template_kwargs": {"foo": "bar", "enable_thinking": True},
    }
    # Original is not mutated.
    assert configured == {
        "temperature": 0.2,
        "chat_template_kwargs": {"foo": "bar"},
    }


def test_operation_extra_body_skips_non_openai_compatible_provider():
    """Gemini/Vertex/Anthropic/LiteLLM must not receive chat_template_kwargs."""
    for non_oai in ("gemini", "vertexai", "anthropic", "litellm", "litellmrouter", "bedrock"):
        result = _operation_extra_body(
            {"temperature": 0.2}, enable_thinking=True, provider=non_oai
        )
        assert result == {"temperature": 0.2}, f"provider={non_oai}"
        assert "chat_template_kwargs" not in result, f"provider={non_oai}"


def test_operation_extra_body_none_provider_is_safe_noop():
    result = _operation_extra_body(
        {"temperature": 0.2}, enable_thinking=True, provider=None
    )
    assert result == {"temperature": 0.2}


def test_operation_extra_body_replaces_malformed_template_kwargs():
    assert _operation_extra_body(
        {"chat_template_kwargs": "invalid"}, enable_thinking=False, provider="ollama"
    ) == {"chat_template_kwargs": {"enable_thinking": False}}


# ── _build_llm integration tests ───────────────────────────────────────────────


def _member(provider="ollama", extra_body=None):
    return LLMMemberConfig(
        provider=provider,
        api_key=None,
        model="m",
        base_url=None,
        reasoning_effort=None,
        extra_body=extra_body,
        default_headers=None,
        bedrock_service_tier=None,
        gemini_service_tier=None,
    )


def test_build_llm_applies_disabled_thinking_to_failover_members():
    member = _member(provider="ollama", extra_body={"temperature": 0.1})
    config = replace(
        HindsightConfig.from_env(),
        llm_members=[member],
        llm_strategy=LLMStrategyConfig(mode="failover"),
    )
    base = LLMProvider(provider="mock", api_key="", base_url="", model="primary")

    result = _build_llm(base, config, "", _NO_CALL_DEFAULTS, enable_thinking=False)

    assert isinstance(result, MultiLLMProvider)
    assert result.members[1].extra_body == {
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_build_llm_applies_enabled_thinking_to_reflect_members():
    member = _member(provider="ollama", extra_body=None)
    config = replace(
        HindsightConfig.from_env(),
        llm_members=[member],
        llm_strategy=LLMStrategyConfig(mode="failover"),
    )
    base = LLMProvider(provider="mock", api_key="", base_url="", model="reflect")

    result = _build_llm(base, config, "", _NO_CALL_DEFAULTS, enable_thinking=True)

    assert isinstance(result, MultiLLMProvider)
    assert result.members[1].extra_body == {
        "chat_template_kwargs": {"enable_thinking": True}
    }
