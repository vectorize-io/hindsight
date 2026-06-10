"""Plumbing tests for the Bedrock service tier (HINDSIGHT_API_LLM_BEDROCK_SERVICE_TIER).

These assert the *wiring* — that a configured tier actually reaches the LiteLLM
call kwargs — rather than just that the env var parses into config (covered by
test_config_validation.py). The config value is threaded
config -> LLMProvider -> create_llm_provider -> LiteLLMLLM, and only LiteLLMLLM
injects ``service_tier`` for ``bedrock/`` models, so the checks live here.
"""

from hindsight_api.engine.llm_wrapper import LLMConfig
from hindsight_api.engine.providers.litellm_llm import LiteLLMLLM

_MESSAGES = [{"role": "user", "content": "hi"}]


def _make_litellm(model: str, tier: str | None) -> LiteLLMLLM:
    return LiteLLMLLM(provider="bedrock", api_key="", base_url="", model=model, bedrock_service_tier=tier)


def test_bedrock_model_injects_service_tier():
    """A configured tier is injected as ``service_tier`` for bedrock/ models."""
    llm = _make_litellm("bedrock/us.amazon.nova-2-lite-v1:0", "flex")
    kwargs = llm._build_common_kwargs(messages=_MESSAGES)
    assert kwargs["service_tier"] == "flex"


def test_bedrock_model_without_tier_omits_service_tier():
    """No tier configured -> no ``service_tier`` key (Bedrock default tier)."""
    llm = _make_litellm("bedrock/us.amazon.nova-2-lite-v1:0", None)
    kwargs = llm._build_common_kwargs(messages=_MESSAGES)
    assert "service_tier" not in kwargs


def test_non_bedrock_model_never_gets_service_tier():
    """The bedrock/ prefix guard keeps the kwarg off non-Bedrock LiteLLM models."""
    llm = LiteLLMLLM(
        provider="litellm",
        api_key="k",
        base_url="",
        model="fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct",
        bedrock_service_tier="flex",
    )
    kwargs = llm._build_common_kwargs(messages=_MESSAGES)
    assert "service_tier" not in kwargs


def test_llm_config_threads_tier_to_provider_impl():
    """End-to-end: LLMConfig -> create_llm_provider -> LiteLLMLLM carries the tier.

    This is the bridge the env var depends on; if MemoryEngine ever stops
    passing ``bedrock_service_tier`` through, the value silently defaults to
    None and the flag becomes inert.
    """
    llm = LLMConfig(
        provider="bedrock",
        api_key="",
        base_url="",
        model="us.amazon.nova-2-lite-v1:0",
        bedrock_service_tier="flex",
    )
    assert llm._provider_impl.bedrock_service_tier == "flex"
