"""Plumbing tests for the per-scope LLM timeout (HINDSIGHT_API_{RETAIN,REFLECT,CONSOLIDATION}_LLM_TIMEOUT).

These assert the *wiring* — that a configured timeout actually reaches the
constructed provider — rather than just that the env var parses into config
(covered by test_config_validation.py). The value is threaded
config -> LLMProvider -> create_llm_provider -> LiteLLMLLM, and LiteLLMLLM wraps
each request in ``asyncio.wait_for(timeout)`` (see test_litellm_timeout.py).

Before this wiring, the per-op timeouts were resolved into config but never passed
to the provider, so ``retain``/``reflect``/``consolidation`` calls silently fell
back to the global ``HINDSIGHT_API_LLM_TIMEOUT`` default and ignored the
operator-set value (issue #2452). If MemoryEngine ever stops passing ``timeout``
through, the per-op knob goes inert again — these checks guard that bridge.
"""

from hindsight_api.config import DEFAULT_LLM_TIMEOUT, ENV_LLM_TIMEOUT
from hindsight_api.engine.llm_wrapper import LLMConfig, create_llm_provider

_LITELLM_MODEL = "litellm_proxy/test-model"


def test_llm_config_threads_timeout_to_provider_impl():
    """End-to-end: LLMConfig -> create_llm_provider -> LiteLLMLLM carries the timeout."""
    llm = LLMConfig(
        provider="litellm",
        api_key="k",
        base_url="",
        model=_LITELLM_MODEL,
        timeout=300.0,
    )
    assert llm._provider_impl.timeout == 300.0


def test_create_llm_provider_threads_timeout():
    """The factory forwards ``timeout`` into the LiteLLM provider."""
    impl = create_llm_provider(
        provider="litellm",
        api_key="k",
        base_url="",
        model=_LITELLM_MODEL,
        reasoning_effort="low",
        timeout=42.0,
    )
    assert impl.timeout == 42.0


def test_bedrock_alias_threads_timeout():
    """The ``bedrock/`` LiteLLM alias path also carries the timeout."""
    llm = LLMConfig(
        provider="bedrock",
        api_key="",
        base_url="",
        model="us.amazon.nova-2-lite-v1:0",
        timeout=77.0,
    )
    assert llm._provider_impl.timeout == 77.0


def test_unset_timeout_falls_back_to_default(monkeypatch):
    """``None`` (no per-op or global override) resolves to the finite default,
    never ``None`` — preserving the existing hard-timeout backstop."""
    monkeypatch.delenv(ENV_LLM_TIMEOUT, raising=False)
    llm = LLMConfig(
        provider="litellm",
        api_key="k",
        base_url="",
        model=_LITELLM_MODEL,
    )
    assert llm._provider_impl.timeout == DEFAULT_LLM_TIMEOUT
