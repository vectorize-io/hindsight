"""A configured reasoning effort is honoured, or reported — in every provider (#3449).

Issue #3449 was filed against the OpenAI-compatible lane, where the setting was gated on
a model-name match and silently dropped. The same setting was dead weight in three more
lanes for a different reason: they accepted `reasoning_effort` into the constructor and
never looked at it again. From the operator's seat the symptom is identical — the
variable is set, documented and visible in the environment, and nothing happens.

Two outcomes are acceptable, and nothing else:
  * **honoured** — LiteLLM (and its Router subclass) forwards it, and LiteLLM translates
    it per target provider (Anthropic thinking budgets, Gemini thinking config, OpenAI's
    flat parameter). `litellm.drop_params = True` discards it for models with no
    reasoning knob rather than raising.
  * **reported** — providers with no reasoning control at all (Gemini and Anthropic
    native SDKs, Claude Code) log a WARNING at startup naming the ignored value.
"""

import logging

import pytest

from hindsight_api.engine.providers.litellm_llm import LiteLLMLLM


def _make_litellm(reasoning_effort: str | None) -> LiteLLMLLM:
    return LiteLLMLLM(
        provider="litellm",
        api_key="unused",
        base_url="http://localhost:0/v1",
        model="litellm_proxy/test-model",
        reasoning_effort=reasoning_effort,
    )


class TestLiteLLMForwardsTheSetting:
    @pytest.mark.parametrize("effort", ["none", "low", "high"])
    def test_configured_effort_reaches_the_completion_kwargs(self, effort):
        # Intention: the lane that CAN honour it must actually do so. LiteLLM owns the
        # per-provider translation, so forwarding the value is the whole fix here.
        # Expected: the value arrives verbatim, including "none".
        kwargs = _make_litellm(effort)._build_common_kwargs(messages=[{"role": "user", "content": "hi"}])
        assert kwargs["reasoning_effort"] == effort

    def test_unconfigured_effort_is_not_forwarded(self):
        # Intention: unset means unset here too — no level invented for the target model.
        # Expected: absent, so the model runs at its own default effort.
        kwargs = _make_litellm(None)._build_common_kwargs(messages=[{"role": "user", "content": "hi"}])
        assert "reasoning_effort" not in kwargs

    def test_the_router_lane_forwards_it_too(self):
        # Intention: LiteLLMRouterLLM builds its own kwargs (entrypoint model, no
        # api_key/base_url), so it does NOT inherit the forwarding — the setting has to
        # be added there as well. A divergent builder silently drops it on the router
        # lane alone, which is exactly the asymmetry #3449 was about.
        from hindsight_api.engine.providers.litellm_router_llm import LiteLLMRouterLLM

        router = LiteLLMRouterLLM(
            provider="litellmrouter",
            api_key="unused",
            base_url="",
            model="default",
            config={"model_list": [{"model_name": "default", "litellm_params": {"model": "openai/gpt-4o-mini"}}]},
            reasoning_effort="high",
        )
        kwargs = router._build_common_kwargs(messages=[{"role": "user", "content": "hi"}])
        assert kwargs["reasoning_effort"] == "high"


class TestProvidersWithoutReasoningControlSaySo:
    """These providers cannot act on the setting; they must not swallow it in silence."""

    def test_gemini_warns(self, caplog):
        from hindsight_api.engine.providers.gemini_llm import GeminiLLM

        with caplog.at_level(logging.WARNING):
            GeminiLLM(provider="gemini", api_key="k", base_url="", model="gemini-2.5-flash", reasoning_effort="high")
        assert any("reasoning_effort" in r.message and "ignored" in r.message for r in caplog.records)

    def test_anthropic_warns(self, caplog):
        from hindsight_api.engine.providers.anthropic_llm import AnthropicLLM

        with caplog.at_level(logging.WARNING):
            AnthropicLLM(
                provider="anthropic",
                api_key="k",
                base_url="",
                model="claude-sonnet-4-20250514",
                reasoning_effort="high",
            )
        assert any("reasoning_effort" in r.message and "ignored" in r.message for r in caplog.records)

    def test_nothing_is_logged_when_nothing_was_configured(self, caplog):
        # Intention: the warning marks a *discarded operator setting*, not a default.
        # Expected: silence when the operator configured nothing.
        from hindsight_api.engine.providers.gemini_llm import GeminiLLM

        with caplog.at_level(logging.WARNING):
            GeminiLLM(provider="gemini", api_key="k", base_url="", model="gemini-2.5-flash", reasoning_effort=None)
        assert not [r for r in caplog.records if "reasoning_effort" in r.message]
