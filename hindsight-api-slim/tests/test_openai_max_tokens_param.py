"""
Tests for OpenAICompatibleLLM._max_tokens_param_name.

Regression coverage for issue #978: Azure OpenAI + GPT-5 models were failing with
"'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."
because PR #858 started sending 'max_tokens' whenever the openai provider had a
custom base_url. Reasoning models only accept 'max_completion_tokens', and Azure
OpenAI is fully OpenAI-API-compatible, so both cases must keep using the new
parameter name.
"""

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM


def _make(provider: str, model: str, base_url: str = "") -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider=provider,
        api_key="test-key",
        base_url=base_url,
        model=model,
    )


class TestMaxTokensParamName:
    def test_native_openai_uses_max_completion_tokens(self):
        llm = _make("openai", "gpt-4o-mini")
        assert llm._max_tokens_param_name() == "max_completion_tokens"

    def test_openai_custom_base_url_falls_back_to_max_tokens(self):
        """Mistral/Together-style OpenAI-compatible endpoints need max_tokens (PR #858)."""
        llm = _make("openai", "mistral-large-latest", base_url="https://api.mistral.ai/v1")
        assert llm._max_tokens_param_name() == "max_tokens"

    def test_azure_openai_uses_max_completion_tokens(self):
        """Regression for #978: Azure is fully OpenAI-API-compatible, not a third-party clone."""
        llm = _make(
            "openai",
            "gpt-4o-mini",
            base_url="https://my-resource.openai.azure.com/openai/v1/",
        )
        assert llm._max_tokens_param_name() == "max_completion_tokens"

    def test_reasoning_model_always_uses_max_completion_tokens(self):
        """Regression for #978: GPT-5/o1/o3 reject max_tokens outright, base_url must not matter."""
        # Azure + GPT-5 (exact reporter setup)
        azure_gpt5 = _make(
            "openai",
            "gpt-5.4-nano",
            base_url="https://my-resource.openai.azure.com/openai/v1/",
        )
        assert azure_gpt5._max_tokens_param_name() == "max_completion_tokens"

        # Even a Mistral-style custom base_url must not downgrade a reasoning model
        for model in ("gpt-5", "gpt-5-mini", "o1-mini", "o3", "deepseek-r1"):
            llm = _make("openai", model, base_url="https://some-proxy.example.com/v1")
            assert llm._max_tokens_param_name() == "max_completion_tokens", model

    def test_groq_uses_max_completion_tokens(self):
        llm = _make("groq", "openai/gpt-oss-120b", base_url="https://api.groq.com/openai/v1")
        assert llm._max_tokens_param_name() == "max_completion_tokens"

    def test_llamacpp_uses_max_completion_tokens(self):
        llm = _make("llamacpp", "some-model", base_url="http://localhost:8080/v1")
        assert llm._max_tokens_param_name() == "max_completion_tokens"

    def test_ollama_uses_max_tokens(self):
        llm = _make("ollama", "gemma3:12b", base_url="http://localhost:11434/v1")
        assert llm._max_tokens_param_name() == "max_tokens"

    def test_lmstudio_uses_max_tokens(self):
        llm = _make("lmstudio", "openai/gpt-oss-20b", base_url="http://localhost:1234/v1")
        assert llm._max_tokens_param_name() == "max_tokens"
