"""Base URL normalization for local OpenAI-compatible providers (LM Studio, Ollama).

LM Studio's server UI advertises its address as a bare host (``http://localhost:1234``),
so users commonly set ``HINDSIGHT_API_LLM_BASE_URL`` to that. Without normalization the
OpenAI SDK POSTs to ``<host>/chat/completions`` and LM Studio rejects it with
``Unexpected endpoint or method`` — its OpenAI-compatible routes live under ``/v1``. See #2922.
"""

import pytest

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM


def _base_url(provider: str, base_url: str, api_key: str = "") -> str:
    return OpenAICompatibleLLM(provider=provider, api_key=api_key, base_url=base_url, model="test-model").base_url


@pytest.mark.parametrize("provider", ["lmstudio", "ollama"])
@pytest.mark.parametrize(
    ("given", "expected"),
    [
        # Bare host (the #2922 footgun) — /v1 is appended.
        ("http://localhost:1234", "http://localhost:1234/v1"),
        # Lone trailing slash is treated as "no path".
        ("http://localhost:1234/", "http://localhost:1234/v1"),
        # Already correct — left unchanged.
        ("http://localhost:1234/v1", "http://localhost:1234/v1"),
        # Explicit trailing slash on /v1 is preserved (SDK strips it anyway).
        ("http://localhost:1234/v1/", "http://localhost:1234/v1/"),
        # An explicit reverse-proxy mount is a deliberate path — untouched.
        ("http://proxy.internal/lmstudio", "http://proxy.internal/lmstudio"),
    ],
)
def test_local_provider_base_url_normalized(provider: str, given: str, expected: str):
    assert _base_url(provider, given) == expected


def test_lmstudio_bare_host_targets_v1_chat_completions():
    """The constructed OpenAI client must POST to /v1/chat/completions, not /chat/completions."""
    llm = OpenAICompatibleLLM(provider="lmstudio", api_key="", base_url="http://localhost:1234", model="qwen")
    assert str(llm._client.base_url).rstrip("/") == "http://localhost:1234/v1"


def test_non_local_provider_base_url_untouched():
    """Cloud/proxy providers keep an explicit host with no path verbatim (path is provider-specific)."""
    assert _base_url("openrouter", "https://gateway.example.com", api_key="sk-x") == "https://gateway.example.com"
