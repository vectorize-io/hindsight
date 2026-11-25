"""OpenRouter app attribution defaults (#6888)."""

from hindsight_api.engine.providers.openai_compatible_headers import OPENROUTER_ATTRIBUTION_HEADERS
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM


def _headers(provider: str, default_headers: dict[str, str] | None = None) -> dict[str, str]:
    llm = OpenAICompatibleLLM(
        provider=provider,
        api_key="test-key",
        base_url="https://example.test/v1",
        model="m",
        default_headers=default_headers,
    )
    return dict(llm._client.default_headers)


def test_openrouter_sends_hindsight_attribution_by_default():
    headers = _headers("openrouter")
    for name, value in OPENROUTER_ATTRIBUTION_HEADERS.items():
        assert headers[name] == value


def test_any_operator_attribution_header_replaces_all_defaults():
    for own in ({"x-title": "Mine"}, {"HTTP-Referer": "https://mine.test"}, {"X-OpenRouter-Title": "Mine"}):
        headers = _headers("openrouter", own)
        assert "Hindsight" not in headers.values()
        assert OPENROUTER_ATTRIBUTION_HEADERS["HTTP-Referer"] not in headers.values()


def test_other_providers_get_no_openrouter_headers():
    headers = _headers("openai")
    assert not set(OPENROUTER_ATTRIBUTION_HEADERS) & set(headers)
