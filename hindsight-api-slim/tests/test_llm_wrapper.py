import pytest

from hindsight_api.engine.llm_wrapper import sanitize_llm_output


@pytest.mark.parametrize(
    "input_text, expected",
    [
        # Null bytes stripped
        ("hello\x00world", "helloworld"),
        ("FIRST\u0000PAGE", "FIRSTPAGE"),
        # Multiple null bytes
        ("\x00\x00text\x00", "text"),
        # Other control characters stripped (non-whitespace)
        ("text\x01\x02\x03end", "textend"),
        ("text\x08end", "textend"),  # backspace
        ("text\x0cend", "textend"),  # form feed
        ("text\x0bend", "textend"),  # vertical tab
        ("text\x1fend", "textend"),  # unit separator
        ("text\x7fend", "textend"),  # DEL
        # Whitespace preserved
        ("hello\tworld", "hello\tworld"),
        ("hello\nworld", "hello\nworld"),
        ("hello\r\nworld", "hello\r\nworld"),
        # Unicode surrogates stripped
        ("text\ud800end", "textend"),
        ("text\udfffend", "textend"),
        # Clean text unchanged
        ("normal text", "normal text"),
        ("unicode: café naïve", "unicode: café naïve"),
        # Edge cases
        ("", ""),
        (None, None),
    ],
)
def test_sanitize_llm_output(input_text, expected):
    assert sanitize_llm_output(input_text) == expected


def test_llm_provider_constructor_ignores_global_config(monkeypatch):
    """The constructor uses only its arguments — it never reads global config.

    Resolving the server-level default for an omitted field is the caller's job
    (MemoryEngine's per-op builds, _member_to_llm, and from_env). Keeping the
    constructor config-free makes a provider's effective settings a pure function
    of its arguments, which is what lets each member of a multi-LLM chain be
    configured independently.
    """
    import json

    from hindsight_api.config import (
        ENV_LLM_DEFAULT_HEADERS,
        ENV_LLM_PROMPT_CACHE_ENABLED,
        clear_config_cache,
    )
    from hindsight_api.engine.llm_wrapper import LLMProvider

    # Global config sets a non-default header map and enables prompt caching...
    monkeypatch.setenv(ENV_LLM_DEFAULT_HEADERS, json.dumps({"x-from": "global"}))
    monkeypatch.setenv(ENV_LLM_PROMPT_CACHE_ENABLED, "true")  # also the global default
    clear_config_cache()

    # ...but a directly-constructed provider that omits them does NOT inherit them.
    provider = LLMProvider(provider="mock", api_key="", base_url="", model="m")
    assert provider.default_headers is None
    assert provider.prompt_cache_enabled is False  # constructor default, not the global True

    clear_config_cache()


def test_llm_provider_constructor_ignores_global_safety_settings(monkeypatch):
    """A Gemini provider built directly does not pull safety settings from config."""
    import json
    from unittest.mock import MagicMock, patch

    from hindsight_api.config import ENV_LLM_GEMINI_SAFETY_SETTINGS, clear_config_cache
    from hindsight_api.engine.llm_wrapper import LLMProvider

    monkeypatch.setenv(
        ENV_LLM_GEMINI_SAFETY_SETTINGS,
        json.dumps([{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}]),
    )
    clear_config_cache()

    with patch("google.genai.Client", return_value=MagicMock()):
        provider = LLMProvider(provider="gemini", api_key="k", base_url="", model="gemini-2.5-flash")

    assert provider.gemini_safety_settings is None  # not inherited from global config

    clear_config_cache()
