from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.engine.llm_wrapper import create_llm_provider, sanitize_llm_output


def test_create_llm_provider_preserves_positional_timeout_compatibility():
    """The new Ollama knob must not steal the old positional timeout slot."""
    impl = create_llm_provider(
        "ollama",
        "",
        "",
        "llama3.2",
        "low",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
        None,
        None,
        7.5,
    )

    assert impl.timeout == 7.5
    assert impl.ollama_num_ctx is None


def test_llm_provider_preserves_positional_timeout_compatibility():
    """The new Ollama knob must not steal the old positional timeout slot."""
    from hindsight_api.engine.llm_wrapper import LLMProvider

    provider = LLMProvider(
        "ollama",
        "",
        "",
        "llama3.2",
        "low",
        None,
        None,
        None,
        None,
        False,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        7.5,
    )

    assert provider.timeout == 7.5
    assert provider.ollama_num_ctx is None


def test_llm_provider_threads_ollama_num_ctx_to_provider_impl():
    """LLMProvider carries the native Ollama context override to the implementation."""
    from hindsight_api.engine.llm_wrapper import LLMProvider

    provider = LLMProvider(
        provider="ollama",
        api_key="",
        base_url="",
        model="llama3.2",
        ollama_num_ctx=65536,
    )

    assert provider.ollama_num_ctx == 65536
    assert provider._provider_impl.ollama_num_ctx == 65536


@pytest.mark.parametrize("bad_value", [0, -1, 1.5, "65536", True])
def test_llm_provider_rejects_invalid_ollama_num_ctx(bad_value):
    """Direct callers should fail before sending invalid Ollama request options."""
    from hindsight_api.engine.llm_wrapper import LLMProvider

    with pytest.raises(ValueError, match="ollama_num_ctx"):
        LLMProvider(
            provider="ollama",
            api_key="",
            base_url="",
            model="llama3.2",
            ollama_num_ctx=bad_value,
        )


@pytest.mark.parametrize("bad_value", [0, -1, 1.5, "65536", True])
def test_openai_compatible_llm_rejects_invalid_ollama_num_ctx(bad_value):
    """The provider implementation also validates direct construction."""
    from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

    with pytest.raises(ValueError, match="ollama_num_ctx"):
        OpenAICompatibleLLM(
            provider="ollama",
            api_key="",
            base_url="",
            model="llama3.2",
            ollama_num_ctx=bad_value,
        )


def test_llm_provider_from_env_reads_ollama_num_ctx(monkeypatch):
    """Direct env construction uses the same optional positive-int parser."""
    from hindsight_api.config import ENV_LLM_OLLAMA_NUM_CTX, clear_config_cache
    from hindsight_api.engine.llm_wrapper import LLMProvider

    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "ollama")
    monkeypatch.setenv(ENV_LLM_OLLAMA_NUM_CTX, "32768")
    clear_config_cache()

    provider = LLMProvider.from_env()

    assert provider.ollama_num_ctx == 32768
    assert provider._provider_impl.ollama_num_ctx == 32768
    clear_config_cache()


@pytest.mark.asyncio
async def test_native_ollama_omits_num_ctx_unless_configured(monkeypatch):
    """Native Ollama calls should not override the model context window by default."""
    from pydantic import BaseModel

    from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

    class Answer(BaseModel):
        ok: bool

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": '{"ok": true}'}}

    calls = []

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json, headers):
            calls.append({"url": url, "json": json, "headers": headers})
            return FakeResponse()

    monkeypatch.setattr("hindsight_api.engine.providers.openai_compatible_llm.httpx.AsyncClient", FakeAsyncClient)

    default_provider = OpenAICompatibleLLM(
        provider="ollama",
        api_key="",
        base_url="",
        model="llama3.2",
    )
    await default_provider._call_ollama_native(
        messages=[{"role": "user", "content": "ping"}],
        response_format=Answer,
        max_completion_tokens=None,
        temperature=None,
        max_retries=0,
        initial_backoff=1,
        max_backoff=1,
        skip_validation=False,
    )

    configured_provider = OpenAICompatibleLLM(
        provider="ollama",
        api_key="",
        base_url="",
        model="llama3.2",
        ollama_num_ctx=65536,
    )
    await configured_provider._call_ollama_native(
        messages=[{"role": "user", "content": "ping"}],
        response_format=Answer,
        max_completion_tokens=None,
        temperature=None,
        max_retries=0,
        initial_backoff=1,
        max_backoff=1,
        skip_validation=False,
    )

    assert "num_ctx" not in calls[0]["json"]["options"]
    assert calls[0]["json"]["options"]["num_batch"] == 512
    assert calls[1]["json"]["options"]["num_ctx"] == 65536


def _openai_response(content: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))],
        usage=None,
    )


@pytest.mark.asyncio
async def test_ollama_num_ctx_applies_to_unstructured_calls_and_verification():
    """OpenAI-compatible Ollama calls, including the startup probe, carry num_ctx."""
    from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

    provider = OpenAICompatibleLLM(
        provider="ollama",
        api_key="",
        base_url="",
        model="llama3.2",
        ollama_num_ctx=24576,
    )
    create = AsyncMock(return_value=_openai_response())
    provider._client.chat.completions.create = create

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        await provider.call(
            messages=[{"role": "user", "content": "Say 'ok'"}],
            max_completion_tokens=64,
            max_retries=0,
        )
        await provider.verify_connection()

    assert create.call_args_list[0].kwargs["extra_body"]["options"]["num_ctx"] == 24576
    assert create.call_args_list[1].kwargs["extra_body"]["options"]["num_ctx"] == 24576


@pytest.mark.asyncio
async def test_ollama_unstructured_num_ctx_preserves_extra_body_precedence_and_unset_behavior():
    """User options win, while an unset context override remains absent."""
    from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

    configured = OpenAICompatibleLLM(
        provider="ollama",
        api_key="",
        base_url="",
        model="llama3.2",
        ollama_num_ctx=24576,
        extra_body={"options": {"num_ctx": 8192, "temperature": 0.2}},
    )
    default = OpenAICompatibleLLM(provider="ollama", api_key="", base_url="", model="llama3.2")
    copied = OpenAICompatibleLLM(
        provider="ollama",
        api_key="",
        base_url="",
        model="llama3.2",
        ollama_num_ctx=24576,
        extra_body={"options": {"temperature": 0.2}},
    )
    configured_create = AsyncMock(return_value=_openai_response())
    default_create = AsyncMock(return_value=_openai_response())
    copied_create = AsyncMock(return_value=_openai_response())
    configured._client.chat.completions.create = configured_create
    default._client.chat.completions.create = default_create
    copied._client.chat.completions.create = copied_create

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        await configured.call(messages=[{"role": "user", "content": "ping"}], max_retries=0)
        await default.call(messages=[{"role": "user", "content": "ping"}], max_retries=0)
        await copied.call(messages=[{"role": "user", "content": "ping"}], max_retries=0)

    configured_options = configured_create.call_args.kwargs["extra_body"]["options"]
    assert configured_options == {"num_ctx": 8192, "temperature": 0.2}
    assert configured._config_extra_body == {"options": {"num_ctx": 8192, "temperature": 0.2}}
    assert "extra_body" not in default_create.call_args.kwargs
    assert copied_create.call_args.kwargs["extra_body"]["options"] == {"num_ctx": 24576, "temperature": 0.2}
    assert copied._config_extra_body == {"options": {"temperature": 0.2}}


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


# --- parse_llm_json: structural repair of malformed LLM JSON (#2547/#2544) ---


def test_parse_llm_json_valid_passthrough():
    from hindsight_api.engine.llm_wrapper import parse_llm_json

    assert parse_llm_json('{"a": 1, "b": "two"}') == {"a": 1, "b": "two"}


def test_parse_llm_json_strips_fences():
    from hindsight_api.engine.llm_wrapper import parse_llm_json

    assert parse_llm_json('```json\n{"a": 1}\n```') == {"a": 1}


@pytest.mark.parametrize(
    "malformed,expected",
    [
        ('{"a": 1,}', {"a": 1}),  # trailing comma
        ('{"a": "unterminated', {"a": "unterminated"}),  # unterminated string
        ("{'a': 'single quotes'}", {"a": "single quotes"}),  # single quotes
        (r'{"path": "C:\Users"}', {"path": "C:\\Users"}),  # invalid \escape (#2504)
        ('```json\n{"a": 1,}\n```', {"a": 1}),  # fenced + trailing comma
    ],
)
def test_parse_llm_json_repairs_structural_malformation(malformed, expected):
    from hindsight_api.engine.llm_wrapper import parse_llm_json

    assert parse_llm_json(malformed) == expected


def test_parse_llm_json_control_char_scrub_still_works():
    from hindsight_api.engine.llm_wrapper import parse_llm_json

    # Raw control char embedded in a string value — scrubbed to a space, no repair.
    assert parse_llm_json('{"a": "line\x01break"}') == {"a": "line break"}


@pytest.mark.parametrize("garbage", ["", "   ", "not json at all !!!", "```json\n```"])
def test_parse_llm_json_unrecoverable_raises(garbage):
    """Repair that yields an empty result must not masquerade as success — the
    contract is to raise so retry ladders / the #1833 fail-loud path can act."""
    import json

    from hindsight_api.engine.llm_wrapper import parse_llm_json

    with pytest.raises(json.JSONDecodeError):
        parse_llm_json(garbage)


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
