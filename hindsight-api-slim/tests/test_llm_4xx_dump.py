"""Unit tests for the provider-agnostic 4xx request-dump diagnostic.

The dump must be a no-op unless ``HINDSIGHT_API_LLM_DEBUG_DUMP_4XX`` is enabled AND
the error carries a 4xx status; when it fires it must serialize the request config
(without message bodies) plus a capped per-message preview, work across the different
request shapes providers assemble (Pydantic config vs kwargs dict), extract the status
across the different SDK error shapes, and never raise.
"""

import logging
from types import SimpleNamespace

from hindsight_api.engine.providers.llm_debug import dump_request_on_4xx, status_code_of

ENV = "HINDSIGHT_API_LLM_DEBUG_DUMP_4XX"


def _dump(**kw):
    kw.setdefault("scope", "consolidation")
    kw.setdefault("provider", "openai")
    kw.setdefault("model", "gpt-x")
    return dump_request_on_4xx(**kw)


# ── status extraction across SDK error shapes ──────────────────────────────────


def test_status_code_of_covers_sdk_shapes():
    assert status_code_of(SimpleNamespace(status_code=400)) == 400  # OpenAI/Anthropic
    assert status_code_of(SimpleNamespace(code=429)) == 429  # google-genai
    assert status_code_of(SimpleNamespace(response=SimpleNamespace(status_code=400))) == 400
    assert status_code_of(SimpleNamespace(message="no code")) is None
    # A stringly-typed code (e.g. google-genai "INVALID_ARGUMENT") is not an int status.
    assert status_code_of(SimpleNamespace(code="INVALID_ARGUMENT")) is None


# ── gating ─────────────────────────────────────────────────────────────────────


def test_dump_is_noop_when_disabled(monkeypatch, caplog):
    monkeypatch.delenv(ENV, raising=False)
    with caplog.at_level(logging.ERROR):
        _dump(err=SimpleNamespace(status_code=400), request={"messages": [{"role": "user", "content": "hi"}]})
    assert "[LLM_4XX_DUMP]" not in caplog.text


def test_dump_is_noop_on_non_4xx(monkeypatch, caplog):
    monkeypatch.setenv(ENV, "true")
    with caplog.at_level(logging.ERROR):
        _dump(err=SimpleNamespace(status_code=500), request={"messages": [{"role": "user", "content": "hi"}]})
        _dump(err=SimpleNamespace(message="no status"), request={"messages": []})
    assert "[LLM_4XX_DUMP]" not in caplog.text


# ── openai/anthropic/litellm shape: kwargs dict carrying messages ──────────────


def test_dump_dict_request_strips_message_bodies_from_config(monkeypatch, caplog):
    monkeypatch.setenv(ENV, "true")
    request = {
        "model": "gpt-x",
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": "hello world"}],
    }
    with caplog.at_level(logging.ERROR):
        _dump(err=SimpleNamespace(status_code=400), request=request)
    assert "[LLM_4XX_DUMP]" in caplog.text
    assert "code=400" in caplog.text
    assert "json_object" in caplog.text  # config is serialized
    assert '"preview": "hello world"' in caplog.text  # message preview
    # The message body must appear only in the capped preview, never in the config view.
    assert '"messages"' not in caplog.text.split("contents=")[0]


def test_dump_handles_anthropic_block_list_content(monkeypatch, caplog):
    monkeypatch.setenv(ENV, "1")
    request = {"messages": [{"role": "user", "content": [{"type": "text", "text": "block text"}]}]}
    with caplog.at_level(logging.ERROR):
        _dump(provider="anthropic", err=SimpleNamespace(status_code=400), request=request)
    assert "block text" in caplog.text


# ── gemini shape: Pydantic config + separate contents ──────────────────────────


def test_dump_pydantic_config_with_separate_contents(monkeypatch, caplog):
    monkeypatch.setenv(ENV, "yes")
    config = SimpleNamespace(model_dump_json=lambda **_: '{"response_schema": {"maxItems": 0}}')
    contents = [SimpleNamespace(role="user", parts=[SimpleNamespace(text="gemini hi")])]
    with caplog.at_level(logging.ERROR):
        _dump(provider="gemini", err=SimpleNamespace(code=400), request=config, messages=contents)
    assert "[LLM_4XX_DUMP]" in caplog.text
    assert "maxItems" in caplog.text  # serialized config
    assert "gemini hi" in caplog.text  # content preview


# ── safety: truncation + never-raise ───────────────────────────────────────────


def test_dump_truncates_long_message_preview(monkeypatch, caplog):
    monkeypatch.setenv(ENV, "on")
    request = {"messages": [{"role": "user", "content": "x" * 5000}]}
    with caplog.at_level(logging.ERROR):
        _dump(err=SimpleNamespace(status_code=400), request=request)
    assert "x" * 1500 in caplog.text
    assert "x" * 1600 not in caplog.text


def test_dump_never_raises_on_unserializable_config(monkeypatch, caplog):
    monkeypatch.setenv(ENV, "true")

    class Bad:
        def model_dump_json(self, **_):
            raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR):
        # Must not raise even though model_dump_json fails (falls back to repr).
        _dump(err=SimpleNamespace(status_code=400), request=Bad(), messages=[])
    assert "[LLM_4XX_DUMP]" in caplog.text
