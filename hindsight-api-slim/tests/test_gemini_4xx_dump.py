"""Unit tests for the opt-in 4xx request-dump diagnostic in the Gemini provider.

The dump must be a no-op unless ``HINDSIGHT_API_LLM_DEBUG_DUMP_4XX`` is enabled,
must serialize the request config + a capped message preview when enabled, and
must never raise (diagnostics can't break the request path).
"""

import logging
from types import SimpleNamespace

from hindsight_api.engine.providers.gemini_llm import _dump_request_on_4xx

ENV = "HINDSIGHT_API_LLM_DEBUG_DUMP_4XX"


def _config(json_str: str = '{"response_schema": {"maxItems": 0}}'):
    return SimpleNamespace(model_dump_json=lambda **_: json_str)


def _contents(text: str):
    return [SimpleNamespace(role="user", parts=[SimpleNamespace(text=text)])]


def _err(code: int = 400):
    return SimpleNamespace(code=code)


def test_dump_is_noop_when_disabled(monkeypatch, caplog):
    monkeypatch.delenv(ENV, raising=False)
    with caplog.at_level(logging.ERROR):
        _dump_request_on_4xx("consolidation", _contents("hello"), _config(), _err())
    assert "[LLM_4XX_DUMP]" not in caplog.text


def test_dump_emits_config_and_preview_when_enabled(monkeypatch, caplog):
    monkeypatch.setenv(ENV, "true")
    with caplog.at_level(logging.ERROR):
        _dump_request_on_4xx("consolidation", _contents("hello world"), _config(), _err(400))
    assert "[LLM_4XX_DUMP]" in caplog.text
    assert "maxItems" in caplog.text  # the serialized config (what's on the wire)
    assert "hello world" in caplog.text  # message preview
    assert "code=400" in caplog.text


def test_dump_truncates_long_message_preview(monkeypatch, caplog):
    monkeypatch.setenv(ENV, "1")
    long_text = "x" * 5000
    with caplog.at_level(logging.ERROR):
        _dump_request_on_4xx("consolidation", _contents(long_text), _config(), _err())
    # preview is capped at 1500 chars, so the full 5000-char body is not logged
    assert "x" * 1500 in caplog.text
    assert "x" * 1600 not in caplog.text


def test_dump_never_raises_on_unserializable_config(monkeypatch, caplog):
    monkeypatch.setenv(ENV, "yes")

    class Bad:
        def model_dump_json(self, **_):
            raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR):
        # must not raise even though model_dump_json fails (falls back to repr)
        _dump_request_on_4xx("consolidation", _contents("hi"), Bad(), _err())
    assert "[LLM_4XX_DUMP]" in caplog.text
