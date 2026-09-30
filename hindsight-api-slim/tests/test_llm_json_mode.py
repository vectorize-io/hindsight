"""
Tests for HINDSIGHT_API_LLM_JSON_MODE / config.llm_json_mode.

The mode chooses how an OpenAI-compatible endpoint is asked for JSON:

- ``auto`` keeps today's per-provider behaviour. lmstudio, ollama, and volcano
  skip ``response_format``, and llamacpp skips it when
  HINDSIGHT_API_LLAMACPP_NO_GRAMMAR is set.
- ``native`` always sends ``response_format``, ignoring that skip list.
- ``prompt`` never sends it. The schema stays in the prompt, for an endpoint
  that ignores ``response_format`` and otherwise burns the whole LLM timeout
  (issue #4935).

It is server-wide. The batch retain path builds its own request body and reads
the same field, so ``prompt`` drops ``response_format`` there too.
"""

import dataclasses
import http.server
import json
import threading
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.config import ENV_LLM_JSON_MODE, HindsightConfig
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

_SCHEMA_HINT = "You must respond with valid JSON matching this schema"


class _Resp(BaseModel):
    ok: bool


def _config_with(json_mode: str) -> object:
    """A real config overriding only llm_json_mode, for the provider to read."""
    from hindsight_api.config import _get_raw_config

    return dataclasses.replace(_get_raw_config(), llm_json_mode=json_mode)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_json_mode_defaults_to_auto(monkeypatch):
    monkeypatch.delenv(ENV_LLM_JSON_MODE, raising=False)
    assert HindsightConfig.from_env().llm_json_mode == "auto"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("native", "native"), (" Prompt ", "prompt"), ("AUTO", "auto")],
)
def test_json_mode_parses_native_and_prompt(monkeypatch, raw, expected):
    monkeypatch.setenv(ENV_LLM_JSON_MODE, raw)
    assert HindsightConfig.from_env().llm_json_mode == expected


def test_json_mode_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv(ENV_LLM_JSON_MODE, "grammar")
    with pytest.raises(ValueError, match="HINDSIGHT_API_LLM_JSON_MODE"):
        HindsightConfig.from_env()


# --------------------------------------------------------------------------- #
# openai-compatible: response_format follows the mode
# --------------------------------------------------------------------------- #


def _openai_response(content: str = '{"ok": true}'):
    choice = SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content=content, tool_calls=None, refusal=None)
    )
    return SimpleNamespace(error=None, usage=None, choices=[choice])


async def _openai_call(*, provider: str, strict: bool, json_mode: str):
    llm = OpenAICompatibleLLM(
        provider=provider, api_key="test-key", base_url="https://example.test/v1", model="local-model"
    )
    create = AsyncMock(return_value=_openai_response())
    llm._client.chat.completions.create = create
    cfg = _config_with(json_mode)  # build before patching to avoid get_config recursion
    with (
        patch("hindsight_api.config.get_config", lambda: cfg),
        patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"),
    ):
        await llm.call(
            messages=[{"role": "user", "content": "Return whether this worked."}],
            response_format=_Resp,
            strict_schema=strict,
            max_retries=0,
        )
    return create.call_args.kwargs


@pytest.mark.asyncio
async def test_auto_soft_sends_json_object():
    kwargs = await _openai_call(provider="openai", strict=False, json_mode="auto")
    assert kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_auto_lmstudio_sends_no_response_format():
    kwargs = await _openai_call(provider="lmstudio", strict=False, json_mode="auto")
    assert kwargs.get("response_format") is None


@pytest.mark.asyncio
async def test_native_lmstudio_sends_json_object():
    kwargs = await _openai_call(provider="lmstudio", strict=False, json_mode="native")
    assert kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_prompt_mode_sends_no_response_format():
    kwargs = await _openai_call(provider="openai", strict=False, json_mode="prompt")
    assert kwargs.get("response_format") is None
    assert _SCHEMA_HINT in kwargs["messages"][0]["content"]


@pytest.mark.asyncio
async def test_prompt_mode_with_strict_schema_sends_no_response_format():
    kwargs = await _openai_call(provider="openai", strict=True, json_mode="prompt")
    assert kwargs.get("response_format") is None
    assert _SCHEMA_HINT in kwargs["messages"][0]["content"]


@pytest.mark.asyncio
async def test_auto_strict_schema_sends_json_schema():
    kwargs = await _openai_call(provider="openai", strict=True, json_mode="auto")
    response_format = kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True


# --------------------------------------------------------------------------- #
# batch retain path
# --------------------------------------------------------------------------- #


def _batch_config(*, json_mode: str, strict_retain: bool):
    return SimpleNamespace(
        retain_max_completion_tokens=None,
        llm_strict_schema=not strict_retain,
        llm_strict_schema_retain=strict_retain,
        llm_temperature_retain=None,
        llm_json_mode=json_mode,
    )


def test_batch_prompt_mode_omits_response_format():
    from hindsight_api.engine.retain.fact_extraction import _build_request_body

    batch_impl = SimpleNamespace(model="gpt-4o-mini", provider="openai", openai_service_tier=None)
    body = _build_request_body(
        batch_impl, _batch_config(json_mode="prompt", strict_retain=False), "system prompt", "user", _Resp
    )
    assert "response_format" not in body
    assert _SCHEMA_HINT in body["messages"][0]["content"]


@pytest.mark.parametrize("strict_retain", [True, False])
def test_batch_auto_strict_follows_retain_config(strict_retain):
    from hindsight_api.engine.retain.fact_extraction import _build_request_body

    batch_impl = SimpleNamespace(model="gpt-4o-mini", provider="openai", openai_service_tier=None)
    body = _build_request_body(
        batch_impl, _batch_config(json_mode="auto", strict_retain=strict_retain), "system prompt", "user", _Resp
    )
    assert body["response_format"]["json_schema"]["strict"] is strict_retain


# --------------------------------------------------------------------------- #
# wire: real OpenAI SDK against a local HTTP server
# --------------------------------------------------------------------------- #


class _ChatHandler(http.server.BaseHTTPRequestHandler):
    bodies: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        body = json.loads(raw.decode())
        type(self).bodies.append({"path": self.path, "body": body})
        model = body.get("model", "local-model")
        payload = {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": '{"ok": true}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args):
        pass


@pytest.fixture
def chat_server() -> Iterator[int]:
    _ChatHandler.bodies = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


async def _wire_call(port: int, monkeypatch, json_mode: str) -> dict:
    llm = OpenAICompatibleLLM(
        provider="openai",
        api_key="test-key",
        base_url=f"http://127.0.0.1:{port}/v1",
        model="local-model",
    )
    cfg = _config_with(json_mode)
    # call() imports get_config locally, so the provider-module binding is not what it reads.
    monkeypatch.setattr("hindsight_api.config.get_config", lambda: cfg)
    await llm.call(
        messages=[{"role": "user", "content": "Return whether this worked."}],
        response_format=_Resp,
        strict_schema=False,
        max_retries=0,
    )
    assert len(_ChatHandler.bodies) == 1
    assert _ChatHandler.bodies[0]["path"] == "/v1/chat/completions"
    return _ChatHandler.bodies[0]["body"]


@pytest.mark.asyncio
async def test_wire_auto_sends_response_format(chat_server, monkeypatch):
    body = await _wire_call(chat_server, monkeypatch, "auto")
    assert body["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_wire_prompt_mode_omits_response_format(chat_server, monkeypatch):
    body = await _wire_call(chat_server, monkeypatch, "prompt")
    assert "response_format" not in body
    assert _SCHEMA_HINT in body["messages"][0]["content"]
