"""Unit tests for the Antigravity (``agy`` CLI) LLM provider.

The CLI is a fake subprocess. Retries and JSON parsing are CursorLLM's and tested
there; what is tested here is the seam specific to ``agy``: the prompt on stdin as a
stream-json message, the native schema flag, the denied tools, the disabled hooks,
and finding the ``result`` event in the output stream.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from hindsight_api.config import PROVIDER_DEFAULT_MODELS
from hindsight_api.engine.llm_wrapper import create_llm_provider, requires_api_key
from hindsight_api.engine.providers.antigravity_llm import AntigravityLLM


class _Answer(BaseModel):
    answer: str


class _FakeProc:
    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self._stdout = stdout.encode()
        self.returncode = returncode
        self.stdin_payload: bytes | None = None

    async def communicate(self, payload: bytes | None = None):
        self.stdin_payload = payload
        return self._stdout, b""


def _stream(response: str, status: str = "SUCCESS", error: str = "", structured: Any = None) -> str:
    result = {
        "status": status,
        "response": response,
        "structured_output": structured,
        "error": error,
        "usage": {"input_tokens": 900, "output_tokens": 12, "thinking_tokens": 0},
    }
    return (
        json.dumps({"event": "init", "tools": ["view_file"]})
        + "\n"
        + json.dumps({"event": "result", "result": result})
        + "\n"
    )


async def _call(stdout: str, **kwargs: Any):
    with patch("hindsight_api.engine.providers.antigravity_llm.shutil.which", return_value="/usr/bin/agy"):
        llm = AntigravityLLM(provider="antigravity", api_key="", base_url="", model="gemini-3.8-flash-low")
    seen: dict[str, Any] = {}

    async def fake_exec(*args, **kw):
        seen["args"], seen["kwargs"] = list(args), kw
        seen["proc"] = _FakeProc(stdout)
        return seen["proc"]

    with patch("hindsight_api.engine.providers.antigravity_llm.asyncio.create_subprocess_exec", fake_exec):
        result = await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0, **kwargs)
    return result, seen


@pytest.mark.asyncio
async def test_prompt_goes_on_stdin_with_tools_denied_and_hooks_off():
    result, seen = await _call(_stream("hello"))
    assert result.content == "hello"
    assert result.usage.input_tokens == 900 and result.usage.output_tokens == 12

    payload = json.loads(seen["proc"].stdin_payload)
    assert payload == {"event": "user", "message": {"role": "user", "content": "hi"}}
    args = seen["args"]
    assert "hi" not in args  # never in argv: Linux caps one argument at 128KB
    assert args[args.index("--model") + 1] == "gemini-3.8-flash-low"
    settings = json.loads(args[args.index("--settings") + 1])
    assert "command(*)" in settings["permissions"]["deny"]
    assert "read_file(*)" in settings["permissions"]["deny"]
    assert seen["kwargs"]["env"]["HINDSIGHT_DISABLE_HOOKS"] == "1"


@pytest.mark.asyncio
async def test_structured_output_passes_the_schema_flag_and_parses_the_reply():
    result, seen = await _call(_stream('{"answer": "42"}'), response_format=_Answer)
    assert result.content == _Answer(answer="42")
    schema = json.loads(seen["args"][seen["args"].index("--json-schema") + 1])
    assert "answer" in schema["properties"]


@pytest.mark.asyncio
async def test_structured_output_wins_over_a_reply_that_repeats_the_json():
    # Seen live: the reply text held the JSON twice, which json.loads rejects.
    doubled = '{"answer": "42"}\n{"answer":"42"}'
    result, _ = await _call(_stream(doubled, structured={"answer": "42"}), response_format=_Answer)
    assert result.content == _Answer(answer="42")


@pytest.mark.asyncio
async def test_error_result_raises():
    with pytest.raises(RuntimeError, match="authentication failed"):
        await _call(_stream("", status="ERROR", error="authentication failed or timed out"))


@pytest.mark.asyncio
async def test_output_without_result_event_raises():
    with pytest.raises(RuntimeError, match="no result"):
        await _call("not json\n")


def test_provider_is_registered():
    assert not requires_api_key("antigravity")
    assert PROVIDER_DEFAULT_MODELS["antigravity"]
    with patch("hindsight_api.engine.providers.antigravity_llm.shutil.which", return_value="/usr/bin/agy"):
        llm = create_llm_provider(provider="antigravity", api_key="", base_url="", model="m", reasoning_effort="low")
    assert isinstance(llm, AntigravityLLM)
