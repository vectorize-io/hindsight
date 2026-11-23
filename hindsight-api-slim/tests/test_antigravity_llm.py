"""Unit tests for the Antigravity (``agy`` CLI) LLM provider.

The CLI is a fake subprocess. Retries and JSON parsing are CursorLLM's and tested
there; what is tested here is the seam specific to ``agy``: the prompt on stdin as a
stream-json message, the native schema flag, the denied tools, the disabled hooks,
and finding the ``result`` event in the output stream.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from hindsight_api.config import PROVIDER_DEFAULT_MODELS
from hindsight_api.engine.llm_interface import ProviderAuthError
from hindsight_api.engine.llm_wrapper import create_llm_provider, requires_api_key
from hindsight_api.engine.providers.antigravity_llm import AntigravityLLM


class _Answer(BaseModel):
    answer: str


class _FakeProc:
    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self._stdout = stdout.encode()
        self.returncode = returncode
        self.stdin_payload: bytes | None = None
        self.killed = False

    async def communicate(self, payload: bytes | None = None):
        self.stdin_payload = payload
        return self._stdout, b""

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> None:
        return None


class _HangingProc(_FakeProc):
    async def communicate(self, payload: bytes | None = None):
        await asyncio.sleep(10)
        raise AssertionError("should have timed out")


@dataclass
class _Spawned:
    """What the provider handed to ``create_subprocess_exec``, one entry per call."""

    procs: list[_FakeProc] = field(default_factory=list)
    args: list[list[str]] = field(default_factory=list)
    envs: list[dict[str, str]] = field(default_factory=list)


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


async def _call(
    stdout: str,
    *,
    spawned: _Spawned,
    proc_cls: type[_FakeProc] = _FakeProc,
    timeout: float | None = None,
    max_retries: int = 0,
    **kwargs: Any,
) -> Any:
    with patch("hindsight_api.engine.providers.antigravity_llm.shutil.which", return_value="/usr/bin/agy"):
        llm = AntigravityLLM(
            provider="antigravity", api_key="", base_url="", model="gemini-3.8-flash-low", timeout=timeout
        )
    seen = spawned

    async def fake_exec(*args, **kw):
        seen.args.append(list(args))
        seen.envs.append(kw["env"])
        seen.procs.append(proc_cls(stdout))
        return seen.procs[-1]

    with patch("hindsight_api.engine.providers.antigravity_llm.asyncio.create_subprocess_exec", fake_exec):
        result = await llm.call(
            messages=[{"role": "user", "content": "hi"}], max_retries=max_retries, initial_backoff=0, **kwargs
        )
    return result


@pytest.mark.asyncio
async def test_prompt_goes_on_stdin_with_tools_denied_and_hooks_off():
    seen = _Spawned()
    result = await _call(_stream("hello"), spawned=seen)
    assert result.content == "hello"
    assert result.usage.input_tokens == 900 and result.usage.output_tokens == 12

    payload = json.loads(seen.procs[0].stdin_payload)
    assert payload == {"event": "user", "message": {"role": "user", "content": "hi"}}
    args = seen.args[0]
    assert "hi" not in args  # never in argv: Linux caps one argument at 128KB
    assert args[args.index("--model") + 1] == "gemini-3.8-flash-low"
    settings = json.loads(args[args.index("--settings") + 1])
    assert "command(*)" in settings["permissions"]["deny"]
    assert "read_file(*)" in settings["permissions"]["deny"]
    assert seen.envs[0]["HINDSIGHT_DISABLE_HOOKS"] == "1"


@pytest.mark.asyncio
async def test_structured_output_passes_the_schema_flag_and_parses_the_reply():
    seen = _Spawned()
    result = await _call(_stream('{"answer": "42"}'), response_format=_Answer, spawned=seen)
    assert result.content == _Answer(answer="42")
    schema = json.loads(seen.args[0][seen.args[0].index("--json-schema") + 1])
    assert "answer" in schema["properties"]


@pytest.mark.asyncio
async def test_structured_output_wins_over_a_reply_that_repeats_the_json():
    # Seen live: the reply text held the JSON twice, which json.loads rejects.
    doubled = '{"answer": "42"}\n{"answer":"42"}'
    result = await _call(_stream(doubled, structured={"answer": "42"}), response_format=_Answer, spawned=_Spawned())
    assert result.content == _Answer(answer="42")


@pytest.mark.asyncio
async def test_error_result_raises():
    with pytest.raises(RuntimeError, match="quota exceeded"):
        await _call(_stream("", status="ERROR", error="quota exceeded"), spawned=_Spawned())


@pytest.mark.asyncio
async def test_missing_login_is_permanent_and_not_retried():
    # Signed out, each attempt waits 60s on a browser login: retrying only multiplies that.
    spawned = _Spawned()
    with pytest.raises(ProviderAuthError, match="sign in"):
        await _call(
            _stream("", status="ERROR", error="authentication failed or timed out"), spawned=spawned, max_retries=3
        )
    assert len(spawned.procs) == 1


@pytest.mark.asyncio
async def test_timeout_kills_the_child_and_names_the_timeout_setting():
    spawned = _Spawned()
    with pytest.raises(TimeoutError, match="HINDSIGHT_API_LLM_TIMEOUT"):
        await _call("", spawned=spawned, proc_cls=_HangingProc, timeout=0.05)
    assert spawned.procs[0].killed


@pytest.mark.asyncio
async def test_output_without_result_event_raises():
    with pytest.raises(RuntimeError, match="no result"):
        await _call("not json\n", spawned=_Spawned())


def test_provider_is_registered():
    assert not requires_api_key("antigravity")
    assert PROVIDER_DEFAULT_MODELS["antigravity"]
    with patch("hindsight_api.engine.providers.antigravity_llm.shutil.which", return_value="/usr/bin/agy"):
        llm = create_llm_provider(provider="antigravity", api_key="", base_url="", model="m", reasoning_effort="low")
    assert isinstance(llm, AntigravityLLM)
