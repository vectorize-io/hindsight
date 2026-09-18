"""Unit tests for the Cursor CLI-backed LLM provider.

The CLI is replaced by a fake subprocess: what is under test is the seam between
Hindsight and `cursor-agent` — how the prompt is assembled, how the emitted
`type: "result"` line is found among the CLI's diagnostics, and how structured
output and tool calls are parsed back out of free-form text.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from hindsight_api.config import PROVIDER_DEFAULT_MODELS
from hindsight_api.engine.llm_interface import (
    LLM_TOOL_CHOICE_NONE,
    LLM_TOOL_CHOICE_REQUIRED,
    LLMToolChoice,
    ProviderContentPolicyError,
)
from hindsight_api.engine.llm_wrapper import create_llm_provider, requires_api_key
from hindsight_api.engine.providers.cursor_llm import CursorLLM

_TOOLS = [
    {
        "function": {
            "name": "recall",
            "description": "Search memory",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    }
]


class _Answer(BaseModel):
    answer: str


class _FakeProc:
    """Stands in for the `cursor-agent` child process."""

    def __init__(self, stdout: str, returncode: int = 0, stderr: str = "") -> None:
        self._stdout = stdout.encode()
        self._stderr = stderr.encode()
        self.returncode = returncode
        self.stdin_payload: bytes | None = None

    async def communicate(self, payload: bytes | None = None):
        self.stdin_payload = payload
        return self._stdout, self._stderr

    def kill(self) -> None:  # pragma: no cover - only on timeout
        pass

    async def wait(self) -> None:  # pragma: no cover - only on timeout
        pass


def _result_line(text: str, *, is_error: bool = False) -> str:
    # The real CLI prints diagnostics on the same stream before its JSON.
    return (
        "cursor-retrieval: tracing to '/tmp/cursor_retrieval.log'\n"
        + json.dumps(
            {
                "type": "result",
                "subtype": "error" if is_error else "success",
                "is_error": is_error,
                "result": text,
                "usage": {"inputTokens": 120, "outputTokens": 7},
            }
        )
        + "\n"
    )


def _provider(
    stdout: str, *, returncode: int = 0, stderr: str = ""
) -> tuple[CursorLLM, list[_FakeProc], list[list[str]]]:
    procs: list[_FakeProc] = []
    arg_lists: list[list[str]] = []

    async def fake_exec(*args, **kwargs):
        arg_lists.append(list(args))
        proc = _FakeProc(stdout, returncode=returncode, stderr=stderr)
        procs.append(proc)
        return proc

    with patch("hindsight_api.engine.providers.cursor_llm.shutil.which", return_value="/usr/bin/cursor-agent"):
        llm = CursorLLM(provider="cursor", api_key="", base_url="", model="auto")
    patcher = patch("hindsight_api.engine.providers.cursor_llm.asyncio.create_subprocess_exec", fake_exec)
    patcher.start()
    llm._stop_patch = patcher.stop  # type: ignore[attr-defined]
    return llm, procs, arg_lists


@pytest.mark.asyncio
async def test_plain_call_sends_flattened_prompt_and_reports_usage():
    llm, procs, arg_lists = _provider(_result_line("hello there"))
    try:
        result = await llm.call(
            messages=[{"role": "system", "content": "Be terse"}, {"role": "user", "content": "Say hi"}],
            max_retries=0,
        )
    finally:
        llm._stop_patch()

    assert result.content == "hello there"
    assert result.usage.input_tokens == 120
    assert result.usage.output_tokens == 7
    prompt = procs[0].stdin_payload.decode()
    assert "Be terse" in prompt and "Say hi" in prompt
    # Headless, read-only, and pinned to the configured model.
    assert arg_lists[0][1:] == ["-p", "--output-format", "json", "--trust", "--mode", "ask", "--model", "auto"]


@pytest.mark.asyncio
async def test_structured_output_injects_schema_and_parses_fenced_json():
    llm, procs, _ = _provider(_result_line('Sure!\n```json\n{"answer": "42"}\n```'))
    try:
        result = await llm.call(messages=[{"role": "user", "content": "q"}], response_format=_Answer, max_retries=0)
    finally:
        llm._stop_patch()

    assert isinstance(result.content, _Answer)
    assert result.content.answer == "42"
    assert "valid JSON matching this schema" in procs[0].stdin_payload.decode()


@pytest.mark.asyncio
async def test_tool_calls_are_parsed_from_the_json_envelope():
    llm, procs, _ = _provider(
        _result_line(json.dumps({"content": None, "tool_calls": [{"name": "recall", "arguments": {"query": "rome"}}]}))
    )
    try:
        result = await llm.call_with_tools(
            messages=[{"role": "user", "content": "what do I know about rome?"}],
            tools=_TOOLS,
            tool_choice=LLM_TOOL_CHOICE_REQUIRED,
            max_retries=0,
        )
    finally:
        llm._stop_patch()

    assert result.finish_reason == "tool_calls"
    assert [(c.name, c.arguments) for c in result.tool_calls] == [("recall", {"query": "rome"})]
    assert "MUST call at least one tool" in procs[0].stdin_payload.decode()


@pytest.mark.asyncio
async def test_named_tool_choice_offers_only_that_tool():
    llm, procs, _ = _provider(_result_line(json.dumps({"tool_calls": [{"name": "recall", "arguments": {}}]})))
    try:
        await llm.call_with_tools(
            messages=[{"role": "user", "content": "go"}],
            tools=_TOOLS + [{"function": {"name": "other", "description": "", "parameters": {}}}],
            tool_choice=LLMToolChoice.named("recall"),
            max_retries=0,
        )
    finally:
        llm._stop_patch()

    prompt = procs[0].stdin_payload.decode()
    assert "MUST call the 'recall' tool" in prompt
    assert '"other"' not in prompt


@pytest.mark.asyncio
async def test_hallucinated_tool_name_is_dropped():
    llm, _, _ = _provider(_result_line(json.dumps({"tool_calls": [{"name": "delete_everything", "arguments": {}}]})))
    try:
        result = await llm.call_with_tools(messages=[{"role": "user", "content": "go"}], tools=_TOOLS, max_retries=0)
    finally:
        llm._stop_patch()

    assert result.tool_calls == []
    assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_tool_choice_none_asks_for_plain_text():
    llm, procs, _ = _provider(_result_line("just text"))
    try:
        result = await llm.call_with_tools(
            messages=[{"role": "user", "content": "go"}],
            tools=_TOOLS,
            tool_choice=LLM_TOOL_CHOICE_NONE,
            max_retries=0,
        )
    finally:
        llm._stop_patch()

    assert result.content == "just text"
    assert result.tool_calls == []
    assert "Do not call any tools" in procs[0].stdin_payload.decode()


@pytest.mark.asyncio
async def test_cli_error_result_raises():
    llm, _, _ = _provider(_result_line("Free plans can only use Auto.", is_error=True))
    try:
        with pytest.raises(RuntimeError, match="Free plans can only use Auto"):
            await llm.call(messages=[{"role": "user", "content": "q"}], max_retries=0)
    finally:
        llm._stop_patch()


@pytest.mark.asyncio
async def test_nonzero_exit_surfaces_stderr():
    llm, _, _ = _provider("", returncode=1, stderr="ActionRequiredError: not logged in")
    try:
        with pytest.raises(RuntimeError, match="not logged in"):
            await llm.call(messages=[{"role": "user", "content": "q"}], max_retries=0)
    finally:
        llm._stop_patch()


@pytest.mark.asyncio
async def test_blocked_prompt_is_permanent_and_not_retried():
    """A moderation block is the same on every replay, so it must not burn the retry budget."""
    llm, procs, _ = _provider(
        "",
        returncode=1,
        stderr=(
            "ActionRequiredError: Request blocked We are unable to complete this request because "
            "it was blocked under the model provider's usage guidelines."
        ),
    )
    try:
        with pytest.raises(ProviderContentPolicyError):
            await llm.call(messages=[{"role": "user", "content": "q"}], max_retries=5)
    finally:
        llm._stop_patch()

    assert len(procs) == 1


@pytest.mark.asyncio
async def test_timeout_error_names_the_timeout_setting():
    llm, _, _ = _provider(_result_line("never read"))
    llm.timeout = 0.01

    async def slow_exec(*args, **kwargs):
        class _Hanging(_FakeProc):
            async def communicate(self, payload: bytes | None = None):
                await asyncio.sleep(5)
                raise AssertionError("should have timed out")

        return _Hanging("")

    with patch("hindsight_api.engine.providers.cursor_llm.asyncio.create_subprocess_exec", slow_exec):
        try:
            with pytest.raises(TimeoutError, match="REFLECT_LLM_TIMEOUT"):
                await llm.call(messages=[{"role": "user", "content": "q"}], max_retries=0)
        finally:
            llm._stop_patch()


def test_missing_cli_raises_actionable_error():
    with patch("hindsight_api.engine.providers.cursor_llm.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="cursor-agent CLI not found"):
            CursorLLM(provider="cursor", api_key="", base_url="", model="auto")


def test_provider_is_registered():
    assert PROVIDER_DEFAULT_MODELS["cursor"] == "auto"
    assert not requires_api_key("cursor")
    with patch("hindsight_api.engine.providers.cursor_llm.shutil.which", return_value="/usr/bin/cursor-agent"):
        llm = create_llm_provider(provider="cursor", api_key="", base_url="", model="auto", reasoning_effort=None)
    assert isinstance(llm, CursorLLM)
