"""Unit tests for the Cursor CLI-backed LLM provider.

The CLI is replaced by a fake subprocess: what is under test is the seam between
Hindsight and `cursor-agent` — how the prompt is assembled, how the child process is
confined, how the emitted `type: "result"` line is found among the CLI's diagnostics,
and how structured output and tool calls are parsed back out of free-form text.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
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
        self.killed = False

    async def communicate(self, payload: bytes | None = None):
        self.stdin_payload = payload
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> None:
        return None


class _HangingProc(_FakeProc):
    """Never answers, so `asyncio.wait_for` in the provider times out."""

    async def communicate(self, payload: bytes | None = None):
        self.stdin_payload = payload
        await asyncio.sleep(10)
        raise AssertionError("should have timed out")


@dataclass
class _Spawned:
    """What the provider handed to `create_subprocess_exec`, per call."""

    llm: CursorLLM
    procs: list[_FakeProc] = field(default_factory=list)
    args: list[list[str]] = field(default_factory=list)
    kwargs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def prompt(self) -> str:
        """The prompt written to the first spawned process's stdin."""
        payload = self.procs[0].stdin_payload
        assert payload is not None
        return payload.decode()


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


@contextmanager
def _provider(
    stdout: str,
    *,
    returncode: int = 0,
    stderr: str = "",
    api_key: str = "",
    proc_cls: type[_FakeProc] = _FakeProc,
) -> Iterator[_Spawned]:
    """Build a CursorLLM whose CLI is a fake process, and always unpatch after."""
    with patch("hindsight_api.engine.providers.cursor_llm.shutil.which", return_value="/usr/bin/cursor-agent"):
        llm = CursorLLM(provider="cursor", api_key=api_key, base_url="", model="auto")
    spawned = _Spawned(llm=llm)

    async def fake_exec(*args, **kwargs):
        spawned.args.append(list(args))
        spawned.kwargs.append(kwargs)
        proc = proc_cls(stdout, returncode=returncode, stderr=stderr)
        spawned.procs.append(proc)
        return proc

    with patch("hindsight_api.engine.providers.cursor_llm.asyncio.create_subprocess_exec", fake_exec):
        yield spawned


@pytest.mark.asyncio
async def test_plain_call_sends_flattened_prompt_and_reports_usage():
    with _provider(_result_line("hello there")) as s:
        result = await s.llm.call(
            messages=[{"role": "system", "content": "Be terse"}, {"role": "user", "content": "Say hi"}],
            max_retries=0,
        )

    assert result.content == "hello there"
    assert result.usage.input_tokens == 120
    assert result.usage.output_tokens == 7
    assert "Be terse" in s.prompt and "Say hi" in s.prompt
    # --mode ask is the CLI's read-only mode and the only tool restriction it offers;
    # it is the whole reason a prompt-injected "run this command" cannot edit or shell.
    assert s.args[0][1:] == ["-p", "--output-format", "json", "--trust", "--mode", "ask", "--model", "auto"]


@pytest.mark.asyncio
async def test_child_is_confined_to_scratch_dirs_and_never_sees_the_key_in_argv():
    """Isolation is the property, not the flag: empty cwd, own config dir, key in env."""
    with _provider(_result_line("ok"), api_key="sk-secret") as s:
        await s.llm.call(messages=[{"role": "user", "content": "q"}], max_retries=0)

    argv, kwargs = s.args[0], s.kwargs[0]
    # A credential in argv is readable by any local user via `ps`.
    assert "sk-secret" not in argv
    assert "--api-key" not in argv
    assert kwargs["env"]["CURSOR_API_KEY"] == "sk-secret"

    # CURSOR_CONFIG_DIR keeps session transcripts out of the operator's ~/.cursor,
    # and keeps the CLI from reading a hooks.json that points back at Hindsight.
    config_dir = kwargs["env"]["CURSOR_CONFIG_DIR"]
    assert "hindsight-cursor-cfg-" in config_dir
    assert "hindsight-cursor-ws-" in kwargs["cwd"]
    assert kwargs["cwd"] != config_dir


def test_isolated_config_denies_every_advertised_tool():
    """The CLI has no tools=[] flag, so the deny list is what keeps it a text completion.

    --mode ask is not a tool switch: without these rules a headless run reads files out
    of its cwd. A single "*" rule is silently ignored by the CLI, so every tool has to be
    named — which is why this asserts the shape, and why the names are worth reviewing
    when the tested CLI version moves.
    """
    from hindsight_api.engine.providers import cursor_llm

    dirs = cursor_llm._get_dirs()
    config = json.loads(Path(dirs.config_dir, "cli-config.json").read_text())

    deny = set(config["permissions"]["deny"])
    assert config["permissions"]["allow"] == []
    # The ones that reach the host or the network, spelled out so a silent drop fails here.
    for tool in ("Shell", "Read", "Write", "Delete", "WebFetch", "CallDynamicTool"):
        assert f"{tool}(*)" in deny, f"{tool} is not denied"
    assert all(rule.endswith("(*)") for rule in deny)


@pytest.mark.asyncio
async def test_structured_output_injects_schema_and_parses_fenced_json():
    with _provider(_result_line('Sure!\n```json\n{"answer": "42"}\n```')) as s:
        result = await s.llm.call(messages=[{"role": "user", "content": "q"}], response_format=_Answer, max_retries=0)

    assert isinstance(result.content, _Answer)
    assert result.content.answer == "42"
    assert "valid JSON matching this schema" in s.prompt


@pytest.mark.asyncio
async def test_structured_output_recovers_json_wrapped_in_prose():
    """The unfenced prose case is the one this provider exists to absorb."""
    with _provider(_result_line('Here is what I found: {"answer": "42"} — hope that helps!')) as s:
        result = await s.llm.call(messages=[{"role": "user", "content": "q"}], response_format=_Answer, max_retries=0)

    assert isinstance(result.content, _Answer)
    assert result.content.answer == "42"


@pytest.mark.asyncio
async def test_unparseable_structured_output_retries_then_raises():
    with _provider(_result_line("I cannot help with that.")) as s:
        with pytest.raises(json.JSONDecodeError):
            await s.llm.call(
                messages=[{"role": "user", "content": "q"}],
                response_format=_Answer,
                max_retries=2,
                initial_backoff=0,
                max_backoff=0,
            )

    # A bad parse is worth another sample — unlike a schema violation, which is not.
    assert len(s.procs) == 3


@pytest.mark.asyncio
async def test_stdout_without_a_result_object_raises():
    """The path a CLI flag or version change lands on."""
    with _provider('cursor-retrieval: tracing to \'/tmp/x.log\'\n{"type":"progress","n":1}\n') as s:
        with pytest.raises(RuntimeError, match="produced no result object"):
            await s.llm.call(messages=[{"role": "user", "content": "q"}], max_retries=0)


@pytest.mark.asyncio
async def test_last_result_object_wins_over_earlier_lines():
    stdout = _result_line("first") + _result_line("second")
    with _provider(stdout) as s:
        result = await s.llm.call(messages=[{"role": "user", "content": "q"}], max_retries=0)

    assert result.content == "second"


@pytest.mark.asyncio
async def test_tool_calls_are_parsed_from_the_json_envelope():
    envelope = json.dumps({"content": None, "tool_calls": [{"name": "recall", "arguments": {"query": "rome"}}]})
    with _provider(_result_line(envelope)) as s:
        result = await s.llm.call_with_tools(
            messages=[{"role": "user", "content": "what do I know about rome?"}],
            tools=_TOOLS,
            tool_choice=LLM_TOOL_CHOICE_REQUIRED,
            max_retries=0,
        )

    assert result.finish_reason == "tool_calls"
    assert [(c.name, c.arguments) for c in result.tool_calls] == [("recall", {"query": "rome"})]
    assert result.input_tokens == 120
    assert result.output_tokens == 7
    assert "MUST call at least one tool" in s.prompt


@pytest.mark.asyncio
async def test_named_tool_choice_offers_only_that_tool():
    with _provider(_result_line(json.dumps({"tool_calls": [{"name": "recall", "arguments": {}}]}))) as s:
        await s.llm.call_with_tools(
            messages=[{"role": "user", "content": "go"}],
            tools=_TOOLS + [{"function": {"name": "other", "description": "", "parameters": {}}}],
            tool_choice=LLMToolChoice.named("recall"),
            max_retries=0,
        )

    assert "MUST call the 'recall' tool" in s.prompt
    assert '"other"' not in s.prompt


@pytest.mark.asyncio
async def test_hallucinated_tool_name_is_dropped():
    with _provider(_result_line(json.dumps({"tool_calls": [{"name": "delete_everything", "arguments": {}}]}))) as s:
        result = await s.llm.call_with_tools(messages=[{"role": "user", "content": "go"}], tools=_TOOLS, max_retries=0)

    assert result.tool_calls == []
    assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_tool_choice_none_asks_for_plain_text():
    with _provider(_result_line("just text")) as s:
        result = await s.llm.call_with_tools(
            messages=[{"role": "user", "content": "go"}],
            tools=_TOOLS,
            tool_choice=LLM_TOOL_CHOICE_NONE,
            max_retries=0,
        )

    assert result.content == "just text"
    assert result.tool_calls == []
    assert "Do not call any tools" in s.prompt


@pytest.mark.asyncio
async def test_cli_error_result_raises():
    with _provider(_result_line("Free plans can only use Auto.", is_error=True)) as s:
        with pytest.raises(RuntimeError, match="Free plans can only use Auto"):
            await s.llm.call(messages=[{"role": "user", "content": "q"}], max_retries=0)


@pytest.mark.asyncio
async def test_nonzero_exit_surfaces_stderr():
    with _provider("", returncode=1, stderr="ActionRequiredError: not logged in") as s:
        with pytest.raises(RuntimeError, match="not logged in"):
            await s.llm.call(messages=[{"role": "user", "content": "q"}], max_retries=0)


@pytest.mark.asyncio
async def test_blocked_prompt_is_permanent_and_not_retried():
    """A moderation block is the same on every replay, so it must not burn the retry budget."""
    blocked = (
        "ActionRequiredError: Request blocked We are unable to complete this request because "
        "it was blocked under the model provider's usage guidelines."
    )
    with _provider("", returncode=1, stderr=blocked) as s:
        with pytest.raises(ProviderContentPolicyError):
            await s.llm.call(messages=[{"role": "user", "content": "q"}], max_retries=5)

    assert len(s.procs) == 1


@pytest.mark.asyncio
async def test_timeout_kills_the_child_and_names_the_timeout_setting():
    with _provider("", proc_cls=_HangingProc) as s:
        s.llm.timeout = 0.01
        with pytest.raises(TimeoutError, match="REFLECT_LLM_TIMEOUT"):
            await s.llm.call(messages=[{"role": "user", "content": "q"}], max_retries=0)

    # A timed-out turn must not leave an orphaned cursor-agent behind.
    assert s.procs[0].killed


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
