"""
Regression tests for Codex structured output (issue #2504).

Before the fix, ``CodexLLM.call(strict_schema=True)`` was a dead no-op: structured
output always went through prompt-injected schema + raw ``json.loads`` on the
model's free-form text. Escape-heavy content (code, serial/CLI commands, Windows
paths, regexes) makes weaker models emit invalid ``\\escape`` sequences, so every
parse attempt fails and retain/consolidation burn all retries and fail.

The fix:
- ``strict_schema=True`` routes structured output through a single forced function
  tool (constrained decoding into the response schema).
- The non-strict fallback now repairs invalid ``\\escape`` sequences before giving up.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.engine.providers.codex_llm import (
    CodexLLM,
    _repair_invalid_json_escapes,
)
from hindsight_api.engine.response_models import LLMToolCall


class _Fact(BaseModel):
    fact: str


def build_llm() -> CodexLLM:
    with patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")):
        return CodexLLM(
            provider="openai-codex",
            api_key="ignored",
            base_url="https://chatgpt.com/backend-api",
            model="gpt-5.4-mini",
        )


# ---------------------------------------------------------------------------
# _repair_invalid_json_escapes — pure unit tests
# ---------------------------------------------------------------------------


def test_repair_fixes_invalid_escape_in_json():
    # `\d` and `\s` are not valid JSON escapes; raw json.loads fails.
    broken = r'{"fact": "regex \d+\s matches digits"}'
    import json

    with pytest.raises(json.JSONDecodeError):
        json.loads(broken)
    repaired = _repair_invalid_json_escapes(broken)
    assert json.loads(repaired) == {"fact": r"regex \d+\s matches digits"}


def test_repair_preserves_valid_escapes():
    import json

    valid = r'{"fact": "line1\nline2\ttab \"quoted\" \\backslash é"}'
    # Already valid — repair must not corrupt it.
    assert json.loads(_repair_invalid_json_escapes(valid)) == json.loads(valid)


def test_repair_handles_windows_paths():
    import json

    # Uses path segments whose first char isn't a valid JSON escape letter
    # (b/f/n/r/t/u), where the repair is unambiguous.
    broken = r'{"path": "C:\Windows\System32\app.exe"}'
    assert json.loads(_repair_invalid_json_escapes(broken)) == {"path": r"C:\Windows\System32\app.exe"}


def test_repair_handles_trailing_backslash():
    # A lone trailing backslash must be escaped, not dropped.
    assert _repair_invalid_json_escapes("abc\\") == "abc\\\\"


# ---------------------------------------------------------------------------
# strict_schema forced-tool path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_strict_schema_uses_forced_function_tool():
    llm = build_llm()
    response = MagicMock()
    response.raise_for_status.return_value = None
    tool_call = LLMToolCall(id="call-1", name="structured_response", arguments={"fact": "the sky is blue"})

    with patch.object(llm._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = response
        with patch.object(llm, "_parse_sse_tool_stream", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = (None, [tool_call])
            result = await llm.call(
                messages=[{"role": "user", "content": "The sky is blue"}],
                response_format=_Fact,
                strict_schema=True,
                max_retries=0,
            )
        sent_payload = mock_post.call_args.kwargs["json"]

    # Forced tool wired into the request payload.
    assert sent_payload["tool_choice"] == {"type": "function", "name": "structured_response"}
    assert len(sent_payload["tools"]) == 1
    assert sent_payload["tools"][0]["name"] == "structured_response"
    assert sent_payload["parallel_tool_calls"] is False
    # No prompt-injected schema in the instructions.
    assert "You must respond with valid JSON" not in sent_payload["instructions"]

    assert isinstance(result, _Fact)
    assert result.fact == "the sky is blue"


@pytest.mark.asyncio
async def test_strict_schema_skip_validation_returns_dict():
    llm = build_llm()
    response = MagicMock()
    response.raise_for_status.return_value = None
    tool_call = LLMToolCall(id="c", name="structured_response", arguments={"fact": "x"})

    with patch.object(llm._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = response
        with patch.object(llm, "_parse_sse_tool_stream", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = (None, [tool_call])
            result = await llm.call(
                messages=[{"role": "user", "content": "hi"}],
                response_format=_Fact,
                strict_schema=True,
                skip_validation=True,
                max_retries=0,
            )

    assert result == {"fact": "x"}


@pytest.mark.asyncio
async def test_strict_schema_retries_when_forced_tool_missing():
    llm = build_llm()
    response = MagicMock()
    response.raise_for_status.return_value = None

    with patch.object(llm._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = response
        # Model returns no tool call at all — should raise after retries exhausted.
        with patch.object(llm, "_parse_sse_tool_stream", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = ("some prose", [])
            with pytest.raises(RuntimeError, match="structured_response"):
                await llm.call(
                    messages=[{"role": "user", "content": "hi"}],
                    response_format=_Fact,
                    strict_schema=True,
                    max_retries=0,
                )


# ---------------------------------------------------------------------------
# Non-strict fallback: escape repair keeps the retry storm from happening
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_strict_repairs_invalid_escapes_without_retrying():
    llm = build_llm()
    response = MagicMock()
    response.raise_for_status.return_value = None
    # Escape-heavy content the model would emit as invalid JSON.
    escape_heavy = r'{"fact": "run rig-control \d serial \s command"}'

    with patch.object(llm._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = response
        with patch.object(llm, "_parse_sse_stream", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = escape_heavy
            result = await llm.call(
                messages=[{"role": "user", "content": "coding transcript"}],
                response_format=_Fact,
                strict_schema=False,
                max_retries=3,
            )

    # Parsed on the first attempt (no retry storm): the SSE stream was read once.
    assert mock_post.await_count == 1
    assert isinstance(result, _Fact)
    assert result.fact == r"run rig-control \d serial \s command"
