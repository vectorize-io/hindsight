"""
Regression tests for Codex structured output (issue #2504).

Before the fix, ``CodexLLM.call(strict_schema=True)`` was a dead no-op: structured
output always went through prompt-injected schema + raw ``json.loads`` on the
model's free-form text. Escape-heavy content (code, serial/CLI commands, Windows
paths, regexes) makes weaker models emit invalid ``\\escape`` sequences, so every
parse attempt fails and retain/consolidation burn all retries and fail.

The fix:
- ``strict_schema=True`` holds the reply to the response schema through the Responses
  API's ``text.format`` json_schema (constrained decoding). It was a single forced
  function tool until no request could carry tools (see ``tools_via_schema``).
- The non-strict fallback now repairs invalid ``\\escape`` sequences before giving up.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.engine.providers.codex_llm import (
    CodexLLM,
    _CodexStreamReply,
    _repair_invalid_json_escapes,
)
from hindsight_api.engine.providers.tools_via_schema import ResponseSchema, schema_format
from tests.codex_stream_stub import CodexReply, stub_codex_stream


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
# strict_schema: text.format json_schema
# ---------------------------------------------------------------------------


async def _strict_call(llm: CodexLLM, reply: str, response_format, **kwargs):
    response = MagicMock(status_code=200)
    response.raise_for_status.return_value = None
    with stub_codex_stream(llm, response) as mock_stream:
        with patch.object(llm, "_parse_sse_stream", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = _CodexStreamReply(text=reply, usage=None)
            result = await llm.call(
                messages=[{"role": "user", "content": "The sky is blue"}],
                response_format=response_format,
                strict_schema=True,
                max_retries=0,
                **kwargs,
            )
    return mock_stream.call_args.kwargs, result.content


@pytest.mark.asyncio
async def test_strict_schema_holds_the_reply_to_the_schema_and_sends_no_tools():
    llm = build_llm()
    sent, result = await _strict_call(llm, '{"fact": "the sky is blue"}', _Fact)
    payload = sent["json"]

    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True
    assert payload["text"]["format"]["schema"]["properties"] == {"fact": {"title": "Fact", "type": "string"}}
    assert not {"tools", "tool_choice", "parallel_tool_calls"} & payload.keys()
    assert sent["headers"]["originator"] == "codex_cli_rs"
    assert sent["headers"]["User-Agent"] == "codex_cli_rs/0.0.0 (Hindsight)"
    # No prompt-injected schema in the instructions.
    assert "You must respond with valid JSON" not in payload["instructions"]
    assert result == _Fact(fact="the sky is blue")


@pytest.mark.asyncio
async def test_strict_schema_skip_validation_returns_dict():
    llm = build_llm()
    span_recorder = MagicMock()
    with patch("hindsight_api.tracing.get_span_recorder", return_value=span_recorder):
        _, result = await _strict_call(llm, '{"fact": "x"}', _Fact, skip_validation=True)

    assert result == {"fact": "x"}
    assert span_recorder.record_llm_call.call_args.kwargs["response_content"] == '{"fact": "x"}'


@pytest.mark.asyncio
async def test_a_tool_turn_schema_reaches_the_request_and_comes_back_as_a_dict():
    """Reflect's tool turns: ``schema_format`` carries a raw JSON schema through ``call``."""
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }
    sent, result = await _strict_call(
        build_llm(),
        '{"query": "editors"}',
        schema_format(ResponseSchema(name="recall", schema=schema)),
        skip_validation=True,
    )

    assert sent["json"]["text"]["format"]["name"] == "recall"
    assert sent["json"]["text"]["format"]["schema"] == schema
    assert result == {"query": "editors"}


@pytest.mark.asyncio
async def test_a_reply_streamed_as_output_text_is_read_once():
    """The Responses API streams text as ``response.output_text.delta`` and repeats it in the
    finished item: the deltas are the reply, the item only a fallback for a stream without them."""
    llm = build_llm()
    deltas = "".join(
        "event: response.output_text.delta\ndata: "
        + json.dumps({"type": "response.output_text.delta", "delta": d})
        + "\n\n"
        for d in ('{"fact": ', '"editors"}')
    )
    item = {"type": "message", "content": [{"type": "output_text", "text": '{"fact": "editors"}'}]}
    done = "event: response.output_item.done\ndata: " + json.dumps({"item": item}) + "\n\n"
    with stub_codex_stream(llm, CodexReply(text=deltas + done)):
        result = await llm.call(
            messages=[{"role": "user", "content": "x"}], response_format=_Fact, strict_schema=True, max_retries=0
        )
    assert result.content == _Fact(fact="editors")


# ---------------------------------------------------------------------------
# Non-strict fallback: escape repair keeps the retry storm from happening
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_strict_repairs_invalid_escapes_without_retrying():
    llm = build_llm()
    response = MagicMock(status_code=200)
    response.raise_for_status.return_value = None
    # Escape-heavy content the model would emit as invalid JSON.
    escape_heavy = r'{"fact": "run rig-control \d serial \s command"}'

    with stub_codex_stream(llm, response) as mock_stream:
        with patch.object(llm, "_parse_sse_stream", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = _CodexStreamReply(text=escape_heavy, usage=None)
            result = (
                await llm.call(
                    messages=[{"role": "user", "content": "coding transcript"}],
                    response_format=_Fact,
                    strict_schema=False,
                    max_retries=3,
                )
            ).content

    # Parsed on the first attempt (no retry storm): the SSE stream was read once.
    assert mock_stream.call_count == 1
    assert isinstance(result, _Fact)
    assert result.fact == r"run rig-control \d serial \s command"


@pytest.mark.asyncio
async def test_non_strict_keeps_a_fence_marker_inside_a_json_value():
    """A ``` inside a string value must not cut the payload short (#4819)."""
    llm = build_llm()
    response = MagicMock(status_code=200)
    response.raise_for_status.return_value = None

    with stub_codex_stream(llm, response):
        with patch.object(llm, "_parse_sse_stream", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = _CodexStreamReply(
                text='```json\n{"fact": "wrap it in ```json fences"}\n```', usage=None
            )
            result = (
                await llm.call(
                    messages=[{"role": "user", "content": "hi"}],
                    response_format=_Fact,
                    strict_schema=False,
                    max_retries=0,
                )
            ).content

    assert result.fact == "wrap it in ```json fences"
