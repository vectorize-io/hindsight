"""Replayed tool calls from other providers must satisfy Gemini 3's signature check.

A multi-LLM failover or round-robin chain can hand a half-finished tool loop to a
Gemini member. The assistant tool calls in that history were made by another
provider and carry no ``thought_signature``; Gemini 3 answers such a request with
``400 Function call is missing a thought_signature in functionCall parts``.
Google's documented answer is a bypass value on the first functionCall of each
step. Calls Gemini made itself keep their real signature, and its unsigned
parallel followers stay unsigned.
"""

import base64

import pytest

pytest.importorskip("google.genai")

from hindsight_api.engine.providers.gemini_llm import _convert_messages_to_gemini  # noqa: E402

_BYPASS = b"skip_thought_signature_validator"


def _call(call_id: str, name: str = "search", signature: str | None = None) -> dict:
    tc: dict = {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}
    if signature is not None:
        tc["thought_signature"] = signature
    return tc


def _assistant(*calls: dict, content: str = "") -> dict:
    return {"role": "assistant", "content": content, "tool_calls": list(calls)}


def _result(call_id: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": "ok"}


def test_foreign_tool_call_gets_bypass_signature():
    contents = _convert_messages_to_gemini(
        [{"role": "user", "content": "q"}, _assistant(_call("a")), _result("a")]
    ).contents

    assert contents[1].parts[0].thought_signature == _BYPASS


def test_foreign_parallel_tool_calls_bypass_only_first():
    contents = _convert_messages_to_gemini(
        [
            {"role": "user", "content": "q"},
            _assistant(_call("a"), _call("b")),
            _result("a"),
            _result("b"),
        ]
    ).contents

    assert [p.thought_signature for p in contents[1].parts] == [_BYPASS, None]


def test_foreign_sequential_steps_each_get_bypass():
    contents = _convert_messages_to_gemini(
        [
            {"role": "user", "content": "q"},
            _assistant(_call("a")),
            _result("a"),
            _assistant(_call("b")),
            _result("b"),
        ]
    ).contents

    assert contents[1].parts[0].thought_signature == _BYPASS
    assert contents[3].parts[0].thought_signature == _BYPASS


def test_gemini_signature_is_decoded_and_preserved():
    signature = base64.b64encode(b"real-sig").decode()
    contents = _convert_messages_to_gemini(
        [
            {"role": "user", "content": "q"},
            _assistant(_call("a", signature=signature), _call("b")),
            _result("a"),
            _result("b"),
        ]
    ).contents

    assert [p.thought_signature for p in contents[1].parts] == [b"real-sig", None]


def test_text_part_before_tool_calls_does_not_shift_bypass():
    contents = _convert_messages_to_gemini(
        [{"role": "user", "content": "q"}, _assistant(_call("a"), content="thinking..."), _result("a")]
    ).contents

    text_part, call_part = contents[1].parts
    assert text_part.thought_signature is None
    assert call_part.function_call is not None
    assert call_part.thought_signature == _BYPASS


def test_cache_delta_slice_converts_identically():
    """The cache path converts ``messages[k:]``; it must equal the tail of the full conversion."""
    messages = [
        {"role": "user", "content": "q"},
        _assistant(_call("a")),
        _result("a"),
        _assistant(_call("b"), _call("c")),
        _result("b"),
        _result("c"),
    ]
    k = 3  # the second assistant message: a step boundary, never a bare tool result

    full = _convert_messages_to_gemini(messages).contents
    delta = _convert_messages_to_gemini(messages[k:]).contents

    assert delta == full[k:]
