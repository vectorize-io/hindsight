"""Gemini 3 requires a thought_signature on every functionCall part of the current turn.

Tool calls replayed from another provider (failover mid tool loop) have none, and the
request is rejected with HTTP 400 unless we attach Google's bypass literal (issue #5122).
"""

import base64
from typing import Any

from google.genai import types as genai_types

from hindsight_api.engine.providers.gemini_llm import (
    _SKIP_THOUGHT_SIGNATURE_VALIDATOR,
    _convert_messages_to_gemini,
)

_MESSAGES = [
    {"role": "user", "content": "weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "18C"},
]


def _function_call_parts(contents: list[genai_types.Content]) -> list[genai_types.Part]:
    return [p for c in contents for p in (c.parts or []) if p.function_call is not None]


def test_unsigned_tool_call_gets_bypass_signature() -> None:
    conv = _convert_messages_to_gemini(_MESSAGES)
    parts = _function_call_parts(conv.contents)
    assert len(parts) == 1
    assert parts[0].thought_signature == _SKIP_THOUGHT_SIGNATURE_VALIDATOR


def test_real_signature_is_preserved() -> None:
    signature = b"\x01\x02real-signature"
    messages: list[dict[str, Any]] = [dict(m) for m in _MESSAGES]
    messages[1] = {
        **messages[1],
        "tool_calls": [
            {
                **messages[1]["tool_calls"][0],
                "thought_signature": base64.b64encode(signature).decode(),
            }
        ],
    }
    parts = _function_call_parts(_convert_messages_to_gemini(messages).contents)
    assert len(parts) == 1
    assert parts[0].thought_signature == signature
