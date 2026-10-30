"""Multimodal content parts reach Responses-API providers in Responses shape (#6172).

Retain builds chat/completions parts (``text`` / ``image_url`` / ``file``); the
Responses API (openai-responses, chatgpt, openai-codex) rejects those type names
with HTTP 400, so each provider must translate them before sending.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.providers.codex_llm import CodexLLM, _CodexStreamReply
from hindsight_api.engine.providers.openai_responses_llm import (
    _messages_to_responses_input,
    _to_responses_content,
)
from tests.codex_stream_stub import stub_codex_stream

DATA_URI = "data:image/png;base64,iVBORw0KGgo="

CHAT_PARTS = [
    {"type": "text", "text": "What colour is this?"},
    {"type": "image_url", "image_url": {"url": DATA_URI}},
    {"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBERi0=", "filename": "a.pdf"}},
]

RESPONSES_PARTS = [
    {"type": "input_text", "text": "What colour is this?"},
    {"type": "input_image", "image_url": DATA_URI, "detail": "auto"},
    {"type": "input_file", "file_data": "data:application/pdf;base64,JVBERi0=", "filename": "a.pdf"},
]


def test_chat_parts_become_responses_parts() -> None:
    assert _to_responses_content(CHAT_PARTS, "user") == RESPONSES_PARTS


def test_plain_string_and_responses_parts_pass_through() -> None:
    assert _to_responses_content("hello", "user") == "hello"
    assert _to_responses_content(RESPONSES_PARTS, "user") == RESPONSES_PARTS


def test_assistant_text_becomes_output_text_and_detail_is_kept() -> None:
    assert _to_responses_content([{"type": "text", "text": "hi"}], "assistant") == [
        {"type": "output_text", "text": "hi"}
    ]
    assert _to_responses_content([{"type": "image_url", "image_url": {"url": DATA_URI, "detail": "high"}}], "user") == [
        {"type": "input_image", "image_url": DATA_URI, "detail": "high"}
    ]


def test_messages_to_responses_input_translates_parts() -> None:
    items = _messages_to_responses_input(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": CHAT_PARTS}]
    )
    assert items == [{"role": "system", "content": "sys"}, {"role": "user", "content": RESPONSES_PARTS}]


@pytest.mark.asyncio
async def test_codex_call_sends_responses_parts() -> None:
    with (
        patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")),
        patch.object(CodexLLM, "_load_codex_refresh_token", return_value=None),
    ):
        llm = CodexLLM(
            provider="openai-codex",
            api_key="ignored",
            base_url="https://chatgpt.com/backend-api",
            model="gpt-5.6-luna",
        )
    response = MagicMock(status_code=200)
    response.raise_for_status.return_value = None

    with (
        stub_codex_stream(llm, response) as mock_stream,
        patch.object(
            llm, "_parse_sse_stream", new_callable=AsyncMock, return_value=_CodexStreamReply(text="red", usage=None)
        ),
    ):
        await llm.call(messages=[{"role": "user", "content": CHAT_PARTS}], max_retries=0)

    sent = mock_stream.call_args.kwargs["json"]["input"]
    assert sent == [{"type": "message", "role": "user", "content": RESPONSES_PARTS}]
