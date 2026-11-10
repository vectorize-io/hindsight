"""The json_object hint goes into a multimodal user turn, never a second user turn after it.

Strict chat templates (Gemma's under llama.cpp or vLLM) refuse two user turns in a row with
"Conversation roles must alternate", and retaining an image in json_object mode sent one (#6349).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.engine.providers.openai_compatible_llm import (
    JSON_MODE_USER_HINT,
    OpenAICompatibleLLM,
    _ensure_json_word_in_user_message,
)

IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}


class _Resp(BaseModel):
    ok: bool


def test_hint_goes_into_the_multimodal_turn() -> None:
    messages = [
        {"role": "system", "content": "Extract facts."},
        {"role": "user", "content": [{"type": "text", "text": "Dan sent a diagram"}, IMAGE]},
    ]
    out = _ensure_json_word_in_user_message(messages)
    assert [m["role"] for m in out] == ["system", "user"]
    assert out[1]["content"][0]["text"].startswith(JSON_MODE_USER_HINT)
    assert out[1]["content"][1] == IMAGE
    assert messages[1]["content"][0]["text"] == "Dan sent a diagram"  # input not mutated


def test_image_only_turn_gets_a_text_part() -> None:
    out = _ensure_json_word_in_user_message([{"role": "user", "content": [IMAGE]}])
    assert out == [{"role": "user", "content": [{"type": "text", "text": JSON_MODE_USER_HINT}, IMAGE]}]


def test_responses_api_parts_get_an_input_text_part() -> None:
    image = {"type": "input_image", "image_url": "data:x", "detail": "auto"}
    out = _ensure_json_word_in_user_message([{"role": "user", "content": [image]}])
    assert out[0]["content"] == [{"type": "input_text", "text": JSON_MODE_USER_HINT}, image]


def test_existing_json_word_in_multimodal_turn_is_left_alone() -> None:
    messages = [{"role": "user", "content": [{"type": "text", "text": "answer in json"}, IMAGE]}]
    assert _ensure_json_word_in_user_message(messages) == messages


@pytest.mark.asyncio
async def test_vision_retain_request_keeps_turns_alternating() -> None:
    llm = OpenAICompatibleLLM(provider="openai", api_key="k", base_url="https://example.test/v1", model="m")
    choice = SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content='{"ok": true}', tool_calls=None, refusal=None)
    )
    create = AsyncMock(return_value=SimpleNamespace(error=None, usage=None, choices=[choice]))
    llm._client.chat.completions.create = create
    # Pinned: CI turns json_object off server-wide (HINDSIGHT_API_LLM_OPENAI_COMPATIBLE_JSON_MODE).
    with (
        patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"),
        patch.object(llm, "_supports_json_mode", return_value=True),
    ):
        await llm.call(
            messages=[
                {"role": "system", "content": "Extract facts."},
                {"role": "user", "content": [{"type": "text", "text": "Dan sent a diagram"}, IMAGE]},
            ],
            response_format=_Resp,
            strict_schema=False,
            max_retries=0,
        )
    kwargs = create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in kwargs["messages"]] == ["system", "user"]
    assert kwargs["messages"][1]["content"][1] == IMAGE
