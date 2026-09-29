"""Regression tests for issue #3599: num_ctx must reach free-form Ollama calls.

Ollama's OpenAI-compatible handler decodes a fixed field set and drops the rest,
so ``num_ctx`` cannot be expressed on ``/v1/chat/completions`` at all — not as a
top-level field and not nested under ``options``. Only the native ``/api/chat``
body carries it. Because Ollama keys a loaded model instance by context size, a
free-form call landing on the compatible endpoint reloads the model at the server
default and re-tunes it for every other consumer of a shared host; the startup
``verify_connection()`` probe did exactly that.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM
from tests.ollama_stub import chat_body, ollama_stub

NUM_CTX = 24576


class _Answer(BaseModel):
    answer: str


def _native_response(content: str) -> dict:
    return chat_body(content, done_reason="stop", prompt_eval_count=11, eval_count=3)


def _make_llm(base_url: str, **kwargs) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="ollama",
        api_key="",
        base_url=base_url,
        model="llama3.2",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_free_form_call_goes_native_and_carries_num_ctx():
    """A configured num_ctx routes free-form calls to the endpoint that honours it."""
    async with ollama_stub(_native_response("hello there")) as stub:
        llm = _make_llm(stub.openai_base_url, ollama_num_ctx=NUM_CTX)
        result = (
            await llm.call(
                messages=[{"role": "user", "content": "hi"}],
                max_completion_tokens=64,
                max_retries=0,
            )
        ).content
        await llm.cleanup()

    payload = stub.requests[-1].json
    assert stub.requests[-1].path == "/api/chat"
    assert payload["options"]["num_ctx"] == NUM_CTX
    assert "format" not in payload  # free-form: no schema enforcement
    assert result == "hello there"


@pytest.mark.asyncio
async def test_verify_connection_probe_carries_num_ctx():
    """The startup probe must not reload the model at the server default (#3599)."""
    async with ollama_stub(_native_response("ok")) as stub:
        llm = _make_llm(stub.openai_base_url, ollama_num_ctx=NUM_CTX)
        await llm.verify_connection()
        await llm.cleanup()

    assert stub.requests[-1].path == "/api/chat"
    assert stub.requests[-1].json["options"]["num_ctx"] == NUM_CTX


@pytest.mark.asyncio
async def test_free_form_call_without_num_ctx_stays_on_openai_endpoint():
    """Unset override keeps the existing transport, so nothing changes by default."""
    async with ollama_stub(_native_response("unused")) as stub:
        llm = _make_llm(stub.openai_base_url)
        create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="hello there"))],
                usage=None,
            )
        )
        llm._client.chat.completions.create = create
        result = (await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)).content
        await llm.cleanup()

    assert result == "hello there"
    assert create.await_count == 1
    assert stub.requests == []


@pytest.mark.asyncio
async def test_structured_call_still_validates_against_the_schema():
    """Routing free-form calls native must not disturb the structured path."""
    async with ollama_stub(_native_response(json.dumps({"answer": "42"}))) as stub:
        llm = _make_llm(stub.openai_base_url, ollama_num_ctx=NUM_CTX)
        result = (
            await llm.call(
                messages=[{"role": "user", "content": "hi"}],
                response_format=_Answer,
                max_retries=0,
            )
        ).content
        await llm.cleanup()

    assert "format" in stub.requests[-1].json
    assert isinstance(result, _Answer)
    assert result.answer == "42"


@pytest.mark.asyncio
async def test_free_form_native_strips_reasoning_tags():
    """Reasoning models leak <think> blocks into the body, as on the other path."""
    async with ollama_stub(_native_response("<think>weighing it up</think>Paris")) as stub:
        llm = _make_llm(stub.openai_base_url, ollama_num_ctx=NUM_CTX)
        result = (await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)).content
        await llm.cleanup()

    assert result == "Paris"


@pytest.mark.asyncio
async def test_free_form_native_retries_empty_content():
    """An empty message is retried rather than returned as a valid empty answer."""
    async with ollama_stub(_native_response(""), _native_response("second try")) as stub:
        llm = _make_llm(stub.openai_base_url, ollama_num_ctx=NUM_CTX)
        result = (
            await llm.call(
                messages=[{"role": "user", "content": "hi"}],
                max_retries=1,
                initial_backoff=0.0,
                max_backoff=0.0,
            )
        ).content
        await llm.cleanup()

    assert result == "second try"
    assert len(stub.requests) == 2
