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
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import BaseModel

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

NUM_CTX = 24576


class _Answer(BaseModel):
    answer: str


def _native_response(content: str) -> httpx.Response:
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    return httpx.Response(
        200,
        json={
            "model": "llama3.2",
            "message": {"role": "assistant", "content": content},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 11,
            "eval_count": 3,
        },
        request=request,
    )


def _make_llm(**kwargs) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="ollama",
        api_key="",
        base_url="http://localhost:11434/v1",
        model="llama3.2",
        **kwargs,
    )


def _patch_native(*responses: httpx.Response):
    """Patch httpx.AsyncClient so native /api/chat posts return ``responses`` in order."""
    client = AsyncMock()
    client.post.side_effect = list(responses)
    client.__aenter__.return_value = client
    return client, patch(
        "hindsight_api.engine.providers.openai_compatible_llm.httpx.AsyncClient",
        return_value=client,
    )


@pytest.mark.asyncio
async def test_free_form_call_goes_native_and_carries_num_ctx():
    """A configured num_ctx routes free-form calls to the endpoint that honours it."""
    llm = _make_llm(ollama_num_ctx=NUM_CTX)
    client, patched = _patch_native(_native_response("hello there"))

    with patched:
        result = await llm.call(
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=64,
            max_retries=0,
        )

    payload = client.post.call_args.kwargs["json"]
    assert client.post.call_args.args[0] == "http://localhost:11434/api/chat"
    assert payload["options"]["num_ctx"] == NUM_CTX
    assert "format" not in payload  # free-form: no schema enforcement
    assert result == "hello there"


@pytest.mark.asyncio
async def test_verify_connection_probe_carries_num_ctx():
    """The startup probe must not reload the model at the server default (#3599)."""
    llm = _make_llm(ollama_num_ctx=NUM_CTX)
    client, patched = _patch_native(_native_response("ok"))

    with patched:
        await llm.verify_connection()

    assert client.post.call_args.args[0] == "http://localhost:11434/api/chat"
    assert client.post.call_args.kwargs["json"]["options"]["num_ctx"] == NUM_CTX


@pytest.mark.asyncio
async def test_free_form_call_without_num_ctx_stays_on_openai_endpoint():
    """Unset override keeps the existing transport, so nothing changes by default."""
    llm = _make_llm()
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="hello there"))],
            usage=None,
        )
    )
    llm._client.chat.completions.create = create
    client, patched = _patch_native(_native_response("unused"))

    with patched:
        result = await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert result == "hello there"
    assert create.await_count == 1
    client.post.assert_not_called()


@pytest.mark.asyncio
async def test_structured_call_still_validates_against_the_schema():
    """Routing free-form calls native must not disturb the structured path."""
    llm = _make_llm(ollama_num_ctx=NUM_CTX)
    client, patched = _patch_native(_native_response(json.dumps({"answer": "42"})))

    with patched:
        result = await llm.call(
            messages=[{"role": "user", "content": "hi"}],
            response_format=_Answer,
            max_retries=0,
        )

    assert "format" in client.post.call_args.kwargs["json"]
    assert isinstance(result, _Answer)
    assert result.answer == "42"


@pytest.mark.asyncio
async def test_free_form_native_strips_reasoning_tags():
    """Reasoning models leak <think> blocks into the body, as on the other path."""
    llm = _make_llm(ollama_num_ctx=NUM_CTX)
    client, patched = _patch_native(_native_response("<think>weighing it up</think>Paris"))

    with patched:
        result = await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert result == "Paris"


@pytest.mark.asyncio
async def test_free_form_native_retries_empty_content():
    """An empty message is retried rather than returned as a valid empty answer."""
    llm = _make_llm(ollama_num_ctx=NUM_CTX)
    client, patched = _patch_native(_native_response(""), _native_response("second try"))

    with patched:
        result = await llm.call(
            messages=[{"role": "user", "content": "hi"}],
            max_retries=1,
            initial_backoff=0.0,
            max_backoff=0.0,
        )

    assert result == "second try"
    assert client.post.await_count == 2
