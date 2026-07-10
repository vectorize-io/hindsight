"""Anthropic Message Batches support for the provider batch interface.

The engine's batch path (retain fact extraction, gated on
``retain_batch_enabled``) speaks the OpenAI batch wire shape: JSONL entries
with ``custom_id``/``method``/``url``/``body`` going in, and
``response.body.choices[0].message.content`` (+ OpenAI-keyed ``usage``) coming
out. ``AnthropicLLM`` translates both directions onto the Message Batches API,
which bills all token usage at 50% of standard price.

Translation rules mirror the provider's synchronous ``call()`` path:
- system messages fold into the ``system`` param;
- ``max_completion_tokens`` becomes ``max_tokens`` (default 4096);
- ``temperature`` is dropped (the sync path never sends it either — current
  Claude models reject non-default sampling params);
- ``response_format`` with ``strict=True`` becomes a single forced tool_use
  tool (native constrained decoding, issue #1002); non-strict injects the
  schema into the system prompt and expects JSON text back.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio


def _make_provider():
    with patch("anthropic.AsyncAnthropic") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        from hindsight_api.engine.providers.anthropic_llm import AnthropicLLM

        provider = AnthropicLLM(
            provider="anthropic",
            api_key="fake-key",
            base_url="",
            model="claude-sonnet-5",
        )
    provider._client = MagicMock()
    return provider


_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
    "required": ["facts"],
}


def _openai_request(custom_id: str, *, strict: bool = True, temperature: float | None = 0.1) -> dict:
    body = {
        "model": "claude-sonnet-5",
        "messages": [
            {"role": "system", "content": "Extract facts."},
            {"role": "user", "content": f"Text for {custom_id}"},
        ],
        "max_completion_tokens": 2000,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "facts", "schema": _SCHEMA, "strict": strict},
        },
    }
    if temperature is not None:
        body["temperature"] = temperature
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}


def _batch(status: str = "in_progress", **counts) -> SimpleNamespace:
    defaults = {"processing": 0, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}
    defaults.update(counts)
    return SimpleNamespace(
        id="msgbatch_test1",
        processing_status=status,
        created_at="2026-07-08T00:00:00Z",
        ended_at="2026-07-08T00:30:00Z" if status == "ended" else None,
        request_counts=SimpleNamespace(**defaults),
    )


class _AsyncIter:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        self._iter = iter(self._items)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None


def _succeeded_entry(custom_id: str, tool_input: dict) -> SimpleNamespace:
    block = SimpleNamespace(type="tool_use", name="structured_response", input=tool_input, text=None)
    message = SimpleNamespace(
        content=[block],
        usage=SimpleNamespace(input_tokens=100, output_tokens=40, cache_read_input_tokens=0),
        stop_reason="tool_use",
    )
    return SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type="succeeded", message=message))


def _errored_entry(custom_id: str) -> SimpleNamespace:
    error = SimpleNamespace(type="invalid_request", message="bad request")
    return SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type="errored", error=error))


async def test_supports_batch_api():
    provider = _make_provider()
    assert await provider.supports_batch_api() is True


async def test_submit_batch_translates_openai_requests():
    provider = _make_provider()
    provider._client.messages.batches.create = AsyncMock(return_value=_batch("in_progress", processing=2))

    requests = [_openai_request("chunk_0"), _openai_request("chunk_1")]
    metadata = await provider.submit_batch(requests)

    provider._client.messages.batches.create.assert_awaited_once()
    submitted = provider._client.messages.batches.create.await_args.kwargs["requests"]
    assert [r["custom_id"] for r in submitted] == ["chunk_0", "chunk_1"]

    params = submitted[0]["params"]
    assert params["model"] == "claude-sonnet-5"
    # System message folded into the system param, not left in messages.
    assert "Extract facts." in params["system"]
    assert all(m["role"] != "system" for m in params["messages"])
    assert params["messages"] == [{"role": "user", "content": "Text for chunk_0"}]
    assert params["max_tokens"] == 2000
    # temperature is dropped, mirroring the sync call() path.
    assert "temperature" not in params
    # strict=True → forced tool_use (native constrained decoding).
    assert params["tools"][0]["input_schema"] == _SCHEMA
    assert params["tool_choice"] == {"type": "tool", "name": "structured_response"}

    assert metadata["batch_id"] == "msgbatch_test1"
    assert metadata["status"] == "in_progress"
    assert metadata["request_count"] == 2


async def test_submit_batch_non_strict_schema_injects_into_system():
    provider = _make_provider()
    provider._client.messages.batches.create = AsyncMock(return_value=_batch("in_progress", processing=1))

    await provider.submit_batch([_openai_request("chunk_0", strict=False)])

    params = provider._client.messages.batches.create.await_args.kwargs["requests"][0]["params"]
    assert "tools" not in params
    assert "tool_choice" not in params
    # Schema is injected into the system prompt for JSON-text output.
    assert "facts" in params["system"]
    assert "valid JSON" in params["system"]


async def test_get_batch_status_in_progress():
    provider = _make_provider()
    provider._client.messages.batches.retrieve = AsyncMock(
        return_value=_batch("in_progress", processing=3, succeeded=1)
    )

    status = await provider.get_batch_status("msgbatch_test1")

    assert status["batch_id"] == "msgbatch_test1"
    assert status["status"] == "in_progress"
    assert status["request_counts"]["total"] == 4
    assert status["request_counts"]["completed"] == 1


async def test_get_batch_status_ended_maps_to_completed():
    """The engine's poll loop breaks on the OpenAI-vocabulary status 'completed'."""
    provider = _make_provider()
    provider._client.messages.batches.retrieve = AsyncMock(return_value=_batch("ended", succeeded=3, errored=1))

    status = await provider.get_batch_status("msgbatch_test1")

    assert status["status"] == "completed"
    assert status["request_counts"]["total"] == 4
    assert status["request_counts"]["completed"] == 4
    assert status["request_counts"]["failed"] == 1
    assert status["completed_at"] == "2026-07-08T00:30:00Z"


async def test_retrieve_batch_results_translates_to_openai_shape():
    provider = _make_provider()
    provider._client.messages.batches.retrieve = AsyncMock(return_value=_batch("ended", succeeded=1, errored=1))
    entries = [
        _succeeded_entry("chunk_0", {"facts": ["Alice is an engineer."]}),
        _errored_entry("chunk_1"),
    ]
    provider._client.messages.batches.results = AsyncMock(return_value=_AsyncIter(entries))

    results = await provider.retrieve_batch_results("msgbatch_test1")

    by_id = {r["custom_id"]: r for r in results}
    ok = by_id["chunk_0"]
    body = ok["response"]["body"]
    # The engine reads choices[0].message.content and json.loads() it.
    assert json.loads(body["choices"][0]["message"]["content"]) == {"facts": ["Alice is an engineer."]}
    # Usage arrives under the OpenAI key names the engine sums.
    assert body["usage"] == {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140}

    failed = by_id["chunk_1"]
    assert failed["error"]
    assert "response" not in failed


async def test_retrieve_batch_results_text_content_passthrough():
    """Non-strict requests come back as text blocks; concatenate them as content."""
    provider = _make_provider()
    provider._client.messages.batches.retrieve = AsyncMock(return_value=_batch("ended", succeeded=1))
    text_block = SimpleNamespace(type="text", text='{"facts": []}')
    message = SimpleNamespace(
        content=[text_block],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0),
        stop_reason="end_turn",
    )
    entry = SimpleNamespace(custom_id="chunk_0", result=SimpleNamespace(type="succeeded", message=message))
    provider._client.messages.batches.results = AsyncMock(return_value=_AsyncIter([entry]))

    results = await provider.retrieve_batch_results("msgbatch_test1")

    assert results[0]["response"]["body"]["choices"][0]["message"]["content"] == '{"facts": []}'


async def test_retrieve_batch_results_raises_when_not_ended():
    provider = _make_provider()
    provider._client.messages.batches.retrieve = AsyncMock(return_value=_batch("in_progress", processing=2))

    with pytest.raises(ValueError, match="not completed"):
        await provider.retrieve_batch_results("msgbatch_test1")
