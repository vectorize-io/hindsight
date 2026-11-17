"""A timed-out attempt that is retried, and a cancelled call, each leave a span (#6624).

Both used to leave only a log line, so a slow request showed a gap with nothing in it.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from openai._models import construct_type
from openai.types.chat import ChatCompletion

from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.providers.litellm_llm import LiteLLMLLM
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM
from hindsight_api.tracing import register_span_recorder, unregister_span_recorder


class _Recorder:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record_llm_call(self, **kwargs: Any) -> None:
        self.records.append(kwargs)


@pytest.fixture
def recorder():
    rec = _Recorder()
    register_span_recorder(rec)
    yield rec
    unregister_span_recorder(rec)


def _completion(content: str | None, finish_reason: str = "stop") -> ChatCompletion:
    choice: dict[str, Any] = {
        "index": 0,
        "finish_reason": finish_reason,
        "message": {"role": "assistant", "content": content},
    }
    if finish_reason == "error":
        choice["error"] = {"code": 502, "message": "upstream connection reset"}
    payload = {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [choice],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
    return construct_type(type_=ChatCompletion, value=payload)  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_litellm_traces_the_timed_out_attempt_before_retrying(recorder):
    llm = LiteLLMLLM(provider="litellm", api_key="unused", base_url="http://localhost:0/v1", model="m")
    llm._acompletion = AsyncMock(side_effect=TimeoutError())  # type: ignore[method-assign]
    with pytest.raises(TimeoutError):
        await llm.call(messages=[{"role": "user", "content": "q"}], scope="reflect", max_retries=1, initial_backoff=0)
    # Two attempts, both timed out: the first is traced here, the last by the wrapper.
    (retried,) = recorder.records
    assert isinstance(retried["error"], TimeoutError)
    assert retried["scope"] == "reflect"


@pytest.mark.asyncio
async def test_openai_compatible_traces_the_timed_out_attempt_before_retrying(recorder):
    llm = OpenAICompatibleLLM(provider="openai", api_key="sk-test", base_url="", model="m")
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.side_effect = [TimeoutError(), _completion("Paris.")]
        result = await llm.call(messages=[{"role": "user", "content": "q"}], max_retries=1, initial_backoff=0)
    assert result.content == "Paris."
    retried, succeeded = recorder.records
    assert isinstance(retried["error"], TimeoutError)
    assert succeeded["error"] is None


@pytest.mark.asyncio
async def test_a_timed_out_attempt_does_not_reuse_the_previous_attempts_tokens(recorder):
    """The timeout produced no reply, so it must not count the earlier attempt's tokens again."""
    llm = OpenAICompatibleLLM(provider="openai", api_key="sk-test", base_url="", model="m")
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.side_effect = [_completion(None, "error"), TimeoutError(), _completion("Paris.")]
        await llm.call(messages=[{"role": "user", "content": "q"}], max_retries=2, initial_backoff=0)
    unusable, timed_out, _ = recorder.records
    assert unusable["input_tokens"] == 3
    assert isinstance(timed_out["error"], TimeoutError)
    assert timed_out["input_tokens"] == 0


class _HangingProvider:
    def supports_attempt_scoped_concurrency(self) -> bool:
        return False

    async def call(self, **_kwargs: Any) -> Any:
        await asyncio.Event().wait()

    async def call_with_tools(self, **_kwargs: Any) -> Any:
        await asyncio.Event().wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["call", "call_with_tools"])
async def test_a_cancelled_call_is_traced_and_still_cancelled(recorder, method):
    llm = LLMProvider(provider="mock", api_key="", base_url="", model="mock")
    llm._provider_impl = _HangingProvider()  # type: ignore[assignment]
    kwargs: dict[str, Any] = {"messages": [{"role": "user", "content": "q"}], "scope": "reflect"}
    if method == "call_with_tools":
        kwargs["tools"] = []
    task = asyncio.create_task(getattr(llm, method)(**kwargs))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    (cancelled,) = recorder.records
    assert isinstance(cancelled["error"], asyncio.CancelledError)
    assert cancelled["scope"] == "reflect"
