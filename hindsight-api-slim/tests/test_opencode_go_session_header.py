"""OpenCode Go's required ``x-opencode-session`` conversation header (#4071).

The header must carry ONE id per logical operation: every LLM call of a single
retain/reflect run — including the provider's internal retries — sends the same
value, while separate runs send different ones. The id comes from the operation's
bound ``trace_id``, so these tests drive the real trace context rather than
asserting on a freshly minted uuid.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.engine.cache_affinity import (
    OPENCODE_SESSION_HEADER,
    apply_opencode_session,
)
from hindsight_api.engine.llm_trace import (
    LLMTraceContext,
    reset_trace_context,
    set_trace_context,
)
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM


class SimpleJsonResponse(BaseModel):
    ok: bool


def _llm(provider: str = "opencode-go") -> OpenAICompatibleLLM:
    base_urls = {
        "opencode-go": "https://opencode.ai/zen/go/v1",
        "openai": "https://api.openai.com/v1",
    }
    return OpenAICompatibleLLM(
        provider=provider,
        api_key="test-key",
        base_url=base_urls[provider],
        model="deepseek-v4-flash" if provider == "opencode-go" else "gpt-4o",
    )


def _response(*, content: str | None = '{"ok": true}'):
    choice = SimpleNamespace(
        finish_reason="stop",
        message=SimpleNamespace(content=content, tool_calls=None, refusal=None),
    )
    return SimpleNamespace(choices=[choice], usage=None, error=None)


def _sent_session_ids(create: AsyncMock) -> list[str | None]:
    """The session header sent on each attempt, in call order."""
    return [(call.kwargs.get("extra_headers") or {}).get(OPENCODE_SESSION_HEADER) for call in create.call_args_list]


@contextmanager
def traced_operation(trace_id: str):
    """Bind an operation trace context, as ConfiguredLLMProvider does per run.

    Must be entered inside the test body: a ContextVar token can only be reset
    in the context that created it, so binding from a sync fixture and resetting
    after an async test raises "created in a different Context".
    """
    token = set_trace_context(LLMTraceContext(bank_id="b1", operation="reflect", trace_id=trace_id))
    try:
        yield
    finally:
        reset_trace_context(token)


@pytest.mark.asyncio
async def test_opencode_go_sends_session_header():
    llm = _llm()
    create = AsyncMock(return_value=_response())
    llm._client.chat.completions.create = create

    with (
        traced_operation("trace-abc"),
        patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"),
    ):
        await llm.call(messages=[{"role": "user", "content": "Hi"}], max_retries=0)

    (session_id,) = _sent_session_ids(create)
    assert session_id, "opencode-go must send the session header"


@pytest.mark.asyncio
async def test_session_id_is_stable_across_provider_retries():
    """A retried call is still ONE logical operation — the id must not change."""
    llm = _llm()
    # First attempt returns unparseable JSON, forcing the provider's own retry.
    create = AsyncMock(side_effect=[_response(content="not json"), _response(content='{"ok": true}')])
    llm._client.chat.completions.create = create

    with (
        traced_operation("trace-retry"),
        patch("hindsight_api.engine.providers.openai_compatible_llm.asyncio.sleep", new=AsyncMock()),
        patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"),
    ):
        result = await llm.call(
            messages=[{"role": "user", "content": "Hi"}],
            response_format=SimpleJsonResponse,
            max_retries=1,
            initial_backoff=0,
        )

    assert result.ok is True
    first, second = _sent_session_ids(create)
    assert first and second
    assert first == second, "retries of one operation must reuse the session id"


@pytest.mark.asyncio
async def test_all_calls_of_one_operation_share_one_session_id():
    """The reflect-session property: separate calls in one run share the id."""
    llm = _llm()
    create = AsyncMock(return_value=_response())
    llm._client.chat.completions.create = create

    with (
        traced_operation("trace-one-run"),
        patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"),
    ):
        # An agent loop makes several LLM calls within a single operation, and
        # the message list grows between them.
        await llm.call(messages=[{"role": "user", "content": "Step 1"}], max_retries=0)
        await llm.call(
            messages=[
                {"role": "user", "content": "Step 1"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "Step 2"},
            ],
            max_retries=0,
        )

    first, second = _sent_session_ids(create)
    assert first and second
    assert first == second, "all calls of one reflect run must share the session id"


@pytest.mark.asyncio
async def test_separate_operations_get_separate_session_ids():
    """Two runs are two conversations, so their ids must differ."""
    llm = _llm()
    create = AsyncMock(return_value=_response())
    llm._client.chat.completions.create = create
    messages = [{"role": "user", "content": "Same prompt both runs"}]

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        for trace_id in ("trace-run-1", "trace-run-2"):
            token = set_trace_context(LLMTraceContext(bank_id="b1", operation="reflect", trace_id=trace_id))
            try:
                await llm.call(messages=messages, max_retries=0)
            finally:
                reset_trace_context(token)

    first, second = _sent_session_ids(create)
    assert first and second
    assert first != second, "separate operations must get separate session ids"


@pytest.mark.asyncio
async def test_tool_calls_send_the_session_header():
    llm = _llm()
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="done", tool_calls=None),
            )
        ],
        usage=None,
        error=None,
    )
    create = AsyncMock(return_value=response)
    llm._client.chat.completions.create = create

    with (
        traced_operation("trace-tools"),
        patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"),
    ):
        await llm.call_with_tools(messages=[{"role": "user", "content": "Hi"}], tools=[], max_retries=0)

    (session_id,) = _sent_session_ids(create)
    assert session_id, "call_with_tools must also send the session header"


@pytest.mark.asyncio
async def test_other_providers_do_not_send_the_header():
    """The header is opencode-go's own protocol, not a generic addition."""
    llm = _llm("openai")
    create = AsyncMock(return_value=_response())
    llm._client.chat.completions.create = create

    with (
        traced_operation("trace-openai"),
        patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"),
    ):
        await llm.call(messages=[{"role": "user", "content": "Hi"}], max_retries=0)

    assert _sent_session_ids(create) == [None]


def test_caller_supplied_header_wins():
    """An explicitly set header is preserved rather than overwritten."""
    request = {
        "messages": [{"role": "user", "content": "Hi"}],
        "extra_headers": {OPENCODE_SESSION_HEADER: "operator-chosen-id"},
    }
    apply_opencode_session(request, "opencode-go")
    assert request["extra_headers"][OPENCODE_SESSION_HEADER] == "operator-chosen-id"


def test_untraced_call_still_sends_a_header():
    """Outside a traced context the id falls back to a message fingerprint."""
    request = {"messages": [{"role": "system", "content": "You are a helper."}]}
    apply_opencode_session(request, "opencode-go")
    assert request["extra_headers"][OPENCODE_SESSION_HEADER]


def test_underivable_id_leaves_the_request_unchanged():
    """Fail-open: a malformed message list must not add a header or raise."""
    request: dict = {"messages": "not-a-list"}
    apply_opencode_session(request, "opencode-go")
    assert "extra_headers" not in request
