"""OpenCode Go's required ``x-opencode-session`` conversation header (#4071).

The header must carry ONE id per logical operation: every LLM call of a single
retain/reflect run — including the provider's internal retries — sends the same
value, while separate runs send different ones. The id comes from the operation's
bound ``trace_id``, so these tests drive the real trace context rather than
asserting on a freshly minted uuid.

Three request shapes coexist on opencode-go (``/v1/chat/completions``,
``/v1/responses``, ``/v1/messages``), and the header is required by the host on
all three — the provider-name check the header helper used to apply
missed the Responses path entirely (so a Docker deployment of
``provider=openai-responses`` + ``base_url=https://opencode.ai/zen/go/v1`` +
``model=muse-spark-1.3-contributor`` was rejected by the backend). The host-based
detection exercised here covers every code path that talks to opencode.ai.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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
from hindsight_api.engine.providers.anthropic_llm import AnthropicLLM
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM
from hindsight_api.engine.providers.openai_responses_llm import OpenAIResponsesLLM


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

    assert result.content.ok is True
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
    apply_opencode_session(request, base_url="https://opencode.ai/zen/go/v1")
    assert request["extra_headers"][OPENCODE_SESSION_HEADER] == "operator-chosen-id"


def test_untraced_call_still_sends_a_header():
    """Outside a traced context the id falls back to a message fingerprint."""
    request = {"messages": [{"role": "system", "content": "You are a helper."}]}
    apply_opencode_session(request, base_url="https://opencode.ai/zen/go/v1")
    assert request["extra_headers"][OPENCODE_SESSION_HEADER]


def test_underivable_id_leaves_the_request_unchanged():
    """Fail-open: a malformed message list must not add a header or raise."""
    request: dict = {"messages": "not-a-list"}
    apply_opencode_session(request, base_url="https://opencode.ai/zen/go/v1")
    assert "extra_headers" not in request


# --------------------------------------------------------------------------- #
# Host-based detection: the header is a host requirement, not a provider one.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "base_url",
    [
        "https://opencode.ai/zen/go/v1",
        "https://opencode.ai/zen/go/v1/responses",
        "https://opencode.ai/zen/go/v1/messages",
    ],
)
def test_opencode_host_triggers_the_header(base_url):
    """Any provider targeting opencode.ai must inject the header.

    Regression for the Docker config
    ``provider=openai-responses`` + ``base_url=https://opencode.ai/zen/go/v1``
    + ``model=muse-spark-1.3-contributor``, which the previous
    provider-name-keyed guard silently skipped.
    """
    request = {"messages": [{"role": "user", "content": "Hi"}]}
    apply_opencode_session(request, base_url=base_url)
    assert OPENCODE_SESSION_HEADER in request["extra_headers"]


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.openai.com/v1",
        "https://api.anthropic.com",
        "https://api.deepseek.com",
        "https://api.minimax.io/v1",
        # Native OpenAI/Anthropic: no base URL configured at all.
        "",
        None,
    ],
)
def test_non_opencode_host_does_not_inject_the_header(base_url):
    """Hosts other than opencode.ai must never carry the header."""
    request = {"messages": [{"role": "user", "content": "Hi"}]}
    apply_opencode_session(request, base_url=base_url)
    assert "extra_headers" not in request


def test_host_suffix_must_match_exact_or_parent_domain():
    """A hostile suffix like ``evil-opencode.ai`` must NOT match.

    ``urlparse('https://evil-opencode.ai').hostname`` ends with ``-opencode.ai``
    but is not a subdomain of ``opencode.ai`` — the parsed-host check guards
    against this. ``hostname`` for that URL is ``evil-opencode.ai`` itself.
    """
    request = {"messages": [{"role": "user", "content": "Hi"}]}
    apply_opencode_session(request, base_url="https://evil-opencode.ai")
    assert "extra_headers" not in request


# --------------------------------------------------------------------------- #
# Regression: openai-responses + base_url=opencode.ai sends the header.
# This is the bug the host-based refactor fixes for the Responses path.
# --------------------------------------------------------------------------- #


def _responses_response():
    return SimpleNamespace(
        output_text="done",
        output=[],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
        status="completed",
    )


@pytest.mark.asyncio
async def test_openai_responses_sends_session_header_when_targeting_opencode():
    """provider=openai-responses + base_url=opencode.ai must inject the header.

    This is the regression pin for #4071 / the Muse Spark Contributor setup:
    before the host-based refactor, the header was gated on the provider name
    (``provider == 'opencode-go'``) so this combination sent no header and
    opencode-go rejected the request.
    """
    llm = OpenAIResponsesLLM(
        provider="openai-responses",
        api_key="test-key",
        base_url="https://opencode.ai/zen/go/v1",
        model="muse-spark-1.3-contributor",
    )
    create = AsyncMock(return_value=_responses_response())
    llm._client.responses.create = create

    with (
        traced_operation("trace-muse"),
        patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"),
    ):
        await llm.call(messages=[{"role": "user", "content": "Hi"}], max_retries=0)

    headers = create.call_args.kwargs.get("extra_headers") or {}
    assert headers.get(OPENCODE_SESSION_HEADER), (
        "opencode-go's /v1/responses rejects requests without x-opencode-session"
    )


@pytest.mark.asyncio
async def test_openai_responses_call_with_tools_sends_session_header_when_targeting_opencode():
    llm = OpenAIResponsesLLM(
        provider="openai-responses",
        api_key="test-key",
        base_url="https://opencode.ai/zen/go/v1",
        model="muse-spark-1.3-contributor",
    )
    create = AsyncMock(return_value=_responses_response())
    llm._client.responses.create = create

    with (
        traced_operation("trace-muse-tools"),
        patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"),
    ):
        await llm.call_with_tools(messages=[{"role": "user", "content": "Hi"}], tools=[], max_retries=0)

    headers = create.call_args.kwargs.get("extra_headers") or {}
    assert headers.get(OPENCODE_SESSION_HEADER)


@pytest.mark.asyncio
async def test_openai_responses_does_not_send_header_against_native_openai():
    """The header is opencode-go specific; native OpenAI must not carry it."""
    llm = OpenAIResponsesLLM(
        provider="openai-responses",
        api_key="test-key",
        base_url="",
        model="gpt-5.6",
    )
    create = AsyncMock(return_value=_responses_response())
    llm._client.responses.create = create

    with (
        traced_operation("trace-native-openai"),
        patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"),
    ):
        await llm.call(messages=[{"role": "user", "content": "Hi"}], max_retries=0)

    headers = create.call_args.kwargs.get("extra_headers") or {}
    assert headers.get(OPENCODE_SESSION_HEADER) is None


# --------------------------------------------------------------------------- #
# Regression: anthropic provider + base_url=opencode.ai sends the header.
# This covers the /v1/messages family (minimax-m3, qwen3.x, union-alpha).
# --------------------------------------------------------------------------- #


def _anthropic_response(content_text: str = "done"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=content_text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0),
    )


@pytest.mark.asyncio
async def test_anthropic_sends_session_header_when_targeting_opencode():
    """provider=anthropic + base_url=opencode.ai must inject the header.

    opencode-go's /v1/messages endpoint serves minimax-m3, qwen3.x and
    union-alpha. Without the header the backend rejects the request.
    """
    with patch("anthropic.AsyncAnthropic") as client_cls:
        client_cls.return_value = MagicMock()
        llm = AnthropicLLM(
            provider="anthropic",
            api_key="test-key",
            base_url="https://opencode.ai/zen/go/v1",
            model="minimax-m3",
        )
    llm._client = MagicMock()
    create = AsyncMock(return_value=_anthropic_response())
    llm._client.messages.create = create

    with (
        traced_operation("trace-anthropic-opencode"),
        patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"),
    ):
        await llm.call(messages=[{"role": "user", "content": "Hi"}], max_retries=0)

    headers = create.call_args.kwargs.get("extra_headers") or {}
    assert headers.get(OPENCODE_SESSION_HEADER), (
        "opencode-go's /v1/messages rejects requests without x-opencode-session"
    )


@pytest.mark.asyncio
async def test_anthropic_does_not_send_header_against_native_anthropic():
    with patch("anthropic.AsyncAnthropic") as client_cls:
        client_cls.return_value = MagicMock()
        llm = AnthropicLLM(
            provider="anthropic",
            api_key="test-key",
            base_url="",
            model="claude-haiku-4-5",
        )
    llm._client = MagicMock()
    create = AsyncMock(return_value=_anthropic_response())
    llm._client.messages.create = create

    with (
        traced_operation("trace-anthropic-native"),
        patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"),
    ):
        await llm.call(messages=[{"role": "user", "content": "Hi"}], max_retries=0)

    headers = create.call_args.kwargs.get("extra_headers") or {}
    assert headers.get(OPENCODE_SESSION_HEADER) is None
