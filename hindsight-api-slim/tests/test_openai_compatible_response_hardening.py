import asyncio
import json
from collections.abc import Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import BaseModel

from hindsight_api.engine.llm_trace import (
    current_response_usage,
    reset_response_usage,
    set_response_usage,
)
from hindsight_api.engine.providers.openai_compatible_llm import (
    OpenAICompatibleLLM,
    OutputTooLongError,
    ProviderResponseError,
    _rate_limit_retry_at,
)
from hindsight_api.worker.stage import StageHolder, bind_holder, set_stage


class SimpleJsonResponse(BaseModel):
    ok: bool


class FactListResponse(BaseModel):
    """A list is what makes a cap dangerous: repair drops entries and still validates."""

    facts: list[str]


def _llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="openai",
        api_key="test-key",
        base_url="https://example.test/v1",
        model="gpt-4o-mini",
    )


def _response(*, content: str | None = '{"ok": true}', choices=None, error=None, finish_reason="stop"):
    response = SimpleNamespace(error=error, usage=None)
    if choices is not None:
        response.choices = choices
        return response

    choice = SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=content, tool_calls=None, refusal=None),
    )
    response.choices = [choice]
    return response


@pytest.mark.asyncio
async def test_attempt_stage_is_published_only_after_permits_are_acquired():
    llm = _llm()
    holder = StageHolder()
    waiting_for_permit = asyncio.Event()
    release_permit = asyncio.Event()

    @asynccontextmanager
    async def blocked_attempt_context():
        waiting_for_permit.set()
        await release_permit.wait()
        yield

    async def create(**_kwargs):
        assert holder.stage == "llm.openai.memory.attempt=1/1"
        return _response(content="ok")

    llm._client.chat.completions.create = AsyncMock(side_effect=create)

    async def invoke():
        bind_holder(holder)
        set_stage("llm.openai.memory.queued")
        return await llm.call(
            messages=[{"role": "user", "content": "hello"}],
            max_retries=0,
            attempt_context=blocked_attempt_context,
        )

    task = asyncio.create_task(invoke())
    await asyncio.wait_for(waiting_for_permit.wait(), timeout=1)
    assert holder.stage == "llm.openai.memory.queued"
    release_permit.set()
    assert await task == "ok"


@pytest.mark.asyncio
async def test_json_object_call_adds_json_hint_to_user_message():
    llm = _llm()
    create = AsyncMock(return_value=_response())
    llm._client.chat.completions.create = create

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        result = await llm.call(
            messages=[{"role": "user", "content": "Return whether this worked."}],
            response_format=SimpleJsonResponse,
            max_retries=0,
        )

    assert result.ok is True
    sent_messages = create.call_args.kwargs["messages"]
    assert sent_messages[0]["content"].startswith("Return valid json only.")


@pytest.mark.asyncio
async def test_json_object_call_strips_gemma_thought_tags_before_parsing():
    llm = _llm()
    create = AsyncMock(
        return_value=_response(content='<thought>\nI should return a compact JSON object.\n</thought>\n{"ok": true}')
    )
    llm._client.chat.completions.create = create

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        result = await llm.call(
            messages=[{"role": "user", "content": "Return whether this worked."}],
            response_format=SimpleJsonResponse,
            max_retries=0,
        )

    assert result.ok is True


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["qwen/qwen3.6-35b-a3b", "openai/gpt-oss-120b"])
async def test_openrouter_verification_uses_larger_reasoning_safe_budget(model: str):
    llm = OpenAICompatibleLLM(
        provider="openrouter",
        api_key="test-key",
        base_url="",
        model=model,
    )
    create = AsyncMock(return_value=_response(content="ok"))
    llm._client.chat.completions.create = create

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        await llm.verify_connection()

    sent = create.call_args.kwargs
    assert sent["model"] == model
    assert sent["messages"] == [{"role": "user", "content": "Say 'ok'"}]
    assert sent["max_tokens"] == 512
    assert "max_completion_tokens" not in sent


@pytest.mark.asyncio
async def test_verification_uses_larger_budget_for_other_compatible_gateways():
    llm = _llm()
    create = AsyncMock(return_value=_response(content="ok"))
    llm._client.chat.completions.create = create

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        await llm.verify_connection()

    sent = create.call_args.kwargs
    assert sent["max_tokens"] == 512
    assert "max_completion_tokens" not in sent


@pytest.mark.asyncio
async def test_error_payload_with_no_choices_raises_clear_provider_error_without_retry():
    llm = _llm()
    create = AsyncMock(
        return_value=_response(
            choices=None,
            error={
                "message": "Response input messages must contain the word 'json'",
                "type": "invalid_request_error",
                "param": "input",
            },
        )
    )
    # Simulate SDK objects where the declared field exists but is null.
    create.return_value.choices = None
    llm._client.chat.completions.create = create

    with pytest.raises(ProviderResponseError, match="Provider returned error payload.*word 'json'"):
        await llm.call(
            messages=[{"role": "user", "content": "Return whether this worked."}],
            response_format=SimpleJsonResponse,
            max_retries=2,
        )

    assert create.await_count == 1


@pytest.mark.asyncio
async def test_missing_choices_are_retryable_provider_response_errors():
    llm = _llm()
    empty_response = _response(choices=[])
    valid_response = _response()
    create = AsyncMock(side_effect=[empty_response, valid_response])
    llm._client.chat.completions.create = create

    with (
        patch("hindsight_api.engine.providers.openai_compatible_llm.asyncio.sleep", new=AsyncMock()) as sleep_mock,
        patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"),
    ):
        result = await llm.call(
            messages=[{"role": "user", "content": "Return whether this worked."}],
            response_format=SimpleJsonResponse,
            max_retries=1,
            initial_backoff=0,
        )

    assert result.ok is True
    assert create.await_count == 2
    sleep_mock.assert_awaited_once()


def _ollama_llm(*, num_ctx: int | None = None) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="ollama",
        api_key="",
        base_url="http://localhost:11434/v1",
        model="qwen3",
        ollama_num_ctx=num_ctx,
    )


def _ollama_response(content: str, *, done_reason: str | None = None) -> httpx.Response:
    body = {
        "model": "qwen3",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }
    if done_reason is not None:
        body["done_reason"] = done_reason
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    return httpx.Response(200, json=body, request=request)


@pytest.mark.asyncio
async def test_repairable_json_is_recovered_instead_of_being_dropped():
    """A malformed but structurally repairable response must not be lost (#3683).

    The trailing comma is what json_repair exists to fix. Before #3683 this
    provider parsed with bare json.loads and raised once the retries ran out,
    so the facts in the response never reached the caller.
    """
    llm = _llm()
    create = AsyncMock(return_value=_response(content='{"ok": true,}'))
    llm._client.chat.completions.create = create

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        result = await llm.call(
            messages=[{"role": "user", "content": "Return whether this worked."}],
            response_format=SimpleJsonResponse,
            max_retries=1,
            initial_backoff=0,
        )

    assert result.ok is True


@pytest.mark.asyncio
async def test_repeated_output_does_not_cut_the_retry_budget_short():
    """Two identical bodies do not establish a deterministic generation (#3683).

    Temperature is set by the caller and never read on this path, so nothing here
    can tell a deterministic model from a stochastic one that repeated twice. The
    configured budget is what the caller asked for, and the third attempt is the
    one that succeeds.
    """
    llm = _llm()
    create = AsyncMock(
        side_effect=[
            _response(content='{"ok": true,}'),
            _response(content='{"ok": true,}'),
            _response(content='{"ok": true}'),
        ]
    )
    llm._client.chat.completions.create = create

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        result = await llm.call(
            messages=[{"role": "user", "content": "Return whether this worked."}],
            response_format=SimpleJsonResponse,
            max_retries=3,
            initial_backoff=0,
        )

    assert result.ok is True
    assert create.await_count == 3


@pytest.mark.asyncio
async def test_a_changed_response_still_earns_a_fresh_generation():
    """Flaky malformed output keeps its re-rolls (#3683).

    Output that differs each time is the case a fresh generation can actually
    fix, so the retry ladder must run to the end of the budget.
    """
    llm = _llm()
    create = AsyncMock(
        side_effect=[
            _response(content='{"ok": true,}'),
            _response(content='{"ok": '),
            _response(content='{"ok": true}'),
        ]
    )
    llm._client.chat.completions.create = create

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        result = await llm.call(
            messages=[{"role": "user", "content": "Return whether this worked."}],
            response_format=SimpleJsonResponse,
            max_retries=3,
            initial_backoff=0,
        )

    assert result.ok is True
    assert create.await_count == 3


@pytest.mark.asyncio
async def test_ollama_native_repairs_malformed_json_instead_of_dropping_it():
    """The native /api/chat path parses the same way and had the same gap (#3683).

    ``done_reason="stop"`` is what real Ollama sends on a completed generation,
    and repair is gated on it.
    """
    llm = _ollama_llm()
    mock_client = AsyncMock()
    mock_client.post.return_value = _ollama_response('{"ok": true,}', done_reason="stop")
    mock_client.__aenter__.return_value = mock_client

    with (
        patch(
            "hindsight_api.engine.providers.openai_compatible_llm.httpx.AsyncClient",
            return_value=mock_client,
        ),
        patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"),
    ):
        result = await llm.call(
            messages=[{"role": "user", "content": "Return whether this worked."}],
            response_format=SimpleJsonResponse,
            max_retries=1,
            initial_backoff=0,
        )

    assert result.ok is True


@pytest.mark.asyncio
async def test_ollama_native_reports_a_truncated_body_rather_than_repairing_it():
    """The native path carries the same risk under the done_reason key."""
    llm = _ollama_llm()
    mock_client = AsyncMock()
    mock_client.post.return_value = _ollama_response('{"facts": ["alpha", "beta", "gam', done_reason="length")
    mock_client.__aenter__.return_value = mock_client

    with (
        patch(
            "hindsight_api.engine.providers.openai_compatible_llm.httpx.AsyncClient",
            return_value=mock_client,
        ),
        patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"),
    ):
        with pytest.raises(OutputTooLongError):
            await llm.call(
                messages=[{"role": "user", "content": "List the facts."}],
                response_format=FactListResponse,
                max_retries=1,
                initial_backoff=0,
            )


@pytest.mark.asyncio
async def test_ollama_native_reports_a_truncated_body_that_still_parses():
    """The native path has the same quiet case under done_reason."""
    llm = _ollama_llm()
    mock_client = AsyncMock()
    mock_client.post.return_value = _ollama_response('{"facts": ["alpha", "beta"]}', done_reason="length")
    mock_client.__aenter__.return_value = mock_client

    with (
        patch(
            "hindsight_api.engine.providers.openai_compatible_llm.httpx.AsyncClient",
            return_value=mock_client,
        ),
        patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"),
    ):
        with pytest.raises(OutputTooLongError):
            await llm.call(
                messages=[{"role": "user", "content": "List the facts."}],
                response_format=FactListResponse,
                max_retries=3,
                initial_backoff=0,
            )

    assert mock_client.post.await_count == 1


@pytest.mark.asyncio
async def test_ollama_native_reports_a_capped_free_form_answer():
    """A cap on a free-form native call fails rather than returning the cut text.

    The guard sits ahead of the free-form/structured split, matching what the
    OpenAI-compatible path has done for every scope since #3827: a truncated
    reflect synthesis or mental-model page is a failure, not a short success.
    """
    # Free-form calls only take the native path once num_ctx is configured.
    llm = _ollama_llm(num_ctx=8192)
    mock_client = AsyncMock()
    mock_client.post.return_value = _ollama_response("The three causes are, first,", done_reason="length")
    mock_client.__aenter__.return_value = mock_client

    with (
        patch(
            "hindsight_api.engine.providers.openai_compatible_llm.httpx.AsyncClient",
            return_value=mock_client,
        ),
        patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"),
    ):
        with pytest.raises(OutputTooLongError):
            await llm.call(
                messages=[{"role": "user", "content": "Explain the causes."}],
                max_retries=3,
                initial_backoff=0,
            )

    assert mock_client.post.await_count == 1


@pytest.mark.asyncio
async def test_ollama_native_does_not_retry_a_capped_empty_free_form_answer():
    """A cap reached before the first visible token must not be re-sent (#3811).

    Empty content on the free-form branch raises a *retryable*
    ProviderResponseError, so without the guard ahead of it the identical request
    goes back out against the identical limit.
    """
    llm = _ollama_llm(num_ctx=8192)
    mock_client = AsyncMock()
    mock_client.post.return_value = _ollama_response("", done_reason="length")
    mock_client.__aenter__.return_value = mock_client

    with (
        patch(
            "hindsight_api.engine.providers.openai_compatible_llm.httpx.AsyncClient",
            return_value=mock_client,
        ),
        patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"),
    ):
        with pytest.raises(OutputTooLongError):
            await llm.call(
                messages=[{"role": "user", "content": "Explain the causes."}],
                max_retries=3,
                initial_backoff=0,
            )

    assert mock_client.post.await_count == 1


@pytest.mark.asyncio
async def test_a_malformed_body_that_is_not_truncated_still_gets_repaired():
    """The guard reads finish_reason, not the shape of the failure.

    Malformed output that was not truncated keeps the repair this PR added.
    """
    llm = _llm()
    create = AsyncMock(return_value=_response(content='{"facts": ["alpha", "beta"],}'))
    llm._client.chat.completions.create = create

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        result = await llm.call(
            messages=[{"role": "user", "content": "List the facts."}],
            response_format=FactListResponse,
            max_retries=1,
            initial_backoff=0,
        )

    assert result.facts == ["alpha", "beta"]


@pytest.mark.parametrize(
    ("headers", "message", "expected_seconds"),
    [
        # OpenAI's Go-style duration headers: a single component ("8.64s")
        # and a compound one ("6m0s") that a naive parser reads as 0s.
        ({"x-ratelimit-reset-tokens": "8.64s"}, "Rate limit reached", 8.64),
        ({"x-ratelimit-reset-requests": "6m0s"}, "Rate limit reached", 360),
        # Requests and tokens can reset at different times; wait for the max.
        (
            {"x-ratelimit-reset-requests": "6m0s", "x-ratelimit-reset-tokens": "233ms"},
            "Rate limit reached",
            360,
        ),
        # A full or nearly-full header budget must not mask a longer quota
        # window reported in the response body.
        ({"x-ratelimit-reset-tokens": "0s"}, "Rate limit reached; try again in 5 hours", 5 * 3600),
        ({"x-ratelimit-reset-tokens": "233ms"}, "Rate limit reached; try again in 5 hours", 5 * 3600),
        # Retry-After is an explicit server instruction and must not be
        # extended by a conflicting free-text body hint.
        (
            {"retry-after": "1", "x-ratelimit-reset-tokens": "233ms"},
            "Rate limit reached; try again in 5 hours",
            1,
        ),
    ],
)
def test_rate_limit_retry_at_uses_latest_header_or_body_reset(
    headers: Mapping[str, str], message: str, expected_seconds: float
) -> None:
    error = SimpleNamespace(body={"message": message}, response=SimpleNamespace(headers=headers))
    retry_at = _rate_limit_retry_at(error)
    assert retry_at is not None
    wait = (retry_at - datetime.now(UTC)).total_seconds()
    assert expected_seconds - 1 < wait <= expected_seconds + 1


@pytest.mark.asyncio
async def test_a_truncated_list_is_not_repaired_when_the_provider_omits_finish_reason():
    """Repair needs positive evidence the generation finished (#3683).

    ``json_repair`` closes an unterminated string and list by inventing the
    terminators, so ``{"facts": ["alpha", "beta", "gam`` becomes a valid
    two-and-a-bit-entry answer. A proxy that omits finish_reason gives no
    evidence either way, and guessing there would turn a truncation into a
    short result reported as complete.
    """
    llm = _llm()
    truncated = '{"facts": ["alpha", "beta", "gam'
    create = AsyncMock(return_value=_response(content=truncated, finish_reason=None))
    llm._client.chat.completions.create = create

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        with pytest.raises(json.JSONDecodeError):
            await llm.call(
                messages=[{"role": "user", "content": "List the facts."}],
                response_format=FactListResponse,
                max_retries=0,
                initial_backoff=0,
            )


@pytest.mark.asyncio
async def test_an_unknown_finish_reason_does_not_earn_a_repair():
    """An unrecognised reason is not a completion signal either."""
    llm = _llm()
    create = AsyncMock(return_value=_response(content='{"facts": ["alpha", "gam', finish_reason="incomplete"))
    llm._client.chat.completions.create = create

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        with pytest.raises(json.JSONDecodeError):
            await llm.call(
                messages=[{"role": "user", "content": "List the facts."}],
                response_format=FactListResponse,
                max_retries=0,
                initial_backoff=0,
            )


@pytest.mark.asyncio
async def test_ollama_does_not_repair_a_truncated_list_without_done_reason():
    """The native path is gated the same way."""
    llm = _ollama_llm()
    mock_client = AsyncMock()
    mock_client.post.return_value = _ollama_response('{"facts": ["alpha", "beta", "gam')
    mock_client.__aenter__.return_value = mock_client

    with (
        patch(
            "hindsight_api.engine.providers.openai_compatible_llm.httpx.AsyncClient",
            return_value=mock_client,
        ),
        patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"),
    ):
        with pytest.raises(json.JSONDecodeError):
            await llm.call(
                messages=[{"role": "user", "content": "List the facts."}],
                response_format=FactListResponse,
                max_retries=0,
                initial_backoff=0,
            )


@pytest.mark.asyncio
async def test_ollama_records_usage_for_a_capped_response_before_raising():
    """A capped call still cost tokens, and those are the expensive calls.

    The done_reason guard raises before the usage extraction further down, so
    without stashing here the most expensive responses would be the ones missing
    from accounting.
    """
    llm = _ollama_llm()
    mock_client = AsyncMock()
    body = _ollama_response('{"facts": ["alpha", "beta", "gam', done_reason="length")
    payload = json.loads(body.content)
    payload["prompt_eval_count"] = 1234
    payload["eval_count"] = 567
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    mock_client.post.return_value = httpx.Response(200, json=payload, request=request)
    mock_client.__aenter__.return_value = mock_client

    token = set_response_usage(None)
    try:
        with (
            patch(
                "hindsight_api.engine.providers.openai_compatible_llm.httpx.AsyncClient",
                return_value=mock_client,
            ),
            patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"),
        ):
            with pytest.raises(OutputTooLongError):
                await llm.call(
                    messages=[{"role": "user", "content": "List the facts."}],
                    response_format=FactListResponse,
                    max_retries=0,
                    initial_backoff=0,
                )
        usage = current_response_usage()
        assert usage is not None
        assert usage.input_tokens == 1234
        assert usage.output_tokens == 567
    finally:
        reset_response_usage(token)
