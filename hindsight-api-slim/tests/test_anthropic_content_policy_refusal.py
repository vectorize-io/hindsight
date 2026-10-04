"""Regression tests for vectorize-io/hindsight#5201.

#3690 made a provider content-policy (AUP) refusal permanent -- one call,
``ProviderContentPolicyError``, no retry at any layer -- but implemented it only
in the Claude Code provider, which sees refusals as an ``is_error``
ResultMessage. The native ``anthropic`` provider receives the same refusal as an
ordinary HTTP 200 carrying ``stop_reason`` in {``refusal``, ``content_filter``}
and an empty content list. Nothing looked at that field, so the empty body fell
through to the invalid-JSON retry ladder (one *paid* call per attempt, up to
``max_retries``) and the worker then re-queued the whole retain. The issue
measured 4-16 paid calls for a single refused chunk.

``test_content_policy_refusal_permanent.py`` already pins the three layers that
consume the permanent error. These tests pin the missing fourth one: that this
provider produces it in the first place, in both call paths.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.engine.llm_interface import ProviderContentPolicyError

REFUSAL_STOP_REASONS = ("refusal", "content_filter")


class _Facts(BaseModel):
    facts: list[str]


def _make_provider():
    with patch("anthropic.AsyncAnthropic") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        from hindsight_api.engine.providers.anthropic_llm import AnthropicLLM

        provider = AnthropicLLM(
            provider="anthropic",
            api_key="fake-key",
            base_url="",
            model="claude-sonnet-4-20250514",
        )
    provider._client = MagicMock()
    return provider


def _refusal_response(stop_reason: str):
    """A refusal as the Messages API actually returns it: 200, empty content."""
    resp = MagicMock()
    resp.content = []
    resp.stop_reason = stop_reason
    resp.usage = MagicMock(input_tokens=8, output_tokens=0, cache_read_input_tokens=0)
    return resp


def _text_response(text: str, stop_reason: str = "end_turn"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    resp.stop_reason = stop_reason
    resp.usage = MagicMock(input_tokens=5, output_tokens=3, cache_read_input_tokens=0)
    return resp


# ---------------------------------------------------------------------------
# call()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", REFUSAL_STOP_REASONS)
async def test_call_raises_permanent_error_without_retrying(stop_reason):
    """A refusal must fail on attempt 1 with the permanent type, not burn the budget."""
    provider = _make_provider()
    create = AsyncMock(return_value=_refusal_response(stop_reason))
    provider._client.messages.create = create

    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        with pytest.raises(ProviderContentPolicyError) as excinfo:
            await provider.call(
                messages=[{"role": "user", "content": "extract facts"}],
                response_format=_Facts,
                scope="retain_extract_facts",
                max_retries=10,
                initial_backoff=0.0,
                max_backoff=0.0,
            )

    assert stop_reason in str(excinfo.value)
    assert create.await_count == 1, f"a refusal must not be replayed, but the API was called {create.await_count} times"


@pytest.mark.asyncio
async def test_call_still_retries_ordinary_errors():
    """The guard must be narrow: an empty body without a refusal stop_reason is still a retryable decode failure."""
    provider = _make_provider()
    create = AsyncMock(return_value=_refusal_response("end_turn"))
    provider._client.messages.create = create

    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        with pytest.raises(Exception) as excinfo:  # noqa: B017 - re-raised json.JSONDecodeError after the budget
            await provider.call(
                messages=[{"role": "user", "content": "extract facts"}],
                response_format=_Facts,
                scope="retain_extract_facts",
                max_retries=2,
                initial_backoff=0.0,
                max_backoff=0.0,
            )

    assert not isinstance(excinfo.value, ProviderContentPolicyError)
    assert create.await_count == 3, f"expected 1 initial attempt + 2 retries, got {create.await_count}"


@pytest.mark.asyncio
async def test_normal_stop_reasons_are_untouched():
    """Only the two refusal values are special -- ``max_tokens`` still returns its text."""
    provider = _make_provider()
    create = AsyncMock(return_value=_text_response('{"facts": ["a"]}', stop_reason="max_tokens"))
    provider._client.messages.create = create

    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        result = await provider.call(
            messages=[{"role": "user", "content": "extract facts"}],
            response_format=_Facts,
            scope="retain_extract_facts",
            max_retries=0,
        )

    assert result.content == _Facts(facts=["a"])
    assert create.await_count == 1


# ---------------------------------------------------------------------------
# call_with_tools()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", REFUSAL_STOP_REASONS)
async def test_call_with_tools_raises_instead_of_returning_an_empty_success(stop_reason):
    """A refusal in the tool path used to come back as a successful empty result.

    ``call_with_tools`` never parses JSON, so it never hit the decode ladder --
    it just reported success with ``content=None`` and no tool calls, and the
    caller carried on as if the model had simply produced nothing.
    """
    provider = _make_provider()
    create = AsyncMock(return_value=_refusal_response(stop_reason))
    provider._client.messages.create = create

    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        with pytest.raises(ProviderContentPolicyError) as excinfo:
            await provider.call_with_tools(
                messages=[{"role": "user", "content": "recall"}],
                tools=[
                    {
                        "function": {
                            "name": "noop",
                            "description": "no-op",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    }
                ],
                scope="recall",
                max_retries=10,
                initial_backoff=0.0,
                max_backoff=0.0,
            )

    assert stop_reason in str(excinfo.value)
    assert create.await_count == 1, f"a refusal must not be replayed, but the API was called {create.await_count} times"


@pytest.mark.asyncio
async def test_call_with_tools_normal_stop_reason_still_returns():
    """The tool path keeps working when the model simply stops without a tool call."""
    provider = _make_provider()
    create = AsyncMock(return_value=_text_response("no tool needed"))
    provider._client.messages.create = create

    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        result = await provider.call_with_tools(
            messages=[{"role": "user", "content": "recall"}],
            tools=[
                {
                    "function": {
                        "name": "noop",
                        "description": "no-op",
                        "parameters": {"type": "object", "properties": {}},
                    }
                }
            ],
            scope="recall",
            max_retries=0,
        )

    assert result.content == "no tool needed"
    assert result.tool_calls == []
    assert create.await_count == 1
