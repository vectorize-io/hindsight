"""Regression tests for issue #3811: truncated non-streaming responses.

``chat.completions.create()`` reports a token-limit truncation only through
``finish_reason``. It never raises ``LengthFinishReasonError``, so the handler in
``call()`` that converts that exception into ``OutputTooLongError`` cannot fire for
these call sites, and a truncated body used to be returned to the caller as if it
were complete. For structured output that surfaced as a JSON parse error; for
free-form output it was returned silently.

The fact-extraction auto-split retries on ``OutputTooLongError``, so the truncation
has to reach it as that class to be recoverable.
"""

import types
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.engine.llm_interface import OutputTooLongError
from hindsight_api.engine.providers.openai_compatible_llm import (
    OpenAICompatibleLLM,
    ProviderResponseError,
    _content_or_error,
)


class _Facts(BaseModel):
    facts: list[str]


def _make_llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="openai",
        api_key="sk-test",
        base_url="",
        model="gpt-4o-mini",
    )


def _response(content: str, finish_reason: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                finish_reason=finish_reason,
                message=types.SimpleNamespace(content=content, tool_calls=None, refusal=None),
            )
        ],
        usage=types.SimpleNamespace(prompt_tokens=800, completion_tokens=4096, total_tokens=4896),
        model="gpt-4o-mini",
    )


# The truncated body is valid JSON up to the cut, which is what made it parse-error
# shaped rather than truncation shaped.
_TRUNCATED_JSON = '{"facts": ["user deployed a three-node cluster", "the rollout'


def test_content_or_error_raises_output_too_long_on_length_finish_reason():
    with pytest.raises(OutputTooLongError) as excinfo:
        _content_or_error(
            _response(_TRUNCATED_JSON, "length"),
            provider="openai",
            model="gpt-4o-mini",
            scope="retain_fact_extraction",
        )

    assert "retain_fact_extraction" in str(excinfo.value)


def test_content_or_error_returns_content_when_generation_stopped_normally():
    content, choice = _content_or_error(
        _response('{"facts": []}', "stop"),
        provider="openai",
        model="gpt-4o-mini",
        scope="retain_fact_extraction",
    )

    assert content == '{"facts": []}'
    assert choice.finish_reason == "stop"


@pytest.mark.asyncio
async def test_structured_call_raises_output_too_long_without_retrying():
    """A truncated structured response is not a transient fault: retrying the same
    prompt against the same limit truncates again, so it must surface at once."""
    llm = _make_llm()

    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = _response(_TRUNCATED_JSON, "length")
        with pytest.raises(OutputTooLongError):
            await llm.call(
                messages=[{"role": "user", "content": "extract facts"}],
                response_format=_Facts,
                max_retries=3,
            )

    assert create.call_count == 1


@pytest.mark.asyncio
async def test_freeform_call_raises_output_too_long_instead_of_returning_truncated_text():
    """Without a response_format there is no parse step, so a truncated body used to
    be returned as a complete answer with nothing to signal the cut.

    This is the shape every free-form scope takes -- reflect synthesis, a mental-model
    page. Those have no ``OutputTooLongError`` handler, so raising here turns a
    silently-cut answer into a failed call. That is the deliberate trade recorded at
    the raise site in ``_content_or_error``, not an oversight.
    """
    llm = _make_llm()

    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = _response("The three main causes are: first, the", "length")
        with pytest.raises(OutputTooLongError):
            await llm.call(
                messages=[{"role": "user", "content": "summarize"}],
                max_retries=3,
            )

    assert create.call_count == 1


def test_content_or_error_raises_output_too_long_when_truncated_before_any_content():
    """A budget exhausted before the first visible token is still a truncation.

    A reasoning model can spend the whole completion budget on hidden reasoning and
    return ``content=""`` with ``finish_reason="length"``. Reading that as empty
    content would raise a retryable ``ProviderResponseError`` instead, which sends
    the same request against the same limit rather than splitting the input.
    """
    with pytest.raises(OutputTooLongError) as excinfo:
        _content_or_error(
            _response("", "length"),
            provider="openai",
            model="gpt-4o-mini",
            scope="retain_fact_extraction",
        )

    assert "retain_fact_extraction" in str(excinfo.value)


def test_content_or_error_still_raises_provider_error_on_empty_content_without_truncation():
    """Control: the empty-content path is unchanged for every other finish_reason."""
    with pytest.raises(ProviderResponseError) as excinfo:
        _content_or_error(
            _response("", "stop"),
            provider="openai",
            model="gpt-4o-mini",
            scope="retain_fact_extraction",
        )

    assert excinfo.value.retryable is True
    assert "empty message content" in str(excinfo.value)


@pytest.mark.asyncio
async def test_empty_truncated_call_is_not_retried_against_the_same_limit():
    """The cost of misclassifying this one: ``call()`` retries a retryable
    ``ProviderResponseError``, so an empty truncation used to re-send the identical
    request until the ladder ran out, and the auto-split never saw it."""
    llm = _make_llm()

    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = _response("", "length")
        with pytest.raises(OutputTooLongError):
            await llm.call(
                messages=[{"role": "user", "content": "extract facts"}],
                response_format=_Facts,
                max_retries=3,
            )

    assert create.call_count == 1
