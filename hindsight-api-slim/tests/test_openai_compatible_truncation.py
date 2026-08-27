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


def _response(content: str, finish_reason: str):
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
    be returned as a complete answer with nothing to signal the cut."""
    llm = _make_llm()

    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = _response("The three main causes are: first, the", "length")
        with pytest.raises(OutputTooLongError):
            await llm.call(
                messages=[{"role": "user", "content": "summarize"}],
                max_retries=3,
            )

    assert create.call_count == 1
