"""
Malformed-JSON recovery on the LiteLLM provider parse path (#2547/#2544).

When a structured-output response is malformed JSON (``json.loads`` cannot parse
it at all — trailing commas, unterminated strings, invalid ``\\escape``), the
provider prefers a clean re-roll first and only falls back to a structural
``json_repair`` pass once the retry budget is exhausted. Unrecoverable output
still fails loudly through the existing retry ladder.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from hindsight_api.engine.providers.litellm_llm import LiteLLMLLM


class _Facts(BaseModel):
    a: int


def _make_provider() -> LiteLLMLLM:
    return LiteLLMLLM(
        provider="litellm",
        api_key="unused",
        base_url="http://localhost:0/v1",
        model="litellm_proxy/test-model",
    )


def _make_response(content: str) -> MagicMock:
    response = MagicMock()
    response.usage = None
    response.model = "test-model"
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = "stop"
    response.choices = [choice]
    return response


async def test_malformed_json_repaired_after_retries_exhausted(monkeypatch):
    """Persistently malformed JSON: retry ladder runs, then repair rescues it."""
    provider = _make_provider()
    calls = 0

    async def _fake(**kwargs):
        nonlocal calls
        calls += 1
        return _make_response('{"a": 1,}')  # trailing comma — never valid

    monkeypatch.setattr(provider, "_acompletion", _fake)

    result = await provider.call(
        messages=[{"role": "user", "content": "hi"}],
        response_format=_Facts,
        skip_validation=True,
        max_retries=1,
        initial_backoff=0.01,
        max_backoff=0.01,
    )

    # A clean re-roll is preferred first: attempt 0 raises, attempt 1 repairs.
    assert calls == 2
    assert result == {"a": 1}


async def test_clean_reroll_preferred_over_repair(monkeypatch):
    """A malformed first response is retried; a valid re-roll wins (no repair)."""
    provider = _make_provider()
    responses = [_make_response('{"a": 1,}'), _make_response('{"a": 2}')]

    async def _fake(**kwargs):
        return responses.pop(0)

    monkeypatch.setattr(provider, "_acompletion", _fake)

    result = await provider.call(
        messages=[{"role": "user", "content": "hi"}],
        response_format=_Facts,
        skip_validation=True,
        max_retries=1,
        initial_backoff=0.01,
        max_backoff=0.01,
    )

    assert result == {"a": 2}  # the clean re-roll, not a repair of the first


async def test_unrecoverable_json_raises_after_retries(monkeypatch):
    """Garbage that repair cannot rescue still fails loudly through the ladder."""
    import json

    provider = _make_provider()
    calls = 0

    async def _fake(**kwargs):
        nonlocal calls
        calls += 1
        return _make_response("not json at all !!!")

    monkeypatch.setattr(provider, "_acompletion", _fake)

    with pytest.raises(json.JSONDecodeError):
        await provider.call(
            messages=[{"role": "user", "content": "hi"}],
            response_format=_Facts,
            skip_validation=True,
            max_retries=1,
            initial_backoff=0.01,
            max_backoff=0.01,
        )

    assert calls == 2
