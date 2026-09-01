"""A stalled Gemini request is abandoned at the deadline and retried once.

Gemini occasionally accepts a request and never answers it. The per-request
deadline is the only thing that ends such a call, and before this the abort was
terminal: ``TimeoutError`` fell through to the generic handler, which re-raised
it, so the caller paid the full deadline *and* got an error. In reflect — the one
interactive operation, several sequential calls behind a waiting client — that
turned one stalled iteration into a 120s request answered from a degraded forced
pass, which is what put the doc-example, TypeScript and Python client suites over
their 120s test budgets.

A deadline abort is the most transient failure there is — nothing was returned, so
nothing about the request is implicated — and the stall really is per-request: CI
logs have calls that burn the whole deadline and then answer in ~3s on the very next
attempt. It gets its own small retry budget, separate from the API-error ladder, so
the worst case stays bounded arithmetic (deadline x attempts) that an operation's
wall budget can be checked against.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("google.genai")

from hindsight_api.engine.providers.gemini_llm import _TIMEOUT_RETRIES  # noqa: E402

_TOOLS = [{"type": "function", "function": {"name": "noop", "description": "n", "parameters": {"type": "object"}}}]


def _make_gemini_provider(timeout: float = 0.05):
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        from hindsight_api.engine.providers.gemini_llm import GeminiLLM

        provider = GeminiLLM(
            provider="gemini",
            api_key="fake-api-key",
            base_url="",
            model="gemini-2.5-flash",
            timeout=timeout,
        )
        provider._client = MagicMock()
        return provider


def _text_response(text: str = "done"):
    """The minimum shape both call paths read off a response."""
    part = MagicMock()
    part.text = text
    part.function_call = None
    part.thought_signature = None
    content = MagicMock()
    content.parts = [part]
    candidate = MagicMock()
    candidate.content = content
    response = MagicMock()
    response.text = text
    response.candidates = [candidate]
    response.usage_metadata = None
    return response


async def _never_answers():
    """A request the provider accepts and never completes."""
    await asyncio.sleep(3600)


async def _answers(response):
    return response


def _generate_content(*behaviours):
    """Stub ``generate_content``: call N returns the Nth behaviour's awaitable.

    A plain MagicMock rather than an AsyncMock, because the provider awaits what the
    call *returns* (``asyncio.wait_for(generate_content(...))``) — an AsyncMock would
    hand back the stalling coroutine as the resolved value instead of running it.
    The last behaviour repeats, so a "stalls forever" stub needs only one entry.
    """
    calls = {"n": 0}

    def _call(*_args, **_kwargs):
        idx = min(calls["n"], len(behaviours) - 1)
        calls["n"] += 1
        return behaviours[idx]()

    return MagicMock(side_effect=_call)


@pytest.mark.asyncio
async def test_call_retries_once_after_the_deadline():
    provider = _make_gemini_provider()
    generate = _generate_content(_never_answers, lambda: _answers(_text_response("recovered")))
    provider._client.aio.models.generate_content = generate

    result = await provider.call(messages=[{"role": "user", "content": "hi"}], scope="reflect", initial_backoff=0.0)

    assert result == "recovered", "the retry's answer must be returned, not the stall"
    assert generate.call_count == 2


@pytest.mark.asyncio
async def test_call_gives_up_once_the_timeout_budget_is_spent():
    """The timeout budget, not the whole API-error ladder — the worst case must stay bounded."""
    provider = _make_gemini_provider()
    generate = _generate_content(_never_answers)
    provider._client.aio.models.generate_content = generate

    with pytest.raises(TimeoutError):
        await provider.call(
            messages=[{"role": "user", "content": "hi"}], scope="reflect", max_retries=5, initial_backoff=0.0
        )

    assert generate.call_count == 1 + _TIMEOUT_RETRIES  # NOT 6 (1 + max_retries)


@pytest.mark.asyncio
async def test_call_with_tools_retries_once_after_the_deadline():
    """The tool path is the one reflect uses, so it carries the same rule."""
    provider = _make_gemini_provider()
    generate = _generate_content(_never_answers, lambda: _answers(_text_response("recovered")))
    provider._client.aio.models.generate_content = generate

    result = await provider.call_with_tools(
        messages=[{"role": "user", "content": "hi"}], tools=_TOOLS, scope="reflect", initial_backoff=0.0
    )

    assert result.content == "recovered"
    assert generate.call_count == 2


@pytest.mark.asyncio
async def test_call_with_tools_gives_up_once_the_timeout_budget_is_spent():
    provider = _make_gemini_provider()
    generate = _generate_content(_never_answers)
    provider._client.aio.models.generate_content = generate

    with pytest.raises(TimeoutError):
        await provider.call_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=_TOOLS,
            scope="reflect",
            max_retries=5,
            initial_backoff=0.0,
        )

    assert generate.call_count == 1 + _TIMEOUT_RETRIES
