"""opencode-go's /v1/responses endpoint rejects every tool_choice value except
the default. muse-spark-1.3-contributor, grok-4.6, and gpt-5.6-luna all return
HTTP 400 for ``"required"``, ``"none"`` and named function choices — see
``engine/providers/openai_responses_llm.py::_drops_tool_choice_required``.

The Responses provider must downgrade any non-``auto`` ``tool_choice`` to
``None`` (omit the field) when targeting ``opencode.ai``, otherwise the reflect
agent's tool-calling loop fails every turn. This file pins that downgrade.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.engine.llm_interface import LLMToolChoice, LLMToolChoiceMode
from hindsight_api.engine.providers.openai_responses_llm import OpenAIResponsesLLM


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


def _llm(base_url: str, provider: str = "openai-responses") -> OpenAIResponsesLLM:
    return OpenAIResponsesLLM(
        provider=provider,
        api_key="test-key",
        base_url=base_url,
        model="muse-spark-1.3-contributor",
    )


def _sent_tool_choice(create: AsyncMock):
    """The ``tool_choice`` kwarg the provider sent to ``responses.create``."""
    return create.call_args.kwargs.get("tool_choice")


# --------------------------------------------------------------------------- #
# Host classification: which base URLs trigger the downgrade.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "base_url",
    [
        "https://opencode.ai/zen/go/v1",
        "https://opencode.ai/zen/go/v1/responses",
    ],
)
def test_drops_tool_choice_required_for_opencode_hosts(base_url):
    """Any ``opencode.ai`` host reports as needing the downgrade."""
    llm = _llm(base_url)
    assert llm._drops_tool_choice_required() is True


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.openai.com/v1",
        "",
    ],
)
def test_drops_tool_choice_required_false_for_native_openai(base_url):
    """Native OpenAI Responses accepts every ``tool_choice`` value — no downgrade."""
    llm = _llm(base_url)
    assert llm._drops_tool_choice_required() is False


def test_drops_tool_choice_required_rejects_evil_opencode_suffix():
    """A hostile suffix like ``evil-opencode.ai`` must NOT match.

    Host parsing already enforces exact-or-parent-domain; this guards the
    downgrade from being silently activated by a typo'd base URL.
    """
    llm = _llm("https://evil-opencode.ai")
    assert llm._drops_tool_choice_required() is False


# --------------------------------------------------------------------------- #
# Regression: tool_choice="required" is downgraded to None on opencode.ai.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_required_tool_choice_is_downgraded_against_opencode():
    """The reflect agent's tool-calling loop uses ``LLM_TOOL_CHOICE_REQUIRED``.

    Without the downgrade opencode-go returns HTTP 400 with
    ``only "auto" is supported for tool_choice`` and the agent fails every
    turn. The downgrade must omit the field entirely.
    """
    llm = _llm("https://opencode.ai/zen/go/v1")
    create = AsyncMock(return_value=_responses_response())
    llm._client.responses.create = create

    with patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"):
        await llm.call_with_tools(
            messages=[{"role": "user", "content": "recall something"}],
            tools=[],
            tool_choice=LLMToolChoice(mode=LLMToolChoiceMode.REQUIRED),
            max_retries=0,
        )

    assert "tool_choice" not in create.call_args.kwargs, (
        "opencode-go rejects tool_choice='required'; must be downgraded to None"
    )


@pytest.mark.asyncio
async def test_named_tool_choice_is_downgraded_against_opencode():
    """A named function choice is also unsupported by opencode-go Responses."""
    llm = _llm("https://opencode.ai/zen/go/v1")
    create = AsyncMock(return_value=_responses_response())
    llm._client.responses.create = create

    with patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"):
        await llm.call_with_tools(
            messages=[{"role": "user", "content": "use recall"}],
            tools=[{"type": "function", "name": "recall"}],
            tool_choice=LLMToolChoice.named("recall"),
            max_retries=0,
        )

    assert "tool_choice" not in create.call_args.kwargs


@pytest.mark.asyncio
async def test_auto_tool_choice_is_left_alone_against_opencode():
    """``tool_choice=None`` (AUTO) is the value opencode-go already expects."""
    llm = _llm("https://opencode.ai/zen/go/v1")
    create = AsyncMock(return_value=_responses_response())
    llm._client.responses.create = create

    with patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"):
        await llm.call_with_tools(
            messages=[{"role": "user", "content": "anything"}],
            tools=[{"type": "function", "name": "recall"}],
            tool_choice=None,  # AUTO mode
            max_retries=0,
        )

    assert "tool_choice" not in create.call_args.kwargs


@pytest.mark.asyncio
async def test_required_tool_choice_passes_through_to_native_openai():
    """Native OpenAI Responses must keep ``tool_choice="required"``.

    The downgrade is host-specific; a ``provider=openai-responses`` deployment
    on the native endpoint (e.g. ``gpt-5.6``) still gets the structured value.
    """
    llm = OpenAIResponsesLLM(
        provider="openai-responses",
        api_key="test-key",
        base_url="",
        model="gpt-5.6",
    )
    create = AsyncMock(return_value=_responses_response())
    llm._client.responses.create = create

    with patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"):
        await llm.call_with_tools(
            messages=[{"role": "user", "content": "anything"}],
            tools=[{"type": "function", "name": "recall"}],
            tool_choice=LLMToolChoice(mode=LLMToolChoiceMode.REQUIRED),
            max_retries=0,
        )

    assert create.call_args.kwargs.get("tool_choice") == "required"


@pytest.mark.asyncio
async def test_named_tool_choice_passes_through_to_native_openai():
    llm = OpenAIResponsesLLM(
        provider="openai-responses",
        api_key="test-key",
        base_url="",
        model="gpt-5.6",
    )
    create = AsyncMock(return_value=_responses_response())
    llm._client.responses.create = create

    with patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"):
        await llm.call_with_tools(
            messages=[{"role": "user", "content": "use recall"}],
            tools=[{"type": "function", "name": "recall"}],
            tool_choice=LLMToolChoice.named("recall"),
            max_retries=0,
        )

    assert create.call_args.kwargs.get("tool_choice") == {
        "type": "function",
        "name": "recall",
    }
