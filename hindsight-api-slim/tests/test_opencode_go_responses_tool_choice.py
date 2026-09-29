"""OpenCode Go's /v1/responses rejects every tool_choice except the default.

muse-spark-1.3-contributor, grok-4.6 and gpt-5.6-luna all return HTTP 400
(``only "auto" is supported``) for ``"required"``, ``"none"`` and named
function choices, so the reflect agent's tool loop failed every turn. The
Responses provider omits the field when the base URL is an ``opencode.ai``
host, and keeps it everywhere else.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.engine.cache_affinity import is_opencode_host
from hindsight_api.engine.llm_interface import LLM_TOOL_CHOICE_AUTO, LLMToolChoice, LLMToolChoiceMode
from hindsight_api.engine.providers.openai_responses_llm import OpenAIResponsesLLM

OPENCODE_URL = "https://opencode.ai/zen/go/v1"
RECALL_TOOL = {"type": "function", "function": {"name": "recall", "parameters": {"type": "object"}}}


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (OPENCODE_URL, True),
        ("https://api.opencode.ai/v1", True),
        ("https://api.openai.com/v1", False),
        ("https://evil-opencode.ai", False),
        ("https://opencode.ai.evil.example", False),
        ("", False),
        (None, False),
    ],
)
def test_is_opencode_host(base_url, expected):
    assert is_opencode_host(base_url) is expected


async def _sent_kwargs(base_url: str, tool_choice: LLMToolChoice) -> dict:
    llm = OpenAIResponsesLLM(provider="openai-responses", api_key="k", base_url=base_url, model="gpt-5.6")
    create = AsyncMock(
        return_value=SimpleNamespace(
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
    )
    llm._client.responses.create = create
    with patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"):
        await llm.call_with_tools(
            messages=[{"role": "user", "content": "recall something"}],
            tools=[RECALL_TOOL],
            tool_choice=tool_choice,
            max_retries=0,
        )
    return create.call_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_choice",
    [
        LLMToolChoice(mode=LLMToolChoiceMode.REQUIRED),
        LLMToolChoice(mode=LLMToolChoiceMode.NONE),
        LLMToolChoice.named("recall"),
        LLM_TOOL_CHOICE_AUTO,
    ],
)
async def test_opencode_omits_tool_choice(tool_choice):
    kwargs = await _sent_kwargs(OPENCODE_URL, tool_choice)
    assert "tool_choice" not in kwargs


@pytest.mark.asyncio
async def test_opencode_named_choice_still_narrows_tools():
    """Without tool_choice, the narrowed tools list is what keeps the call forced."""
    kwargs = await _sent_kwargs(OPENCODE_URL, LLMToolChoice.named("recall"))
    assert [tool["name"] for tool in kwargs["tools"]] == ["recall"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        (LLMToolChoice(mode=LLMToolChoiceMode.REQUIRED), "required"),
        (LLMToolChoice.named("recall"), {"type": "function", "name": "recall"}),
    ],
)
async def test_native_openai_keeps_tool_choice(tool_choice, expected):
    kwargs = await _sent_kwargs("", tool_choice)
    assert kwargs["tool_choice"] == expected
