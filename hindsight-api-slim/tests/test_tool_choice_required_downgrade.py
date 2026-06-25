"""Regression tests for the silent-drop of ``tool_choice="required"``.

Reproduces issues #1563 (LM Studio), #1179 (LM Studio + Qwen) and #1877 (vLLM
with ``--enable-auto-tool-choice``). These self-hosted OpenAI-compatible servers
advertise ``tool_choice="required"`` but silently ignore it — returning a
``finish_reason`` of ``"stop"``/``"tool_calls"`` with an EMPTY ``tool_calls``
array and no HTTP error. Reflect's agent loop then sees no tool call, runs
synthesis with no retrieval, and answers "I don't have information" even when the
bank holds the answer.

The fix downgrades ``"required"`` to auto (omitted) for these endpoints so the
model still gets to call a tool. Named ``tool_choice`` dicts are normalized to
``"required"`` + a single filtered tool first, so forced calls stay practically
forced even under auto. The real OpenAI API and cloud providers honor
``"required"`` and must be left untouched.

These are fast, deterministic unit tests: the OpenAI client's ``create`` is
mocked and we assert on the exact kwargs sent over the wire.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Recall semantic memories",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Finish and return the answer",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    },
]


def _make_llm(provider: str, base_url: str) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider=provider,
        api_key="local" if provider in ("ollama", "lmstudio") else "sk-test",
        base_url=base_url,
        model="qwen3",
    )


def _tool_call_response() -> MagicMock:
    """A populated tool call — what these servers return under auto (not required)."""
    tc = MagicMock()
    tc.id = "call_abc123"
    tc.function.name = "recall"
    tc.function.arguments = json.dumps({"query": "test"})

    resp = MagicMock()
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    resp.usage.total_tokens = 15
    # Explicit None: an auto-MagicMock here is truthy, so the reasoning-token
    # accounting (#2378) would do arithmetic on a MagicMock and crash.
    resp.usage.completion_tokens_details = None
    resp.choices[0].finish_reason = "tool_calls"
    resp.choices[0].message.content = None
    resp.choices[0].message.tool_calls = [tc]
    return resp


async def _capture_call(llm: OpenAICompatibleLLM, tool_choice) -> dict:
    """Invoke call_with_tools and return the kwargs sent to the OpenAI client."""
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = _tool_call_response()
        await llm.call_with_tools(
            messages=[{"role": "user", "content": "What did claude say?"}],
            tools=TOOLS,
            tool_choice=tool_choice,
            max_retries=0,
        )
    return mock_create.call_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,base_url",
    [
        ("lmstudio", "http://localhost:1234/v1"),  # #1563 / #1179
        ("ollama", "http://localhost:11434/v1"),  # #1563
        ("openai", "http://vllm:8000/v1"),  # #1877 — self-hosted vLLM via openai provider
    ],
)
async def test_required_is_downgraded_for_self_hosted(provider: str, base_url: str):
    """``required`` is omitted (downgraded to auto) for servers that drop it."""
    sent = await _capture_call(_make_llm(provider, base_url), "required")
    assert "tool_choice" not in sent, (
        f"{provider} should not receive tool_choice='required' (it silently drops it); got {sent.get('tool_choice')!r}"
    )


@pytest.mark.asyncio
async def test_required_preserved_for_real_openai():
    """The real OpenAI API honors ``required`` — no base_url override, so leave it."""
    sent = await _capture_call(_make_llm("openai", ""), "required")
    assert sent["tool_choice"] == "required"


@pytest.mark.asyncio
async def test_required_preserved_for_llama_server():
    """llama-server (llamacpp) honors ``required`` and is excluded (#1179)."""
    sent = await _capture_call(_make_llm("llamacpp", "http://localhost:8080/v1"), "required")
    assert sent["tool_choice"] == "required"


@pytest.mark.asyncio
async def test_required_preserved_for_cloud_provider():
    """Cloud providers (e.g. groq) honor ``required`` and keep it."""
    sent = await _capture_call(_make_llm("groq", ""), "required")
    assert sent["tool_choice"] == "required"


@pytest.mark.asyncio
async def test_named_tool_choice_forced_call_survives_downgrade():
    """Reflect forces a tool via a named dict; it must still effectively force.

    The dict is normalized to ``required`` + a single filtered tool, then the
    downgrade drops ``required`` to auto. With only one tool available the call
    stays practically forced, so the model still emits the tool call instead of
    the empty-tool_calls failure mode.
    """
    llm = _make_llm("lmstudio", "http://localhost:1234/v1")
    named = {"type": "function", "function": {"name": "recall"}}
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = _tool_call_response()
        result = await llm.call_with_tools(
            messages=[{"role": "user", "content": "What did claude say?"}],
            tools=TOOLS,
            tool_choice=named,
            max_retries=0,
        )

    sent = mock_create.call_args.kwargs
    # required was dropped (the silent-drop trigger is gone) ...
    assert "tool_choice" not in sent
    # ... but tools were narrowed to just the forced one, keeping the call forced.
    assert len(sent["tools"]) == 1
    assert sent["tools"][0]["function"]["name"] == "recall"
    # and the model returns the tool call rather than an empty array.
    assert [tc.name for tc in result.tool_calls] == ["recall"]


def test_drops_tool_choice_required_classification():
    """Direct unit check of the endpoint classifier."""
    assert _make_llm("lmstudio", "http://localhost:1234/v1")._drops_tool_choice_required()
    assert _make_llm("ollama", "http://localhost:11434/v1")._drops_tool_choice_required()
    assert _make_llm("openai", "http://vllm:8000/v1")._drops_tool_choice_required()
    # Real OpenAI, llama-server, and cloud providers are not affected.
    assert not _make_llm("openai", "")._drops_tool_choice_required()
    assert not _make_llm("llamacpp", "http://localhost:8080/v1")._drops_tool_choice_required()
    assert not _make_llm("groq", "")._drops_tool_choice_required()
