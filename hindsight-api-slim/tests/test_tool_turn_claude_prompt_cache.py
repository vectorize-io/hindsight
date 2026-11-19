"""Tool turns on Claude: one reply shape for every turn, and a cache marker each next turn reads.

Claude caches a prompt up to a ``cache_control`` part, and drops that cache when the reply
schema changes. Measured on Bedrock (Haiku 4.5) before this: a reflect's forced searches, free
turns and closing ``done`` each had their own schema and no turn read a cache back; after it,
each turn read the conversation so far and the closing call read all but its own request.
"""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from hindsight_api.config import ENV_LLM_LITELLM_PROMPT_CACHE, clear_config_cache
from hindsight_api.engine.llm_interface import LLM_TOOL_CHOICE_REQUIRED, LLMToolChoice
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.providers.litellm_llm import LiteLLMLLM
from hindsight_api.engine.response_models import LLMCallResult, TokenUsage

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} over the bank",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    }
    for name in ("search_observations", "recall")
] + [
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Give the answer",
            "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
        },
    }
]
LOOP_TOOLS = TOOLS[:2]


def _reply(name: str, arguments: dict[str, Any]) -> LLMCallResult:
    return LLMCallResult(
        content={"call": {"name": name, "arguments": arguments}},
        usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
    )


def _claude(*replies: LLMCallResult) -> LLMProvider:
    """A Claude provider whose model calls answer with ``replies``; read them on ``_provider_impl.call``."""
    llm = LLMProvider(provider="anthropic", api_key="k", base_url="", model="claude-haiku-4-5")
    llm._provider_impl.call = AsyncMock(side_effect=list(replies))  # type: ignore[method-assign]
    return llm


def _without_markers(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """As Claude reads them: a string is one text part, and a marker is not content."""

    def strip(content: Any) -> Any:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        return [{k: v for k, v in part.items() if k != "cache_control"} for part in content]

    return [{**m, "content": strip(m["content"])} for m in messages]


def _marked(messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
    return [
        (i, j)
        for i, m in enumerate(messages)
        if isinstance(m["content"], list)
        for j, part in enumerate(m["content"])
        if "cache_control" in part
    ]


@pytest.mark.asyncio
async def test_forced_free_and_closing_turns_share_one_reply_schema():
    llm = _claude(
        _reply("search_observations", {"query": "q"}),
        _reply("recall", {"query": "q"}),
        _reply("done", {"answer": "a"}),
    )
    call = llm._provider_impl.call
    messages = [{"role": "user", "content": "Who is on call?"}]
    for choice, tools in (
        (LLMToolChoice.named("search_observations"), LOOP_TOOLS),
        (LLM_TOOL_CHOICE_REQUIRED, LOOP_TOOLS),
        (LLMToolChoice.named("done"), TOOLS),
    ):
        await llm.call_with_tools(messages=messages, tools=tools, tool_choice=choice, declared_tools=TOOLS)

    schemas = [c.kwargs["response_format"].model_json_schema() for c in call.await_args_list]
    assert schemas[0] == schemas[1] == schemas[2]
    assert {o["properties"]["name"]["enum"][0] for o in schemas[0]["properties"]["call"]["anyOf"]} == {
        "search_observations",
        "recall",
        "done",
    }
    # The forced turn says, in words, which function it must call.
    assert "`search_observations`" in call.await_args_list[0].kwargs["messages"][-1]["content"][-1]["text"]


@pytest.mark.asyncio
async def test_a_forced_turn_answered_with_another_function_is_corrected_once():
    llm = _claude(_reply("recall", {"query": "q"}), _reply("search_observations", {"query": "q"}))
    call = llm._provider_impl.call

    result = await llm.call_with_tools(
        messages=[{"role": "user", "content": "Who is on call?"}],
        tools=LOOP_TOOLS,
        tool_choice=LLMToolChoice.named("search_observations"),
        declared_tools=TOOLS,
    )

    assert [c.name for c in result.tool_calls] == ["search_observations"]
    assert "must call `search_observations`" in call.await_args_list[1].kwargs["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_a_free_turn_names_its_functions_and_refuses_one_it_leaves_out():
    """Reflect's free turns leave ``done`` out; under the shared schema the request says so."""
    llm = _claude(_reply("done", {"answer": "a"}), _reply("recall", {"query": "q"}))
    call = llm._provider_impl.call

    result = await llm.call_with_tools(
        messages=[{"role": "user", "content": "Who is on call?"}], tools=LOOP_TOOLS, declared_tools=TOOLS
    )

    assert [c.name for c in result.tool_calls] == ["recall"]
    first, correction = (c.kwargs["messages"][-1]["content"] for c in call.await_args_list)
    assert "(`search_observations`, `recall`)" in first[-1]["text"]
    assert "`done` can't be called on this turn" in correction


@pytest.mark.asyncio
async def test_each_turn_resends_the_last_turns_cached_prefix_unchanged():
    """What a turn marked is, part for part, the start of what the next turn sends."""
    llm = _claude(_reply("recall", {"query": "q"}), _reply("done", {"answer": "a"}))
    call = llm._provider_impl.call
    messages: list[dict[str, Any]] = [{"role": "system", "content": "Mission."}, {"role": "user", "content": "Q?"}]
    await llm.call_with_tools(messages=messages, tools=LOOP_TOOLS, declared_tools=TOOLS)
    messages += [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "recall", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "Mara is on call."},
    ]
    # The closing call adds its own request as a user turn, joined onto the result.
    await llm.call_with_tools(
        messages=[*messages, {"role": "user", "content": "Write the answer."}],
        tools=TOOLS,
        tool_choice=LLMToolChoice.named("done"),
        declared_tools=TOOLS,
    )

    for earlier, later in zip(call.await_args_list, call.await_args_list[1:]):
        sent, next_sent = earlier.kwargs["messages"], later.kwargs["messages"]
        ((i, j),) = _marked(sent)
        cached = _without_markers(sent)[: i + 1]
        cached[i] = {**cached[i], "content": cached[i]["content"][: j + 1]}
        resent = _without_markers(next_sent)[: i + 1]
        resent[i] = {**resent[i], "content": resent[i]["content"][: j + 1]}
        assert resent == cached
    # And the closing call marks the result it was given, so the request after it is all it writes.
    closing = call.await_args_list[-1].kwargs["messages"]
    ((i, j),) = _marked(closing)
    assert closing[i]["content"][j]["text"] == "Write the answer."
    assert closing[i]["content"][j - 1]["text"].startswith("Result of `recall`")


@pytest.mark.asyncio
async def test_other_providers_keep_a_schema_per_forced_turn():
    llm = LLMProvider(provider="openai", api_key="k", base_url="http://vllm:8000/v1", model="gemma-4")
    call = AsyncMock(return_value=LLMCallResult(content={"query": "q"}, usage=TokenUsage()))
    llm._provider_impl.call = call  # type: ignore[method-assign]

    await llm.call_with_tools(
        messages=[{"role": "user", "content": "Q?"}], tools=LOOP_TOOLS, tool_choice=LLMToolChoice.named("recall")
    )

    sent = call.await_args.kwargs
    assert "call" not in sent["response_format"].model_json_schema()["properties"]
    assert not _marked(sent["messages"])


@pytest.mark.parametrize(
    ("model", "marks"),
    [
        ("bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0", True),
        ("vertex_ai/claude-sonnet-4-5", True),
        ("bedrock/us.amazon.nova-lite-v1:0", False),
        ("azure/gpt-4o", False),
    ],
)
def test_litellm_marks_only_claude_models(monkeypatch, model: str, marks: bool):
    monkeypatch.delenv(ENV_LLM_LITELLM_PROMPT_CACHE, raising=False)
    clear_config_cache()
    llm = LiteLLMLLM(provider="litellm", api_key="", base_url="", model=model)
    assert llm.caches_marked_prefix() is marks


@pytest.mark.parametrize(("setting", "marks"), [("true", True), ("false", False)])
def test_the_setting_overrides_the_model_name(monkeypatch, setting: str, marks: bool):
    """A Claude model behind an application inference profile ARN has no "claude" in its name."""
    monkeypatch.setenv(ENV_LLM_LITELLM_PROMPT_CACHE, setting)
    clear_config_cache()
    try:
        arn = "bedrock/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"
        assert LiteLLMLLM(provider="bedrock", api_key="", base_url="", model=arn).caches_marked_prefix() is marks
        claude = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
        assert LiteLLMLLM(provider="bedrock", api_key="", base_url="", model=claude).caches_marked_prefix() is marks
    finally:
        clear_config_cache()  # read lazily, after monkeypatch has put the env back
