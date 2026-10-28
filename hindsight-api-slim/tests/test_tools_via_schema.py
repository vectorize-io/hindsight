"""Reflect's tool calls through a response schema (HINDSIGHT_API_LLM_TOOLS_VIA_SCHEMA).

Some OpenAI-compatible servers parse tool calls but do not enforce ``tool_choice``:
vLLM's gemma4 parser skips the grammar for "required" and named choices. Reflect's
forced steps then come back as the stop signal or prose, and its closing ``done`` as
another tool. Such servers do enforce a response schema, so with the setting on a named
choice is asked for in words and through its arguments' schema, and the JSON reply is
handed back as the call. In document mode ``done`` then takes the page as markdown.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.llm_interface import LLM_TOOL_CHOICE_REQUIRED, LLMToolChoice
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM
from hindsight_api.engine.reflect.agent import run_reflect_agent
from hindsight_api.engine.reflect.prompts import build_system_prompt_for_tools
from hindsight_api.engine.reflect.tools_schema import get_reflect_tools
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult
from tests.test_reflect_agent import _StepCacheProvider, mock_functions  # noqa: F401 - the fixture

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Search raw facts. Use it for details",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}, "query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer_ready",
            "description": "Stop retrieving",
            "parameters": {"type": "object", "properties": {"answer_found": {"type": "boolean"}}},
        },
    },
]


@pytest.fixture
def via_schema(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_LLM_TOOLS_VIA_SCHEMA", "true")


def _response(content: str | None, tool_calls=None, finish_reason: str = "stop") -> MagicMock:
    resp = MagicMock()
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 20
    resp.usage.total_tokens = 120
    resp.usage.completion_tokens_details = None
    resp.error = None
    resp.model_dump.return_value = {}
    resp.choices[0].finish_reason = finish_reason
    resp.choices[0].message.content = content
    resp.choices[0].message.tool_calls = tool_calls
    return resp


async def _call(tool_choice, reply: MagicMock):
    llm = OpenAICompatibleLLM(provider="openai", api_key="k", base_url="http://vllm:8000/v1", model="gemma-4")
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = reply
        result = await llm.call_with_tools(
            messages=[{"role": "user", "content": "What does the user prefer?"}],
            tools=TOOLS,
            tool_choice=tool_choice,
            max_retries=0,
        )
        return result, create.call_args.kwargs


class TestTheProvider:
    @pytest.mark.asyncio
    async def test_a_named_choice_is_asked_for_through_its_arguments_schema(self, via_schema):
        result, sent = await _call(LLMToolChoice.named("recall"), _response('{"query": "tooling preferences"}'))

        # Every tool stays declared, so the prompt keeps the prefix the earlier turns cached.
        assert [t["function"]["name"] for t in sent["tools"]] == ["recall", "answer_ready"]
        assert sent["tool_choice"] == "none"
        schema = sent["response_format"]["json_schema"]["schema"]
        assert sent["response_format"]["json_schema"]["strict"] is True
        # Strict: every property required, optional ones nullable. The reason goes last: written
        # first, it sent Gemma 4 into a query that never closed.
        assert list(schema["properties"]) == ["query", "reason"]
        assert schema["required"] == ["query", "reason"]
        assert schema["additionalProperties"] is False
        # No string bounds: under xgrammar a bounded string admits no escape, so no "\\n" in a page.
        assert "minLength" not in schema["properties"]["query"]
        # Asked for in words too: a schema on a reply nobody asked for gave degenerate arguments.
        ask = sent["messages"][-1]
        assert ask["role"] == "user" and "`recall`" in ask["content"] and "Search raw facts" in ask["content"]

        assert [(c.name, c.arguments) for c in result.tool_calls] == [("recall", {"query": "tooling preferences"})]
        assert result.content is None

    @pytest.mark.asyncio
    async def test_an_unparseable_reply_rides_as_raw_arguments(self, via_schema):
        result, _ = await _call(LLMToolChoice.named("recall"), _response('{"query": "cut of', finish_reason="length"))
        assert result.tool_calls[0].arguments == {"_raw": '{"query": "cut of'}
        assert result.finish_reason == "length"

    @pytest.mark.asyncio
    async def test_a_free_turn_is_one_of_the_tools_with_its_arguments(self, via_schema):
        reply = '{"call": {"name": "recall", "arguments": {"query": "editors", "reason": null}}}'
        result, sent = await _call(LLM_TOOL_CHOICE_REQUIRED, _response(reply))

        assert sent["tool_choice"] == "none"
        options = sent["response_format"]["json_schema"]["schema"]["properties"]["call"]["anyOf"]
        assert [o["properties"]["name"]["enum"] for o in options] == [["recall"], ["answer_ready"]]
        # Strict subset: an argument the tool left optional is nullable, and a null is dropped.
        assert options[0]["properties"]["arguments"]["properties"]["reason"]["type"] == ["string", "null"]
        assert "ONE of the declared functions" in sent["messages"][-1]["content"]
        assert [(c.name, c.arguments) for c in result.tool_calls] == [("recall", {"query": "editors"})]

    @pytest.mark.asyncio
    async def test_a_reply_naming_no_call_is_a_turn_without_one(self, via_schema):
        result, _ = await _call(LLM_TOOL_CHOICE_REQUIRED, _response('{"call": {}}'))
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_a_reply_cut_off_at_the_cap_is_sampled_once_more(self, via_schema):
        llm = OpenAICompatibleLLM(provider="openai", api_key="k", base_url="http://vllm:8000/v1", model="gemma-4")
        looped = _response('{"query": "owner owner owner owner', finish_reason="length")
        with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
            create.side_effect = [looped, _response('{"query": "ranking service owner"}')]
            result = await llm.call_with_tools(
                messages=[{"role": "user", "content": "Who owns ranking?"}],
                tools=TOOLS,
                tool_choice=LLMToolChoice.named("recall"),
                max_retries=0,
            )
        assert create.await_count == 2
        assert result.tool_calls[0].arguments == {"query": "ranking service owner"}

    @pytest.mark.asyncio
    async def test_off_by_default_a_named_choice_narrows_the_tools(self):
        _, sent = await _call(LLMToolChoice.named("recall"), _response(None))
        assert [t["function"]["name"] for t in sent["tools"]] == ["recall"]
        assert sent["tool_choice"] == "required"
        assert "response_format" not in sent


class TestTheMarkdownDocument:
    def test_done_takes_markdown_then_the_cited_ids(self):
        done = next(
            t
            for t in get_reflect_tools(answer_as_document=True, answer_ready=True, document_as_markdown=True)
            if t["function"]["name"] == "done"
        )
        params = done["function"]["parameters"]
        # Field order is writing order under a schema: the citations come after the text.
        assert list(params["properties"])[:2] == ["answer_found", "markdown"]
        assert list(params["properties"])[-3:] == ["memory_ids", "mental_model_ids", "observation_ids"]
        assert "document" not in params["properties"]
        # Required, ids included: held to a schema, a model skips whatever it may skip.
        assert params["required"] == list(params["properties"])
        assert "as a markdown document" in done["function"]["description"]

    def test_the_system_prompt_asks_for_markdown_and_not_for_sections(self):
        prompt = build_system_prompt_for_tools(
            {"name": "B", "mission": ""}, answer_as_document=True, document_as_markdown=True
        )
        assert "Output Format: Markdown Document" in prompt
        assert "'sections' array" not in prompt


class TestTheRefresh:
    @pytest.mark.asyncio
    async def test_a_page_written_as_markdown_is_stored_as_its_sections(self, via_schema, mock_functions):  # noqa: F811
        provider = _StepCacheProvider(
            scripted=[
                LLMToolCallResult(
                    tool_calls=[LLMToolCall(id="1", name="recall", arguments={"query": "q"})],
                    finish_reason="tool_calls",
                ),
                LLMToolCallResult(
                    tool_calls=[LLMToolCall(id="2", name="answer_ready", arguments={"answer_found": True})],
                    finish_reason="tool_calls",
                ),
                LLMToolCallResult(
                    tool_calls=[
                        LLMToolCall(
                            id="3",
                            name="done",
                            arguments={
                                "answer_found": True,
                                "markdown": "## Tools\n\nThe user prefers ruff.\n\n## Editors\n\n- vim\n- helix",
                                "memory_ids": ["f1"],
                            },
                        )
                    ],
                    finish_reason="tool_calls",
                ),
            ]
        )
        sent_tools: list[list[dict]] = []
        original = provider.call_with_tools

        max_tokens: list = []

        async def _record(*, tools, **kwargs):
            sent_tools.append(tools)
            max_tokens.append(kwargs.get("max_completion_tokens"))
            return await original(tools=tools, **kwargs)

        provider.call_with_tools = _record

        result = await run_reflect_agent(
            llm_config=provider,
            bank_id="b",
            query="What tools does the user prefer?",
            bank_profile={"name": "T", "mission": "M"},
            has_mental_models=False,
            budget="high",
            max_iterations=8,
            answer_as_document=True,
            **mock_functions,
        )

        assert [s.heading for s in result.document.sections] == ["Tools", "Editors"]
        # A loop turn only names a search here, so it is capped; the closing done is not.
        assert max_tokens[:-1] and all(cap == 1024 for cap in max_tokens[:-1]), max_tokens
        assert max_tokens[-1] != 1024
        assert result.used_memory_ids == ["mem-1"], "f1 resolves to the memory recall returned"
        # The closing done may cite only the handles the results gave out.
        closing_done = next(t for t in sent_tools[-1] if t["function"]["name"] == "done")
        cited = closing_done["function"]["parameters"]["properties"]
        assert cited["memory_ids"]["items"]["enum"] == ["f1"]
        assert cited["observation_ids"]["maxItems"] == 0
        assert json.dumps(closing_done).count('"markdown"') >= 1


class TestCodex:
    @pytest.mark.asyncio
    async def test_codex_asks_for_the_call_through_text_format(self, via_schema):
        from hindsight_api.engine.providers.codex_llm import CodexLLM, _CodexStreamReply
        from tests.codex_stream_stub import stub_codex_stream

        with patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")):
            llm = CodexLLM(provider="openai-codex", api_key="x", base_url="https://chatgpt.com/backend-api", model="m")
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        with stub_codex_stream(llm, response) as stream:
            with patch.object(llm, "_parse_sse_tool_stream", new_callable=AsyncMock) as parse:
                parse.return_value = _CodexStreamReply(text='{"query": "editors"}', tool_calls=[], usage=None)
                result = await llm.call_with_tools(
                    messages=[{"role": "user", "content": "What editors?"}],
                    tools=TOOLS,
                    tool_choice=LLMToolChoice.named("recall"),
                    max_retries=0,
                )
            sent = stream.call_args.kwargs["json"]

        assert sent["tool_choice"] == "none"
        assert [t["name"] for t in sent["tools"]] == ["recall", "answer_ready"], "the tools stay declared"
        assert sent["text"]["format"]["type"] == "json_schema"
        assert sent["input"][-1]["content"].startswith("Call `recall` now")
        assert [(c.name, c.arguments) for c in result.tool_calls] == [("recall", {"query": "editors"})]


@pytest.mark.asyncio
async def test_codex_reads_a_schema_reply_streamed_as_output_text(via_schema):
    """The Responses API streams text as response.output_text.delta: the tool-call parser
    used to read only response.text.delta, so every schema-held reply came back empty."""
    from hindsight_api.engine.providers.codex_llm import CodexLLM
    from tests.codex_stream_stub import CodexReply, stub_codex_stream

    with (
        patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")),
        patch.object(CodexLLM, "_load_codex_refresh_token", return_value=None),
    ):
        llm = CodexLLM(provider="openai-codex", api_key="x", base_url="https://chatgpt.com/backend-api", model="m")
    stream = "".join(
        "event: response.output_text.delta\ndata: "
        + json.dumps({"type": "response.output_text.delta", "delta": d})
        + "\n\n"
        for d in ('{"query": ', '"editors"}')
    )
    with stub_codex_stream(llm, CodexReply(text=stream)):
        result = await llm.call_with_tools(
            messages=[{"role": "user", "content": "What editors?"}],
            tools=TOOLS,
            tool_choice=LLMToolChoice.named("recall"),
            max_retries=0,
        )
    assert [(c.name, c.arguments) for c in result.tool_calls] == [("recall", {"query": "editors"})]
