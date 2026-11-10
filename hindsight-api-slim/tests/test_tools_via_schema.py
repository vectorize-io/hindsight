"""Tool calls as replies to a response schema: the one way every provider makes them.

``LLMProvider.call_with_tools`` never sends ``tools`` or ``tool_choice``. It replays the
conversation as plain turns with the functions declared in the system prompt, asks for the
turn's call in words, and holds the reply to a schema through the provider's ``call``: a
named choice to that function's arguments, a free one to "one of the functions, with its
arguments". The JSON reply is handed back as the call. In document mode ``done`` takes the
page as markdown.
"""

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.llm_interface import LLM_TOOL_CHOICE_REQUIRED, LLMToolChoice, OutputTooLongError
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.providers.tools_via_schema import _arguments_schema
from hindsight_api.engine.reflect.agent import run_reflect_agent
from hindsight_api.engine.reflect.prompts import build_system_prompt_for_tools
from hindsight_api.engine.reflect.tools_schema import get_reflect_tools
from hindsight_api.engine.response_models import LLMCallResult, LLMToolCall, LLMToolCallResult, TokenUsage
from tests.test_reflect_agent import _ScriptedProvider, mock_functions  # noqa: F401 - the fixture

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


def _reply(content) -> LLMCallResult:
    return LLMCallResult(content=content, usage=TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120))


async def _call(tool_choice, *replies, messages=None):
    llm = LLMProvider(provider="openai", api_key="k", base_url="http://vllm:8000/v1", model="gemma-4")
    llm._provider_impl.call = AsyncMock(side_effect=list(replies))
    result = await llm.call_with_tools(
        messages=messages or [{"role": "user", "content": "What does the user prefer?"}],
        tools=TOOLS,
        tool_choice=tool_choice,
        max_retries=0,
    )
    return result, llm._provider_impl.call


class TestTheCall:
    @pytest.mark.asyncio
    async def test_a_named_choice_is_asked_for_through_its_arguments_schema(self):
        result, call = await _call(LLMToolChoice.named("recall"), _reply({"query": "tooling preferences"}))

        sent = call.await_args.kwargs
        assert "tools" not in sent and "tool_choice" not in sent
        assert sent["strict_schema"] is True and sent["skip_validation"] is True
        schema = sent["response_format"].model_json_schema()
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
        # The functions are declared in the system prompt instead of as native tools.
        assert sent["messages"][0]["role"] == "system" and '"name": "answer_ready"' in sent["messages"][0]["content"]

        assert [(c.name, c.arguments) for c in result.tool_calls] == [("recall", {"query": "tooling preferences"})]
        assert result.content is None
        assert (result.input_tokens, result.output_tokens) == (100, 20)

    @pytest.mark.asyncio
    async def test_an_unparseable_reply_rides_as_raw_arguments(self):
        result, _ = await _call(LLMToolChoice.named("recall"), _reply('{"query": "cut of'))
        assert result.tool_calls[0].arguments == {"_raw": '{"query": "cut of'}

    @pytest.mark.asyncio
    async def test_a_free_turn_is_one_of_the_tools_with_its_arguments(self):
        reply = {"call": {"name": "recall", "arguments": {"query": "editors", "reason": None}}}
        result, call = await _call(LLM_TOOL_CHOICE_REQUIRED, _reply(reply))

        sent = call.await_args.kwargs
        options = sent["response_format"].model_json_schema()["properties"]["call"]["anyOf"]
        assert [o["properties"]["name"]["enum"] for o in options] == [["recall"], ["answer_ready"]]
        # Strict subset: an argument the tool left optional is nullable, and a null is dropped.
        assert options[0]["properties"]["arguments"]["properties"]["reason"] == {
            "anyOf": [{"type": "string"}, {"type": "null"}]
        }
        assert "ONE of the declared functions" in sent["messages"][-1]["content"]
        assert [(c.name, c.arguments) for c in result.tool_calls] == [("recall", {"query": "editors"})]

    @pytest.mark.asyncio
    async def test_declared_tools_are_the_prompt_and_tools_the_choice(self):
        """A turn may narrow what it calls and keep the prompt (and its cached prefix) unchanged."""
        llm = LLMProvider(provider="openai", api_key="k", base_url="http://vllm:8000/v1", model="gemma-4")
        llm._provider_impl.call = AsyncMock(
            return_value=_reply({"call": {"name": "recall", "arguments": {"query": "q"}}})
        )
        await llm.call_with_tools(
            messages=[{"role": "user", "content": "q"}],
            tools=TOOLS[:1],
            declared_tools=TOOLS,
            tool_choice=LLM_TOOL_CHOICE_REQUIRED,
            max_retries=0,
        )

        sent = llm._provider_impl.call.await_args.kwargs
        assert '"name": "answer_ready"' in sent["messages"][0]["content"]
        options = sent["response_format"].model_json_schema()["properties"]["call"]["anyOf"]
        assert [o["properties"]["name"]["enum"] for o in options] == [["recall"]]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "reply",
        [{"recall": {"query": "editors"}}, {"call": {"name": "recall", "arguments": {"query": "editors"}}}],
        ids=["under-its-name", "free-turn-shape"],
    )
    async def test_a_forced_reply_wrapped_in_the_functions_name_is_unwrapped(self, reply):
        """Seen on a provider that does not enforce additionalProperties: the page came back empty."""
        result, _ = await _call(LLMToolChoice.named("recall"), _reply(reply))
        assert result.tool_calls[0].arguments == {"query": "editors"}

    @pytest.mark.asyncio
    async def test_prose_where_the_schema_asked_for_json_is_a_turn_without_a_call(self):
        """A model that does not follow the schema; the caller names it instead of a parse error."""
        prose = "I think Alice lives in Berlin."
        result, _ = await _call(LLMToolChoice.named("recall"), json.JSONDecodeError("Expecting value", prose, 0))
        assert result.tool_calls == [] and result.finish_reason == "stop"
        assert result.content == prose

    @pytest.mark.asyncio
    async def test_a_reply_naming_no_call_is_a_turn_without_one(self):
        result, call = await _call(LLM_TOOL_CHOICE_REQUIRED, _reply({"call": {}}), _reply({"call": {}}))
        assert call.await_count == 2  # asked once more, with what was wrong
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_arguments_flattened_next_to_the_name_are_corrected_once(self):
        """#6420: an endpoint that does not enforce the schema. The flat call reached the agent as
        a successful ``done`` and failed there as "no answer"; now the model is told and re-asked."""
        flat = {"call": {"name": "recall", "query": "editors"}}
        wrapped = {"call": {"name": "recall", "arguments": {"query": "editors"}}}
        result, call = await _call(LLM_TOOL_CHOICE_REQUIRED, _reply(flat), _reply(wrapped))

        assert call.await_count == 2
        assert (result.input_tokens, result.output_tokens) == (200, 40)  # both calls billed
        correction = call.await_args.kwargs["messages"][-1]["content"]
        assert 'under "arguments"' in correction and "ONE of the declared functions" in correction
        assert [(c.name, c.arguments) for c in result.tool_calls] == [("recall", {"query": "editors"})]

    @pytest.mark.asyncio
    async def test_arguments_sent_as_a_string_are_corrected_once(self):
        """#6302: the arguments arrived as a JSON string, the tool read no query and burned a turn."""
        stringified = {"call": {"name": "recall", "arguments": json.dumps({"query": "editors"})}}
        wrapped = {"call": {"name": "recall", "arguments": {"query": "editors"}}}
        result, call = await _call(LLM_TOOL_CHOICE_REQUIRED, _reply(stringified), _reply(wrapped))
        assert call.await_count == 2
        assert [(c.name, c.arguments) for c in result.tool_calls] == [("recall", {"query": "editors"})]

    @pytest.mark.asyncio
    async def test_a_forced_reply_with_none_of_its_parameters_is_corrected_once(self):
        """A forced ``done`` written in the free-turn shape with its answer beside the name (#6420)."""
        flat = {"call": {"name": "recall", "query": "editors"}}
        result, call = await _call(LLMToolChoice.named("recall"), _reply(flat), _reply({"query": "editors"}))
        assert call.await_count == 2
        correction = call.await_args.kwargs["messages"][-1]["content"]
        assert "none of them, only call" in correction and "Call `recall` now" in correction
        assert result.tool_calls[0].arguments == {"query": "editors"}

    @pytest.mark.asyncio
    async def test_an_argument_left_out_is_left_to_the_tool(self):
        """``required`` holds fields the agent does without (``reason``); only a broken shape is re-asked."""
        result, call = await _call(LLMToolChoice.named("recall"), _reply({"reason": "look"}))
        assert call.await_count == 1
        assert result.tool_calls[0].arguments == {"reason": "look"}

    @pytest.mark.asyncio
    async def test_a_reply_still_wrong_after_the_correction_is_a_turn_without_a_call(self):
        """So the caller reports a model that does not follow the schema, not a missing answer."""
        flat = {"call": {"name": "recall", "query": "editors"}}
        result, call = await _call(LLM_TOOL_CHOICE_REQUIRED, _reply(flat), _reply(flat))
        assert call.await_count == 2
        assert result.tool_calls == [] and result.content == json.dumps(flat)

    @pytest.mark.asyncio
    async def test_a_reply_cut_off_at_the_cap_is_sampled_once_more(self):
        result, call = await _call(
            LLMToolChoice.named("recall"), OutputTooLongError("cap"), _reply({"query": "ranking service owner"})
        )
        assert call.await_count == 2
        assert result.tool_calls[0].arguments == {"query": "ranking service owner"}

    @pytest.mark.asyncio
    async def test_earlier_calls_and_results_are_replayed_as_plain_turns(self):
        history = [
            {"role": "system", "content": "You are reflect."},
            {"role": "user", "content": "What does the user prefer?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "recall", "arguments": '{"query": "tools"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": '{"memories": []}'},
        ]
        _, call = await _call(
            LLM_TOOL_CHOICE_REQUIRED, _reply({"call": {"name": "answer_ready", "arguments": {}}}), messages=history
        )

        sent = call.await_args.kwargs["messages"]
        assert sent[0]["content"].startswith("You are reflect.")
        assert sent[2] == {
            "role": "assistant",
            "content": json.dumps({"call": {"name": "recall", "arguments": {"query": "tools"}}}),
        }
        # The result and this turn's request are one user turn (see the next test).
        assert sent[3]["role"] == "user"
        assert sent[3]["content"].startswith('Result of `recall`:\n{"memories": []}\n\nDecide your next step')
        assert all(set(m) == {"role", "content"} for m in sent)

    @pytest.mark.asyncio
    async def test_no_two_user_turns_in_a_row(self):
        """Gemma's chat template under llama.cpp/vLLM refuses them, so every reflect failed there (#6349).

        The question and the first request, a result and the next request, and the results of
        a reply that made several calls each become one user turn, in order.
        """
        history = [
            {"role": "system", "content": "You are reflect."},
            {"role": "user", "content": "What does the user prefer?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "recall", "arguments": '{"query": "a"}'}},
                    {"id": "c2", "type": "function", "function": {"name": "recall", "arguments": '{"query": "b"}'}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "first"},
            {"role": "tool", "tool_call_id": "c2", "content": "second"},
        ]
        # A reply that names a function: one naming none is asked again (#6469), and these tests
        # script a single reply, since what they check is the messages sent, not the reply.
        for messages in (history[:2], history):
            _, call = await _call(
                LLM_TOOL_CHOICE_REQUIRED, _reply({"call": {"name": "answer_ready", "arguments": {}}}), messages=messages
            )
            roles = [m["role"] for m in call.await_args.kwargs["messages"]]
            assert all(not (a == b == "user") for a, b in zip(roles, roles[1:])), roles

        sent = call.await_args.kwargs["messages"]
        assert sent[-1]["content"].index("first") < sent[-1]["content"].index("second")
        assert sent[-1]["content"].index("second") < sent[-1]["content"].index("Decide your next step")

    @pytest.mark.asyncio
    async def test_a_later_turn_extends_the_earlier_turns_text(self):
        """The prompt cache reads a prefix: a turn joined onto the one before keeps it."""
        first = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
        second = [
            *first,
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "recall", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
        ]
        # A reply that names a function, as above: one naming none is asked again (#6469).
        _, call_1 = await _call(
            LLM_TOOL_CHOICE_REQUIRED, _reply({"call": {"name": "answer_ready", "arguments": {}}}), messages=first
        )
        _, call_2 = await _call(
            LLM_TOOL_CHOICE_REQUIRED, _reply({"call": {"name": "answer_ready", "arguments": {}}}), messages=second
        )
        sent_1, sent_2 = call_1.await_args.kwargs["messages"], call_2.await_args.kwargs["messages"]

        assert sent_2[0] == sent_1[0]
        assert sent_1[1]["content"].startswith(sent_2[1]["content"])


def _type_lists(node: Any) -> Iterator[list[str]]:
    if isinstance(node, dict):
        if isinstance(node.get("type"), list):
            yield node["type"]
        for value in node.values():
            yield from _type_lists(value)
    elif isinstance(node, list):
        for value in node:
            yield from _type_lists(value)


@pytest.mark.parametrize("answer_as_document", [False, True])
def test_no_reflect_tool_sends_a_type_list(answer_as_document: bool) -> None:
    """MiniMax answers ``"type": ["integer", "null"]`` with HTTP 400 (#6321); Anthropic refuses
    ``["array", "null"]`` beside ``items`` (#6305). An optional argument is an ``anyOf`` instead."""
    tools = get_reflect_tools(answer_as_document=answer_as_document, answer_ready=True)
    schemas = [_arguments_schema(t) for t in tools]
    assert [t for s in schemas for t in _type_lists(s)] == []
    recall = next(s for t, s in zip(tools, schemas) if t["function"]["name"] == "recall")
    assert recall["properties"]["max_tokens"]["anyOf"][-1] == {"type": "null"}


class TestTheMarkdownDocument:
    def test_done_takes_markdown_then_the_cited_ids(self):
        done = next(
            t for t in get_reflect_tools(answer_as_document=True, answer_ready=True) if t["function"]["name"] == "done"
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
        prompt = build_system_prompt_for_tools({"name": "B", "mission": ""}, answer_as_document=True)
        assert "Output Format: Markdown Document" in prompt
        assert "'sections' array" not in prompt


class TestTheRefresh:
    @pytest.mark.asyncio
    async def test_a_page_written_as_markdown_is_stored_as_its_sections(self, mock_functions):  # noqa: F811
        provider = _ScriptedProvider(
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


class TestAFlatDone:
    @pytest.mark.asyncio
    async def test_a_done_with_its_answer_beside_the_name_ends_in_that_answer(self, mock_functions):  # noqa: F811
        """#6420: Ollama Cloud does not enforce the schema, and DeepSeek wrote the closing call as
        ``{"call": {"name": "done", "answer": ...}}``. Reflect failed with "no answer" though the
        model wrote one; told what was wrong, the model puts it under ``arguments``."""
        replies: list[str] = []

        def _fill(schema: dict[str, Any]) -> Any:
            """A reply that fits ``schema``, as an enforcing endpoint would hold one to."""
            if "anyOf" in schema:
                return _fill(schema["anyOf"][0])
            kind = schema.get("type")
            if kind == "object":
                return {k: _fill(v) for k, v in (schema.get("properties") or {}).items()}
            if kind == "array":
                enum = (schema.get("items") or {}).get("enum")
                return enum[:1] if enum and schema.get("maxItems", 1) else []
            if kind == "boolean":
                return True
            if kind in ("integer", "number"):
                return 1000
            return schema["enum"][0] if "enum" in schema else "The user prefers ruff."

        async def _model(*, messages, response_format=None, **kwargs):
            usage = TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120)
            if response_format is None:
                return LLMCallResult(content="The user prefers ruff.", usage=usage)
            schema = response_format.model_json_schema()
            searched = any(m["content"].startswith("Result of") for m in messages if m["role"] == "user")
            options = (
                {o["properties"]["name"]["enum"][0]: o for o in schema["properties"]["call"]["anyOf"]}
                if (response_format.__name__ == "next_call")
                else {}
            )
            if response_format.__name__ == "done":
                arguments = _fill(schema)
                corrected = "does not fit the required format" in messages[-1]["content"]
                reply = arguments if corrected else {"call": {"name": "done", **arguments}}  # the flat call
            elif options:
                name = "answer_ready" if searched and "answer_ready" in options else next(iter(options))
                reply = {"call": {"name": name, "arguments": _fill(options[name]["properties"]["arguments"])}}
            else:
                reply = _fill(schema)
            replies.append(json.dumps(reply))
            return LLMCallResult(content=reply, usage=usage)

        llm = LLMProvider(provider="openai", api_key="k", base_url="http://vllm:8000/v1", model="deepseek")
        llm._provider_impl.call = AsyncMock(side_effect=_model)

        result = await run_reflect_agent(
            llm_config=llm,
            bank_id="b",
            query="What tools does the user prefer?",
            bank_profile={"name": "T", "mission": "M"},
            has_mental_models=False,
            budget="high",
            max_iterations=8,
            **mock_functions,
        )

        assert any('"name": "done", "answer"' in r for r in replies), replies  # the flat call was made
        assert result.text == "The user prefers ruff."
