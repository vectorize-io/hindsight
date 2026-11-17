"""The pictures behind knowledge-bank hits reach the reflect model as images, and nothing else.

``tool_recall`` returns them under ``KB_IMAGES_KEY``; the agent takes them out before the trace, the
presenter or the pruning read the output, and sends them as one user turn after the results.
"""

from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.reflect.agent import run_reflect_agent
from hindsight_api.engine.reflect.tools import KB_IMAGES_KEY
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult
from tests.test_reflect_agent import _ScriptedProvider, mock_functions  # noqa: F401  (fixture)

_IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


class _RecordingProvider(_ScriptedProvider):
    def __init__(self, scripted):
        super().__init__(scripted)
        self.messages: list[list[dict]] = []

    async def call_with_tools(self, *, messages, **kwargs):
        self.messages.append(list(messages))
        return await super().call_with_tools(messages=messages, **kwargs)


def _recall_turn() -> LLMToolCallResult:
    return LLMToolCallResult(
        tool_calls=[LLMToolCall(id="0", name="recall", arguments={"query": "q"})], finish_reason="tool_calls"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("with_pictures", [True, False])
async def test_knowledge_pictures_follow_the_results_as_images(mock_functions, with_pictures):  # noqa: F811
    output = {
        "memories": [{"id": "mem-1", "text": "a fact"}],
        "documents": [{"source": "keys.png", "text": "[image passage: no text]", "picture": "Picture of keys.png"}],
    }
    if with_pictures:
        output[KB_IMAGES_KEY] = [{"type": "text", "text": "Picture of keys.png"}, _IMAGE]
    provider = _RecordingProvider(
        [
            _recall_turn(),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="1", name="done", arguments={"answer": "A", "memory_ids": []})],
                finish_reason="tool_calls",
            ),
        ]
    )
    result = await run_reflect_agent(
        llm_config=provider,
        bank_id="b",
        query="q",
        bank_profile={"name": "T", "mission": "M"},
        has_mental_models=False,
        include_observations=False,
        include_recall=True,
        budget="high",
        max_iterations=4,
        **{**mock_functions, "recall_fn": AsyncMock(return_value=output)},
    )

    after_recall = provider.messages[-1]
    picture_turns = [m for m in after_recall if m["role"] == "user" and isinstance(m["content"], list)]
    if with_pictures:
        assert len(picture_turns) == 1
        parts = picture_turns[0]["content"]
        assert parts[1:] == [{"type": "text", "text": "Picture of keys.png"}, _IMAGE]
        # The turn comes right after the tool result it describes.
        assert after_recall[after_recall.index(picture_turns[0]) - 1]["role"] == "tool"
    else:
        assert picture_turns == []
    # Neither the trace nor the text the model reads carries the base64.
    recall_trace = next(t for t in result.tool_trace if t.tool == "recall")
    assert KB_IMAGES_KEY not in recall_trace.output
    tool_text = "".join(m["content"] for m in after_recall if m["role"] == "tool")
    assert "base64" not in tool_text and "Picture of keys.png" in tool_text


@pytest.mark.asyncio
async def test_the_standalone_synthesis_gets_the_pictures_back(mock_functions):  # noqa: F811
    """It rebuilds its prompt from the text of the results, which holds no image (a real model then
    answered "the data does not contain the code" that only the picture showed)."""
    from hindsight_api.engine.response_models import LLMCallResult, TokenUsage

    output = {"memories": [], "documents": [{"source": "door.png", "picture": "Picture of door.png"}]}
    output[KB_IMAGES_KEY] = [{"type": "text", "text": "Picture of door.png"}, _IMAGE]
    provider = _RecordingProvider(
        [
            _recall_turn(),
            LLMToolCallResult(content="prose", finish_reason="stop"),
            LLMToolCallResult(content="prose again", finish_reason="stop"),
        ]
    )
    provider.call = AsyncMock(
        return_value=LLMCallResult(content="4817", usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2))
    )
    result = await run_reflect_agent(
        llm_config=provider,
        bank_id="b",
        query="q",
        bank_profile={"name": "T", "mission": "M"},
        has_mental_models=False,
        include_observations=False,
        include_recall=True,
        budget="high",
        max_iterations=4,
        **{**mock_functions, "recall_fn": AsyncMock(return_value=output)},
    )

    assert result.text == "4817"
    user_turn = provider.call.await_args.kwargs["messages"][-1]
    assert user_turn["role"] == "user" and _IMAGE in user_turn["content"]
