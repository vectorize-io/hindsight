"""Reflect's serialized history must satisfy Anthropic's tool_result protocol.

Anthropic (and strict gateways in front of it) require every ``tool_use`` block
in an assistant turn to be answered by ``tool_result`` blocks in the SINGLE
immediately-following user message, in the same order. Two layers have to agree
for that to hold: the reflect agent emits one ``role:tool`` message per result
in the model's tool_call order, and ``AnthropicLLM`` groups that consecutive run
into one user message. Unit tests cover each layer; this pins the invariant they
exist to uphold, so a regression in either one fails here.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.reflect.agent import run_reflect_agent

pytestmark = pytest.mark.asyncio


def _assert_tool_protocol_valid(messages: list[dict]) -> None:
    for index, message in enumerate(messages):
        blocks = message["content"]
        if message["role"] != "assistant" or not isinstance(blocks, list):
            continue
        tool_use_ids = [b["id"] for b in blocks if b.get("type") == "tool_use"]
        if not tool_use_ids:
            continue
        assert index + 1 < len(messages), f"assistant tool_use turn at {index} has no following message"
        following = messages[index + 1]
        assert following["role"] == "user", f"message after tool_use turn {index} is {following['role']}"
        result_ids = [b["tool_use_id"] for b in following["content"] if b.get("type") == "tool_result"]
        assert result_ids == tool_use_ids, f"tool_result ids {result_ids} != tool_use ids {tool_use_ids}"


def _tool_use_block(block_id: str, name: str, **arguments):
    block = MagicMock()
    block.type = "tool_use"
    block.id = block_id
    block.name = name
    block.input = arguments
    return block


def _response(*blocks):
    response = MagicMock()
    response.content = list(blocks)
    response.usage = MagicMock(input_tokens=10, output_tokens=2, cache_read_input_tokens=0)
    response.stop_reason = "tool_use"
    return response


async def test_parallel_and_hallucinated_tool_batch_serializes_validly():
    """A parallel batch with a hallucinated call in the middle stays protocol-valid."""
    from hindsight_api.engine.providers.anthropic_llm import AnthropicLLM

    with patch("anthropic.AsyncAnthropic") as client_cls:
        client_cls.return_value = MagicMock()
        provider = AnthropicLLM(provider="anthropic", api_key="fake-key", base_url="", model="claude-sonnet-5")
    provider._client = MagicMock()

    create = AsyncMock(
        side_effect=[
            _response(
                _tool_use_block("t1", "search_observations", query="q"),
                _tool_use_block("t2", "web_search", query="q"),
                _tool_use_block("t3", "recall", query="q"),
            ),
            _response(_tool_use_block("t4", "recall", query="q")),
            _response(_tool_use_block("t5", "done", answer="A", memory_ids=["mem-1"])),
        ]
    )
    sent_histories: list[list[dict]] = []

    async def _capturing_create(**kwargs):
        import copy

        sent_histories.append(copy.deepcopy(kwargs["messages"]))
        return await create(**kwargs)

    provider._client.messages.create = _capturing_create

    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        result = await run_reflect_agent(
            llm_config=provider,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            search_mental_models_fn=AsyncMock(return_value={"mental_models": []}),
            search_observations_fn=AsyncMock(return_value={"observations": []}),
            recall_fn=AsyncMock(return_value={"memories": [{"id": "mem-1", "content": "test memory"}]}),
            expand_fn=AsyncMock(return_value={"memories": []}),
        )

    assert result.text == "A"
    # The later turns are the ones carrying tool history; all must be valid.
    assert len(sent_histories) >= 2
    for history in sent_histories:
        _assert_tool_protocol_valid(history)
