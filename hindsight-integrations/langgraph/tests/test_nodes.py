"""Unit tests for Hindsight LangGraph nodes."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from hindsight_langgraph import create_recall_node, create_retain_node
from hindsight_langgraph.errors import HindsightError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def _mock_client():
    client = MagicMock()
    client.aretain = AsyncMock()
    client.arecall = AsyncMock()
    return client


def _mock_recall_response(texts: list[str]):
    response = MagicMock()
    results = []
    for t in texts:
        r = MagicMock()
        r.text = t
        results.append(r)
    response.results = results
    return response


class TestRecallNode:
    @pytest.mark.asyncio
    async def test_injects_memories_as_system_message(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(
            ["User likes Python", "User is in NYC"]
        )
        node = create_recall_node(bank_id="test-bank", client=client)

        state = {"messages": [HumanMessage(content="What do you remember about me?")]}
        result = await node(state)

        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert isinstance(msg, SystemMessage)
        assert "User likes Python" in msg.content
        assert "User is in NYC" in msg.content

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_human_message(self):
        client = _mock_client()
        node = create_recall_node(bank_id="test-bank", client=client)

        state = {"messages": [SystemMessage(content="You are a helpful assistant")]}
        result = await node(state)

        assert result["messages"] == []
        client.arecall.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_results(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response([])
        node = create_recall_node(bank_id="test-bank", client=client)

        state = {"messages": [HumanMessage(content="hello")]}
        result = await node(state)

        assert result["messages"] == []

    @pytest.mark.asyncio
    async def test_respects_max_results(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(
            ["fact1", "fact2", "fact3", "fact4", "fact5"]
        )
        node = create_recall_node(bank_id="test-bank", client=client, max_results=2)

        state = {"messages": [HumanMessage(content="query")]}
        result = await node(state)

        msg = result["messages"][0]
        assert "1. fact1" in msg.content
        assert "2. fact2" in msg.content
        assert "3." not in msg.content

    @pytest.mark.asyncio
    async def test_resolves_bank_id_from_config(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])
        node = create_recall_node(client=client, bank_id_from_config="user_id")

        state = {"messages": [HumanMessage(content="hello")]}
        config = {"configurable": {"user_id": "user-456"}}
        await node(state, config=config)

        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["bank_id"] == "user-456"

    @pytest.mark.asyncio
    async def test_skips_when_no_bank_id(self):
        client = _mock_client()
        node = create_recall_node(client=client)

        state = {"messages": [HumanMessage(content="hello")]}
        result = await node(state)

        assert result["messages"] == []
        client.arecall.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_on_recall_error(self):
        client = _mock_client()
        client.arecall.side_effect = RuntimeError("connection refused")
        node = create_recall_node(bank_id="test-bank", client=client)

        state = {"messages": [HumanMessage(content="hello")]}
        with pytest.raises(HindsightError, match="Recall node failed"):
            await node(state)

    @pytest.mark.asyncio
    async def test_passes_tags(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])
        node = create_recall_node(
            bank_id="test-bank",
            client=client,
            tags=["scope:user"],
            tags_match="all",
        )

        state = {"messages": [HumanMessage(content="hello")]}
        await node(state)

        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["tags"] == ["scope:user"]
        assert call_kwargs["tags_match"] == "all"


class TestRecallNodeOutputKey:
    @pytest.mark.asyncio
    async def test_output_key_returns_memory_text(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(
            ["User likes Python", "User is in NYC"]
        )
        node = create_recall_node(
            bank_id="test-bank", client=client, output_key="memory_context"
        )

        state = {"messages": [HumanMessage(content="What do you remember?")]}
        result = await node(state)

        assert "messages" not in result
        assert "memory_context" in result
        assert "User likes Python" in result["memory_context"]
        assert "User is in NYC" in result["memory_context"]

    @pytest.mark.asyncio
    async def test_output_key_returns_none_when_no_results(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response([])
        node = create_recall_node(
            bank_id="test-bank", client=client, output_key="memory_context"
        )

        state = {"messages": [HumanMessage(content="hello")]}
        result = await node(state)

        assert result == {"memory_context": None}

    @pytest.mark.asyncio
    async def test_output_key_raises_on_error(self):
        client = _mock_client()
        client.arecall.side_effect = RuntimeError("connection refused")
        node = create_recall_node(
            bank_id="test-bank", client=client, output_key="memory_context"
        )

        state = {"messages": [HumanMessage(content="hello")]}
        with pytest.raises(HindsightError, match="Recall node failed"):
            await node(state)


class TestRetainNode:
    @pytest.mark.asyncio
    async def test_retains_human_messages(self):
        client = _mock_client()
        node = create_retain_node(bank_id="test-bank", client=client)

        state = {
            "messages": [
                HumanMessage(content="I like pizza"),
                AIMessage(content="Got it!"),
            ]
        }
        await node(state)

        client.aretain.assert_called_once()
        call_kwargs = client.aretain.call_args[1]
        assert call_kwargs["bank_id"] == "test-bank"
        assert "I like pizza" in call_kwargs["content"]
        assert "Got it!" not in call_kwargs["content"]

    @pytest.mark.asyncio
    async def test_retains_both_when_configured(self):
        client = _mock_client()
        node = create_retain_node(
            bank_id="test-bank", client=client, retain_human=True, retain_ai=True
        )

        state = {
            "messages": [
                HumanMessage(content="I like pizza"),
                AIMessage(content="Got it!"),
            ]
        }
        await node(state)

        call_kwargs = client.aretain.call_args[1]
        assert "I like pizza" in call_kwargs["content"]
        assert "Got it!" in call_kwargs["content"]

    @pytest.mark.asyncio
    async def test_skips_when_no_messages_match(self):
        client = _mock_client()
        node = create_retain_node(
            bank_id="test-bank", client=client, retain_human=False, retain_ai=False
        )

        state = {"messages": [HumanMessage(content="hello")]}
        await node(state)

        client.aretain.assert_not_called()

    @pytest.mark.asyncio
    async def test_passes_tags(self):
        client = _mock_client()
        node = create_retain_node(
            bank_id="test-bank", client=client, tags=["source:chat"]
        )

        state = {"messages": [HumanMessage(content="hello")]}
        await node(state)

        call_kwargs = client.aretain.call_args[1]
        assert call_kwargs["tags"] == ["source:chat"]

    @pytest.mark.asyncio
    async def test_resolves_bank_id_from_config(self):
        client = _mock_client()
        node = create_retain_node(client=client, bank_id_from_config="user_id")

        state = {"messages": [HumanMessage(content="hello")]}
        config = {"configurable": {"user_id": "user-789"}}
        await node(state, config=config)

        call_kwargs = client.aretain.call_args[1]
        assert call_kwargs["bank_id"] == "user-789"

    @pytest.mark.asyncio
    async def test_raises_on_retain_error(self):
        client = _mock_client()
        client.aretain.side_effect = RuntimeError("connection refused")
        node = create_retain_node(bank_id="test-bank", client=client)

        state = {"messages": [HumanMessage(content="hello")]}
        with pytest.raises(HindsightError, match="Retain node failed"):
            await node(state)

    @pytest.mark.asyncio
    async def test_passes_metadata_and_document_id(self):
        client = _mock_client()
        node = create_retain_node(
            bank_id="test-bank",
            client=client,
            metadata={"source": "chat"},
            document_id="session-1",
        )

        state = {"messages": [HumanMessage(content="hello")]}
        await node(state)

        call_kwargs = client.aretain.call_args[1]
        assert call_kwargs["metadata"] == {"source": "chat"}
        assert call_kwargs["document_id"] == "session-1"


class TestRecallNodeNewParams:
    @pytest.mark.asyncio
    async def test_passes_recall_types(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])
        node = create_recall_node(
            bank_id="test-bank",
            client=client,
            recall_types=["world", "experience"],
        )

        state = {"messages": [HumanMessage(content="hello")]}
        await node(state)

        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["types"] == ["world", "experience"]

    @pytest.mark.asyncio
    async def test_passes_recall_include_entities(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])
        node = create_recall_node(
            bank_id="test-bank",
            client=client,
            recall_include_entities=True,
        )

        state = {"messages": [HumanMessage(content="hello")]}
        await node(state)

        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["include_entities"] is True

    @pytest.mark.asyncio
    async def test_recall_types_not_passed_when_none(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])
        node = create_recall_node(bank_id="test-bank", client=client)

        state = {"messages": [HumanMessage(content="hello")]}
        await node(state)

        call_kwargs = client.arecall.call_args[1]
        assert "types" not in call_kwargs
        assert "include_entities" not in call_kwargs
