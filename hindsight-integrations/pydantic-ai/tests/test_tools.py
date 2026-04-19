"""Unit tests for Hindsight Pydantic AI tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_pydantic_ai import (
    configure,
    create_hindsight_tools,
    memory_instructions,
    reset_config,
)
from hindsight_pydantic_ai.errors import HindsightError


def _mock_client():
    """Create a mock Hindsight client with async methods."""
    client = MagicMock()
    client.aretain = AsyncMock()
    client.arecall = AsyncMock()
    client.areflect = AsyncMock()
    return client


def _mock_recall_result(text: str):
    """Create a mock RecallResult."""
    result = MagicMock()
    result.text = text
    return result


def _mock_recall_response(texts: list[str]):
    """Create a mock RecallResponse with results."""
    response = MagicMock()
    response.results = [_mock_recall_result(t) for t in texts]
    return response


def _mock_reflect_response(text: str):
    """Create a mock ReflectResponse."""
    response = MagicMock()
    response.text = text
    return response


def _mock_retain_response():
    """Create a mock RetainResponse."""
    response = MagicMock()
    response.success = True
    return response


class TestCreateHindsightTools:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_returns_three_tools_by_default(self):
        client = _mock_client()
        tools = create_hindsight_tools(bank_id="test", client=client)
        assert len(tools) == 3

    def test_include_retain_only(self):
        client = _mock_client()
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            include_retain=True,
            include_recall=False,
            include_reflect=False,
        )
        assert len(tools) == 1
        assert tools[0].name == "hindsight_retain"

    def test_include_recall_only(self):
        client = _mock_client()
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            include_retain=False,
            include_recall=True,
            include_reflect=False,
        )
        assert len(tools) == 1
        assert tools[0].name == "hindsight_recall"

    def test_include_reflect_only(self):
        client = _mock_client()
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            include_retain=False,
            include_recall=False,
            include_reflect=True,
        )
        assert len(tools) == 1
        assert tools[0].name == "hindsight_reflect"

    def test_no_tools_when_all_excluded(self):
        client = _mock_client()
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            include_retain=False,
            include_recall=False,
            include_reflect=False,
        )
        assert len(tools) == 0

    def test_raises_without_client_or_config(self):
        with pytest.raises(HindsightError, match="No Hindsight API URL"):
            create_hindsight_tools(bank_id="test")

    def test_falls_back_to_global_config(self):
        configure(hindsight_api_url="http://localhost:8888")
        with patch("hindsight_pydantic_ai.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            tools = create_hindsight_tools(bank_id="test")
            assert len(tools) == 3
            mock_cls.assert_called_once()
            assert mock_cls.call_args.kwargs["base_url"] == "http://localhost:8888"
            assert mock_cls.call_args.kwargs["timeout"] == 30.0

    def test_explicit_url_overrides_config(self):
        configure(hindsight_api_url="http://config:8888")
        with patch("hindsight_pydantic_ai.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            create_hindsight_tools(
                bank_id="test", hindsight_api_url="http://explicit:9999"
            )
            mock_cls.assert_called_once()
            assert mock_cls.call_args.kwargs["base_url"] == "http://explicit:9999"
            assert mock_cls.call_args.kwargs["timeout"] == 30.0


class TestRetainTool:
    @pytest.mark.asyncio
    async def test_retain_stores_memory(self):
        client = _mock_client()
        client.aretain.return_value = _mock_retain_response()
        tools = create_hindsight_tools(
            bank_id="test-bank",
            client=client,
            include_recall=False,
            include_reflect=False,
        )
        tool_fn = tools[0].function

        result = await tool_fn("The user likes Python")

        assert result == "Memory stored successfully."
        client.aretain.assert_called_once_with(bank_id="test-bank", content="The user likes Python")

    @pytest.mark.asyncio
    async def test_retain_passes_tags(self):
        client = _mock_client()
        client.aretain.return_value = _mock_retain_response()
        tools = create_hindsight_tools(
            bank_id="test-bank",
            client=client,
            tags=["source:chat"],
            include_recall=False,
            include_reflect=False,
        )
        tool_fn = tools[0].function

        await tool_fn("some content")

        call_kwargs = client.aretain.call_args[1]
        assert call_kwargs["tags"] == ["source:chat"]

    @pytest.mark.asyncio
    async def test_retain_raises_hindsight_error(self):
        client = _mock_client()
        client.aretain.side_effect = RuntimeError("connection refused")
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            include_recall=False,
            include_reflect=False,
        )
        tool_fn = tools[0].function

        with pytest.raises(HindsightError, match="Retain failed"):
            await tool_fn("content")


class TestRecallTool:
    @pytest.mark.asyncio
    async def test_recall_returns_numbered_results(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(
            ["User likes Python", "User is in NYC"]
        )
        tools = create_hindsight_tools(
            bank_id="test-bank",
            client=client,
            include_retain=False,
            include_reflect=False,
        )
        tool_fn = tools[0].function

        result = await tool_fn("user preferences")

        assert "1. User likes Python" in result
        assert "2. User is in NYC" in result
        client.arecall.assert_called_once()

    @pytest.mark.asyncio
    async def test_recall_empty_results(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response([])
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            include_retain=False,
            include_reflect=False,
        )
        tool_fn = tools[0].function

        result = await tool_fn("anything")

        assert result == "No relevant memories found."

    @pytest.mark.asyncio
    async def test_recall_passes_budget_and_max_tokens(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            budget="high",
            max_tokens=2048,
            include_retain=False,
            include_reflect=False,
        )
        tool_fn = tools[0].function

        await tool_fn("query")

        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["budget"] == "high"
        assert call_kwargs["max_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_recall_passes_tags(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            recall_tags=["scope:user"],
            recall_tags_match="all",
            include_retain=False,
            include_reflect=False,
        )
        tool_fn = tools[0].function

        await tool_fn("query")

        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["tags"] == ["scope:user"]
        assert call_kwargs["tags_match"] == "all"

    @pytest.mark.asyncio
    async def test_recall_raises_hindsight_error(self):
        client = _mock_client()
        client.arecall.side_effect = RuntimeError("timeout")
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            include_retain=False,
            include_reflect=False,
        )
        tool_fn = tools[0].function

        with pytest.raises(HindsightError, match="Recall failed"):
            await tool_fn("query")


class TestReflectTool:
    @pytest.mark.asyncio
    async def test_reflect_returns_text(self):
        client = _mock_client()
        client.areflect.return_value = _mock_reflect_response(
            "The user is a Python developer who prefers functional patterns."
        )
        tools = create_hindsight_tools(
            bank_id="test-bank",
            client=client,
            include_retain=False,
            include_recall=False,
        )
        tool_fn = tools[0].function

        result = await tool_fn("What do you know about the user?")

        assert result == "The user is a Python developer who prefers functional patterns."
        client.areflect.assert_called_once()

    @pytest.mark.asyncio
    async def test_reflect_empty_returns_fallback(self):
        client = _mock_client()
        client.areflect.return_value = _mock_reflect_response("")
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            include_retain=False,
            include_recall=False,
        )
        tool_fn = tools[0].function

        result = await tool_fn("anything")

        assert result == "No relevant memories found."

    @pytest.mark.asyncio
    async def test_reflect_passes_budget(self):
        client = _mock_client()
        client.areflect.return_value = _mock_reflect_response("answer")
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            budget="high",
            include_retain=False,
            include_recall=False,
        )
        tool_fn = tools[0].function

        await tool_fn("query")

        call_kwargs = client.areflect.call_args[1]
        assert call_kwargs["budget"] == "high"

    @pytest.mark.asyncio
    async def test_reflect_raises_hindsight_error(self):
        client = _mock_client()
        client.areflect.side_effect = RuntimeError("timeout")
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            include_retain=False,
            include_recall=False,
        )
        tool_fn = tools[0].function

        with pytest.raises(HindsightError, match="Reflect failed"):
            await tool_fn("query")


class TestMemoryInstructions:
    @pytest.mark.asyncio
    async def test_returns_formatted_memories(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(
            ["Likes Python", "Lives in NYC", "Prefers dark mode"]
        )
        instructions_fn = memory_instructions(
            bank_id="test-bank", client=client
        )

        # Instructions functions receive RunContext — mock it
        mock_ctx = MagicMock()
        result = await instructions_fn(mock_ctx)

        assert "Relevant memories:" in result
        assert "1. Likes Python" in result
        assert "2. Lives in NYC" in result
        assert "3. Prefers dark mode" in result

    @pytest.mark.asyncio
    async def test_respects_max_results(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(
            ["fact1", "fact2", "fact3", "fact4", "fact5"]
        )
        instructions_fn = memory_instructions(
            bank_id="test", client=client, max_results=2
        )

        mock_ctx = MagicMock()
        result = await instructions_fn(mock_ctx)

        assert "1. fact1" in result
        assert "2. fact2" in result
        assert "3." not in result

    @pytest.mark.asyncio
    async def test_custom_prefix(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])
        instructions_fn = memory_instructions(
            bank_id="test", client=client, prefix="Memory context:\n"
        )

        mock_ctx = MagicMock()
        result = await instructions_fn(mock_ctx)

        assert result.startswith("Memory context:")

    @pytest.mark.asyncio
    async def test_empty_results_returns_empty_string(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response([])
        instructions_fn = memory_instructions(
            bank_id="test", client=client
        )

        mock_ctx = MagicMock()
        result = await instructions_fn(mock_ctx)

        assert result == ""

    @pytest.mark.asyncio
    async def test_error_returns_empty_string(self):
        client = _mock_client()
        client.arecall.side_effect = RuntimeError("connection error")
        instructions_fn = memory_instructions(
            bank_id="test", client=client
        )

        mock_ctx = MagicMock()
        result = await instructions_fn(mock_ctx)

        assert result == ""

    @pytest.mark.asyncio
    async def test_passes_query_and_budget(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])
        instructions_fn = memory_instructions(
            bank_id="test",
            client=client,
            query="user preferences and context",
            budget="high",
        )

        mock_ctx = MagicMock()
        await instructions_fn(mock_ctx)

        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["query"] == "user preferences and context"
        assert call_kwargs["budget"] == "high"

    @pytest.mark.asyncio
    async def test_passes_tags(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])
        instructions_fn = memory_instructions(
            bank_id="test",
            client=client,
            tags=["scope:user"],
            tags_match="all",
        )

        mock_ctx = MagicMock()
        await instructions_fn(mock_ctx)

        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["tags"] == ["scope:user"]
        assert call_kwargs["tags_match"] == "all"

    def test_raises_without_client_or_config(self):
        reset_config()
        with pytest.raises(HindsightError, match="No Hindsight API URL"):
            memory_instructions(bank_id="test")
