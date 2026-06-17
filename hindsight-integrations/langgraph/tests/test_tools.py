"""Unit tests for Hindsight LangGraph tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hindsight_langgraph import (
    configure,
    create_hindsight_tools,
    memory_instructions,
    reset_config,
)
from hindsight_langgraph.errors import HindsightError


def _mock_client():
    """Create a mock Hindsight client with async methods."""
    client = MagicMock()
    client.aretain = AsyncMock()
    client.arecall = AsyncMock()
    client.areflect = AsyncMock()
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


def _mock_reflect_response(text: str):
    response = MagicMock()
    response.text = text
    return response


def _mock_retain_response():
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

    def test_defaults_to_cloud_without_config(self, monkeypatch):
        """With no client, config, or explicit URL, defaults to the cloud URL."""
        from hindsight_langgraph.config import DEFAULT_HINDSIGHT_API_URL

        monkeypatch.delenv("HINDSIGHT_API_KEY", raising=False)
        with patch("hindsight_langgraph._client.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            tools = create_hindsight_tools(bank_id="test")
            assert len(tools) == 3
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["base_url"] == DEFAULT_HINDSIGHT_API_URL
            # No key configured → none passed; it only fails at call time.
            assert "api_key" not in call_kwargs

    def test_reads_api_key_from_env_without_config(self, monkeypatch):
        """HINDSIGHT_API_KEY is honoured even when configure() was never called."""
        monkeypatch.setenv("HINDSIGHT_API_KEY", "sk-from-env")
        with patch("hindsight_langgraph._client.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            create_hindsight_tools(bank_id="test")
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["api_key"] == "sk-from-env"

    def test_falls_back_to_global_config(self):
        configure(hindsight_api_url="http://localhost:8888")
        with patch("hindsight_langgraph._client.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            tools = create_hindsight_tools(bank_id="test")
            assert len(tools) == 3
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["base_url"] == "http://localhost:8888"
            assert call_kwargs["timeout"] == 30.0
            assert "user_agent" in call_kwargs

    def test_explicit_url_overrides_config(self):
        configure(hindsight_api_url="http://config:8888")
        with patch("hindsight_langgraph._client.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            create_hindsight_tools(bank_id="test", hindsight_api_url="http://explicit:9999")
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["base_url"] == "http://explicit:9999"
            assert call_kwargs["timeout"] == 30.0


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
        result = await tools[0].ainvoke("The user likes Python")
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
        await tools[0].ainvoke("some content")
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
        with pytest.raises(HindsightError, match="Retain failed"):
            await tools[0].ainvoke("content")


class TestRecallTool:
    @pytest.mark.asyncio
    async def test_recall_returns_numbered_results(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["User likes Python", "User is in NYC"])
        tools = create_hindsight_tools(
            bank_id="test-bank",
            client=client,
            include_retain=False,
            include_reflect=False,
        )
        result = await tools[0].ainvoke("user preferences")
        assert "1. User likes Python" in result
        assert "2. User is in NYC" in result

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
        result = await tools[0].ainvoke("anything")
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
        await tools[0].ainvoke("query")
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
        await tools[0].ainvoke("query")
        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["tags"] == ["scope:user"]
        assert call_kwargs["tags_match"] == "all"


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
        result = await tools[0].ainvoke("What do you know about the user?")
        assert result == "The user is a Python developer who prefers functional patterns."

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
        result = await tools[0].ainvoke("anything")
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
        await tools[0].ainvoke("query")
        call_kwargs = client.areflect.call_args[1]
        assert call_kwargs["budget"] == "high"

    @pytest.mark.asyncio
    async def test_reflect_passes_context(self):
        client = _mock_client()
        client.areflect.return_value = _mock_reflect_response("answer")
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            reflect_context="The user is asking about project setup",
            include_retain=False,
            include_recall=False,
        )
        await tools[0].ainvoke("query")
        call_kwargs = client.areflect.call_args[1]
        assert call_kwargs["context"] == "The user is asking about project setup"

    @pytest.mark.asyncio
    async def test_reflect_passes_max_tokens_and_response_schema(self):
        client = _mock_client()
        client.areflect.return_value = _mock_reflect_response("answer")
        schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            reflect_max_tokens=2048,
            reflect_response_schema=schema,
            include_retain=False,
            include_recall=False,
        )
        await tools[0].ainvoke("query")
        call_kwargs = client.areflect.call_args[1]
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["response_schema"] == schema

    @pytest.mark.asyncio
    async def test_reflect_passes_tags(self):
        client = _mock_client()
        client.areflect.return_value = _mock_reflect_response("answer")
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            reflect_tags=["scope:global"],
            reflect_tags_match="all",
            include_retain=False,
            include_recall=False,
        )
        await tools[0].ainvoke("query")
        call_kwargs = client.areflect.call_args[1]
        assert call_kwargs["tags"] == ["scope:global"]
        assert call_kwargs["tags_match"] == "all"


class TestRetainExtendedParams:
    @pytest.mark.asyncio
    async def test_retain_passes_metadata(self):
        client = _mock_client()
        client.aretain.return_value = _mock_retain_response()
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            retain_metadata={"source": "chat", "session": "abc"},
            include_recall=False,
            include_reflect=False,
        )
        await tools[0].ainvoke("content")
        call_kwargs = client.aretain.call_args[1]
        assert call_kwargs["metadata"] == {"source": "chat", "session": "abc"}

    @pytest.mark.asyncio
    async def test_retain_passes_document_id(self):
        client = _mock_client()
        client.aretain.return_value = _mock_retain_response()
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            retain_document_id="session-123",
            include_recall=False,
            include_reflect=False,
        )
        await tools[0].ainvoke("content")
        call_kwargs = client.aretain.call_args[1]
        assert call_kwargs["document_id"] == "session-123"


class TestRecallExtendedParams:
    @pytest.mark.asyncio
    async def test_recall_passes_types(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            recall_types=["world", "experience"],
            include_retain=False,
            include_reflect=False,
        )
        await tools[0].ainvoke("query")
        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["types"] == ["world", "experience"]

    @pytest.mark.asyncio
    async def test_recall_passes_include_entities(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            recall_include_entities=True,
            include_retain=False,
            include_reflect=False,
        )
        await tools[0].ainvoke("query")
        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["include_entities"] is True


class TestMemoryInstructions:
    @pytest.mark.asyncio
    async def test_returns_base_instructions_when_no_memories(self):
        client = _mock_client()
        response = MagicMock()
        response.results = []
        client.arecall.return_value = response

        fn = memory_instructions(
            bank_id="test",
            client=client,
            base_instructions="You are helpful.",
        )
        result = await fn()
        assert result == "You are helpful."

    @pytest.mark.asyncio
    async def test_appends_memories_to_base_instructions(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["likes Python", "uses VS Code"])

        fn = memory_instructions(
            bank_id="test",
            client=client,
            base_instructions="You are helpful.",
        )
        result = await fn()
        assert result.startswith("You are helpful.")
        assert "likes Python" in result
        assert "uses VS Code" in result

    @pytest.mark.asyncio
    async def test_passes_recall_params(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])

        fn = memory_instructions(
            bank_id="test",
            client=client,
            budget="high",
            max_tokens=2048,
            tags=["scope:user"],
            tags_match="all",
        )
        await fn()

        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["bank_id"] == "test"
        assert call_kwargs["budget"] == "high"
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["tags"] == ["scope:user"]
        assert call_kwargs["tags_match"] == "all"

    @pytest.mark.asyncio
    async def test_respects_max_results(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["a", "b", "c", "d"])

        fn = memory_instructions(
            bank_id="test",
            client=client,
            max_results=2,
        )
        result = await fn()
        # Should only include 2 memories
        assert "1." in result
        assert "2." in result
        assert "3." not in result

    @pytest.mark.asyncio
    async def test_custom_prefix(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])

        fn = memory_instructions(
            bank_id="test",
            client=client,
            prefix="\n\nContext:\n",
        )
        result = await fn()
        assert "\n\nContext:\n" in result

    @pytest.mark.asyncio
    async def test_falls_back_on_error(self):
        client = _mock_client()
        client.arecall.side_effect = RuntimeError("connection refused")

        fn = memory_instructions(bank_id="test", client=client, base_instructions="You are helpful.")
        result = await fn()
        assert result == "You are helpful."
