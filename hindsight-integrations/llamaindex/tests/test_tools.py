"""Unit tests for Hindsight LlamaIndex tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hindsight_llamaindex import (
    HindsightToolSpec,
    configure,
    create_hindsight_tools,
    reset_config,
)
from hindsight_llamaindex.errors import HindsightError


def _mock_client():
    """Create a mock Hindsight client with sync and async methods."""
    client = MagicMock()
    client.retain = MagicMock()
    client.recall = MagicMock()
    client.reflect = MagicMock()
    client.create_bank = MagicMock()
    client.aretain = AsyncMock()
    client.arecall = AsyncMock()
    client.areflect = AsyncMock()
    client.acreate_bank = AsyncMock()
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


class TestHindsightToolSpec:
    def test_spec_functions_list(self):
        assert HindsightToolSpec.spec_functions == [
            ("retain_memory", "aretain_memory"),
            ("recall_memory", "arecall_memory"),
            ("reflect_on_memory", "areflect_on_memory"),
        ]

    def test_to_tool_list_returns_three_tools(self):
        client = _mock_client()
        spec = HindsightToolSpec(bank_id="test", client=client)
        tools = spec.to_tool_list()
        assert len(tools) == 3

    def test_to_tool_list_selective(self):
        client = _mock_client()
        spec = HindsightToolSpec(bank_id="test", client=client)
        tools = spec.to_tool_list(spec_functions=[("recall_memory", "arecall_memory")])
        assert len(tools) == 1
        assert tools[0].metadata.name == "recall_memory"


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
        assert tools[0].metadata.name == "retain_memory"

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
        assert tools[0].metadata.name == "recall_memory"

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
        assert tools[0].metadata.name == "reflect_on_memory"

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
        with patch("hindsight_llamaindex._client.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            tools = create_hindsight_tools(bank_id="test")
            assert len(tools) == 3
            mock_cls.assert_called_once()
            assert mock_cls.call_args.kwargs["base_url"] == "http://localhost:8888"
            assert mock_cls.call_args.kwargs["timeout"] == 30.0

    def test_explicit_url_overrides_config(self):
        configure(hindsight_api_url="http://config:8888")
        with patch("hindsight_llamaindex._client.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            create_hindsight_tools(
                bank_id="test", hindsight_api_url="http://explicit:9999"
            )
            mock_cls.assert_called_once()
            assert mock_cls.call_args.kwargs["base_url"] == "http://explicit:9999"
            assert mock_cls.call_args.kwargs["timeout"] == 30.0


class TestRetainTool:
    def test_retain_stores_memory(self):
        client = _mock_client()
        client.retain.return_value = _mock_retain_response()
        spec = HindsightToolSpec(bank_id="test-bank", client=client)
        result = spec.retain_memory("The user likes Python")
        assert result == "Memory stored successfully."
        call_kwargs = client.retain.call_args[1]
        assert call_kwargs["bank_id"] == "test-bank"
        assert call_kwargs["content"] == "The user likes Python"
        assert call_kwargs["context"] == "llamaindex"

    def test_retain_passes_tags(self):
        client = _mock_client()
        client.retain.return_value = _mock_retain_response()
        spec = HindsightToolSpec(
            bank_id="test-bank", client=client, tags=["source:chat"]
        )
        spec.retain_memory("some content")
        call_kwargs = client.retain.call_args[1]
        assert call_kwargs["tags"] == ["source:chat"]

    def test_retain_passes_metadata(self):
        client = _mock_client()
        client.retain.return_value = _mock_retain_response()
        spec = HindsightToolSpec(
            bank_id="test",
            client=client,
            retain_metadata={"source": "chat", "session": "abc"},
        )
        spec.retain_memory("content")
        call_kwargs = client.retain.call_args[1]
        assert call_kwargs["metadata"] == {"source": "chat", "session": "abc"}

    def test_retain_passes_explicit_document_id(self):
        client = _mock_client()
        client.retain.return_value = _mock_retain_response()
        spec = HindsightToolSpec(
            bank_id="test", client=client, retain_document_id="session-123"
        )
        spec.retain_memory("content")
        call_kwargs = client.retain.call_args[1]
        assert call_kwargs["document_id"] == "session-123"

    def test_retain_auto_generates_document_id(self):
        client = _mock_client()
        client.retain.return_value = _mock_retain_response()
        spec = HindsightToolSpec(bank_id="test", client=client)
        spec.retain_memory("content")
        call_kwargs = client.retain.call_args[1]
        doc_id = call_kwargs["document_id"]
        # Auto-generated format: {session_id}-{uuid_hex_12}
        assert "-" in doc_id
        suffix = doc_id.rsplit("-", 1)[1]
        assert len(suffix) == 12

    def test_retain_passes_context_label(self):
        client = _mock_client()
        client.retain.return_value = _mock_retain_response()
        spec = HindsightToolSpec(bank_id="test", client=client, retain_context="my-app")
        spec.retain_memory("content")
        call_kwargs = client.retain.call_args[1]
        assert call_kwargs["context"] == "my-app"

    def test_retain_defaults_to_llamaindex_context(self):
        client = _mock_client()
        client.retain.return_value = _mock_retain_response()
        spec = HindsightToolSpec(bank_id="test", client=client)
        spec.retain_memory("content")
        call_kwargs = client.retain.call_args[1]
        assert call_kwargs["context"] == "llamaindex"

    def test_retain_returns_error_message_on_failure(self):
        """Errors are returned gracefully, not raised."""
        client = _mock_client()
        client.retain.side_effect = RuntimeError("connection refused")
        spec = HindsightToolSpec(bank_id="test", client=client)
        result = spec.retain_memory("content")
        assert "Failed to store memory" in result
        assert "connection refused" in result


class TestRecallTool:
    def test_recall_returns_numbered_results(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(
            ["User likes Python", "User is in NYC"]
        )
        spec = HindsightToolSpec(bank_id="test-bank", client=client)
        result = spec.recall_memory("user preferences")
        assert "1. User likes Python" in result
        assert "2. User is in NYC" in result

    def test_recall_empty_results(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response([])
        spec = HindsightToolSpec(bank_id="test", client=client)
        result = spec.recall_memory("anything")
        assert result == "No relevant memories found."

    def test_recall_passes_budget_and_max_tokens(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        spec = HindsightToolSpec(
            bank_id="test", client=client, budget="high", max_tokens=2048
        )
        spec.recall_memory("query")
        call_kwargs = client.recall.call_args[1]
        assert call_kwargs["budget"] == "high"
        assert call_kwargs["max_tokens"] == 2048

    def test_recall_passes_tags(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        spec = HindsightToolSpec(
            bank_id="test",
            client=client,
            recall_tags=["scope:user"],
            recall_tags_match="all",
        )
        spec.recall_memory("query")
        call_kwargs = client.recall.call_args[1]
        assert call_kwargs["tags"] == ["scope:user"]
        assert call_kwargs["tags_match"] == "all"

    def test_recall_passes_types(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        spec = HindsightToolSpec(
            bank_id="test",
            client=client,
            recall_types=["world", "experience"],
        )
        spec.recall_memory("query")
        call_kwargs = client.recall.call_args[1]
        assert call_kwargs["types"] == ["world", "experience"]

    def test_recall_passes_include_entities(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        spec = HindsightToolSpec(
            bank_id="test", client=client, recall_include_entities=True
        )
        spec.recall_memory("query")
        call_kwargs = client.recall.call_args[1]
        assert call_kwargs["include_entities"] is True

    def test_recall_returns_error_message_on_failure(self):
        """Errors are returned gracefully, not raised."""
        client = _mock_client()
        client.recall.side_effect = RuntimeError("timeout")
        spec = HindsightToolSpec(bank_id="test", client=client)
        result = spec.recall_memory("query")
        assert "Failed to search memory" in result


class TestReflectTool:
    def test_reflect_returns_text(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response(
            "The user is a Python developer who prefers functional patterns."
        )
        spec = HindsightToolSpec(bank_id="test-bank", client=client)
        result = spec.reflect_on_memory("What do you know about the user?")
        assert (
            result == "The user is a Python developer who prefers functional patterns."
        )

    def test_reflect_empty_returns_fallback(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("")
        spec = HindsightToolSpec(bank_id="test", client=client)
        result = spec.reflect_on_memory("anything")
        assert result == "No relevant memories found."

    def test_reflect_passes_budget(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("answer")
        spec = HindsightToolSpec(bank_id="test", client=client, budget="high")
        spec.reflect_on_memory("query")
        call_kwargs = client.reflect.call_args[1]
        assert call_kwargs["budget"] == "high"

    def test_reflect_passes_context(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("answer")
        spec = HindsightToolSpec(
            bank_id="test",
            client=client,
            reflect_context="The user is asking about project setup",
        )
        spec.reflect_on_memory("query")
        call_kwargs = client.reflect.call_args[1]
        assert call_kwargs["context"] == "The user is asking about project setup"

    def test_reflect_passes_max_tokens_and_response_schema(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("answer")
        schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
        spec = HindsightToolSpec(
            bank_id="test",
            client=client,
            reflect_max_tokens=2048,
            reflect_response_schema=schema,
        )
        spec.reflect_on_memory("query")
        call_kwargs = client.reflect.call_args[1]
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["response_schema"] == schema

    def test_reflect_passes_tags(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("answer")
        spec = HindsightToolSpec(
            bank_id="test",
            client=client,
            reflect_tags=["scope:global"],
            reflect_tags_match="all",
        )
        spec.reflect_on_memory("query")
        call_kwargs = client.reflect.call_args[1]
        assert call_kwargs["tags"] == ["scope:global"]
        assert call_kwargs["tags_match"] == "all"

    def test_reflect_falls_back_to_recall_tags(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("answer")
        spec = HindsightToolSpec(
            bank_id="test",
            client=client,
            recall_tags=["scope:user"],
            recall_tags_match="any",
        )
        spec.reflect_on_memory("query")
        call_kwargs = client.reflect.call_args[1]
        assert call_kwargs["tags"] == ["scope:user"]
        assert call_kwargs["tags_match"] == "any"

    def test_reflect_returns_error_message_on_failure(self):
        """Errors are returned gracefully, not raised."""
        client = _mock_client()
        client.reflect.side_effect = RuntimeError("timeout")
        spec = HindsightToolSpec(bank_id="test", client=client)
        result = spec.reflect_on_memory("query")
        assert "Failed to reflect on memory" in result


class TestBankMission:
    def test_creates_bank_with_mission_on_first_use(self):
        client = _mock_client()
        client.retain.return_value = _mock_retain_response()
        spec = HindsightToolSpec(
            bank_id="test-bank", client=client, mission="Track user preferences"
        )
        spec.retain_memory("content")
        client.create_bank.assert_called_once_with(
            bank_id="test-bank",
            name="test-bank",
            mission="Track user preferences",
        )

    def test_bank_creation_is_idempotent(self):
        client = _mock_client()
        client.retain.return_value = _mock_retain_response()
        client.recall.return_value = _mock_recall_response(["fact"])
        spec = HindsightToolSpec(
            bank_id="test-bank", client=client, mission="my mission"
        )
        spec.retain_memory("content")
        spec.recall_memory("query")
        # create_bank should only be called once
        assert client.create_bank.call_count == 1

    def test_bank_creation_failure_is_graceful(self):
        client = _mock_client()
        client.create_bank.side_effect = RuntimeError("already exists")
        client.retain.return_value = _mock_retain_response()
        spec = HindsightToolSpec(
            bank_id="test-bank", client=client, mission="my mission"
        )
        # Should not raise
        result = spec.retain_memory("content")
        assert result == "Memory stored successfully."

    def test_no_bank_creation_without_mission(self):
        client = _mock_client()
        client.retain.return_value = _mock_retain_response()
        spec = HindsightToolSpec(bank_id="test-bank", client=client)
        spec.retain_memory("content")
        client.create_bank.assert_not_called()

    def test_mission_from_config(self):
        reset_config()
        configure(
            hindsight_api_url="http://localhost:8888",
            mission="config mission",
        )
        with patch("hindsight_llamaindex._client.Hindsight") as mock_cls:
            mock_instance = _mock_client()
            mock_cls.return_value = mock_instance
            mock_instance.retain.return_value = _mock_retain_response()

            spec = HindsightToolSpec(bank_id="test")
            spec.retain_memory("content")
            mock_instance.create_bank.assert_called_once_with(
                bank_id="test",
                name="test",
                mission="config mission",
            )
        reset_config()


class TestLlamaIndexCompatibility:
    """Verify tools integrate correctly with LlamaIndex agent classes."""

    def test_tools_have_correct_metadata(self):
        """Each tool should have name, description, and fn_schema."""
        client = _mock_client()
        spec = HindsightToolSpec(bank_id="test", client=client)
        tools = spec.to_tool_list()

        for tool in tools:
            assert tool.metadata.name is not None
            assert tool.metadata.description is not None
            assert tool.metadata.fn_schema is not None

    def test_tool_names_match_spec_functions(self):
        client = _mock_client()
        spec = HindsightToolSpec(bank_id="test", client=client)
        tools = spec.to_tool_list()
        tool_names = {t.metadata.name for t in tools}
        assert tool_names == {"retain_memory", "recall_memory", "reflect_on_memory"}

    def test_tools_accepted_by_react_agent(self):
        """ReActAgent should accept our tools without error."""
        from llama_index.core.agent import ReActAgent
        from llama_index.core.llms import MockLLM

        client = _mock_client()
        spec = HindsightToolSpec(bank_id="test", client=client)
        tools = spec.to_tool_list()

        # Should not raise — verifies tool format is compatible
        agent = ReActAgent(tools=tools, llm=MockLLM())
        assert agent is not None

    def test_tools_have_both_sync_and_async(self):
        """Each tool should have both sync fn and async fn."""
        client = _mock_client()
        spec = HindsightToolSpec(bank_id="test", client=client)
        tools = spec.to_tool_list()

        for tool in tools:
            assert tool._fn is not None, f"{tool.metadata.name} missing sync fn"
            assert tool._async_fn is not None, f"{tool.metadata.name} missing async fn"

    def test_retain_tool_callable_via_function_tool(self):
        """FunctionTool.call() should invoke retain_memory correctly."""
        client = _mock_client()
        client.retain.return_value = _mock_retain_response()
        spec = HindsightToolSpec(bank_id="test", client=client)
        tools = spec.to_tool_list(spec_functions=[("retain_memory", "aretain_memory")])
        tool = tools[0]

        result = tool.call(content="test memory")
        assert "stored successfully" in str(result)
        client.retain.assert_called_once()

    def test_recall_tool_callable_via_function_tool(self):
        """FunctionTool.call() should invoke recall_memory correctly."""
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["some fact"])
        spec = HindsightToolSpec(bank_id="test", client=client)
        tools = spec.to_tool_list(spec_functions=[("recall_memory", "arecall_memory")])
        tool = tools[0]

        result = tool.call(query="test query")
        assert "some fact" in str(result)
        client.recall.assert_called_once()


class TestConfigFallback:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_budget_falls_back_to_config(self):
        configure(hindsight_api_url="http://localhost:8888")
        with patch("hindsight_llamaindex._client.Hindsight") as mock_cls:
            mock_instance = _mock_client()
            mock_cls.return_value = mock_instance
            mock_instance.recall.return_value = _mock_recall_response(["fact"])

            # Configure with custom budget
            reset_config()
            configure(hindsight_api_url="http://localhost:8888", budget="high")

            spec = HindsightToolSpec(bank_id="test")
            spec.recall_memory("query")
            call_kwargs = mock_instance.recall.call_args[1]
            assert call_kwargs["budget"] == "high"

    def test_explicit_budget_overrides_config(self):
        configure(hindsight_api_url="http://localhost:8888", budget="high")
        with patch("hindsight_llamaindex._client.Hindsight") as mock_cls:
            mock_instance = _mock_client()
            mock_cls.return_value = mock_instance
            mock_instance.recall.return_value = _mock_recall_response(["fact"])

            spec = HindsightToolSpec(bank_id="test", budget="low")
            spec.recall_memory("query")
            call_kwargs = mock_instance.recall.call_args[1]
            assert call_kwargs["budget"] == "low"

    def test_tags_fall_back_to_config(self):
        configure(hindsight_api_url="http://localhost:8888", tags=["config:tag"])
        with patch("hindsight_llamaindex._client.Hindsight") as mock_cls:
            mock_instance = _mock_client()
            mock_cls.return_value = mock_instance
            mock_instance.retain.return_value = _mock_retain_response()

            spec = HindsightToolSpec(bank_id="test")
            spec.retain_memory("content")
            call_kwargs = mock_instance.retain.call_args[1]
            assert call_kwargs["tags"] == ["config:tag"]

    def test_context_falls_back_to_config(self):
        configure(hindsight_api_url="http://localhost:8888", context="my-app")
        with patch("hindsight_llamaindex._client.Hindsight") as mock_cls:
            mock_instance = _mock_client()
            mock_cls.return_value = mock_instance
            mock_instance.retain.return_value = _mock_retain_response()

            spec = HindsightToolSpec(bank_id="test")
            spec.retain_memory("content")
            call_kwargs = mock_instance.retain.call_args[1]
            assert call_kwargs["context"] == "my-app"
