"""Unit tests for Hindsight AG2 tools."""

import inspect
from typing import Annotated, get_type_hints
from unittest.mock import MagicMock, patch

import pytest
from hindsight_ag2 import (
    configure,
    create_hindsight_tools,
    register_hindsight_tools,
    reset_config,
)
from hindsight_ag2.errors import HindsightError


def _mock_client():
    """Create a mock Hindsight client with sync methods."""
    client = MagicMock()
    client.retain = MagicMock(return_value=None)
    client.recall = MagicMock()
    client.reflect = MagicMock()
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


class TestImports:
    def test_imports(self):
        from hindsight_ag2 import (  # noqa: F401
            HindsightAG2Config,
            HindsightError,
            configure,
            create_hindsight_tools,
            get_config,
            register_hindsight_tools,
            reset_config,
        )


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
        assert tools[0].__name__ == "hindsight_retain"

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
        assert tools[0].__name__ == "hindsight_recall"

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
        assert tools[0].__name__ == "hindsight_reflect"

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

    def test_tool_names(self):
        client = _mock_client()
        tools = create_hindsight_tools(bank_id="test", client=client)
        names = [fn.__name__ for fn in tools]
        assert names == ["hindsight_retain", "hindsight_recall", "hindsight_reflect"]

    def test_tool_docstrings(self):
        client = _mock_client()
        tools = create_hindsight_tools(bank_id="test", client=client)
        for fn in tools:
            assert fn.__doc__ is not None
            assert len(fn.__doc__) > 0

    def test_raises_without_client_or_config(self):
        with pytest.raises(HindsightError, match="No Hindsight API URL"):
            create_hindsight_tools(bank_id="test")

    def test_falls_back_to_global_config(self):
        configure(hindsight_api_url="http://localhost:8888")
        with patch("hindsight_ag2._client.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            tools = create_hindsight_tools(bank_id="test")
            assert len(tools) == 3
            mock_cls.assert_called_once()
            assert mock_cls.call_args.kwargs["base_url"] == "http://localhost:8888"
            assert mock_cls.call_args.kwargs["timeout"] == 30.0

    def test_explicit_url_overrides_config(self):
        configure(hindsight_api_url="http://config:8888")
        with patch("hindsight_ag2._client.Hindsight") as mock_cls:
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
        tools = create_hindsight_tools(
            bank_id="test-bank",
            client=client,
            include_recall=False,
            include_reflect=False,
        )
        result = tools[0]("The user likes Python")
        assert result == "Memory stored successfully."
        client.retain.assert_called_once_with(
            bank_id="test-bank", content="The user likes Python"
        )

    def test_retain_passes_tags(self):
        client = _mock_client()
        tools = create_hindsight_tools(
            bank_id="test-bank",
            client=client,
            tags=["source:chat"],
            include_recall=False,
            include_reflect=False,
        )
        tools[0]("some content")
        call_kwargs = client.retain.call_args[1]
        assert call_kwargs["tags"] == ["source:chat"]

    def test_retain_passes_metadata(self):
        client = _mock_client()
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            retain_metadata={"source": "chat", "session": "abc"},
            include_recall=False,
            include_reflect=False,
        )
        tools[0]("content")
        call_kwargs = client.retain.call_args[1]
        assert call_kwargs["metadata"] == {"source": "chat", "session": "abc"}

    def test_retain_passes_document_id(self):
        client = _mock_client()
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            retain_document_id="session-123",
            include_recall=False,
            include_reflect=False,
        )
        tools[0]("content")
        call_kwargs = client.retain.call_args[1]
        assert call_kwargs["document_id"] == "session-123"

    def test_retain_raises_hindsight_error(self):
        client = _mock_client()
        client.retain.side_effect = RuntimeError("connection refused")
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            include_recall=False,
            include_reflect=False,
        )
        with pytest.raises(HindsightError, match="Retain failed"):
            tools[0]("content")


class TestRecallTool:
    def test_recall_returns_numbered_results(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(
            ["User likes Python", "User is in NYC"]
        )
        tools = create_hindsight_tools(
            bank_id="test-bank",
            client=client,
            include_retain=False,
            include_reflect=False,
        )
        result = tools[0]("user preferences")
        assert "1. User likes Python" in result
        assert "2. User is in NYC" in result

    def test_recall_empty_results(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response([])
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            include_retain=False,
            include_reflect=False,
        )
        result = tools[0]("anything")
        assert result == "No relevant memories found."

    def test_recall_passes_budget_and_max_tokens(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            budget="high",
            max_tokens=2048,
            include_retain=False,
            include_reflect=False,
        )
        tools[0]("query")
        call_kwargs = client.recall.call_args[1]
        assert call_kwargs["budget"] == "high"
        assert call_kwargs["max_tokens"] == 2048

    def test_recall_passes_tags(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            recall_tags=["scope:user"],
            recall_tags_match="all",
            include_retain=False,
            include_reflect=False,
        )
        tools[0]("query")
        call_kwargs = client.recall.call_args[1]
        assert call_kwargs["tags"] == ["scope:user"]
        assert call_kwargs["tags_match"] == "all"

    def test_recall_passes_types(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            recall_types=["world", "experience"],
            include_retain=False,
            include_reflect=False,
        )
        tools[0]("query")
        call_kwargs = client.recall.call_args[1]
        assert call_kwargs["types"] == ["world", "experience"]

    def test_recall_passes_include_entities(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            recall_include_entities=True,
            include_retain=False,
            include_reflect=False,
        )
        tools[0]("query")
        call_kwargs = client.recall.call_args[1]
        assert call_kwargs["include_entities"] is True

    def test_recall_raises_hindsight_error(self):
        client = _mock_client()
        client.recall.side_effect = RuntimeError("connection refused")
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            include_retain=False,
            include_reflect=False,
        )
        with pytest.raises(HindsightError, match="Recall failed"):
            tools[0]("query")


class TestReflectTool:
    def test_reflect_returns_text(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response(
            "The user is a Python developer who prefers functional patterns."
        )
        tools = create_hindsight_tools(
            bank_id="test-bank",
            client=client,
            include_retain=False,
            include_recall=False,
        )
        result = tools[0]("What do you know about the user?")
        assert result == "The user is a Python developer who prefers functional patterns."

    def test_reflect_empty_returns_fallback(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("")
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            include_retain=False,
            include_recall=False,
        )
        result = tools[0]("anything")
        assert result == "No relevant memories found."

    def test_reflect_passes_budget(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("answer")
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            budget="high",
            include_retain=False,
            include_recall=False,
        )
        tools[0]("query")
        call_kwargs = client.reflect.call_args[1]
        assert call_kwargs["budget"] == "high"

    def test_reflect_passes_context(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("answer")
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            reflect_context="The user is asking about project setup",
            include_retain=False,
            include_recall=False,
        )
        tools[0]("query")
        call_kwargs = client.reflect.call_args[1]
        assert call_kwargs["context"] == "The user is asking about project setup"

    def test_reflect_passes_max_tokens_and_response_schema(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("answer")
        schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            reflect_max_tokens=2048,
            reflect_response_schema=schema,
            include_retain=False,
            include_recall=False,
        )
        tools[0]("query")
        call_kwargs = client.reflect.call_args[1]
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["response_schema"] == schema

    def test_reflect_passes_tags(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("answer")
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            reflect_tags=["scope:global"],
            reflect_tags_match="all",
            include_retain=False,
            include_recall=False,
        )
        tools[0]("query")
        call_kwargs = client.reflect.call_args[1]
        assert call_kwargs["tags"] == ["scope:global"]
        assert call_kwargs["tags_match"] == "all"

    def test_reflect_raises_hindsight_error(self):
        client = _mock_client()
        client.reflect.side_effect = RuntimeError("connection refused")
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            include_retain=False,
            include_recall=False,
        )
        with pytest.raises(HindsightError, match="Reflect failed"):
            tools[0]("query")


class TestAnnotatedTypes:
    def test_retain_has_annotated_parameter(self):
        client = _mock_client()
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            include_recall=False,
            include_reflect=False,
        )
        hints = get_type_hints(tools[0], include_extras=True)
        assert "content" in hints
        # Check it's Annotated
        assert hasattr(hints["content"], "__metadata__")

    def test_recall_has_annotated_parameter(self):
        client = _mock_client()
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            include_retain=False,
            include_reflect=False,
        )
        hints = get_type_hints(tools[0], include_extras=True)
        assert "query" in hints
        assert hasattr(hints["query"], "__metadata__")

    def test_reflect_has_annotated_parameter(self):
        client = _mock_client()
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            include_retain=False,
            include_recall=False,
        )
        hints = get_type_hints(tools[0], include_extras=True)
        assert "query" in hints
        assert hasattr(hints["query"], "__metadata__")


class TestRegisterHindsightTools:
    def test_registers_all_tools(self):
        client = _mock_client()
        agent = MagicMock()
        executor = MagicMock()

        # Make register_for_llm return a decorator that returns the function
        agent.register_for_llm.return_value = lambda fn: fn
        executor.register_for_execution.return_value = lambda fn: fn

        tools = register_hindsight_tools(
            agent, executor, bank_id="test", client=client
        )

        assert len(tools) == 3
        assert agent.register_for_llm.call_count == 3
        assert executor.register_for_execution.call_count == 3

    def test_registers_with_docstring_descriptions(self):
        client = _mock_client()
        agent = MagicMock()
        executor = MagicMock()

        agent.register_for_llm.return_value = lambda fn: fn
        executor.register_for_execution.return_value = lambda fn: fn

        register_hindsight_tools(agent, executor, bank_id="test", client=client)

        # Each register_for_llm call should have a description kwarg
        for call in agent.register_for_llm.call_args_list:
            assert "description" in call.kwargs
            assert call.kwargs["description"] is not None
            assert len(call.kwargs["description"]) > 0

    def test_passes_kwargs_to_create_tools(self):
        client = _mock_client()
        agent = MagicMock()
        executor = MagicMock()

        agent.register_for_llm.return_value = lambda fn: fn
        executor.register_for_execution.return_value = lambda fn: fn

        tools = register_hindsight_tools(
            agent,
            executor,
            bank_id="test",
            client=client,
            include_retain=True,
            include_recall=False,
            include_reflect=False,
        )

        assert len(tools) == 1
        assert tools[0].__name__ == "hindsight_retain"


class TestConfigDefaults:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_default_budget(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            include_retain=False,
            include_reflect=False,
        )
        tools[0]("query")
        call_kwargs = client.recall.call_args[1]
        assert call_kwargs["budget"] == "mid"

    def test_default_max_tokens(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            include_retain=False,
            include_reflect=False,
        )
        tools[0]("query")
        call_kwargs = client.recall.call_args[1]
        assert call_kwargs["max_tokens"] == 4096

    def test_config_budget_used_when_no_explicit(self):
        configure(hindsight_api_url="http://localhost:8888", budget="low")
        with patch("hindsight_ag2._client.Hindsight") as mock_cls:
            mock_client = _mock_client()
            mock_client.recall.return_value = _mock_recall_response(["fact"])
            mock_cls.return_value = mock_client
            tools = create_hindsight_tools(
                bank_id="test",
                include_retain=False,
                include_reflect=False,
            )
            tools[0]("query")
            call_kwargs = mock_client.recall.call_args[1]
            assert call_kwargs["budget"] == "low"

    def test_explicit_budget_overrides_config(self):
        configure(hindsight_api_url="http://localhost:8888", budget="low")
        with patch("hindsight_ag2._client.Hindsight") as mock_cls:
            mock_client = _mock_client()
            mock_client.recall.return_value = _mock_recall_response(["fact"])
            mock_cls.return_value = mock_client
            tools = create_hindsight_tools(
                bank_id="test",
                budget="high",
                include_retain=False,
                include_reflect=False,
            )
            tools[0]("query")
            call_kwargs = mock_client.recall.call_args[1]
            assert call_kwargs["budget"] == "high"
