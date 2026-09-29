"""Unit tests for Hindsight Haystack tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack.dataclasses import ChatMessage
from hindsight_haystack import (
    HindsightMemoryWrapper,
    configure,
    create_hindsight_tools,
    reset_config,
)
from hindsight_haystack.errors import HindsightError


def _mock_client():
    """Create a mock Hindsight client with async methods."""
    client = MagicMock()
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


def _mock_reflect_response(text: str, structured_output: dict | None = None):
    response = MagicMock()
    response.text = text
    response.structured_output = structured_output
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
        assert tools[0].name == "retain_memory"

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
        assert tools[0].name == "recall_memory"

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
        assert tools[0].name == "reflect_on_memory"

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
        from hindsight_haystack.config import DEFAULT_HINDSIGHT_API_URL

        monkeypatch.delenv("HINDSIGHT_API_KEY", raising=False)
        with patch("hindsight_haystack._client.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            tools = create_hindsight_tools(bank_id="test")
            assert len(tools) == 3
            assert mock_cls.call_args.kwargs["base_url"] == DEFAULT_HINDSIGHT_API_URL
            # No key configured -> none passed; it only fails at call time.
            assert "api_key" not in mock_cls.call_args.kwargs

    def test_reads_api_key_from_env_without_config(self, monkeypatch):
        """HINDSIGHT_API_KEY is honoured even when configure() was never called."""
        monkeypatch.setenv("HINDSIGHT_API_KEY", "sk-from-env")
        with patch("hindsight_haystack._client.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            create_hindsight_tools(bank_id="test")
            assert mock_cls.call_args.kwargs["api_key"] == "sk-from-env"

    def test_falls_back_to_global_config(self):
        configure(hindsight_api_url="http://localhost:8888")
        with patch("hindsight_haystack._client.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            tools = create_hindsight_tools(bank_id="test")
            assert len(tools) == 3
            mock_cls.assert_called_once()
            assert mock_cls.call_args.kwargs["base_url"] == "http://localhost:8888"
            assert mock_cls.call_args.kwargs["timeout"] == 30.0

    def test_explicit_url_overrides_config(self):
        configure(hindsight_api_url="http://config:8888")
        with patch("hindsight_haystack._client.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            create_hindsight_tools(bank_id="test", hindsight_api_url="http://explicit:9999")
            mock_cls.assert_called_once()
            assert mock_cls.call_args.kwargs["base_url"] == "http://explicit:9999"
            assert mock_cls.call_args.kwargs["timeout"] == 30.0


class TestRetainTool:
    def test_retain_stores_memory(self):
        client = _mock_client()
        client.aretain.return_value = _mock_retain_response()
        tools = create_hindsight_tools(bank_id="test-bank", client=client)
        retain_tool = [t for t in tools if t.name == "retain_memory"][0]
        result = retain_tool.invoke(content="The user likes Python")
        assert result == "Memory stored successfully."
        call_kwargs = client.aretain.call_args[1]
        assert call_kwargs["bank_id"] == "test-bank"
        assert call_kwargs["content"] == "The user likes Python"
        assert call_kwargs["context"] == "haystack"

    def test_retain_passes_tags(self):
        client = _mock_client()
        client.aretain.return_value = _mock_retain_response()
        tools = create_hindsight_tools(bank_id="test-bank", client=client, tags=["source:chat"])
        retain_tool = [t for t in tools if t.name == "retain_memory"][0]
        retain_tool.invoke(content="some content")
        call_kwargs = client.aretain.call_args[1]
        assert call_kwargs["tags"] == ["source:chat"]

    def test_retain_passes_metadata(self):
        client = _mock_client()
        client.aretain.return_value = _mock_retain_response()
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            retain_metadata={"source": "chat", "session": "abc"},
        )
        retain_tool = [t for t in tools if t.name == "retain_memory"][0]
        retain_tool.invoke(content="content")
        call_kwargs = client.aretain.call_args[1]
        assert call_kwargs["metadata"] == {"source": "chat", "session": "abc"}

    def test_retain_passes_explicit_document_id(self):
        client = _mock_client()
        client.aretain.return_value = _mock_retain_response()
        tools = create_hindsight_tools(bank_id="test", client=client, retain_document_id="session-123")
        retain_tool = [t for t in tools if t.name == "retain_memory"][0]
        retain_tool.invoke(content="content")
        call_kwargs = client.aretain.call_args[1]
        assert call_kwargs["document_id"] == "session-123"

    def test_retain_auto_generates_document_id(self):
        client = _mock_client()
        client.aretain.return_value = _mock_retain_response()
        tools = create_hindsight_tools(bank_id="test", client=client)
        retain_tool = [t for t in tools if t.name == "retain_memory"][0]
        retain_tool.invoke(content="content")
        call_kwargs = client.aretain.call_args[1]
        doc_id = call_kwargs["document_id"]
        # Auto-generated format: {session_id}-{uuid_hex_12}
        assert "-" in doc_id
        suffix = doc_id.rsplit("-", 1)[1]
        assert len(suffix) == 12

    def test_retain_passes_context_label(self):
        client = _mock_client()
        client.aretain.return_value = _mock_retain_response()
        tools = create_hindsight_tools(bank_id="test", client=client, retain_context="my-app")
        retain_tool = [t for t in tools if t.name == "retain_memory"][0]
        retain_tool.invoke(content="content")
        call_kwargs = client.aretain.call_args[1]
        assert call_kwargs["context"] == "my-app"

    def test_retain_defaults_to_haystack_context(self):
        client = _mock_client()
        client.aretain.return_value = _mock_retain_response()
        tools = create_hindsight_tools(bank_id="test", client=client)
        retain_tool = [t for t in tools if t.name == "retain_memory"][0]
        retain_tool.invoke(content="content")
        call_kwargs = client.aretain.call_args[1]
        assert call_kwargs["context"] == "haystack"

    def test_retain_returns_error_message_on_failure(self):
        """Errors are returned gracefully, not raised."""
        client = _mock_client()
        client.aretain.side_effect = RuntimeError("connection refused")
        tools = create_hindsight_tools(bank_id="test", client=client)
        retain_tool = [t for t in tools if t.name == "retain_memory"][0]
        result = retain_tool.invoke(content="content")
        assert "Failed to store memory" in result
        assert "connection refused" in result


class TestRecallTool:
    def test_recall_returns_numbered_results(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["User likes Python", "User is in NYC"])
        tools = create_hindsight_tools(bank_id="test-bank", client=client)
        recall_tool = [t for t in tools if t.name == "recall_memory"][0]
        result = recall_tool.invoke(query="user preferences")
        assert "1. User likes Python" in result
        assert "2. User is in NYC" in result

    def test_recall_empty_results(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response([])
        tools = create_hindsight_tools(bank_id="test", client=client)
        recall_tool = [t for t in tools if t.name == "recall_memory"][0]
        result = recall_tool.invoke(query="anything")
        assert result == "No relevant memories found."

    def test_recall_passes_budget_and_max_tokens(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])
        tools = create_hindsight_tools(bank_id="test", client=client, budget="high", max_tokens=2048)
        recall_tool = [t for t in tools if t.name == "recall_memory"][0]
        recall_tool.invoke(query="query")
        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["budget"] == "high"
        assert call_kwargs["max_tokens"] == 2048

    def test_recall_passes_tags(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            recall_tags=["scope:user"],
            recall_tags_match="all",
        )
        recall_tool = [t for t in tools if t.name == "recall_memory"][0]
        recall_tool.invoke(query="query")
        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["tags"] == ["scope:user"]
        assert call_kwargs["tags_match"] == "all"

    def test_recall_passes_types(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            recall_types=["world", "experience"],
        )
        recall_tool = [t for t in tools if t.name == "recall_memory"][0]
        recall_tool.invoke(query="query")
        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["types"] == ["world", "experience"]

    def test_recall_passes_include_entities(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])
        tools = create_hindsight_tools(bank_id="test", client=client, recall_include_entities=True)
        recall_tool = [t for t in tools if t.name == "recall_memory"][0]
        recall_tool.invoke(query="query")
        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["include_entities"] is True

    def test_recall_returns_error_message_on_failure(self):
        """Errors are returned gracefully, not raised."""
        client = _mock_client()
        client.arecall.side_effect = RuntimeError("timeout")
        tools = create_hindsight_tools(bank_id="test", client=client)
        recall_tool = [t for t in tools if t.name == "recall_memory"][0]
        result = recall_tool.invoke(query="query")
        assert "Failed to search memory" in result


class TestReflectTool:
    def test_reflect_returns_text(self):
        client = _mock_client()
        client.areflect.return_value = _mock_reflect_response(
            "The user is a Python developer who prefers functional patterns."
        )
        tools = create_hindsight_tools(bank_id="test-bank", client=client)
        reflect_tool = [t for t in tools if t.name == "reflect_on_memory"][0]
        result = reflect_tool.invoke(query="What do you know about the user?")
        assert result == "The user is a Python developer who prefers functional patterns."

    def test_reflect_empty_returns_fallback(self):
        client = _mock_client()
        client.areflect.return_value = _mock_reflect_response("")
        tools = create_hindsight_tools(bank_id="test", client=client)
        reflect_tool = [t for t in tools if t.name == "reflect_on_memory"][0]
        result = reflect_tool.invoke(query="anything")
        assert result == "No relevant memories found."

    def test_reflect_passes_budget(self):
        client = _mock_client()
        client.areflect.return_value = _mock_reflect_response("answer")
        tools = create_hindsight_tools(bank_id="test", client=client, budget="high")
        reflect_tool = [t for t in tools if t.name == "reflect_on_memory"][0]
        reflect_tool.invoke(query="query")
        call_kwargs = client.areflect.call_args[1]
        assert call_kwargs["budget"] == "high"

    def test_reflect_passes_context(self):
        client = _mock_client()
        client.areflect.return_value = _mock_reflect_response("answer")
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            reflect_context="The user is asking about project setup",
        )
        reflect_tool = [t for t in tools if t.name == "reflect_on_memory"][0]
        reflect_tool.invoke(query="query")
        call_kwargs = client.areflect.call_args[1]
        assert call_kwargs["context"] == "The user is asking about project setup"

    def test_reflect_passes_max_tokens_and_response_schema(self):
        client = _mock_client()
        client.areflect.return_value = _mock_reflect_response("answer")
        schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            reflect_max_tokens=2048,
            reflect_response_schema=schema,
        )
        reflect_tool = [t for t in tools if t.name == "reflect_on_memory"][0]
        reflect_tool.invoke(query="query")
        call_kwargs = client.areflect.call_args[1]
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["response_schema"] == schema

    def test_reflect_passes_tags(self):
        client = _mock_client()
        client.areflect.return_value = _mock_reflect_response("answer")
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            reflect_tags=["scope:global"],
            reflect_tags_match="all",
        )
        reflect_tool = [t for t in tools if t.name == "reflect_on_memory"][0]
        reflect_tool.invoke(query="query")
        call_kwargs = client.areflect.call_args[1]
        assert call_kwargs["tags"] == ["scope:global"]
        assert call_kwargs["tags_match"] == "all"

    def test_reflect_falls_back_to_recall_tags(self):
        client = _mock_client()
        client.areflect.return_value = _mock_reflect_response("answer")
        tools = create_hindsight_tools(
            bank_id="test",
            client=client,
            recall_tags=["scope:user"],
            recall_tags_match="any",
        )
        reflect_tool = [t for t in tools if t.name == "reflect_on_memory"][0]
        reflect_tool.invoke(query="query")
        call_kwargs = client.areflect.call_args[1]
        assert call_kwargs["tags"] == ["scope:user"]
        assert call_kwargs["tags_match"] == "any"

    def test_reflect_returns_error_message_on_failure(self):
        """Errors are returned gracefully, not raised."""
        client = _mock_client()
        client.areflect.side_effect = RuntimeError("timeout")
        tools = create_hindsight_tools(bank_id="test", client=client)
        reflect_tool = [t for t in tools if t.name == "reflect_on_memory"][0]
        result = reflect_tool.invoke(query="query")
        assert "Failed to reflect on memory" in result


class TestBankMission:
    def test_creates_bank_with_mission_on_first_use(self):
        client = _mock_client()
        client.aretain.return_value = _mock_retain_response()
        tools = create_hindsight_tools(bank_id="test-bank", client=client, mission="Track user preferences")
        retain_tool = [t for t in tools if t.name == "retain_memory"][0]
        retain_tool.invoke(content="content")
        client.acreate_bank.assert_called_once_with(
            bank_id="test-bank",
            name="test-bank",
            mission="Track user preferences",
        )

    def test_bank_creation_is_idempotent(self):
        client = _mock_client()
        client.aretain.return_value = _mock_retain_response()
        client.arecall.return_value = _mock_recall_response(["fact"])
        tools = create_hindsight_tools(bank_id="test-bank", client=client, mission="my mission")
        retain_tool = [t for t in tools if t.name == "retain_memory"][0]
        recall_tool = [t for t in tools if t.name == "recall_memory"][0]
        retain_tool.invoke(content="content")
        recall_tool.invoke(query="query")
        # create_bank should only be called once
        assert client.acreate_bank.call_count == 1

    def test_bank_creation_failure_is_graceful(self):
        client = _mock_client()
        client.acreate_bank.side_effect = RuntimeError("already exists")
        client.aretain.return_value = _mock_retain_response()
        tools = create_hindsight_tools(bank_id="test-bank", client=client, mission="my mission")
        retain_tool = [t for t in tools if t.name == "retain_memory"][0]
        # Should not raise
        result = retain_tool.invoke(content="content")
        assert result == "Memory stored successfully."

    def test_no_bank_creation_without_mission(self):
        client = _mock_client()
        client.aretain.return_value = _mock_retain_response()
        tools = create_hindsight_tools(bank_id="test-bank", client=client)
        retain_tool = [t for t in tools if t.name == "retain_memory"][0]
        retain_tool.invoke(content="content")
        client.acreate_bank.assert_not_called()

    def test_mission_from_config(self):
        reset_config()
        configure(
            hindsight_api_url="http://localhost:8888",
            mission="config mission",
        )
        with patch("hindsight_haystack._client.Hindsight") as mock_cls:
            mock_instance = _mock_client()
            mock_cls.return_value = mock_instance
            mock_instance.aretain.return_value = _mock_retain_response()

            tools = create_hindsight_tools(bank_id="test")
            retain_tool = [t for t in tools if t.name == "retain_memory"][0]
            retain_tool.invoke(content="content")
            mock_instance.acreate_bank.assert_called_once_with(
                bank_id="test",
                name="test",
                mission="config mission",
            )
        reset_config()


class TestHaystackCompatibility:
    """Verify tools integrate correctly with Haystack's Tool interface."""

    def test_tools_have_correct_name_and_description(self):
        """Each tool should have name and description."""
        client = _mock_client()
        tools = create_hindsight_tools(bank_id="test", client=client)

        for tool in tools:
            assert tool.name is not None
            assert tool.description is not None
            assert len(tool.description) > 0

    def test_tool_names_match_expected_set(self):
        client = _mock_client()
        tools = create_hindsight_tools(bank_id="test", client=client)
        tool_names = {t.name for t in tools}
        assert tool_names == {"retain_memory", "recall_memory", "reflect_on_memory"}

    def test_tools_have_parameters_schema(self):
        """Each tool should have a valid parameters JSON schema."""
        client = _mock_client()
        tools = create_hindsight_tools(bank_id="test", client=client)

        for tool in tools:
            assert tool.parameters is not None
            assert tool.parameters["type"] == "object"
            assert "properties" in tool.parameters
            assert "required" in tool.parameters

    def test_retain_tool_callable(self):
        """Tool.invoke() should call retain_memory correctly."""
        client = _mock_client()
        client.aretain.return_value = _mock_retain_response()
        tools = create_hindsight_tools(bank_id="test", client=client, include_recall=False, include_reflect=False)
        tool = tools[0]

        result = tool.invoke(content="test memory")
        assert "stored successfully" in str(result)
        client.aretain.assert_called_once()

    def test_recall_tool_callable(self):
        """Tool.invoke() should call recall_memory correctly."""
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["some fact"])
        tools = create_hindsight_tools(bank_id="test", client=client, include_retain=False, include_reflect=False)
        tool = tools[0]

        result = tool.invoke(query="test query")
        assert "some fact" in str(result)
        client.arecall.assert_called_once()

    def test_tools_are_serializable(self):
        """Tool.to_dict() should succeed (not raise SerializationError)."""
        client = _mock_client()
        tools = create_hindsight_tools(bank_id="test", client=client)

        for tool in tools:
            d = tool.to_dict()
            assert d["type"] == "hindsight_haystack.tools._HindsightTool"
            assert d["data"]["name"] == tool.name
            assert "backend_kwargs" in d["data"]

    def test_tools_round_trip_serialization_with_config(self):
        """Tool.from_dict(tool.to_dict()) should produce a working tool (config path)."""
        configure(hindsight_api_url="http://localhost:8888")
        with patch("hindsight_haystack._client.Hindsight") as mock_cls:
            mock_instance = _mock_client()
            mock_cls.return_value = mock_instance
            mock_instance.aretain.return_value = _mock_retain_response()

            tools = create_hindsight_tools(bank_id="test")
            retain_tool = [t for t in tools if t.name == "retain_memory"][0]

            from hindsight_haystack.tools import _HindsightTool

            d = retain_tool.to_dict()
            restored = _HindsightTool.from_dict(d)

            assert restored.name == "retain_memory"
            result = restored.invoke(content="test")
            assert result == "Memory stored successfully."
        reset_config()

    def test_tools_round_trip_serialization_with_client(self, monkeypatch):
        """Round-trip works when tools were created with an explicit client.

        The base_url is serialized (so a deserialized pipeline still points at
        the right host). The api_key is NOT serialized — Haystack pipelines
        get dumped to YAML and a serialized key leaks into every dump. The
        from_dict path picks the key back up from the HINDSIGHT_API_KEY env
        var via resolve_client.
        """
        from hindsight_client import Hindsight

        reset_config()
        # Create a real-ish client to extract base_url from
        with patch("hindsight_haystack._client.Hindsight") as mock_cls:
            mock_instance = _mock_client()
            mock_instance._base_url = "http://from-client:8888"
            mock_instance._api_key = "client-key"
            mock_cls.return_value = mock_instance

            # Pass the mock as the explicit client
            tools = create_hindsight_tools(bank_id="test", client=mock_instance)
            retain_tool = [t for t in tools if t.name == "retain_memory"][0]

            # Verify serialized kwargs captured the URL but NOT the key
            d = retain_tool.to_dict()
            assert d["data"]["backend_kwargs"]["hindsight_api_url"] == "http://from-client:8888"
            assert "api_key" not in d["data"]["backend_kwargs"], (
                "api_key must not appear in serialized backend_kwargs — would leak to YAML pipeline dumps"
            )

            # And it shouldn't be hiding under another name anywhere in the dump
            import json

            assert "client-key" not in json.dumps(d), "the literal api_key value leaked into the serialized dict"

            # Round-trip: from_dict picks the URL from the serialized dict and the key
            # from the env var (HINDSIGHT_API_KEY) via resolve_client.
            monkeypatch.setenv("HINDSIGHT_API_KEY", "env-key")
            from hindsight_haystack.tools import _HindsightTool

            restored = _HindsightTool.from_dict(d)
            assert restored.name == "retain_memory"
            mock_cls.assert_called()
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["base_url"] == "http://from-client:8888"
            assert call_kwargs["api_key"] == "env-key"

    def test_serialization_preserves_explicit_url_over_client(self):
        """When both client= and hindsight_api_url= are passed, explicit URL wins."""
        client = _mock_client()
        client._base_url = "http://from-client:8888"

        tools = create_hindsight_tools(bank_id="test", client=client, hindsight_api_url="http://explicit:9999")
        d = tools[0].to_dict()
        # Explicit URL should take precedence
        assert d["data"]["backend_kwargs"]["hindsight_api_url"] == "http://explicit:9999"


class TestConfigFallback:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_budget_falls_back_to_config(self):
        configure(hindsight_api_url="http://localhost:8888")
        with patch("hindsight_haystack._client.Hindsight") as mock_cls:
            mock_instance = _mock_client()
            mock_cls.return_value = mock_instance
            mock_instance.arecall.return_value = _mock_recall_response(["fact"])

            # Configure with custom budget
            reset_config()
            configure(hindsight_api_url="http://localhost:8888", budget="high")

            tools = create_hindsight_tools(bank_id="test")
            recall_tool = [t for t in tools if t.name == "recall_memory"][0]
            recall_tool.invoke(query="query")
            call_kwargs = mock_instance.arecall.call_args[1]
            assert call_kwargs["budget"] == "high"

    def test_explicit_budget_overrides_config(self):
        configure(hindsight_api_url="http://localhost:8888", budget="high")
        with patch("hindsight_haystack._client.Hindsight") as mock_cls:
            mock_instance = _mock_client()
            mock_cls.return_value = mock_instance
            mock_instance.arecall.return_value = _mock_recall_response(["fact"])

            tools = create_hindsight_tools(bank_id="test", budget="low")
            recall_tool = [t for t in tools if t.name == "recall_memory"][0]
            recall_tool.invoke(query="query")
            call_kwargs = mock_instance.arecall.call_args[1]
            assert call_kwargs["budget"] == "low"

    def test_tags_fall_back_to_config(self):
        configure(hindsight_api_url="http://localhost:8888", tags=["config:tag"])
        with patch("hindsight_haystack._client.Hindsight") as mock_cls:
            mock_instance = _mock_client()
            mock_cls.return_value = mock_instance
            mock_instance.aretain.return_value = _mock_retain_response()

            tools = create_hindsight_tools(bank_id="test")
            retain_tool = [t for t in tools if t.name == "retain_memory"][0]
            retain_tool.invoke(content="content")
            call_kwargs = mock_instance.aretain.call_args[1]
            assert call_kwargs["tags"] == ["config:tag"]

    def test_context_falls_back_to_config(self):
        configure(hindsight_api_url="http://localhost:8888", context="my-app")
        with patch("hindsight_haystack._client.Hindsight") as mock_cls:
            mock_instance = _mock_client()
            mock_cls.return_value = mock_instance
            mock_instance.aretain.return_value = _mock_retain_response()

            tools = create_hindsight_tools(bank_id="test")
            retain_tool = [t for t in tools if t.name == "retain_memory"][0]
            retain_tool.invoke(content="content")
            call_kwargs = mock_instance.aretain.call_args[1]
            assert call_kwargs["context"] == "my-app"


class TestRunSync:
    """Verify the _run_sync helper works in both event-loop contexts."""

    def test_run_sync_without_running_loop(self):
        """Direct asyncio.run path — no event loop active."""
        from hindsight_haystack.tools import _run_sync

        async def coro():
            return 42

        assert _run_sync(coro()) == 42

    def test_run_sync_inside_running_loop(self):
        """Thread-pool path — called from within a running event loop."""
        import asyncio

        from hindsight_haystack.tools import _run_sync

        async def inner():
            return "from_thread"

        async def outer():
            # _run_sync is called while an event loop is running
            return _run_sync(inner())

        result = asyncio.run(outer())
        assert result == "from_thread"

    def test_run_sync_propagates_errors(self):
        """Exceptions from the coroutine should propagate through."""
        from hindsight_haystack.tools import _run_sync

        async def failing():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            _run_sync(failing())


class TestReflectStructuredOutput:
    """Verify reflect returns structured JSON when response_schema is set."""

    def test_reflect_returns_structured_output_when_schema_set(self):
        client = _mock_client()
        schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
        structured = {"summary": "User prefers Python"}
        client.areflect.return_value = _mock_reflect_response("Some prose", structured_output=structured)
        tools = create_hindsight_tools(bank_id="test", client=client, reflect_response_schema=schema)
        reflect_tool = [t for t in tools if t.name == "reflect_on_memory"][0]
        result = reflect_tool.invoke(query="query")
        import json

        assert json.loads(result) == structured

    def test_reflect_returns_text_when_no_schema(self):
        client = _mock_client()
        client.areflect.return_value = _mock_reflect_response("Plain prose", structured_output={"summary": "data"})
        tools = create_hindsight_tools(bank_id="test", client=client)
        reflect_tool = [t for t in tools if t.name == "reflect_on_memory"][0]
        result = reflect_tool.invoke(query="query")
        assert result == "Plain prose"

    def test_reflect_falls_back_to_text_when_structured_output_is_none(self):
        client = _mock_client()
        schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
        client.areflect.return_value = _mock_reflect_response("Fallback prose", structured_output=None)
        tools = create_hindsight_tools(bank_id="test", client=client, reflect_response_schema=schema)
        reflect_tool = [t for t in tools if t.name == "reflect_on_memory"][0]
        result = reflect_tool.invoke(query="query")
        assert result == "Fallback prose"


class TestBankCreationRetry:
    """Verify _ensure_bank retries on transient errors but not on 'already exists'."""

    def test_retries_on_transient_error(self):
        client = _mock_client()
        client.acreate_bank.side_effect = [
            RuntimeError("connection refused"),
            None,  # succeeds on second try
        ]
        client.aretain.return_value = _mock_retain_response()
        tools = create_hindsight_tools(bank_id="test", client=client, mission="my mission")
        retain_tool = [t for t in tools if t.name == "retain_memory"][0]
        # First call — bank creation fails transiently
        retain_tool.invoke(content="first")
        # Second call — bank creation should be retried and succeed
        retain_tool.invoke(content="second")
        assert client.acreate_bank.call_count == 2

    def test_no_retry_on_already_exists(self):
        client = _mock_client()
        client.acreate_bank.side_effect = RuntimeError("bank already exists")
        client.aretain.return_value = _mock_retain_response()
        tools = create_hindsight_tools(bank_id="test", client=client, mission="my mission")
        retain_tool = [t for t in tools if t.name == "retain_memory"][0]
        retain_tool.invoke(content="first")
        retain_tool.invoke(content="second")
        # Should only try once because "already exists" marks as initialized
        assert client.acreate_bank.call_count == 1

    def test_no_retry_on_409_conflict(self):
        client = _mock_client()
        client.acreate_bank.side_effect = RuntimeError("409 Conflict")
        client.aretain.return_value = _mock_retain_response()
        tools = create_hindsight_tools(bank_id="test", client=client, mission="my mission")
        retain_tool = [t for t in tools if t.name == "retain_memory"][0]
        retain_tool.invoke(content="first")
        retain_tool.invoke(content="second")
        assert client.acreate_bank.call_count == 1


class TestHindsightMemoryWrapper:
    """Test the HindsightMemoryWrapper class."""

    def test_creates_three_tools_by_default(self):
        client = _mock_client()
        toolset = HindsightMemoryWrapper(bank_id="test", client=client)
        assert len(toolset) == 3
        names = {t.name for t in toolset}
        assert names == {"retain_memory", "recall_memory", "reflect_on_memory"}

    def test_include_flags(self):
        client = _mock_client()
        toolset = HindsightMemoryWrapper(
            bank_id="test",
            client=client,
            include_retain=True,
            include_recall=False,
            include_reflect=False,
        )
        assert len(toolset) == 1
        assert toolset[0].name == "retain_memory"

    def test_serialization_round_trip(self):
        client = _mock_client()
        client._base_url = "http://test:8888"
        client._api_key = "test-key"
        toolset = HindsightMemoryWrapper(
            bank_id="test",
            client=client,
            auto_recall=True,
            auto_retain=True,
            mission="my mission",
        )

        d = toolset.to_dict()
        assert d["data"]["auto_recall"] is True
        assert d["data"]["auto_retain"] is True
        assert d["data"]["backend_kwargs"]["bank_id"] == "test"

        with patch("hindsight_haystack._client.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            restored = HindsightMemoryWrapper.from_dict(d)
            assert len(restored) == 3
            assert restored._auto_recall is True
            assert restored._auto_retain is True


class TestToolsetAutoRecall:
    """Test auto-recall behavior in HindsightMemoryWrapper.run()."""

    def test_auto_recall_enriches_system_prompt(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["User likes Python", "User is in NYC"])
        toolset = HindsightMemoryWrapper(bank_id="test", client=client, auto_recall=True)

        # Create a mock agent
        agent = MagicMock()
        agent.system_prompt = "You are helpful."
        agent.run.return_value = {
            "messages": [ChatMessage.from_assistant("response")],
            "last_message": ChatMessage.from_assistant("response"),
        }

        toolset.run(agent, messages=[ChatMessage.from_user("What do you know about me?")])

        # Agent.run should have been called with enriched system_prompt
        call_kwargs = agent.run.call_args[1]
        assert "User likes Python" in call_kwargs["system_prompt"]
        assert "User is in NYC" in call_kwargs["system_prompt"]
        assert "You are helpful." in call_kwargs["system_prompt"]

    def test_auto_recall_uses_agent_system_prompt_when_none_provided(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["some memory"])
        toolset = HindsightMemoryWrapper(bank_id="test", client=client, auto_recall=True)

        agent = MagicMock()
        agent.system_prompt = "Agent default prompt."
        agent.run.return_value = {
            "messages": [],
            "last_message": ChatMessage.from_assistant("ok"),
        }

        toolset.run(agent, messages=[ChatMessage.from_user("hello")])
        call_kwargs = agent.run.call_args[1]
        assert "Agent default prompt." in call_kwargs["system_prompt"]

    def test_auto_recall_overrides_with_explicit_system_prompt(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["memory"])
        toolset = HindsightMemoryWrapper(bank_id="test", client=client, auto_recall=True)

        agent = MagicMock()
        agent.system_prompt = "Agent default."
        agent.run.return_value = {
            "messages": [],
            "last_message": ChatMessage.from_assistant("ok"),
        }

        toolset.run(
            agent,
            messages=[ChatMessage.from_user("hi")],
            system_prompt="Custom prompt.",
        )
        call_kwargs = agent.run.call_args[1]
        # Should use explicit prompt, not agent's default
        assert "Custom prompt." in call_kwargs["system_prompt"]
        assert "Agent default." not in call_kwargs["system_prompt"]

    def test_auto_recall_skips_when_no_user_message(self):
        client = _mock_client()
        toolset = HindsightMemoryWrapper(bank_id="test", client=client, auto_recall=True)

        agent = MagicMock()
        agent.system_prompt = "base prompt"
        agent.run.return_value = {
            "messages": [],
            "last_message": ChatMessage.from_assistant("ok"),
        }

        toolset.run(agent, messages=[ChatMessage.from_system("system msg")])
        # No recall should have been attempted
        client.arecall.assert_not_called()

    def test_auto_recall_disabled_by_default(self):
        client = _mock_client()
        toolset = HindsightMemoryWrapper(bank_id="test", client=client)

        agent = MagicMock()
        agent.system_prompt = "base"
        agent.run.return_value = {
            "messages": [],
            "last_message": ChatMessage.from_assistant("ok"),
        }

        toolset.run(agent, messages=[ChatMessage.from_user("hello")])
        client.arecall.assert_not_called()

    def test_auto_recall_respects_max_recall_results(self):
        client = _mock_client()
        # Return many results
        client.arecall.return_value = _mock_recall_response([f"Memory {i}" for i in range(20)])
        toolset = HindsightMemoryWrapper(
            bank_id="test",
            client=client,
            auto_recall=True,
            max_recall_results=3,
        )

        agent = MagicMock()
        agent.system_prompt = "base"
        agent.run.return_value = {
            "messages": [],
            "last_message": ChatMessage.from_assistant("ok"),
        }

        toolset.run(agent, messages=[ChatMessage.from_user("hello")])
        call_kwargs = agent.run.call_args[1]
        prompt = call_kwargs["system_prompt"]
        # Should only have 3 memories, not 20
        assert "3. Memory 2" in prompt
        assert "4." not in prompt

    def test_auto_recall_no_memories_passes_base_prompt(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response([])
        toolset = HindsightMemoryWrapper(bank_id="test", client=client, auto_recall=True)

        agent = MagicMock()
        agent.system_prompt = "base prompt"
        agent.run.return_value = {
            "messages": [],
            "last_message": ChatMessage.from_assistant("ok"),
        }

        toolset.run(agent, messages=[ChatMessage.from_user("hello")])
        call_kwargs = agent.run.call_args[1]
        assert call_kwargs["system_prompt"] == "base prompt"


class TestToolsetAutoRetain:
    """Test auto-retain behavior in HindsightMemoryWrapper.run()."""

    def test_auto_retain_stores_user_and_assistant_messages(self):
        client = _mock_client()
        client.aretain.return_value = _mock_retain_response()
        toolset = HindsightMemoryWrapper(bank_id="test", client=client, auto_retain=True)

        agent = MagicMock()
        agent.system_prompt = None
        last_msg = ChatMessage.from_assistant("I'll remember that.")
        agent.run.return_value = {
            "messages": [last_msg],
            "last_message": last_msg,
        }

        toolset.run(agent, messages=[ChatMessage.from_user("My favorite color is blue.")])

        # Should have retained user message + assistant response
        assert client.aretain.call_count == 2
        calls = [c[1]["content"] for c in client.aretain.call_args_list]
        assert "My favorite color is blue." in calls
        assert "I'll remember that." in calls

    def test_auto_retain_includes_role_metadata(self):
        client = _mock_client()
        client.aretain.return_value = _mock_retain_response()
        toolset = HindsightMemoryWrapper(bank_id="test", client=client, auto_retain=True)

        agent = MagicMock()
        agent.system_prompt = None
        last_msg = ChatMessage.from_assistant("ok")
        agent.run.return_value = {
            "messages": [last_msg],
            "last_message": last_msg,
        }

        toolset.run(agent, messages=[ChatMessage.from_user("hello")])

        # Check metadata includes role
        user_call = client.aretain.call_args_list[0][1]
        assert user_call["metadata"]["role"] == "user"
        assert user_call["metadata"]["source"] == "haystack"
        assistant_call = client.aretain.call_args_list[1][1]
        assert assistant_call["metadata"]["role"] == "assistant"

    def test_auto_retain_skips_system_messages(self):
        client = _mock_client()
        client.aretain.return_value = _mock_retain_response()
        toolset = HindsightMemoryWrapper(bank_id="test", client=client, auto_retain=True)

        agent = MagicMock()
        agent.system_prompt = None
        last_msg = ChatMessage.from_assistant("ok")
        agent.run.return_value = {
            "messages": [last_msg],
            "last_message": last_msg,
        }

        toolset.run(
            agent,
            messages=[
                ChatMessage.from_system("system context"),
                ChatMessage.from_user("hello"),
            ],
        )

        # Only user + assistant, not system
        assert client.aretain.call_count == 2

    def test_auto_retain_disabled_by_default(self):
        client = _mock_client()
        toolset = HindsightMemoryWrapper(bank_id="test", client=client)

        agent = MagicMock()
        agent.system_prompt = None
        agent.run.return_value = {
            "messages": [],
            "last_message": ChatMessage.from_assistant("ok"),
        }

        toolset.run(agent, messages=[ChatMessage.from_user("hello")])
        client.aretain.assert_not_called()

    def test_auto_retain_handles_error_gracefully(self):
        client = _mock_client()
        client.aretain.side_effect = RuntimeError("connection refused")
        toolset = HindsightMemoryWrapper(bank_id="test", client=client, auto_retain=True)

        agent = MagicMock()
        agent.system_prompt = None
        last_msg = ChatMessage.from_assistant("ok")
        agent.run.return_value = {
            "messages": [last_msg],
            "last_message": last_msg,
        }

        # Should not raise even though retain fails
        result = toolset.run(agent, messages=[ChatMessage.from_user("hello")])
        assert result is not None


class TestToolsetAutoRecallAndRetain:
    """Test combined auto-recall + auto-retain behavior."""

    def test_both_auto_recall_and_retain(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["remembered fact"])
        client.aretain.return_value = _mock_retain_response()
        toolset = HindsightMemoryWrapper(
            bank_id="test",
            client=client,
            auto_recall=True,
            auto_retain=True,
        )

        agent = MagicMock()
        agent.system_prompt = "You are helpful."
        last_msg = ChatMessage.from_assistant("Here's what I know.")
        agent.run.return_value = {
            "messages": [last_msg],
            "last_message": last_msg,
        }

        toolset.run(agent, messages=[ChatMessage.from_user("Tell me about myself")])

        # Recall was called for prompt enrichment
        client.arecall.assert_called_once()
        # Retain was called for user + assistant messages
        assert client.aretain.call_count == 2
        # System prompt was enriched
        call_kwargs = agent.run.call_args[1]
        assert "remembered fact" in call_kwargs["system_prompt"]
