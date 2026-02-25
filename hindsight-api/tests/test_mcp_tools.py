"""Tests for the shared MCP tools module."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.mcp_tools import (
    MCPToolsConfig,
    _validate_mental_model_inputs,
    build_content_dict,
    parse_timestamp,
    register_mcp_tools,
)


class TestParseTimestamp:
    """Tests for parse_timestamp function."""

    def test_parse_iso_format_with_z(self):
        """Test parsing ISO format with Z suffix."""
        result = parse_timestamp("2024-01-15T10:30:00Z")
        assert result == datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

    def test_parse_iso_format_with_offset(self):
        """Test parsing ISO format with timezone offset."""
        result = parse_timestamp("2024-01-15T10:30:00+00:00")
        assert result == datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

    def test_parse_iso_format_without_tz(self):
        """Test parsing ISO format without timezone."""
        result = parse_timestamp("2024-01-15T10:30:00")
        assert result == datetime(2024, 1, 15, 10, 30, 0)

    def test_parse_invalid_format_raises(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_timestamp("not-a-date")
        assert "Invalid timestamp format" in str(exc_info.value)


class TestBuildContentDict:
    """Tests for build_content_dict function."""

    def test_basic_content(self):
        """Test building content dict with just content and context."""
        result, error = build_content_dict("test content", "test_context")
        assert error is None
        assert result == {"content": "test content", "context": "test_context"}

    def test_with_valid_timestamp(self):
        """Test building content dict with valid timestamp."""
        result, error = build_content_dict("test content", "test_context", "2024-01-15T10:30:00Z")
        assert error is None
        assert result["content"] == "test content"
        assert result["context"] == "test_context"
        assert result["event_date"] == datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

    def test_with_invalid_timestamp(self):
        """Test building content dict with invalid timestamp."""
        result, error = build_content_dict("test content", "test_context", "invalid")
        assert error is not None
        assert "Invalid timestamp format" in error
        assert result == {}

    def test_with_none_timestamp(self):
        """Test building content dict with None timestamp."""
        result, error = build_content_dict("test content", "test_context", None)
        assert error is None
        assert "event_date" not in result


# =========================================================================
# Mental Model MCP Tool Tests
# =========================================================================


@pytest.fixture
def mock_memory():
    """Create a mock MemoryEngine with all MCP tool methods."""
    memory = MagicMock()
    # Mental model methods
    memory.list_mental_models = AsyncMock(
        return_value=[
            {"id": "mm-1", "name": "Coding Prefs", "source_query": "coding preferences?", "content": "Prefers Python"},
            {"id": "mm-2", "name": "Goals", "source_query": "current goals?", "content": "Ship v2"},
        ]
    )
    memory.get_mental_model = AsyncMock(
        return_value={
            "id": "mm-1",
            "name": "Coding Prefs",
            "source_query": "coding preferences?",
            "content": "Prefers Python",
        }
    )
    memory.create_mental_model = AsyncMock(return_value={"id": "mm-new"})
    memory.submit_async_refresh_mental_model = AsyncMock(return_value={"operation_id": "op-123"})
    memory.update_mental_model = AsyncMock(
        return_value={
            "id": "mm-1",
            "name": "Updated Name",
            "source_query": "new query?",
            "content": "Updated",
        }
    )
    memory.delete_mental_model = AsyncMock(return_value=True)

    # Retain/recall/reflect
    memory.retain_batch_async = AsyncMock()
    memory.submit_async_retain = AsyncMock(return_value={"operation_id": "op-retain"})
    memory.recall_async = AsyncMock(return_value=MagicMock(model_dump_json=lambda indent=None: '{"results": []}', model_dump=lambda: {"results": []}))
    memory.reflect_async = AsyncMock(return_value=MagicMock(model_dump_json=lambda indent=None: '{"text": "reflection"}', model_dump=lambda: {"text": "reflection"}, structured_output=None))

    # Directive methods
    memory.list_directives = AsyncMock(return_value=[{"id": "dir-1", "name": "Be concise", "content": "Keep responses short"}])
    memory.create_directive = AsyncMock(return_value={"id": "dir-new", "name": "Test", "content": "Test content"})
    memory.delete_directive = AsyncMock(return_value=True)

    # Memory browsing methods
    memory.list_memory_units = AsyncMock(return_value={"items": [{"id": "mem-1", "content": "Test"}], "total": 1})
    memory.get_memory_unit = AsyncMock(return_value={"id": "mem-1", "content": "Test memory"})
    memory.delete_memory_unit = AsyncMock(return_value={"deleted_count": 1})

    # Document methods
    memory.list_documents = AsyncMock(return_value={"items": [{"id": "doc-1", "name": "Test Doc"}], "total": 1})
    memory.get_document = AsyncMock(return_value={"id": "doc-1", "name": "Test Doc"})
    memory.delete_document = AsyncMock(return_value={"deleted_memories": 5})

    # Operation methods
    memory.list_operations = AsyncMock(return_value={"items": [{"id": "op-1", "status": "completed"}]})
    memory.get_operation_status = AsyncMock(return_value={"id": "op-1", "status": "completed", "progress": 100})
    memory.cancel_operation = AsyncMock(return_value={"id": "op-1", "status": "cancelled"})

    # Tags & bank methods
    memory.list_tags = AsyncMock(return_value={"items": ["tag1", "tag2"], "total": 2})
    memory.get_bank_profile = AsyncMock(return_value={"id": "test-bank", "name": "Test Bank", "mission": "Testing"})
    memory.get_bank_stats = AsyncMock(return_value={"nodes": 100, "links": 50})
    memory.update_bank = AsyncMock(return_value={"id": "test-bank", "name": "Updated"})
    memory.delete_bank = AsyncMock(return_value={"deleted_memories": 10, "deleted_entities": 5})
    memory.list_banks = AsyncMock(return_value=[])

    return memory


@pytest.fixture
def mcp_server_with_mental_models(mock_memory):
    """Create a FastMCP server with mental model tools registered (multi-bank mode)."""
    from fastmcp import FastMCP

    mcp = FastMCP("test", stateless_http=True)
    config = MCPToolsConfig(
        bank_id_resolver=lambda: "test-bank",
        include_bank_id_param=True,
        tools={
            "list_mental_models",
            "get_mental_model",
            "create_mental_model",
            "update_mental_model",
            "delete_mental_model",
            "refresh_mental_model",
        },
    )
    register_mcp_tools(mcp, mock_memory, config)
    return mcp


@pytest.fixture
def mcp_server_single_bank(mock_memory):
    """Create a FastMCP server with mental model tools registered (single-bank mode)."""
    from fastmcp import FastMCP

    mcp = FastMCP("test")
    config = MCPToolsConfig(
        bank_id_resolver=lambda: "fixed-bank",
        include_bank_id_param=False,
        tools={
            "list_mental_models",
            "get_mental_model",
            "create_mental_model",
            "update_mental_model",
            "delete_mental_model",
            "refresh_mental_model",
        },
    )
    register_mcp_tools(mcp, mock_memory, config)
    return mcp


class TestMentalModelToolRegistration:
    """Test that mental model tools are registered correctly."""

    def test_tools_registered_multi_bank(self, mcp_server_with_mental_models):
        tools = mcp_server_with_mental_models._tool_manager._tools
        expected = {
            "list_mental_models",
            "get_mental_model",
            "create_mental_model",
            "update_mental_model",
            "delete_mental_model",
            "refresh_mental_model",
        }
        assert expected == set(tools.keys())

    def test_tools_registered_single_bank(self, mcp_server_single_bank):
        tools = mcp_server_single_bank._tool_manager._tools
        expected = {
            "list_mental_models",
            "get_mental_model",
            "create_mental_model",
            "update_mental_model",
            "delete_mental_model",
            "refresh_mental_model",
        }
        assert expected == set(tools.keys())

    @pytest.mark.asyncio
    async def test_list_mental_models_propagates_request_context(self, mock_memory):
        from fastmcp import FastMCP

        mcp = FastMCP("test", stateless_http=True)
        config = MCPToolsConfig(
            bank_id_resolver=lambda: "test-bank",
            api_key_resolver=lambda: "test-api-key",
            include_bank_id_param=True,
            tools={"list_mental_models"},
        )
        register_mcp_tools(mcp, mock_memory, config)
        await _tools(mcp)["list_mental_models"].fn()
        request_context = mock_memory.list_mental_models.call_args.kwargs["request_context"]
        assert request_context.api_key == "test-api-key"

    @pytest.mark.asyncio
    async def test_create_mental_model_propagates_request_context(self, mock_memory):
        from fastmcp import FastMCP

        mcp = FastMCP("test", stateless_http=True)
        config = MCPToolsConfig(
            bank_id_resolver=lambda: "test-bank",
            api_key_resolver=lambda: "test-api-key",
            include_bank_id_param=True,
            tools={"create_mental_model"},
        )
        register_mcp_tools(mcp, mock_memory, config)
        await _tools(mcp)["create_mental_model"].fn(name="Test", source_query="query")
        request_context = mock_memory.create_mental_model.call_args.kwargs["request_context"]
        assert request_context.api_key == "test-api-key"

    def test_mental_model_tools_in_default_set(self):
        """All tools should be in the default tools set when config.tools is None."""
        from fastmcp import FastMCP

        memory = MagicMock()
        # Mock all engine methods that tools reference
        memory.retain_batch_async = AsyncMock()
        memory.submit_async_retain = AsyncMock(return_value={"operation_id": "op"})
        memory.recall_async = AsyncMock(return_value=MagicMock(results=[]))
        memory.reflect_async = AsyncMock()
        memory.list_banks = AsyncMock(return_value=[])
        memory.get_bank_profile = AsyncMock(return_value={})
        memory.update_bank = AsyncMock()
        memory.list_mental_models = AsyncMock(return_value=[])
        memory.get_mental_model = AsyncMock()
        memory.create_mental_model = AsyncMock()
        memory.submit_async_refresh_mental_model = AsyncMock()
        memory.update_mental_model = AsyncMock()
        memory.delete_mental_model = AsyncMock()
        memory.list_directives = AsyncMock(return_value=[])
        memory.create_directive = AsyncMock()
        memory.delete_directive = AsyncMock()
        memory.list_memory_units = AsyncMock(return_value={})
        memory.get_memory_unit = AsyncMock()
        memory.delete_memory_unit = AsyncMock()
        memory.list_documents = AsyncMock(return_value={})
        memory.get_document = AsyncMock()
        memory.delete_document = AsyncMock()
        memory.list_operations = AsyncMock(return_value={})
        memory.get_operation_status = AsyncMock()
        memory.cancel_operation = AsyncMock()
        memory.list_tags = AsyncMock(return_value={})
        memory.get_bank_stats = AsyncMock(return_value={})
        memory.delete_bank = AsyncMock(return_value={})

        mcp = FastMCP("test", stateless_http=True)
        config = MCPToolsConfig(
            bank_id_resolver=lambda: "bank",
            include_bank_id_param=True,
            tools=None,  # Default - all tools
        )
        register_mcp_tools(mcp, memory, config)
        tools = mcp._tool_manager._tools
        assert "list_mental_models" in tools
        assert "create_mental_model" in tools
        assert "refresh_mental_model" in tools
        # New tools
        assert "list_directives" in tools
        assert "list_memories" in tools
        assert "list_documents" in tools
        assert "list_operations" in tools
        assert "list_tags" in tools
        assert "get_bank" in tools
        assert "get_bank_stats" in tools
        assert "update_bank" in tools
        assert "delete_bank" in tools
        assert "clear_memories" in tools
        assert len(tools) == 29


@pytest.fixture
def no_bank_mcp_server(mock_memory):
    """Create a multi-bank MCP server where bank_id_resolver returns None."""
    from fastmcp import FastMCP

    mcp = FastMCP("test", stateless_http=True)
    config = MCPToolsConfig(
        bank_id_resolver=lambda: None,
        include_bank_id_param=True,
        tools={
            "list_mental_models",
            "get_mental_model",
            "create_mental_model",
            "update_mental_model",
            "delete_mental_model",
            "refresh_mental_model",
        },
    )
    register_mcp_tools(mcp, mock_memory, config)
    return mcp


def _tools(mcp_server):
    """Helper to get tools dict from MCP server."""
    return mcp_server._tool_manager._tools


@pytest.mark.asyncio
class TestListMentalModels:
    async def test_list_multi_bank(self, mcp_server_with_mental_models, mock_memory):
        result = await _tools(mcp_server_with_mental_models)["list_mental_models"].fn()
        assert '"mm-1"' in result
        assert '"mm-2"' in result
        mock_memory.list_mental_models.assert_called_once()
        assert mock_memory.list_mental_models.call_args.kwargs["bank_id"] == "test-bank"

    async def test_list_with_bank_id_override(self, mcp_server_with_mental_models, mock_memory):
        """Explicit bank_id should override the resolver."""
        await _tools(mcp_server_with_mental_models)["list_mental_models"].fn(bank_id="other-bank")
        assert mock_memory.list_mental_models.call_args.kwargs["bank_id"] == "other-bank"

    async def test_list_with_tags(self, mcp_server_with_mental_models, mock_memory):
        await _tools(mcp_server_with_mental_models)["list_mental_models"].fn(tags=["work"])
        assert mock_memory.list_mental_models.call_args.kwargs["tags"] == ["work"]

    async def test_list_single_bank(self, mcp_server_single_bank, mock_memory):
        result = await _tools(mcp_server_single_bank)["list_mental_models"].fn()
        assert isinstance(result, dict)
        assert len(result["items"]) == 2
        assert mock_memory.list_mental_models.call_args.kwargs["bank_id"] == "fixed-bank"

    async def test_list_no_bank_returns_error(self, no_bank_mcp_server):
        result = await _tools(no_bank_mcp_server)["list_mental_models"].fn()
        assert "error" in result

    async def test_list_engine_error_multi_bank(self, mcp_server_with_mental_models, mock_memory):
        mock_memory.list_mental_models.side_effect = RuntimeError("DB connection lost")
        result = await _tools(mcp_server_with_mental_models)["list_mental_models"].fn()
        assert "error" in result
        assert "DB connection lost" in result

    async def test_list_engine_error_single_bank(self, mcp_server_single_bank, mock_memory):
        mock_memory.list_mental_models.side_effect = RuntimeError("DB connection lost")
        result = await _tools(mcp_server_single_bank)["list_mental_models"].fn()
        assert isinstance(result, dict)
        assert "error" in result


@pytest.mark.asyncio
class TestGetMentalModel:
    async def test_get_multi_bank(self, mcp_server_with_mental_models, mock_memory):
        result = await _tools(mcp_server_with_mental_models)["get_mental_model"].fn(mental_model_id="mm-1")
        assert '"mm-1"' in result
        assert mock_memory.get_mental_model.call_args.kwargs["mental_model_id"] == "mm-1"

    async def test_get_with_bank_id_override(self, mcp_server_with_mental_models, mock_memory):
        await _tools(mcp_server_with_mental_models)["get_mental_model"].fn(mental_model_id="mm-1", bank_id="other-bank")
        assert mock_memory.get_mental_model.call_args.kwargs["bank_id"] == "other-bank"

    async def test_get_not_found_multi_bank(self, mcp_server_with_mental_models, mock_memory):
        mock_memory.get_mental_model.return_value = None
        result = await _tools(mcp_server_with_mental_models)["get_mental_model"].fn(mental_model_id="missing")
        assert "not found" in result

    async def test_get_not_found_single_bank(self, mcp_server_single_bank, mock_memory):
        mock_memory.get_mental_model.return_value = None
        result = await _tools(mcp_server_single_bank)["get_mental_model"].fn(mental_model_id="missing")
        assert isinstance(result, dict)
        assert "not found" in result["error"]

    async def test_get_single_bank(self, mcp_server_single_bank, mock_memory):
        result = await _tools(mcp_server_single_bank)["get_mental_model"].fn(mental_model_id="mm-1")
        assert isinstance(result, dict)
        assert result["id"] == "mm-1"

    async def test_get_no_bank_returns_error(self, no_bank_mcp_server):
        result = await _tools(no_bank_mcp_server)["get_mental_model"].fn(mental_model_id="mm-1")
        assert "error" in result

    async def test_get_engine_error(self, mcp_server_with_mental_models, mock_memory):
        mock_memory.get_mental_model.side_effect = RuntimeError("DB error")
        result = await _tools(mcp_server_with_mental_models)["get_mental_model"].fn(mental_model_id="mm-1")
        assert "error" in result


@pytest.mark.asyncio
class TestCreateMentalModel:
    async def test_create_multi_bank(self, mcp_server_with_mental_models, mock_memory):
        result = await _tools(mcp_server_with_mental_models)["create_mental_model"].fn(
            name="Test Model",
            source_query="What are the user's preferences?",
        )
        assert '"mm-new"' in result
        assert '"op-123"' in result
        mock_memory.create_mental_model.assert_called_once()
        call_kwargs = mock_memory.create_mental_model.call_args.kwargs
        assert call_kwargs["name"] == "Test Model"
        assert call_kwargs["source_query"] == "What are the user's preferences?"
        assert call_kwargs["content"] == "Generating content..."
        # Verify async refresh was scheduled
        mock_memory.submit_async_refresh_mental_model.assert_called_once()
        assert mock_memory.submit_async_refresh_mental_model.call_args.kwargs["mental_model_id"] == "mm-new"

    async def test_create_with_custom_id(self, mcp_server_with_mental_models, mock_memory):
        await _tools(mcp_server_with_mental_models)["create_mental_model"].fn(
            name="Test", source_query="query", mental_model_id="custom-id"
        )
        assert mock_memory.create_mental_model.call_args.kwargs["mental_model_id"] == "custom-id"

    async def test_create_with_tags_and_max_tokens(self, mcp_server_with_mental_models, mock_memory):
        await _tools(mcp_server_with_mental_models)["create_mental_model"].fn(
            name="Test", source_query="query", tags=["work", "coding"], max_tokens=4096
        )
        call_kwargs = mock_memory.create_mental_model.call_args.kwargs
        assert call_kwargs["tags"] == ["work", "coding"]
        assert call_kwargs["max_tokens"] == 4096

    async def test_create_with_bank_id_override(self, mcp_server_with_mental_models, mock_memory):
        await _tools(mcp_server_with_mental_models)["create_mental_model"].fn(
            name="Test", source_query="query", bank_id="other-bank"
        )
        assert mock_memory.create_mental_model.call_args.kwargs["bank_id"] == "other-bank"
        assert mock_memory.submit_async_refresh_mental_model.call_args.kwargs["bank_id"] == "other-bank"

    async def test_create_single_bank(self, mcp_server_single_bank, mock_memory):
        result = await _tools(mcp_server_single_bank)["create_mental_model"].fn(name="Test", source_query="query")
        assert isinstance(result, dict)
        assert result["mental_model_id"] == "mm-new"
        assert result["operation_id"] == "op-123"

    async def test_create_no_bank_returns_error(self, no_bank_mcp_server):
        result = await _tools(no_bank_mcp_server)["create_mental_model"].fn(name="Test", source_query="query")
        assert "error" in result

    async def test_create_value_error_multi_bank(self, mcp_server_with_mental_models, mock_memory):
        """ValueError from engine (e.g. invalid ID format) should return error, not crash."""
        mock_memory.create_mental_model.side_effect = ValueError("ID must be alphanumeric lowercase")
        result = await _tools(mcp_server_with_mental_models)["create_mental_model"].fn(
            name="Test", source_query="query", mental_model_id="INVALID!!"
        )
        assert "alphanumeric" in result

    async def test_create_value_error_single_bank(self, mcp_server_single_bank, mock_memory):
        mock_memory.create_mental_model.side_effect = ValueError("ID must be alphanumeric lowercase")
        result = await _tools(mcp_server_single_bank)["create_mental_model"].fn(
            name="Test", source_query="query", mental_model_id="INVALID!!"
        )
        assert isinstance(result, dict)
        assert "alphanumeric" in result["error"]

    async def test_create_engine_error(self, mcp_server_with_mental_models, mock_memory):
        mock_memory.create_mental_model.side_effect = RuntimeError("DB error")
        result = await _tools(mcp_server_with_mental_models)["create_mental_model"].fn(
            name="Test", source_query="query"
        )
        assert "error" in result


@pytest.mark.asyncio
class TestUpdateMentalModel:
    async def test_update_multi_bank(self, mcp_server_with_mental_models, mock_memory):
        result = await _tools(mcp_server_with_mental_models)["update_mental_model"].fn(
            mental_model_id="mm-1", name="Updated Name"
        )
        assert '"Updated Name"' in result
        call_kwargs = mock_memory.update_mental_model.call_args.kwargs
        assert call_kwargs["name"] == "Updated Name"
        assert call_kwargs["source_query"] is None  # Not updated

    async def test_update_multiple_fields(self, mcp_server_with_mental_models, mock_memory):
        await _tools(mcp_server_with_mental_models)["update_mental_model"].fn(
            mental_model_id="mm-1", name="New Name", source_query="new query?", tags=["updated"], max_tokens=4096
        )
        call_kwargs = mock_memory.update_mental_model.call_args.kwargs
        assert call_kwargs["name"] == "New Name"
        assert call_kwargs["source_query"] == "new query?"
        assert call_kwargs["tags"] == ["updated"]
        assert call_kwargs["max_tokens"] == 4096

    async def test_update_with_bank_id_override(self, mcp_server_with_mental_models, mock_memory):
        await _tools(mcp_server_with_mental_models)["update_mental_model"].fn(
            mental_model_id="mm-1", name="X", bank_id="other-bank"
        )
        assert mock_memory.update_mental_model.call_args.kwargs["bank_id"] == "other-bank"

    async def test_update_not_found_multi_bank(self, mcp_server_with_mental_models, mock_memory):
        mock_memory.update_mental_model.return_value = None
        result = await _tools(mcp_server_with_mental_models)["update_mental_model"].fn(
            mental_model_id="missing", name="X"
        )
        assert "not found" in result

    async def test_update_single_bank(self, mcp_server_single_bank, mock_memory):
        result = await _tools(mcp_server_single_bank)["update_mental_model"].fn(mental_model_id="mm-1", name="Updated")
        assert isinstance(result, dict)
        assert mock_memory.update_mental_model.call_args.kwargs["bank_id"] == "fixed-bank"

    async def test_update_not_found_single_bank(self, mcp_server_single_bank, mock_memory):
        mock_memory.update_mental_model.return_value = None
        result = await _tools(mcp_server_single_bank)["update_mental_model"].fn(mental_model_id="missing", name="X")
        assert isinstance(result, dict)
        assert "not found" in result["error"]

    async def test_update_no_bank_returns_error(self, no_bank_mcp_server):
        result = await _tools(no_bank_mcp_server)["update_mental_model"].fn(mental_model_id="mm-1", name="X")
        assert "error" in result

    async def test_update_engine_error(self, mcp_server_with_mental_models, mock_memory):
        mock_memory.update_mental_model.side_effect = RuntimeError("DB error")
        result = await _tools(mcp_server_with_mental_models)["update_mental_model"].fn(mental_model_id="mm-1", name="X")
        assert "error" in result


@pytest.mark.asyncio
class TestDeleteMentalModel:
    async def test_delete_multi_bank(self, mcp_server_with_mental_models, mock_memory):
        result = await _tools(mcp_server_with_mental_models)["delete_mental_model"].fn(mental_model_id="mm-1")
        assert '"deleted"' in result
        assert mock_memory.delete_mental_model.call_args.kwargs["mental_model_id"] == "mm-1"

    async def test_delete_with_bank_id_override(self, mcp_server_with_mental_models, mock_memory):
        await _tools(mcp_server_with_mental_models)["delete_mental_model"].fn(
            mental_model_id="mm-1", bank_id="other-bank"
        )
        assert mock_memory.delete_mental_model.call_args.kwargs["bank_id"] == "other-bank"

    async def test_delete_not_found_multi_bank(self, mcp_server_with_mental_models, mock_memory):
        mock_memory.delete_mental_model.return_value = False
        result = await _tools(mcp_server_with_mental_models)["delete_mental_model"].fn(mental_model_id="missing")
        assert "not found" in result

    async def test_delete_not_found_single_bank(self, mcp_server_single_bank, mock_memory):
        mock_memory.delete_mental_model.return_value = False
        result = await _tools(mcp_server_single_bank)["delete_mental_model"].fn(mental_model_id="missing")
        assert isinstance(result, dict)
        assert "not found" in result["error"]

    async def test_delete_single_bank(self, mcp_server_single_bank, mock_memory):
        result = await _tools(mcp_server_single_bank)["delete_mental_model"].fn(mental_model_id="mm-1")
        assert isinstance(result, dict)
        assert result["status"] == "deleted"

    async def test_delete_no_bank_returns_error(self, no_bank_mcp_server):
        result = await _tools(no_bank_mcp_server)["delete_mental_model"].fn(mental_model_id="mm-1")
        assert "error" in result

    async def test_delete_engine_error(self, mcp_server_with_mental_models, mock_memory):
        mock_memory.delete_mental_model.side_effect = RuntimeError("DB error")
        result = await _tools(mcp_server_with_mental_models)["delete_mental_model"].fn(mental_model_id="mm-1")
        assert "error" in result


@pytest.mark.asyncio
class TestRefreshMentalModel:
    async def test_refresh_multi_bank(self, mcp_server_with_mental_models, mock_memory):
        result = await _tools(mcp_server_with_mental_models)["refresh_mental_model"].fn(mental_model_id="mm-1")
        assert '"op-123"' in result
        assert '"queued"' in result

    async def test_refresh_with_bank_id_override(self, mcp_server_with_mental_models, mock_memory):
        await _tools(mcp_server_with_mental_models)["refresh_mental_model"].fn(
            mental_model_id="mm-1", bank_id="other-bank"
        )
        assert mock_memory.submit_async_refresh_mental_model.call_args.kwargs["bank_id"] == "other-bank"

    async def test_refresh_not_found_multi_bank(self, mcp_server_with_mental_models, mock_memory):
        mock_memory.submit_async_refresh_mental_model.side_effect = ValueError("Mental model 'missing' not found")
        result = await _tools(mcp_server_with_mental_models)["refresh_mental_model"].fn(mental_model_id="missing")
        assert "not found" in result

    async def test_refresh_not_found_single_bank(self, mcp_server_single_bank, mock_memory):
        mock_memory.submit_async_refresh_mental_model.side_effect = ValueError("not found")
        result = await _tools(mcp_server_single_bank)["refresh_mental_model"].fn(mental_model_id="missing")
        assert isinstance(result, dict)
        assert "not found" in result["error"]

    async def test_refresh_single_bank(self, mcp_server_single_bank, mock_memory):
        result = await _tools(mcp_server_single_bank)["refresh_mental_model"].fn(mental_model_id="mm-1")
        assert isinstance(result, dict)
        assert result["operation_id"] == "op-123"

    async def test_refresh_no_bank_returns_error(self, no_bank_mcp_server):
        result = await _tools(no_bank_mcp_server)["refresh_mental_model"].fn(mental_model_id="mm-1")
        assert "error" in result

    async def test_refresh_engine_error(self, mcp_server_with_mental_models, mock_memory):
        mock_memory.submit_async_refresh_mental_model.side_effect = RuntimeError("DB error")
        result = await _tools(mcp_server_with_mental_models)["refresh_mental_model"].fn(mental_model_id="mm-1")
        assert "error" in result


class TestValidateMentalModelInputs:
    """Tests for the _validate_mental_model_inputs helper."""

    def test_valid_inputs(self):
        assert _validate_mental_model_inputs(name="Test", source_query="query", max_tokens=2048) is None

    def test_none_inputs(self):
        assert _validate_mental_model_inputs() is None

    def test_empty_name(self):
        result = _validate_mental_model_inputs(name="")
        assert result == "name cannot be empty"

    def test_whitespace_name(self):
        result = _validate_mental_model_inputs(name="   ")
        assert result == "name cannot be empty"

    def test_empty_source_query(self):
        result = _validate_mental_model_inputs(source_query="")
        assert result == "source_query cannot be empty"

    def test_whitespace_source_query(self):
        result = _validate_mental_model_inputs(source_query="  \t  ")
        assert result == "source_query cannot be empty"

    def test_max_tokens_too_low(self):
        result = _validate_mental_model_inputs(max_tokens=0)
        assert "max_tokens must be between 256 and 8192" in result

    def test_max_tokens_too_high(self):
        result = _validate_mental_model_inputs(max_tokens=10000)
        assert "max_tokens must be between 256 and 8192" in result

    def test_max_tokens_at_lower_bound(self):
        assert _validate_mental_model_inputs(max_tokens=256) is None

    def test_max_tokens_at_upper_bound(self):
        assert _validate_mental_model_inputs(max_tokens=8192) is None


@pytest.mark.asyncio
class TestMentalModelInputValidation:
    """Tests that validation is applied in create/update tools before engine calls."""

    async def test_create_empty_name_returns_error_multi_bank(self, mcp_server_with_mental_models, mock_memory):
        result = await _tools(mcp_server_with_mental_models)["create_mental_model"].fn(name="", source_query="query")
        assert "name cannot be empty" in result
        mock_memory.create_mental_model.assert_not_called()

    async def test_create_empty_source_query_returns_error_multi_bank(self, mcp_server_with_mental_models, mock_memory):
        result = await _tools(mcp_server_with_mental_models)["create_mental_model"].fn(name="Test", source_query="")
        assert "source_query cannot be empty" in result
        mock_memory.create_mental_model.assert_not_called()

    async def test_create_max_tokens_too_low_multi_bank(self, mcp_server_with_mental_models, mock_memory):
        result = await _tools(mcp_server_with_mental_models)["create_mental_model"].fn(
            name="Test", source_query="query", max_tokens=0
        )
        assert "max_tokens must be between 256 and 8192" in result
        mock_memory.create_mental_model.assert_not_called()

    async def test_create_max_tokens_too_high_single_bank(self, mcp_server_single_bank, mock_memory):
        result = await _tools(mcp_server_single_bank)["create_mental_model"].fn(
            name="Test", source_query="query", max_tokens=10000
        )
        assert isinstance(result, dict)
        assert "max_tokens must be between 256 and 8192" in result["error"]
        mock_memory.create_mental_model.assert_not_called()

    async def test_update_empty_name_returns_error_multi_bank(self, mcp_server_with_mental_models, mock_memory):
        result = await _tools(mcp_server_with_mental_models)["update_mental_model"].fn(mental_model_id="mm-1", name="")
        assert "name cannot be empty" in result
        mock_memory.update_mental_model.assert_not_called()

    async def test_update_empty_name_returns_error_single_bank(self, mcp_server_single_bank, mock_memory):
        result = await _tools(mcp_server_single_bank)["update_mental_model"].fn(mental_model_id="mm-1", name="  ")
        assert isinstance(result, dict)
        assert "name cannot be empty" in result["error"]
        mock_memory.update_mental_model.assert_not_called()

    async def test_not_found_error_includes_bank_id_multi_bank(self, mcp_server_with_mental_models, mock_memory):
        mock_memory.get_mental_model.return_value = None
        result = await _tools(mcp_server_with_mental_models)["get_mental_model"].fn(mental_model_id="missing")
        assert "test-bank" in result

    async def test_not_found_error_includes_bank_id_single_bank(self, mcp_server_single_bank, mock_memory):
        mock_memory.get_mental_model.return_value = None
        result = await _tools(mcp_server_single_bank)["get_mental_model"].fn(mental_model_id="missing")
        assert isinstance(result, dict)
        assert "fixed-bank" in result["error"]


# =========================================================================
# New Parameter Tests for Existing Tools
# =========================================================================


def _make_mcp_server(mock_memory, tools, include_bank_id=True):
    """Helper to create an MCP server with specific tools."""
    from fastmcp import FastMCP

    mcp = FastMCP("test", stateless_http=True)
    config = MCPToolsConfig(
        bank_id_resolver=lambda: "test-bank",
        include_bank_id_param=include_bank_id,
        tools=tools,
    )
    register_mcp_tools(mcp, mock_memory, config)
    return mcp


@pytest.mark.asyncio
class TestRetainNewParams:
    """Tests for new retain parameters: tags, metadata, document_id."""

    async def test_retain_with_tags(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"retain"})
        await _tools(mcp)["retain"].fn(content="test", tags=["user:123", "project:alpha"])
        call_args = mock_memory.submit_async_retain.call_args
        contents = call_args.kwargs["contents"]
        assert contents[0]["tags"] == ["user:123", "project:alpha"]

    async def test_retain_with_metadata(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"retain"})
        await _tools(mcp)["retain"].fn(content="test", metadata={"source": "slack"})
        call_args = mock_memory.submit_async_retain.call_args
        contents = call_args.kwargs["contents"]
        assert contents[0]["metadata"] == {"source": "slack"}

    async def test_retain_with_document_id(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"retain"})
        await _tools(mcp)["retain"].fn(content="test", document_id="doc-1")
        call_args = mock_memory.submit_async_retain.call_args
        contents = call_args.kwargs["contents"]
        assert contents[0]["document_id"] == "doc-1"

    async def test_retain_without_new_params_backward_compat(self, mock_memory):
        """Existing behavior preserved when new params not provided."""
        mcp = _make_mcp_server(mock_memory, {"retain"})
        await _tools(mcp)["retain"].fn(content="test")
        call_args = mock_memory.submit_async_retain.call_args
        contents = call_args.kwargs["contents"]
        assert "tags" not in contents[0]
        assert "metadata" not in contents[0]
        assert "document_id" not in contents[0]


@pytest.mark.asyncio
class TestRecallNewParams:
    """Tests for new recall parameters: budget, types, tags, tags_match, query_timestamp."""

    async def test_recall_default_budget_high(self, mock_memory):
        """Default budget should be HIGH (backward compat)."""
        from hindsight_api.engine.memory_engine import Budget

        mcp = _make_mcp_server(mock_memory, {"recall"})
        await _tools(mcp)["recall"].fn(query="test")
        call_kwargs = mock_memory.recall_async.call_args.kwargs
        assert call_kwargs["budget"] == Budget.HIGH

    async def test_recall_budget_low(self, mock_memory):
        from hindsight_api.engine.memory_engine import Budget

        mcp = _make_mcp_server(mock_memory, {"recall"})
        await _tools(mcp)["recall"].fn(query="test", budget="low")
        call_kwargs = mock_memory.recall_async.call_args.kwargs
        assert call_kwargs["budget"] == Budget.LOW

    async def test_recall_with_types(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"recall"})
        await _tools(mcp)["recall"].fn(query="test", types=["world"])
        call_kwargs = mock_memory.recall_async.call_args.kwargs
        assert call_kwargs["fact_type"] == ["world"]

    async def test_recall_default_types_all(self, mock_memory):
        from hindsight_api.engine.response_models import VALID_RECALL_FACT_TYPES

        mcp = _make_mcp_server(mock_memory, {"recall"})
        await _tools(mcp)["recall"].fn(query="test")
        call_kwargs = mock_memory.recall_async.call_args.kwargs
        assert call_kwargs["fact_type"] == list(VALID_RECALL_FACT_TYPES)

    async def test_recall_with_tags(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"recall"})
        await _tools(mcp)["recall"].fn(query="test", tags=["project:x"])
        call_kwargs = mock_memory.recall_async.call_args.kwargs
        assert call_kwargs["tags"] == ["project:x"]
        assert call_kwargs["tags_match"] == "any"

    async def test_recall_with_query_timestamp(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"recall"})
        await _tools(mcp)["recall"].fn(query="test", query_timestamp="2024-01-01T00:00:00Z")
        call_kwargs = mock_memory.recall_async.call_args.kwargs
        assert call_kwargs["question_date"] == datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
class TestReflectNewParams:
    """Tests for new reflect parameters: max_tokens, response_schema, tags, tags_match."""

    async def test_reflect_with_max_tokens(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"reflect"})
        await _tools(mcp)["reflect"].fn(query="test", max_tokens=2048)
        call_kwargs = mock_memory.reflect_async.call_args.kwargs
        assert call_kwargs["max_tokens"] == 2048

    async def test_reflect_with_response_schema(self, mock_memory):
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        mock_memory.reflect_async = AsyncMock(
            return_value=MagicMock(
                model_dump_json=lambda indent=None: '{"text": "reflection"}',
                model_dump=lambda: {"text": "reflection"},
                structured_output={"answer": "yes"},
            )
        )
        mcp = _make_mcp_server(mock_memory, {"reflect"})
        result = await _tools(mcp)["reflect"].fn(query="test", response_schema=schema)
        call_kwargs = mock_memory.reflect_async.call_args.kwargs
        assert call_kwargs["response_schema"] == schema
        # Multi-bank returns JSON string
        import json

        parsed = json.loads(result)
        assert parsed["structured_output"] == {"answer": "yes"}

    async def test_reflect_with_tags(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"reflect"})
        await _tools(mcp)["reflect"].fn(query="test", tags=["scope:work"], tags_match="all")
        call_kwargs = mock_memory.reflect_async.call_args.kwargs
        assert call_kwargs["tags"] == ["scope:work"]
        assert call_kwargs["tags_match"] == "all"

    async def test_reflect_without_tags_no_tags_in_kwargs(self, mock_memory):
        """When tags not provided, they should not be passed to engine."""
        mcp = _make_mcp_server(mock_memory, {"reflect"})
        await _tools(mcp)["reflect"].fn(query="test")
        call_kwargs = mock_memory.reflect_async.call_args.kwargs
        assert "tags" not in call_kwargs


@pytest.mark.asyncio
class TestMentalModelTrigger:
    """Tests for trigger_refresh_after_consolidation on create/update mental model."""

    async def test_create_with_trigger(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"create_mental_model"})
        await _tools(mcp)["create_mental_model"].fn(
            name="Test", source_query="query", trigger_refresh_after_consolidation=True
        )
        call_kwargs = mock_memory.create_mental_model.call_args.kwargs
        assert call_kwargs["trigger"] == {"refresh_after_consolidation": True}

    async def test_create_default_trigger_false(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"create_mental_model"})
        await _tools(mcp)["create_mental_model"].fn(name="Test", source_query="query")
        call_kwargs = mock_memory.create_mental_model.call_args.kwargs
        assert call_kwargs["trigger"] == {"refresh_after_consolidation": False}

    async def test_update_with_trigger(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"update_mental_model"})
        await _tools(mcp)["update_mental_model"].fn(
            mental_model_id="mm-1", trigger_refresh_after_consolidation=True
        )
        call_kwargs = mock_memory.update_mental_model.call_args.kwargs
        assert call_kwargs["trigger"] == {"refresh_after_consolidation": True}

    async def test_update_without_trigger_no_trigger_in_kwargs(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"update_mental_model"})
        await _tools(mcp)["update_mental_model"].fn(mental_model_id="mm-1", name="New Name")
        call_kwargs = mock_memory.update_mental_model.call_args.kwargs
        assert "trigger" not in call_kwargs


# =========================================================================
# Directive Tool Tests
# =========================================================================


@pytest.mark.asyncio
class TestDirectiveTools:
    async def test_list_directives_multi_bank(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"list_directives"}, include_bank_id=True)
        result = await _tools(mcp)["list_directives"].fn()
        assert '"dir-1"' in result
        mock_memory.list_directives.assert_called_once()
        assert mock_memory.list_directives.call_args[0][0] == "test-bank"

    async def test_list_directives_single_bank(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"list_directives"}, include_bank_id=False)
        result = await _tools(mcp)["list_directives"].fn()
        assert isinstance(result, dict)
        assert len(result["items"]) == 1

    async def test_create_directive(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"create_directive"}, include_bank_id=True)
        result = await _tools(mcp)["create_directive"].fn(name="Test", content="Be concise", priority=5)
        assert '"dir-new"' in result
        call_args = mock_memory.create_directive.call_args
        assert call_args[0][0] == "test-bank"
        assert call_args.kwargs["name"] == "Test"
        assert call_args.kwargs["content"] == "Be concise"
        assert call_args.kwargs["priority"] == 5

    async def test_delete_directive(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"delete_directive"}, include_bank_id=True)
        result = await _tools(mcp)["delete_directive"].fn(directive_id="dir-1")
        assert '"deleted"' in result
        assert mock_memory.delete_directive.call_args[0][1] == "dir-1"

    async def test_delete_directive_not_found(self, mock_memory):
        mock_memory.delete_directive.return_value = False
        mcp = _make_mcp_server(mock_memory, {"delete_directive"}, include_bank_id=True)
        result = await _tools(mcp)["delete_directive"].fn(directive_id="missing")
        assert "not found" in result


# =========================================================================
# Memory Browsing Tool Tests
# =========================================================================


@pytest.mark.asyncio
class TestMemoryBrowsingTools:
    async def test_list_memories_default(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"list_memories"}, include_bank_id=True)
        result = await _tools(mcp)["list_memories"].fn()
        assert '"mem-1"' in result
        call_kwargs = mock_memory.list_memory_units.call_args.kwargs
        assert call_kwargs["limit"] == 100
        assert call_kwargs["offset"] == 0

    async def test_list_memories_with_filters(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"list_memories"}, include_bank_id=True)
        await _tools(mcp)["list_memories"].fn(type="world", q="test query", limit=50)
        call_kwargs = mock_memory.list_memory_units.call_args.kwargs
        assert call_kwargs["fact_type"] == "world"
        assert call_kwargs["search_query"] == "test query"
        assert call_kwargs["limit"] == 50

    async def test_get_memory(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"get_memory"}, include_bank_id=True)
        result = await _tools(mcp)["get_memory"].fn(memory_id="mem-1")
        assert '"mem-1"' in result

    async def test_get_memory_not_found(self, mock_memory):
        mock_memory.get_memory_unit.return_value = None
        mcp = _make_mcp_server(mock_memory, {"get_memory"}, include_bank_id=True)
        result = await _tools(mcp)["get_memory"].fn(memory_id="missing")
        assert "not found" in result

    async def test_delete_memory(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"delete_memory"}, include_bank_id=True)
        result = await _tools(mcp)["delete_memory"].fn(memory_id="mem-1")
        assert '"deleted"' in result
        assert mock_memory.delete_memory_unit.call_args.kwargs["unit_id"] == "mem-1"

    async def test_list_memories_single_bank(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"list_memories"}, include_bank_id=False)
        result = await _tools(mcp)["list_memories"].fn()
        assert isinstance(result, dict)


# =========================================================================
# Document Tool Tests
# =========================================================================


@pytest.mark.asyncio
class TestDocumentTools:
    async def test_list_documents(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"list_documents"}, include_bank_id=True)
        result = await _tools(mcp)["list_documents"].fn()
        assert '"doc-1"' in result

    async def test_get_document(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"get_document"}, include_bank_id=True)
        result = await _tools(mcp)["get_document"].fn(document_id="doc-1")
        assert '"doc-1"' in result

    async def test_get_document_not_found(self, mock_memory):
        mock_memory.get_document.return_value = None
        mcp = _make_mcp_server(mock_memory, {"get_document"}, include_bank_id=True)
        result = await _tools(mcp)["get_document"].fn(document_id="missing")
        assert "not found" in result

    async def test_delete_document(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"delete_document"}, include_bank_id=True)
        result = await _tools(mcp)["delete_document"].fn(document_id="doc-1")
        assert '"deleted"' in result

    async def test_list_documents_single_bank(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"list_documents"}, include_bank_id=False)
        result = await _tools(mcp)["list_documents"].fn()
        assert isinstance(result, dict)


# =========================================================================
# Operation Tool Tests
# =========================================================================


@pytest.mark.asyncio
class TestOperationTools:
    async def test_list_operations(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"list_operations"}, include_bank_id=True)
        result = await _tools(mcp)["list_operations"].fn()
        assert '"op-1"' in result

    async def test_list_operations_with_status(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"list_operations"}, include_bank_id=True)
        await _tools(mcp)["list_operations"].fn(status="completed", limit=10)
        call_kwargs = mock_memory.list_operations.call_args.kwargs
        assert call_kwargs["status"] == "completed"
        assert call_kwargs["limit"] == 10

    async def test_get_operation(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"get_operation"}, include_bank_id=True)
        result = await _tools(mcp)["get_operation"].fn(operation_id="op-1")
        assert '"op-1"' in result

    async def test_cancel_operation(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"cancel_operation"}, include_bank_id=True)
        result = await _tools(mcp)["cancel_operation"].fn(operation_id="op-1")
        assert '"cancelled"' in result

    async def test_list_operations_single_bank(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"list_operations"}, include_bank_id=False)
        result = await _tools(mcp)["list_operations"].fn()
        assert isinstance(result, dict)


# =========================================================================
# Tags & Bank Tool Tests
# =========================================================================


@pytest.mark.asyncio
class TestTagsAndBankTools:
    async def test_list_tags(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"list_tags"}, include_bank_id=True)
        result = await _tools(mcp)["list_tags"].fn(q="project:*", limit=50)
        call_kwargs = mock_memory.list_tags.call_args.kwargs
        assert call_kwargs["pattern"] == "project:*"
        assert call_kwargs["limit"] == 50

    async def test_get_bank(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"get_bank"}, include_bank_id=True)
        result = await _tools(mcp)["get_bank"].fn()
        assert '"test-bank"' in result or "test-bank" in result

    async def test_get_bank_stats(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"get_bank_stats"}, include_bank_id=True)
        result = await _tools(mcp)["get_bank_stats"].fn()
        assert "100" in result  # nodes count

    async def test_update_bank(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"update_bank"}, include_bank_id=True)
        result = await _tools(mcp)["update_bank"].fn(name="New Name", mission="New Mission")
        call_kwargs = mock_memory.update_bank.call_args.kwargs
        assert call_kwargs["name"] == "New Name"
        assert call_kwargs["mission"] == "New Mission"

    async def test_delete_bank(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"delete_bank"}, include_bank_id=True)
        result = await _tools(mcp)["delete_bank"].fn()
        assert '"deleted"' in result
        mock_memory.delete_bank.assert_called_once()

    async def test_clear_memories(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"clear_memories"}, include_bank_id=True)
        result = await _tools(mcp)["clear_memories"].fn()
        assert '"cleared"' in result
        mock_memory.delete_bank.assert_called_once()

    async def test_clear_memories_with_type_filter(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"clear_memories"}, include_bank_id=True)
        await _tools(mcp)["clear_memories"].fn(type="world")
        call_kwargs = mock_memory.delete_bank.call_args.kwargs
        assert call_kwargs["fact_type"] == "world"

    async def test_list_tags_single_bank(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"list_tags"}, include_bank_id=False)
        result = await _tools(mcp)["list_tags"].fn()
        assert isinstance(result, dict)

    async def test_get_bank_single_bank(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"get_bank"}, include_bank_id=False)
        result = await _tools(mcp)["get_bank"].fn()
        assert isinstance(result, dict)

    async def test_delete_bank_single_bank(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"delete_bank"}, include_bank_id=False)
        result = await _tools(mcp)["delete_bank"].fn()
        assert isinstance(result, dict)
        assert result["status"] == "deleted"

    async def test_clear_memories_single_bank(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"clear_memories"}, include_bank_id=False)
        result = await _tools(mcp)["clear_memories"].fn()
        assert isinstance(result, dict)
        assert result["status"] == "cleared"


# =========================================================================
# Additional Error Handling & Edge Case Tests
# =========================================================================


@pytest.mark.asyncio
class TestOperationErrorHandling:
    """Error handling tests for operation tools."""

    async def test_get_operation_engine_error(self, mock_memory):
        mock_memory.get_operation_status.side_effect = RuntimeError("Operation not found")
        mcp = _make_mcp_server(mock_memory, {"get_operation"}, include_bank_id=True)
        result = await _tools(mcp)["get_operation"].fn(operation_id="missing")
        assert "error" in result
        assert "Operation not found" in result

    async def test_get_operation_engine_error_single_bank(self, mock_memory):
        mock_memory.get_operation_status.side_effect = RuntimeError("Operation not found")
        mcp = _make_mcp_server(mock_memory, {"get_operation"}, include_bank_id=False)
        result = await _tools(mcp)["get_operation"].fn(operation_id="missing")
        assert isinstance(result, dict)
        assert "Operation not found" in result["error"]

    async def test_cancel_operation_engine_error(self, mock_memory):
        mock_memory.cancel_operation.side_effect = RuntimeError("Cannot cancel completed operation")
        mcp = _make_mcp_server(mock_memory, {"cancel_operation"}, include_bank_id=True)
        result = await _tools(mcp)["cancel_operation"].fn(operation_id="op-done")
        assert "error" in result
        assert "Cannot cancel" in result

    async def test_cancel_operation_engine_error_single_bank(self, mock_memory):
        mock_memory.cancel_operation.side_effect = RuntimeError("Cannot cancel")
        mcp = _make_mcp_server(mock_memory, {"cancel_operation"}, include_bank_id=False)
        result = await _tools(mcp)["cancel_operation"].fn(operation_id="op-done")
        assert isinstance(result, dict)
        assert "Cannot cancel" in result["error"]


@pytest.mark.asyncio
class TestDeleteErrorHandling:
    """Error handling tests for delete operations."""

    async def test_delete_memory_engine_error(self, mock_memory):
        mock_memory.delete_memory_unit.side_effect = RuntimeError("DB error")
        mcp = _make_mcp_server(mock_memory, {"delete_memory"}, include_bank_id=True)
        result = await _tools(mcp)["delete_memory"].fn(memory_id="mem-1")
        assert "error" in result
        assert "DB error" in result

    async def test_delete_memory_single_bank(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"delete_memory"}, include_bank_id=False)
        result = await _tools(mcp)["delete_memory"].fn(memory_id="mem-1")
        assert isinstance(result, dict)
        assert result["status"] == "deleted"

    async def test_delete_document_engine_error(self, mock_memory):
        mock_memory.delete_document.side_effect = RuntimeError("DB error")
        mcp = _make_mcp_server(mock_memory, {"delete_document"}, include_bank_id=True)
        result = await _tools(mcp)["delete_document"].fn(document_id="doc-1")
        assert "error" in result
        assert "DB error" in result

    async def test_delete_document_single_bank(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"delete_document"}, include_bank_id=False)
        result = await _tools(mcp)["delete_document"].fn(document_id="doc-1")
        assert isinstance(result, dict)
        assert result["status"] == "deleted"


@pytest.mark.asyncio
class TestUpdateBankVariants:
    """Additional tests for update_bank tool."""

    async def test_update_bank_single_bank(self, mock_memory):
        mcp = _make_mcp_server(mock_memory, {"update_bank"}, include_bank_id=False)
        result = await _tools(mcp)["update_bank"].fn(name="New Name")
        assert isinstance(result, dict)
        call_kwargs = mock_memory.update_bank.call_args.kwargs
        assert call_kwargs["name"] == "New Name"

    async def test_update_bank_engine_error(self, mock_memory):
        mock_memory.update_bank.side_effect = RuntimeError("DB error")
        mcp = _make_mcp_server(mock_memory, {"update_bank"}, include_bank_id=True)
        result = await _tools(mcp)["update_bank"].fn(name="X")
        assert "error" in result

    async def test_get_bank_stats_engine_error(self, mock_memory):
        mock_memory.get_bank_stats.side_effect = RuntimeError("DB error")
        mcp = _make_mcp_server(mock_memory, {"get_bank_stats"}, include_bank_id=True)
        result = await _tools(mcp)["get_bank_stats"].fn()
        assert "error" in result


@pytest.mark.asyncio
class TestEmptyListReturns:
    """Tests that empty lists are handled gracefully."""

    async def test_list_memories_empty(self, mock_memory):
        mock_memory.list_memory_units.return_value = {"items": [], "total": 0}
        mcp = _make_mcp_server(mock_memory, {"list_memories"}, include_bank_id=True)
        result = await _tools(mcp)["list_memories"].fn()
        assert '"items": []' in result or "[]" in result

    async def test_list_documents_empty(self, mock_memory):
        mock_memory.list_documents.return_value = {"items": [], "total": 0}
        mcp = _make_mcp_server(mock_memory, {"list_documents"}, include_bank_id=True)
        result = await _tools(mcp)["list_documents"].fn()
        assert '"items": []' in result or "[]" in result

    async def test_list_operations_empty(self, mock_memory):
        mock_memory.list_operations.return_value = {"items": []}
        mcp = _make_mcp_server(mock_memory, {"list_operations"}, include_bank_id=True)
        result = await _tools(mcp)["list_operations"].fn()
        assert '"items": []' in result or "[]" in result

    async def test_list_directives_empty(self, mock_memory):
        mock_memory.list_directives.return_value = []
        mcp = _make_mcp_server(mock_memory, {"list_directives"}, include_bank_id=True)
        result = await _tools(mcp)["list_directives"].fn()
        assert "[]" in result

    async def test_list_tags_empty(self, mock_memory):
        mock_memory.list_tags.return_value = {"items": [], "total": 0}
        mcp = _make_mcp_server(mock_memory, {"list_tags"}, include_bank_id=True)
        result = await _tools(mcp)["list_tags"].fn()
        assert '"items": []' in result or "[]" in result

# =========================================================================
# Bank-Level Tool Filtering Tests
# =========================================================================


@pytest.fixture
def mock_memory_with_resolver():
    """Create a mock MemoryEngine with config resolver for bank filtering tests."""
    memory = MagicMock()
    memory.retain_batch_async = AsyncMock()
    memory.recall_async = AsyncMock(
        return_value=MagicMock(
            model_dump_json=lambda indent=None: '{"results": []}',
            model_dump=lambda: {"results": []},
        )
    )
    memory._config_resolver = MagicMock()
    memory._config_resolver.get_bank_config = AsyncMock(return_value={})
    return memory


class TestBankToolFiltering:
    """Tests for bank-level mcp_enabled_tools filtering via _apply_bank_tool_filtering."""

    @pytest.mark.asyncio
    async def test_disallowed_tool_raises_error(self, mock_memory_with_resolver):
        """Tool not in bank's mcp_enabled_tools list is hidden from get_tools()."""
        from fastmcp import FastMCP

        mock_memory_with_resolver._config_resolver.get_bank_config = AsyncMock(
            return_value={"mcp_enabled_tools": ["retain"]}
        )

        mcp = FastMCP("test")
        config = MCPToolsConfig(
            bank_id_resolver=lambda: "test-bank",
            include_bank_id_param=False,
            tools={"retain", "recall"},
        )
        register_mcp_tools(mcp, mock_memory_with_resolver, config)

        # Both tools are registered in the manager's internal dict
        assert "recall" in mcp._tool_manager._tools

        # But get_tools() (used by tools/list and tools/call) filters it out
        visible = await mcp._tool_manager.get_tools()
        assert "retain" in visible
        assert "recall" not in visible

    @pytest.mark.asyncio
    async def test_allowed_tool_remains_visible(self, mock_memory_with_resolver):
        """Tool in bank's mcp_enabled_tools list stays visible in get_tools()."""
        from fastmcp import FastMCP

        mock_memory_with_resolver._config_resolver.get_bank_config = AsyncMock(
            return_value={"mcp_enabled_tools": ["retain", "recall"]}
        )

        mcp = FastMCP("test")
        config = MCPToolsConfig(
            bank_id_resolver=lambda: "test-bank",
            include_bank_id_param=False,
            tools={"retain", "recall"},
        )
        register_mcp_tools(mcp, mock_memory_with_resolver, config)

        visible = await mcp._tool_manager.get_tools()
        assert "retain" in visible
        assert "recall" in visible

    @pytest.mark.asyncio
    async def test_no_filter_when_mcp_enabled_tools_absent(self, mock_memory_with_resolver):
        """When bank config has no mcp_enabled_tools key, all tools remain visible."""
        from fastmcp import FastMCP

        mock_memory_with_resolver._config_resolver.get_bank_config = AsyncMock(return_value={})

        mcp = FastMCP("test")
        config = MCPToolsConfig(
            bank_id_resolver=lambda: "test-bank",
            include_bank_id_param=False,
            tools={"retain", "recall"},
        )
        register_mcp_tools(mcp, mock_memory_with_resolver, config)

        visible = await mcp._tool_manager.get_tools()
        assert "retain" in visible
        assert "recall" in visible

    @pytest.mark.asyncio
    async def test_filter_skipped_when_no_bank_id(self, mock_memory_with_resolver):
        """When bank_id resolver returns None, config is not fetched and all tools are visible."""
        from fastmcp import FastMCP

        mock_memory_with_resolver._config_resolver.get_bank_config = AsyncMock(
            return_value={"mcp_enabled_tools": ["retain"]}  # Would block recall
        )

        mcp = FastMCP("test")
        config = MCPToolsConfig(
            bank_id_resolver=lambda: None,  # No bank_id context
            include_bank_id_param=False,
            tools={"retain", "recall"},
        )
        register_mcp_tools(mcp, mock_memory_with_resolver, config)

        visible = await mcp._tool_manager.get_tools()
        # Filter bypassed — config resolver was never consulted, all tools visible
        assert "recall" in visible
        mock_memory_with_resolver._config_resolver.get_bank_config.assert_not_called()
