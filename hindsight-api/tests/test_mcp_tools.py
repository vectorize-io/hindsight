"""Tests for the shared MCP tools module."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.mcp_tools import MCPToolsConfig, build_content_dict, parse_timestamp, register_mcp_tools


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
    """Create a mock MemoryEngine with mental model methods."""
    memory = MagicMock()
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

    def test_mental_model_tools_in_default_set(self):
        """Mental model tools should be in the default tools set when config.tools is None."""
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
