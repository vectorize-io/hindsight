"""Tests for MCPExtension loading and tool registration."""

from unittest.mock import MagicMock, patch

import pytest
from fastmcp import FastMCP

from hindsight_api import MemoryEngine
from hindsight_api.extensions.mcp import MCPExtension


class MockMCPExtension(MCPExtension):
    """Test extension that registers a custom tool."""

    def __init__(self, config=None):
        super().__init__(config)
        self.register_tools_called = False
        self.registered_mcp = None
        self.registered_memory = None

    def register_tools(self, mcp: FastMCP, memory: MemoryEngine) -> None:
        """Register a test tool to verify extension was called."""
        self.register_tools_called = True
        self.registered_mcp = mcp
        self.registered_memory = memory

        @mcp.tool()
        async def test_extension_tool(query: str) -> str:
            """A test tool registered by the extension."""
            return f"Extension tool received: {query}"


class TestMCPExtensionBase:
    """Tests for MCPExtension base class."""

    def test_mcp_extension_is_abstract(self):
        """MCPExtension.register_tools is abstract and must be implemented."""
        with pytest.raises(TypeError, match="abstract method"):
            MCPExtension()

    def test_subclass_can_be_instantiated(self):
        """Subclass implementing register_tools can be instantiated."""
        ext = MockMCPExtension()
        assert ext is not None
        assert ext.register_tools_called is False

    def test_register_tools_receives_mcp_and_memory(self):
        """register_tools receives FastMCP and MemoryEngine instances."""
        ext = MockMCPExtension()
        mcp = FastMCP("test")
        memory = MagicMock(spec=MemoryEngine)

        ext.register_tools(mcp, memory)

        assert ext.register_tools_called is True
        assert ext.registered_mcp is mcp
        assert ext.registered_memory is memory


class TestMCPExtensionLoading:
    """Tests for MCPExtension loading in create_mcp_server."""

    @pytest.fixture
    def mock_memory(self):
        """Create a mock MemoryEngine."""
        memory = MagicMock()
        memory._tenant_extension = MagicMock()
        memory._tenant_extension.authenticate_mcp = MagicMock()
        return memory

    def test_create_mcp_server_without_extension(self, mock_memory):
        """create_mcp_server works without MCPExtension configured."""
        from hindsight_api.api.mcp import create_mcp_server

        with patch("hindsight_api.api.mcp.load_extension", return_value=None):
            mcp = create_mcp_server(mock_memory)

        # Core tools should be registered
        tools = mcp._tool_manager._tools
        assert "retain" in tools
        assert "recall" in tools
        assert "reflect" in tools
        # Extension tool should NOT be present
        assert "test_extension_tool" not in tools

    def test_create_mcp_server_with_extension(self, mock_memory):
        """create_mcp_server loads and calls MCPExtension when configured."""
        from hindsight_api.api.mcp import create_mcp_server

        mock_ext = MockMCPExtension()

        with patch("hindsight_api.api.mcp.load_extension", return_value=mock_ext):
            mcp = create_mcp_server(mock_memory)

        # Extension should have been called
        assert mock_ext.register_tools_called is True

        # Core tools should still be registered
        tools = mcp._tool_manager._tools
        assert "retain" in tools
        assert "recall" in tools

        # Extension tool should also be registered
        assert "test_extension_tool" in tools

    @pytest.mark.asyncio
    async def test_extension_tool_is_callable(self, mock_memory):
        """Tool registered by extension can be called."""
        from hindsight_api.api.mcp import create_mcp_server

        mock_ext = MockMCPExtension()

        with patch("hindsight_api.api.mcp.load_extension", return_value=mock_ext):
            mcp = create_mcp_server(mock_memory)

        # Get and call the extension tool
        tools = mcp._tool_manager._tools
        test_tool = tools["test_extension_tool"]
        result = await test_tool.fn(query="hello world")

        assert result == "Extension tool received: hello world"

    def test_load_extension_called_with_correct_args(self, mock_memory):
        """load_extension is called with 'MCP' prefix and MCPExtension class."""
        from hindsight_api.api.mcp import create_mcp_server

        with patch("hindsight_api.api.mcp.load_extension") as mock_load:
            mock_load.return_value = None
            create_mcp_server(mock_memory)

        mock_load.assert_called_once_with("MCP", MCPExtension)


class TestMCPExtensionIntegration:
    """Integration tests verifying extension tools work end-to-end."""

    @pytest.fixture
    def mock_memory(self):
        """Create a mock MemoryEngine with required methods."""
        memory = MagicMock()
        memory.retain_batch_async = MagicMock()
        memory.submit_async_retain = MagicMock(return_value={"operation_id": "test-op"})
        memory.recall_async = MagicMock(return_value=MagicMock(results=[]))
        memory.reflect_async = MagicMock(return_value=MagicMock(text="reflection"))
        memory.list_banks = MagicMock(return_value=[])
        memory.get_bank_profile = MagicMock(return_value={"id": "test"})
        memory._tenant_extension = MagicMock()
        return memory

    def test_extension_tools_coexist_with_core_tools(self, mock_memory):
        """Extension tools are added alongside core tools, not replacing them."""
        from hindsight_api.api.mcp import create_mcp_server

        mock_ext = MockMCPExtension()

        with patch("hindsight_api.api.mcp.load_extension", return_value=mock_ext):
            mcp = create_mcp_server(mock_memory)

        tools = mcp._tool_manager._tools
        # All core tools present
        assert "retain" in tools
        assert "recall" in tools
        assert "reflect" in tools
        assert "list_banks" in tools
        assert "create_bank" in tools
        # Extension tool also present
        assert "test_extension_tool" in tools
        # Total: 5 core + 1 extension = 6 tools
        assert len(tools) == 6
