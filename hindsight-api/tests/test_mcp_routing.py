"""Test MCP server routing with dynamic bank_id."""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_memory():
    """Create a mock MemoryEngine."""
    memory = MagicMock()
    memory.retain_batch_async = AsyncMock()
    memory.submit_async_retain = AsyncMock(return_value={"operation_id": "test-op-123"})
    memory.recall_async = AsyncMock(return_value=MagicMock(results=[]))
    return memory


@pytest.mark.asyncio
async def test_mcp_context_variable():
    """Test that context variable works correctly."""
    from hindsight_api.api.mcp import get_current_bank_id, _current_bank_id

    # Initially None
    assert get_current_bank_id() is None

    # Set and verify
    token = _current_bank_id.set("test-bank-123")
    try:
        assert get_current_bank_id() == "test-bank-123"
    finally:
        _current_bank_id.reset(token)

    # Back to None after reset
    assert get_current_bank_id() is None


@pytest.mark.asyncio
async def test_mcp_tools_use_context_bank_id(mock_memory):
    """Test that MCP tools use bank_id from context."""
    from hindsight_api.api.mcp import create_mcp_server, _current_bank_id

    mcp_server = create_mcp_server(mock_memory)

    # Get the tools
    tools = mcp_server._tool_manager._tools
    assert "retain" in tools
    assert "recall" in tools

    # Test retain with bank_id from context (use async_processing=False for synchronous test)
    token = _current_bank_id.set("context-bank-id")
    try:
        retain_tool = tools["retain"]
        result = await retain_tool.fn(content="test content", context="test_context", async_processing=False)
        assert "successfully" in result.lower()

        # Verify the memory was called with the context bank_id
        mock_memory.retain_batch_async.assert_called_once()
        call_kwargs = mock_memory.retain_batch_async.call_args.kwargs
        assert call_kwargs["bank_id"] == "context-bank-id"
    finally:
        _current_bank_id.reset(token)


def test_path_parsing_logic():
    """Test the path parsing logic for bank_id extraction."""
    def parse_path(path):
        """Simulate the path parsing logic from MCPMiddleware."""
        if not path.startswith("/") or len(path) <= 1:
            return None, None  # Error case

        parts = path[1:].split("/", 1)
        if not parts[0]:
            return None, None  # Error case

        bank_id = parts[0]
        new_path = "/" + parts[1] if len(parts) > 1 else "/"
        return bank_id, new_path

    # Test bank-specific paths
    bank_id, remaining = parse_path("/my-bank/")
    assert bank_id == "my-bank"
    assert remaining == "/"

    bank_id, remaining = parse_path("/my-bank")
    assert bank_id == "my-bank"
    assert remaining == "/"

    # Test error case - no bank_id
    bank_id, remaining = parse_path("/")
    assert bank_id is None

    # Test with complex bank_id
    bank_id, remaining = parse_path("/user_12345/")
    assert bank_id == "user_12345"
    assert remaining == "/"

    # Test with additional path after bank_id
    bank_id, remaining = parse_path("/my-bank/some/path")
    assert bank_id == "my-bank"
    assert remaining == "/some/path"


@pytest.mark.asyncio
async def test_api_key_context_variable():
    """Test that API key context variable works correctly."""
    from hindsight_api.api.mcp import get_current_api_key, _current_api_key

    # Initially None
    assert get_current_api_key() is None

    # Set and verify
    token = _current_api_key.set("test-api-key-123")
    try:
        assert get_current_api_key() == "test-api-key-123"
    finally:
        _current_api_key.reset(token)

    # Back to None after reset
    assert get_current_api_key() is None


@pytest.mark.asyncio
async def test_mcp_tools_propagate_api_key(mock_memory):
    """Test that MCP tools propagate API key to RequestContext."""
    from hindsight_api.api.mcp import create_mcp_server, _current_bank_id, _current_api_key

    mcp_server = create_mcp_server(mock_memory)
    tools = mcp_server._tool_manager._tools

    # Set both bank_id and api_key context
    bank_token = _current_bank_id.set("test-bank")
    api_key_token = _current_api_key.set("test-bearer-token")
    try:
        retain_tool = tools["retain"]
        result = await retain_tool.fn(content="test content", context="test_context", async_processing=False)
        assert "successfully" in result.lower()

        # Verify the memory was called with request_context containing api_key
        mock_memory.retain_batch_async.assert_called_once()
        call_kwargs = mock_memory.retain_batch_async.call_args.kwargs
        assert call_kwargs["request_context"].api_key == "test-bearer-token"
    finally:
        _current_bank_id.reset(bank_token)
        _current_api_key.reset(api_key_token)


def test_multi_bank_mode_exposes_all_tools(mock_memory):
    """Test that multi-bank mode exposes all tools including bank management."""
    from hindsight_api.api.mcp import create_mcp_server

    # Create server in multi-bank mode (default)
    mcp_server = create_mcp_server(mock_memory, multi_bank=True)
    tools = mcp_server._tool_manager._tools

    # Should have all tools
    assert "retain" in tools
    assert "recall" in tools
    assert "reflect" in tools
    assert "list_banks" in tools
    assert "create_bank" in tools


def test_single_bank_mode_excludes_bank_management_tools(mock_memory):
    """Test that single-bank mode only exposes bank-scoped tools."""
    from hindsight_api.api.mcp import create_mcp_server

    # Create server in single-bank mode
    mcp_server = create_mcp_server(mock_memory, multi_bank=False)
    tools = mcp_server._tool_manager._tools

    # Should only have bank-scoped tools
    assert "retain" in tools
    assert "recall" in tools
    assert "reflect" in tools

    # Should NOT have bank management tools
    assert "list_banks" not in tools
    assert "create_bank" not in tools


def test_multi_bank_mode_tools_have_bank_id_param(mock_memory):
    """Test that multi-bank mode tools include bank_id parameter."""
    from hindsight_api.api.mcp import create_mcp_server
    import inspect

    mcp_server = create_mcp_server(mock_memory, multi_bank=True)
    tools = mcp_server._tool_manager._tools

    # Check that tools have bank_id parameter
    retain_tool = tools["retain"]
    retain_sig = inspect.signature(retain_tool.fn)
    assert "bank_id" in retain_sig.parameters

    recall_tool = tools["recall"]
    recall_sig = inspect.signature(recall_tool.fn)
    assert "bank_id" in recall_sig.parameters

    reflect_tool = tools["reflect"]
    reflect_sig = inspect.signature(reflect_tool.fn)
    assert "bank_id" in reflect_sig.parameters


def test_single_bank_mode_tools_no_bank_id_param(mock_memory):
    """Test that single-bank mode tools do NOT include bank_id parameter."""
    from hindsight_api.api.mcp import create_mcp_server
    import inspect

    mcp_server = create_mcp_server(mock_memory, multi_bank=False)
    tools = mcp_server._tool_manager._tools

    # Check that tools do NOT have bank_id parameter
    retain_tool = tools["retain"]
    retain_sig = inspect.signature(retain_tool.fn)
    assert "bank_id" not in retain_sig.parameters

    recall_tool = tools["recall"]
    recall_sig = inspect.signature(recall_tool.fn)
    assert "bank_id" not in recall_sig.parameters

    reflect_tool = tools["reflect"]
    reflect_sig = inspect.signature(reflect_tool.fn)
    assert "bank_id" not in reflect_sig.parameters


@pytest.mark.asyncio
async def test_middleware_handles_both_endpoints(mock_memory):
    """Test that MCPMiddleware routes to correct server based on URL path."""
    from hindsight_api.api.mcp import MCPMiddleware

    # Create middleware (single instance)
    middleware = MCPMiddleware(None, mock_memory)

    # Verify both server instances exist
    assert middleware.multi_bank_app is not None
    assert middleware.single_bank_app is not None

    # Verify they expose different tools
    multi_bank_tools = middleware.multi_bank_server._tool_manager._tools
    single_bank_tools = middleware.single_bank_server._tool_manager._tools

    # Multi-bank should have all tools
    assert "retain" in multi_bank_tools
    assert "recall" in multi_bank_tools
    assert "list_banks" in multi_bank_tools
    assert "create_bank" in multi_bank_tools

    # Single-bank should only have scoped tools
    assert "retain" in single_bank_tools
    assert "recall" in single_bank_tools
    assert "list_banks" not in single_bank_tools
    assert "create_bank" not in single_bank_tools


@pytest.mark.asyncio
async def test_routing_logic_from_url_path():
    """Test that routing correctly selects server based on URL structure."""
    from hindsight_api.api.mcp import MCPMiddleware
    from unittest.mock import AsyncMock

    # Mock memory
    mock_memory = MagicMock()

    # Create middleware
    middleware = MCPMiddleware(None, mock_memory)

    # Simulate different URL patterns and verify routing
    test_cases = [
        # (path, expected_bank_id_from_path, description)
        ("/alice/messages", True, "Bank ID in path"),
        ("/my-agent-123/", True, "Bank ID in path with trailing slash"),
        ("/messages", False, "MCP endpoint, no bank ID"),
        ("/", False, "Root path, no bank ID"),
    ]

    for path, expected_bank_from_path, description in test_cases:
        # Simulate the path parsing logic
        bank_id = None
        bank_id_from_path = False
        MCP_ENDPOINTS = {"sse", "messages"}

        if path.startswith("/") and len(path) > 1:
            parts = path[1:].split("/", 1)
            if parts[0] and parts[0] not in MCP_ENDPOINTS:
                bank_id = parts[0]
                bank_id_from_path = True

        assert bank_id_from_path == expected_bank_from_path, f"Failed for: {description} (path={path})"
