"""Test MCP server routing with dynamic bank_id."""

from unittest.mock import AsyncMock, MagicMock

import pytest


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
    from hindsight_api.api.mcp import _current_bank_id, get_current_bank_id

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
    from hindsight_api.api.mcp import _current_bank_id, create_mcp_server

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
    from hindsight_api.api.mcp import _current_api_key, get_current_api_key

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
    from hindsight_api.api.mcp import _current_api_key, _current_bank_id, create_mcp_server

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


@pytest.mark.asyncio
async def test_tenant_id_context_variable():
    """Test that tenant_id and api_key_id context variables work correctly."""
    from hindsight_api.api.mcp import (
        _current_api_key_id,
        _current_tenant_id,
        get_current_api_key_id,
        get_current_tenant_id,
    )

    # Initially None
    assert get_current_tenant_id() is None
    assert get_current_api_key_id() is None

    # Set and verify
    tenant_token = _current_tenant_id.set("org-123")
    key_id_token = _current_api_key_id.set("key-456")
    try:
        assert get_current_tenant_id() == "org-123"
        assert get_current_api_key_id() == "key-456"
    finally:
        _current_tenant_id.reset(tenant_token)
        _current_api_key_id.reset(key_id_token)

    # Back to None after reset
    assert get_current_tenant_id() is None
    assert get_current_api_key_id() is None


@pytest.mark.asyncio
async def test_mcp_tools_propagate_tenant_id_and_api_key_id(mock_memory):
    """Test that MCP tools propagate tenant_id and api_key_id to RequestContext.

    This is the critical test for usage metering: the UsageMeteringValidator reads
    request_context.tenant_id to identify the org for billing. Without this,
    MCP operations get tenant_id="unknown" and billing is skipped entirely.
    """
    from hindsight_api.api.mcp import (
        _current_api_key,
        _current_api_key_id,
        _current_bank_id,
        _current_tenant_id,
        create_mcp_server,
    )

    mcp_server = create_mcp_server(mock_memory)
    tools = mcp_server._tool_manager._tools

    # Set all context vars (simulating what MCPMiddleware does after authenticate_mcp)
    bank_token = _current_bank_id.set("test-bank")
    api_key_token = _current_api_key.set("hsk_test_key")
    tenant_token = _current_tenant_id.set("org-billing-123")
    key_id_token = _current_api_key_id.set("key-uuid-456")
    try:
        retain_tool = tools["retain"]
        await retain_tool.fn(content="test content", context="test_context", async_processing=False)

        # Verify the RequestContext passed to memory engine has all auth fields
        mock_memory.retain_batch_async.assert_called_once()
        request_context = mock_memory.retain_batch_async.call_args.kwargs["request_context"]
        assert request_context.api_key == "hsk_test_key"
        assert request_context.tenant_id == "org-billing-123"
        assert request_context.api_key_id == "key-uuid-456"
    finally:
        _current_bank_id.reset(bank_token)
        _current_api_key.reset(api_key_token)
        _current_tenant_id.reset(tenant_token)
        _current_api_key_id.reset(key_id_token)


def test_multi_bank_mode_exposes_all_tools(mock_memory):
    """Test that multi-bank mode exposes all tools including bank management and mental models."""
    from hindsight_api.api.mcp import create_mcp_server

    # Create server in multi-bank mode (default)
    mcp_server = create_mcp_server(mock_memory, multi_bank=True)
    tools = mcp_server._tool_manager._tools

    # Core tools
    assert "retain" in tools
    assert "recall" in tools
    assert "reflect" in tools
    assert "list_banks" in tools
    assert "create_bank" in tools

    # Mental model tools
    assert "list_mental_models" in tools
    assert "get_mental_model" in tools
    assert "create_mental_model" in tools
    assert "update_mental_model" in tools
    assert "delete_mental_model" in tools
    assert "refresh_mental_model" in tools


def test_single_bank_mode_excludes_bank_management_tools(mock_memory):
    """Test that single-bank mode only exposes bank-scoped tools."""
    from hindsight_api.api.mcp import create_mcp_server

    # Create server in single-bank mode
    mcp_server = create_mcp_server(mock_memory, multi_bank=False)
    tools = mcp_server._tool_manager._tools

    # Should have bank-scoped tools
    assert "retain" in tools
    assert "recall" in tools
    assert "reflect" in tools

    # Mental model tools should also be present (they're bank-scoped)
    assert "list_mental_models" in tools
    assert "get_mental_model" in tools
    assert "create_mental_model" in tools
    assert "update_mental_model" in tools
    assert "delete_mental_model" in tools
    assert "refresh_mental_model" in tools

    # Should NOT have bank management tools
    assert "list_banks" not in tools
    assert "create_bank" not in tools


def test_multi_bank_mode_tools_have_bank_id_param(mock_memory):
    """Test that multi-bank mode tools include bank_id parameter."""
    import inspect

    from hindsight_api.api.mcp import create_mcp_server

    mcp_server = create_mcp_server(mock_memory, multi_bank=True)
    tools = mcp_server._tool_manager._tools

    # All bank-scoped tools should have bank_id parameter in multi-bank mode
    bank_scoped_tools = [
        "retain",
        "recall",
        "reflect",
        "list_mental_models",
        "get_mental_model",
        "create_mental_model",
        "update_mental_model",
        "delete_mental_model",
        "refresh_mental_model",
    ]
    for tool_name in bank_scoped_tools:
        tool = tools[tool_name]
        sig = inspect.signature(tool.fn)
        assert "bank_id" in sig.parameters, f"{tool_name} should have bank_id param in multi-bank mode"


def test_single_bank_mode_tools_no_bank_id_param(mock_memory):
    """Test that single-bank mode tools do NOT include bank_id parameter."""
    import inspect

    from hindsight_api.api.mcp import create_mcp_server

    mcp_server = create_mcp_server(mock_memory, multi_bank=False)
    tools = mcp_server._tool_manager._tools

    # No bank-scoped tool should have bank_id parameter in single-bank mode
    bank_scoped_tools = [
        "retain",
        "recall",
        "reflect",
        "list_mental_models",
        "get_mental_model",
        "create_mental_model",
        "update_mental_model",
        "delete_mental_model",
        "refresh_mental_model",
    ]
    for tool_name in bank_scoped_tools:
        tool = tools[tool_name]
        sig = inspect.signature(tool.fn)
        assert "bank_id" not in sig.parameters, f"{tool_name} should NOT have bank_id param in single-bank mode"


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
    assert "list_mental_models" in multi_bank_tools
    assert "create_mental_model" in multi_bank_tools

    # Single-bank should only have scoped tools
    assert "retain" in single_bank_tools
    assert "recall" in single_bank_tools
    assert "list_mental_models" in single_bank_tools
    assert "create_mental_model" in single_bank_tools
    assert "list_banks" not in single_bank_tools
    assert "create_bank" not in single_bank_tools


@pytest.mark.asyncio
async def test_routing_logic_from_url_path():
    """Test that routing correctly selects server based on URL structure."""
    from unittest.mock import AsyncMock

    from hindsight_api.api.mcp import MCPMiddleware

    # Mock memory
    mock_memory = MagicMock()

    # Create middleware
    middleware = MCPMiddleware(None, mock_memory)

    # Simulate different URL patterns and verify routing
    test_cases = [
        # (path_after_stripping_mcp, expected_bank_id_from_path, expected_bank_id, description)
        ("/alice/messages", True, "alice", "Bank ID in path with endpoint"),
        ("/my-agent-123/", True, "my-agent-123", "Bank ID in path with trailing slash"),
        ("ciccio/messages", True, "ciccio", "Bank ID without leading slash (after mount strip)"),
        ("bob", True, "bob", "Bank ID only, no leading slash"),
        ("/messages", False, None, "MCP endpoint, no bank ID"),
        ("/", False, None, "Root path, no bank ID"),
    ]

    for path, expected_bank_from_path, expected_bank_id, description in test_cases:
        # Simulate the path parsing logic with leading slash normalization
        if path and not path.startswith("/"):
            path = "/" + path

        bank_id = None
        bank_id_from_path = False
        MCP_ENDPOINTS = {"sse", "messages"}

        if path.startswith("/") and len(path) > 1:
            parts = path[1:].split("/", 1)
            if parts[0] and parts[0] not in MCP_ENDPOINTS:
                bank_id = parts[0]
                bank_id_from_path = True

        assert bank_id_from_path == expected_bank_from_path, f"Failed for: {description} (path={path})"
        assert bank_id == expected_bank_id, f"Failed bank_id for: {description} (path={path}, got={bank_id})"
