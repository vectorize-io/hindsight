"""Test MCP server routing with dynamic bank_id."""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from hindsight_api.extensions.builtin.tenant import ApiKeyTenantExtension, DefaultTenantExtension


@pytest.fixture
def mock_memory():
    """Create a mock MemoryEngine."""
    memory = MagicMock()
    memory.retain_batch_async = AsyncMock()
    memory.submit_async_retain = AsyncMock(return_value={"operation_id": "test-op-123"})
    memory.recall_async = AsyncMock(return_value=MagicMock(results=[]))
    return memory


def _make_scope(path="/mcp", headers=None):
    """Build a minimal ASGI HTTP scope."""
    raw_headers = []
    for k, v in (headers or {}).items():
        raw_headers.append((k.lower().encode(), v.encode()))
    # MCP requires Accept header
    raw_headers.append((b"accept", b"application/json, text/event-stream"))
    raw_headers.append((b"content-type", b"application/json"))
    return {
        "type": "http",
        "path": path,
        "root_path": "",
        "headers": raw_headers,
    }


async def _collect_response(middleware, scope, body=b""):
    """Send a request through the middleware and collect the response status and body."""
    status = None
    response_body = b""

    async def receive():
        return {"type": "http.request", "body": body}

    async def send(message):
        nonlocal status, response_body
        if message["type"] == "http.response.start":
            status = message["status"]
        elif message["type"] == "http.response.body":
            response_body += message.get("body", b"")

    await middleware(scope, receive, send)
    return status, response_body


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


# --- Middleware authentication tests ---


@pytest.fixture
def memory_with_api_key_auth():
    """Create a mock MemoryEngine with ApiKeyTenantExtension."""
    memory = MagicMock()
    memory._tenant_extension = ApiKeyTenantExtension({"api_key": "test-secret-123"})
    return memory


@pytest.fixture
def memory_with_default_auth():
    """Create a mock MemoryEngine with DefaultTenantExtension (no auth)."""
    memory = MagicMock()
    memory._tenant_extension = DefaultTenantExtension({})
    return memory


@pytest.mark.asyncio
async def test_mcp_middleware_rejects_no_auth(memory_with_api_key_auth):
    """MCP middleware returns 401 when no Authorization header is provided."""
    from hindsight_api.api.mcp import MCPMiddleware

    middleware = MCPMiddleware(None, memory_with_api_key_auth)
    scope = _make_scope(path="/mcp")
    status, body = await _collect_response(middleware, scope)

    assert status == 401
    assert b"Authentication failed" in body


@pytest.mark.asyncio
async def test_mcp_middleware_rejects_wrong_key(memory_with_api_key_auth):
    """MCP middleware returns 401 when an invalid API key is provided."""
    from hindsight_api.api.mcp import MCPMiddleware

    middleware = MCPMiddleware(None, memory_with_api_key_auth)
    scope = _make_scope(path="/mcp", headers={"Authorization": "Bearer wrong-key"})
    status, body = await _collect_response(middleware, scope)

    assert status == 401
    assert b"Authentication failed" in body


@pytest.mark.asyncio
async def test_mcp_middleware_accepts_valid_key(memory_with_api_key_auth):
    """MCP middleware passes through when a valid API key is provided."""
    from hindsight_api.api.mcp import MCPMiddleware

    middleware = MCPMiddleware(None, memory_with_api_key_auth)
    scope = _make_scope(
        path="/mcp",
        headers={"Authorization": "Bearer test-secret-123"},
    )
    # FastMCP raises RuntimeError because its lifespan isn't initialized in unit tests.
    # If we get that error, auth passed — the request made it past the middleware.
    with pytest.raises(RuntimeError, match="Task group is not initialized"):
        await _collect_response(
            middleware,
            scope,
            body=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode(),
        )


@pytest.mark.asyncio
async def test_mcp_middleware_default_tenant_no_auth_required(memory_with_default_auth):
    """MCP middleware passes through with no auth when DefaultTenantExtension is used."""
    from hindsight_api.api.mcp import MCPMiddleware

    middleware = MCPMiddleware(None, memory_with_default_auth)
    scope = _make_scope(path="/mcp")
    # Same as above — RuntimeError means auth passed and request reached FastMCP internals.
    with pytest.raises(RuntimeError, match="Task group is not initialized"):
        await _collect_response(
            middleware,
            scope,
            body=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode(),
        )


class MultiTenantTestExtension:
    """Test extension that maps API keys to tenant schemas."""

    def __init__(self, key_to_schema: dict[str, str]):
        self.key_to_schema = key_to_schema

    async def authenticate(self, context):
        from hindsight_api.extensions.tenant import AuthenticationError, TenantContext

        if not context.api_key:
            raise AuthenticationError("API key required")
        schema = self.key_to_schema.get(context.api_key)
        if not schema:
            raise AuthenticationError("Invalid API key")
        return TenantContext(schema_name=schema)

    async def authenticate_mcp(self, context):
        """MCP auth delegates to authenticate by default."""
        return await self.authenticate(context)


@pytest.mark.asyncio
async def test_mcp_middleware_sets_schema_from_tenant_context():
    """MCP middleware sets _current_schema from tenant context for multi-tenant isolation."""
    from hindsight_api.api.mcp import MCPMiddleware
    from hindsight_api.engine.memory_engine import _current_schema

    # Create extension that maps keys to different schemas
    tenant_ext = MultiTenantTestExtension({
        "key-for-tenant-alpha": "tenant_alpha",
        "key-for-tenant-beta": "tenant_beta",
    })

    memory = MagicMock()
    memory._tenant_extension = tenant_ext

    middleware = MCPMiddleware(None, memory)

    # Track what schema was set during request processing
    captured_schema = None

    # Patch the mcp_app to capture the schema instead of actually processing
    async def mock_mcp_app(scope, receive, send):
        nonlocal captured_schema
        captured_schema = _current_schema.get()
        # Send a minimal response
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    middleware.mcp_app = mock_mcp_app

    # Test tenant alpha
    scope = _make_scope(path="/mcp", headers={"Authorization": "Bearer key-for-tenant-alpha"})
    await _collect_response(middleware, scope)
    assert captured_schema == "tenant_alpha", f"Expected tenant_alpha, got {captured_schema}"

    # Test tenant beta
    scope = _make_scope(path="/mcp", headers={"Authorization": "Bearer key-for-tenant-beta"})
    await _collect_response(middleware, scope)
    assert captured_schema == "tenant_beta", f"Expected tenant_beta, got {captured_schema}"


@pytest.mark.asyncio
async def test_mcp_legacy_auth_token(monkeypatch):
    """MCP middleware supports legacy MCP_AUTH_TOKEN for backwards compatibility."""
    import hindsight_api.api.mcp as mcp_module
    from hindsight_api.api.mcp import MCPMiddleware

    # Set legacy auth token
    monkeypatch.setattr(mcp_module, "MCP_AUTH_TOKEN", "legacy-secret-token")

    memory = MagicMock()
    # Even with ApiKeyTenantExtension, legacy token should work
    memory._tenant_extension = ApiKeyTenantExtension({"api_key": "different-key"})

    middleware = MCPMiddleware(None, memory)

    # Wrong token should fail
    scope = _make_scope(path="/mcp", headers={"Authorization": "Bearer wrong-token"})
    status, body = await _collect_response(middleware, scope)
    assert status == 401
    assert b"Invalid authentication token" in body

    # Correct legacy token should pass
    scope = _make_scope(path="/mcp", headers={"Authorization": "Bearer legacy-secret-token"})
    with pytest.raises(RuntimeError, match="Task group is not initialized"):
        await _collect_response(
            middleware,
            scope,
            body=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode(),
        )


@pytest.mark.asyncio
async def test_mcp_auth_disabled_flag():
    """ApiKeyTenantExtension with mcp_auth_disabled=true skips MCP auth."""
    from hindsight_api.api.mcp import MCPMiddleware

    memory = MagicMock()
    # Create extension with MCP auth disabled
    memory._tenant_extension = ApiKeyTenantExtension({
        "api_key": "test-secret-123",
        "mcp_auth_disabled": "true",
    })

    middleware = MCPMiddleware(None, memory)

    # No auth should pass when mcp_auth_disabled=true
    scope = _make_scope(path="/mcp")
    with pytest.raises(RuntimeError, match="Task group is not initialized"):
        await _collect_response(
            middleware,
            scope,
            body=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode(),
        )
