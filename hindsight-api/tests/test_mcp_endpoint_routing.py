"""Integration test for MCP endpoint routing.

This test verifies that /mcp/ and /mcp/{bank_id}/ expose different tool sets,
and that URLs with or without trailing slashes both work (no 307 redirect).
"""

import httpx
import pytest
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client


@pytest.mark.asyncio
async def test_mcp_endpoint_routing_integration(memory):
    """Test that multi-bank and single-bank endpoints expose different tools using StreamableHTTP.

    This is a regression test for issue #317 where /mcp/{bank_id}/ was incorrectly
    exposing all tools (including list_banks) and bank_id parameters.
    """
    from hindsight_api.api import create_app

    # Create app with MCP enabled
    app = create_app(memory, mcp_api_enabled=True, initialize_memory=False)

    # Use the app's lifespan context to properly initialize MCP servers
    async with app.router.lifespan_context(app):
        # Create an HTTPX client that routes to our ASGI app
        from httpx import ASGITransport

        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client:
            # Test 1: Multi-bank endpoint /mcp/
            async with streamable_http_client("http://test/mcp/", http_client=http_client) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    multi_result = await session.list_tools()

                    multi_tools = {t.name for t in multi_result.tools}

                    # Multi-bank should have all tools including bank management and mental models
                    assert "retain" in multi_tools
                    assert "recall" in multi_tools
                    assert "reflect" in multi_tools
                    assert "list_banks" in multi_tools, "Multi-bank should expose list_banks"
                    assert "create_bank" in multi_tools, "Multi-bank should expose create_bank"
                    assert "list_mental_models" in multi_tools, "Multi-bank should expose list_mental_models"
                    assert "create_mental_model" in multi_tools, "Multi-bank should expose create_mental_model"
                    assert "get_mental_model" in multi_tools, "Multi-bank should expose get_mental_model"
                    assert "update_mental_model" in multi_tools, "Multi-bank should expose update_mental_model"
                    assert "delete_mental_model" in multi_tools, "Multi-bank should expose delete_mental_model"
                    assert "refresh_mental_model" in multi_tools, "Multi-bank should expose refresh_mental_model"

                    # Multi-bank retain should have bank_id parameter
                    retain_tool = next((t for t in multi_result.tools if t.name == "retain"), None)
                    assert retain_tool is not None
                    multi_params = set(retain_tool.inputSchema.get("properties", {}).keys())
                    assert "bank_id" in multi_params, "Multi-bank retain should have bank_id parameter"

            # Test 2: Single-bank endpoint /mcp/test-bank/
            async with streamable_http_client("http://test/mcp/test-bank/", http_client=http_client) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    single_result = await session.list_tools()

                    single_tools = {t.name for t in single_result.tools}

                    # Single-bank should have scoped tools including mental models (no bank management)
                    assert "retain" in single_tools
                    assert "recall" in single_tools
                    assert "reflect" in single_tools
                    assert "list_mental_models" in single_tools, "Single-bank should expose list_mental_models"
                    assert "create_mental_model" in single_tools, "Single-bank should expose create_mental_model"
                    assert "list_banks" not in single_tools, "Single-bank should NOT expose list_banks"
                    assert "create_bank" not in single_tools, "Single-bank should NOT expose create_bank"

                    # Single-bank retain should NOT have bank_id parameter
                    retain_tool = next((t for t in single_result.tools if t.name == "retain"), None)
                    assert retain_tool is not None
                    single_params = set(retain_tool.inputSchema.get("properties", {}).keys())
                    assert "bank_id" not in single_params, "Single-bank retain should NOT have bank_id parameter"


@pytest.mark.asyncio
async def test_mcp_no_trailing_slash_works(memory):
    """Test that /mcp (no trailing slash) discovers tools without 307 redirect.

    Starlette's Mount class redirects /mcp to /mcp/ with a 307 Temporary Redirect.
    Many MCP clients don't follow POST redirects, causing 0 tools to be discovered.
    MCPMiddleware wraps the app directly (no Mount), so the redirect never happens.
    """
    from hindsight_api.api import create_app

    app = create_app(memory, mcp_api_enabled=True, initialize_memory=False)

    async with app.router.lifespan_context(app):
        from httpx import ASGITransport

        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client:
            # /mcp (no slash) should work the same as /mcp/
            async with streamable_http_client("http://test/mcp", http_client=http_client) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.list_tools()

                    tools = {t.name for t in result.tools}
                    assert len(tools) >= 11, f"Expected at least 11 tools from /mcp, got {len(tools)}: {tools}"
                    assert "retain" in tools
                    assert "recall" in tools
                    assert "list_banks" in tools

            # /mcp/my-bank (single-bank, no slash) should also work
            async with streamable_http_client("http://test/mcp/my-bank", http_client=http_client) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.list_tools()

                    tools = {t.name for t in result.tools}
                    assert "retain" in tools
                    assert "list_banks" not in tools, "Single-bank /mcp/my-bank should NOT expose list_banks"


@pytest.mark.asyncio
async def test_mcp_tool_execution_through_client(memory):
    """Test that tools can be called (not just discovered) through the MCP client.

    This verifies the full pipeline: HTTP → middleware → FastMCP → tool → engine → response.
    Previous tests only checked tool discovery (list_tools), not actual execution.
    """
    from httpx import ASGITransport

    from hindsight_api.api import create_app

    app = create_app(memory, mcp_api_enabled=True, initialize_memory=False)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client:
            async with streamable_http_client("http://test/mcp/", http_client=http_client) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    # Execute list_banks tool
                    result = await session.call_tool("list_banks", arguments={})
                    assert result is not None
                    assert len(result.content) > 0
                    # The result text should be valid JSON with a "banks" key
                    import json

                    response_text = result.content[0].text
                    parsed = json.loads(response_text)
                    assert "banks" in parsed


@pytest.mark.asyncio
async def test_mcp_mental_model_validation_through_client(memory):
    """Test that input validation works through the real MCP transport.

    Verifies that invalid inputs return error messages without crashing,
    and that the engine is never called with invalid data.
    """
    from httpx import ASGITransport

    from hindsight_api.api import create_app

    app = create_app(memory, mcp_api_enabled=True, initialize_memory=False)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client:
            async with streamable_http_client("http://test/mcp/", http_client=http_client) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    # Test: empty name should return validation error
                    import json

                    result = await session.call_tool(
                        "create_mental_model",
                        arguments={"name": "", "source_query": "test query"},
                    )
                    assert result is not None
                    parsed = json.loads(result.content[0].text)
                    assert "error" in parsed
                    assert "name cannot be empty" in parsed["error"]

                    # Test: max_tokens out of range should return validation error
                    result = await session.call_tool(
                        "create_mental_model",
                        arguments={"name": "Test", "source_query": "test query", "max_tokens": 0},
                    )
                    parsed = json.loads(result.content[0].text)
                    assert "error" in parsed
                    assert "max_tokens must be between 256 and 8192" in parsed["error"]


@pytest.mark.asyncio
async def test_mcp_bank_named_sse_routes_to_single_bank(memory):
    """Test that a bank named 'sse' routes to single-bank mode.

    Regression test: the old MCP_ENDPOINTS blocklist prevented banks named 'sse'
    or 'messages' from being accessed via path routing. They fell through to
    multi-bank mode instead.
    """
    from httpx import ASGITransport

    from hindsight_api.api import create_app

    app = create_app(memory, mcp_api_enabled=True, initialize_memory=False)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client:
            async with streamable_http_client("http://test/mcp/sse/", http_client=http_client) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    tools = {t.name for t in result.tools}

                    # Should be single-bank mode (no bank management tools)
                    assert "retain" in tools
                    assert "recall" in tools
                    assert "list_banks" not in tools, "Bank 'sse' should route to single-bank mode"
                    assert "create_bank" not in tools

                    # retain should NOT have bank_id parameter (single-bank mode)
                    retain_tool = next(t for t in result.tools if t.name == "retain")
                    params = set(retain_tool.inputSchema.get("properties", {}).keys())
                    assert "bank_id" not in params


@pytest.mark.asyncio
async def test_mcp_bank_named_messages_routes_to_single_bank(memory):
    """Test that a bank named 'messages' routes to single-bank mode.

    Same regression test as test_mcp_bank_named_sse_routes_to_single_bank but for 'messages'.
    """
    from httpx import ASGITransport

    from hindsight_api.api import create_app

    app = create_app(memory, mcp_api_enabled=True, initialize_memory=False)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client:
            async with streamable_http_client("http://test/mcp/messages/", http_client=http_client) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    tools = {t.name for t in result.tools}

                    assert "retain" in tools
                    assert "list_banks" not in tools, "Bank 'messages' should route to single-bank mode"
