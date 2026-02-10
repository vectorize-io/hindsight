"""Integration test for MCP endpoint routing.

This test verifies that /mcp/ and /mcp/{bank_id}/ expose different tool sets.
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
