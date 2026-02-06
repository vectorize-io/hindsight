"""Integration test for MCP endpoint routing.

This test verifies that /mcp/ and /mcp/{bank_id}/ expose different tool sets.
"""

import pytest
import json
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_mcp_endpoint_routing_integration(memory):
    """Test that multi-bank and single-bank endpoints expose different tools.

    This is a regression test for issue #317 where /mcp/{bank_id}/ was incorrectly
    exposing all tools (including list_banks) and bank_id parameters.
    """
    from hindsight_api.api import create_app

    # Create app with MCP enabled
    app = create_app(memory, mcp_api_enabled=True, initialize_memory=False)

    # Use the app's lifespan context to properly initialize MCP servers
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:

            # Test 1: Multi-bank endpoint /mcp/
            response = await client.post(
                "/mcp/",
                json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
                headers={"Accept": "application/json, text/event-stream"},
            )

            assert response.status_code == 200, f"Multi-bank endpoint failed: {response.text[:200]}"

            # Parse SSE response
            multi_data = _parse_sse_response(response.text)
            multi_tools = {t["name"] for t in multi_data.get("result", {}).get("tools", [])}

            # Multi-bank should have all tools including bank management
            assert "retain" in multi_tools
            assert "recall" in multi_tools
            assert "reflect" in multi_tools
            assert "list_banks" in multi_tools, "Multi-bank should expose list_banks"
            assert "create_bank" in multi_tools, "Multi-bank should expose create_bank"

            # Multi-bank retain should have bank_id parameter
            retain_tool = next((t for t in multi_data["result"]["tools"] if t["name"] == "retain"), None)
            assert retain_tool is not None
            multi_params = set(retain_tool["inputSchema"]["properties"].keys())
            assert "bank_id" in multi_params, "Multi-bank retain should have bank_id parameter"

            # Test 2: Single-bank endpoint /mcp/test-bank/
            response = await client.post(
                "/mcp/test-bank/",
                json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
                headers={"Accept": "application/json, text/event-stream"},
            )

            assert response.status_code == 200, f"Single-bank endpoint failed: {response.text[:200]}"

            # Parse SSE response
            single_data = _parse_sse_response(response.text)
            single_tools = {t["name"] for t in single_data.get("result", {}).get("tools", [])}

            # Single-bank should only have scoped tools (no bank management)
            assert "retain" in single_tools
            assert "recall" in single_tools
            assert "reflect" in single_tools
            assert "list_banks" not in single_tools, "Single-bank should NOT expose list_banks"
            assert "create_bank" not in single_tools, "Single-bank should NOT expose create_bank"

            # Single-bank retain should NOT have bank_id parameter
            retain_tool = next((t for t in single_data["result"]["tools"] if t["name"] == "retain"), None)
            assert retain_tool is not None
            single_params = set(retain_tool["inputSchema"]["properties"].keys())
            assert "bank_id" not in single_params, "Single-bank retain should NOT have bank_id parameter"


def _parse_sse_response(text: str) -> dict:
    """Parse SSE (Server-Sent Events) response format.

    FastMCP returns responses in SSE format:
    event: message
    data: {"jsonrpc": "2.0", ...}
    """
    for line in text.split('\n'):
        if line.startswith('data: '):
            json_str = line[6:]  # Remove "data: " prefix
            return json.loads(json_str)
    raise ValueError(f"No data line found in SSE response: {text[:200]}")
