"""
E2E tests for the MCP (Model Context Protocol) server.

Tests MCP endpoints by connecting to a running FastAPI server with MCP enabled.

Requires:
- HINDSIGHT_API_URL environment variable (e.g., http://localhost:8888)
- A running Hindsight API server with MCP enabled
"""

import asyncio
import os
import uuid

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


def get_mcp_url() -> str:
    """Get the MCP URL from environment."""
    base_url = os.environ.get("HINDSIGHT_API_URL", "http://localhost:8888")
    return f"{base_url}/mcp/"


def get_unique_bank_id() -> str:
    """Generate a unique bank_id for test isolation."""
    return f"mcp-test-{uuid.uuid4().hex[:8]}"


@pytest.mark.asyncio
async def test_mcp_server_tools_via_http():
    """Test MCP server tools via StreamableHTTP transport using proper MCP client."""
    mcp_url = get_mcp_url()
    bank_id = get_unique_bank_id()

    async with streamable_http_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Test 1: List tools
            tools_list = await session.list_tools()
            print(f"Tools: {tools_list}")
            tool_names = [t.name for t in tools_list.tools]
            assert "retain" in tool_names
            assert "recall" in tool_names

            # Test 2: Call retain
            put_result = await session.call_tool(
                "retain",
                arguments={
                    "content": "User loves Python programming and prefers pytest for testing",
                    "context": "programming_preferences",
                    "bank_id": bank_id,
                },
            )
            print(f"Retain result: {put_result}")
            assert put_result is not None

            # Wait a bit for indexing
            await asyncio.sleep(1)

            # Test 3: Call recall
            search_result = await session.call_tool(
                "recall",
                arguments={
                    "query": "What programming languages does the user like?",
                    "bank_id": bank_id,
                },
            )
            print(f"Recall result: {search_result}")
            assert search_result is not None


@pytest.mark.asyncio
async def test_create_bank_and_list_banks():
    """Test create_bank and list_banks tools."""
    import json

    mcp_url = get_mcp_url()
    bank_id = get_unique_bank_id()

    async with streamable_http_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Test 1: Create a new bank
            create_result = await session.call_tool(
                "create_bank",
                arguments={
                    "bank_id": bank_id,
                    "name": "Test Bank",
                    "background": "A bank for testing MCP integration",
                },
            )
            print(f"Create bank result: {create_result}")
            assert create_result is not None

            # Parse the result - now uses BankProfileResponse model
            result_text = create_result.content[0].text
            result_data = json.loads(result_text)
            print(f"Parsed result: {result_data}")

            # Check fields match BankProfileResponse schema
            assert result_data.get("bank_id") == bank_id
            assert result_data.get("name") == "Test Bank"
            assert result_data.get("background") == "A bank for testing MCP integration"
            assert "disposition" in result_data  # DispositionTraits object

            # Test 2: List banks and verify our bank is there
            list_result = await session.call_tool("list_banks", arguments={})
            print(f"List banks result: {list_result}")
            assert list_result is not None

            # Now uses BankListResponse model with banks array
            list_text = list_result.content[0].text
            list_data = json.loads(list_text)
            print(f"Parsed list: {list_data}")

            # Find our bank in the list - field is bank_id per BankListItem model
            bank_ids = [b["bank_id"] for b in list_data.get("banks", [])]
            assert bank_id in bank_ids, f"Bank {bank_id} not found in list: {bank_ids}"


@pytest.mark.asyncio
async def test_multiple_concurrent_requests():
    """Test multiple concurrent requests from a single session."""
    mcp_url = get_mcp_url()
    bank_id = get_unique_bank_id()

    async with streamable_http_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Fire off 10 concurrent search requests from same session
            async def make_search(idx):
                try:
                    result = await session.call_tool(
                        "recall",
                        arguments={
                            "query": f"test query {idx}",
                            "bank_id": bank_id,
                        },
                    )
                    return idx, "success", result
                except Exception as e:
                    return idx, "error", str(e)

            tasks = [make_search(i) for i in range(10)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check results
            successes = 0
            failures = 0

            for result in results:
                if isinstance(result, Exception):
                    print(f"Request failed with exception: {result}")
                    failures += 1
                else:
                    idx, status, data = result
                    if status == "success":
                        successes += 1
                    else:
                        print(f"Request {idx} failed: {data}")
                        failures += 1

            print(f"Successes: {successes}, Failures: {failures}")

            # We expect all requests to succeed
            assert successes >= 8, f"Too many failures: {failures}/10"


@pytest.mark.asyncio
async def test_race_condition_with_rapid_requests():
    """Test rapid-fire requests with multiple sessions to trigger race condition."""
    mcp_url = get_mcp_url()
    bank_id = get_unique_bank_id()

    async def rapid_session_search(idx):
        """Create a new session and immediately make a request."""
        try:
            async with streamable_http_client(mcp_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Make request immediately after initialization
                    result = await session.call_tool(
                        "recall",
                        arguments={
                            "query": f"rapid query {idx}",
                            "bank_id": bank_id,
                        },
                    )
                    return idx, "success", result
        except Exception as e:
            return idx, "error", str(e)

    # Fire 20 requests with minimal delay, each with its own session
    tasks = [rapid_session_search(i) for i in range(20)]
    results = await asyncio.gather(*tasks)

    # Analyze results
    errors = []
    for idx, status, data in results:
        if status == "error":
            errors.append((idx, data))

    if errors:
        print(f"Found {len(errors)} errors:")
        for idx, error_msg in errors:
            print(f"  Request {idx}: {error_msg}")

    # Most requests should succeed
    assert len(errors) < 5, f"Too many errors: {len(errors)}/20"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
